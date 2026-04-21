"""MCP tools for IBKR functionality."""

import json
import time
from typing import Any, Sequence

from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolRequest

from dataclasses import asdict

from .client import ibkr_client
from .config import settings
from .orders import StagedOrder, staged_store
from .utils import validate_symbol, validate_symbols, IBKRError


LIVE_PORTS = {7496, 4001}
MAX_QUOTE_DRIFT = 0.30  # reject staging/confirming if limit is >30% away from last
                        # MUST match MAX_SUBMIT_DRIFT_PCT in /Users/ttang/Trader/scripts/compute_signals.py
                        # so the scanner pre-filter and this server-side gate agree.

# Per-symbol quote cache shared by stage_order / confirm_order validation.
# During a scan, a single symbol's tiers (T1/T2/T3) all validate against the
# same reference price — and confirm_order fires seconds after stage_order.
# Without this, each validation pays ~3s in reqMktData poll + 1s rate limit.
# TTL is short enough that a price move large enough to matter (>30% drift)
# can't hide inside the cache window.
_QUOTE_CACHE: dict[str, tuple[float, dict]] = {}
_QUOTE_CACHE_TTL = 30.0  # seconds


async def _cached_get_quote(symbol: str) -> dict:
    """Return a recent quote for symbol, reusing a cached one if <TTL old.

    Only caches successful lookups — error responses always re-fetch so a
    transient IBKR hiccup doesn't poison the cache for 30s.
    """
    key = symbol.upper()
    now = time.monotonic()
    entry = _QUOTE_CACHE.get(key)
    if entry is not None:
        fetched_at, cached = entry
        if now - fetched_at < _QUOTE_CACHE_TTL:
            return cached
    quote = await ibkr_client.get_quote(symbol)
    if "error" not in quote:
        _QUOTE_CACHE[key] = (now, quote)
    return quote


async def _validate_order_inputs(symbol: str, action: str, quantity: int,
                                 limit_price: float,
                                 require_quote: bool = True) -> dict:
    """Shared validation. Returns {'ok': True} or {'ok': False, 'error': ...}.

    When require_quote=False (used by stage_order), skips the quote-drift check
    if IBKR is not connected — allows offline staging. confirm_order always
    passes require_quote=True so the drift check runs at submission time.
    """
    action = action.upper()
    if action not in ("BUY", "SELL"):
        return {"ok": False, "error": f"action must be BUY or SELL (got {action!r})"}
    if quantity <= 0:
        return {"ok": False, "error": "quantity must be positive"}
    if quantity > settings.max_order_size:
        return {"ok": False,
                "error": f"quantity {quantity} exceeds MAX_ORDER_SIZE ({settings.max_order_size})"}
    if limit_price <= 0:
        return {"ok": False, "error": "limit_price must be positive"}

    # Quote sanity — reject tier prices wildly off from last.
    # Uses a short-TTL per-symbol cache so multi-tier stage_order + confirm_order
    # sequences don't each pay the full reqMktData poll cost.
    try:
        quote = await _cached_get_quote(symbol)
    except Exception:
        if require_quote:
            return {"ok": False, "error": "IBKR not connected — cannot validate quote"}
        return {"ok": True, "reference_price": None, "drift_pct": None,
                "warning": "IBKR offline — quote-drift check skipped, will re-validate at confirm time"}

    if "error" in quote:
        if require_quote:
            return {"ok": False, "error": f"quote lookup failed: {quote['error']}"}
        return {"ok": True, "reference_price": None, "drift_pct": None,
                "warning": f"quote lookup failed ({quote['error']}) — will re-validate at confirm time"}

    ref = quote.get("last") or quote.get("close") or 0
    source = quote.get("source", "unknown")
    if ref <= 0:
        # Fallback: bid/ask midpoint. Often populated after-hours for major
        # stocks even when last/close aren't, and valid for a drift check.
        bid = quote.get("bid") or 0
        ask = quote.get("ask") or 0
        if bid > 0 and ask > 0:
            ref = (bid + ask) / 2
            source = f"{source} (bid/ask mid)"
    if ref <= 0:
        if require_quote:
            return {"ok": False, "error": "could not obtain reference price for symbol"}
        return {"ok": True, "reference_price": None, "drift_pct": None,
                "warning": "no reference price available — will re-validate at confirm time"}

    drift = abs(limit_price - ref) / ref
    if drift > MAX_QUOTE_DRIFT:
        return {"ok": False,
                "error": f"limit ${limit_price:.2f} is {drift*100:.1f}% from last ${ref:.2f} "
                         f"(max {MAX_QUOTE_DRIFT*100:.0f}%, source={source}). Refusing."}
    return {"ok": True, "reference_price": ref, "drift_pct": round(drift * 100, 2),
            "reference_source": source}


# Create the server instance
server = Server("ibkr-mcp")


# Define all tools
TOOLS = [
    Tool(
        name="get_portfolio",
        description="Retrieve current portfolio positions and P&L from IBKR",
        inputSchema={
            "type": "object",
            "properties": {
                "account": {"type": "string", "description": "Account ID (optional, uses current account if not specified)"}
            },
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_account_summary", 
        description="Get account balances and key metrics from IBKR",
        inputSchema={
            "type": "object",
            "properties": {
                "account": {"type": "string", "description": "Account ID (optional, uses current account if not specified)"}
            },
            "additionalProperties": False
        }
    ),
    Tool(
        name="switch_account",
        description="Switch between IBKR accounts",
        inputSchema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account ID to switch to"}
            },
            "required": ["account_id"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_accounts",
        description="Get available IBKR accounts and current account", 
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
    ),
    Tool(
        name="check_shortable_shares",
        description="Check short selling availability for securities",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {"type": "string", "description": "Comma-separated list of symbols"},
                "account": {"type": "string", "description": "Account ID (optional, uses current account if not specified)"}
            },
            "required": ["symbols"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_margin_requirements",
        description="Get margin requirements for securities",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {"type": "string", "description": "Comma-separated list of symbols"},
                "account": {"type": "string", "description": "Account ID (optional, uses current account if not specified)"}
            },
            "required": ["symbols"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="short_selling_analysis",
        description="Complete short selling analysis: availability, margin requirements, and summary",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {"type": "string", "description": "Comma-separated list of symbols"},
                "account": {"type": "string", "description": "Account ID (optional, uses current account if not specified)"}
            },
            "required": ["symbols"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_connection_status",
        description="Check IBKR TWS/Gateway connection status and account information",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
    ),
    Tool(
        name="stage_order",
        description=("Validate and stage a limit order for later approval. Does NOT submit to IBKR. "
                     "Applies MAX_ORDER_SIZE and quote-drift safety gates. Returns a staged_id for confirm_order."),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "action": {"type": "string", "enum": ["BUY", "SELL"]},
                "quantity": {"type": "integer", "minimum": 1},
                "limit_price": {"type": "number", "exclusiveMinimum": 0},
                "tif": {"type": "string", "enum": ["DAY", "GTC", "IOC", "FOK"], "default": "DAY"},
                "outside_rth": {"type": "boolean", "default": False,
                                 "description": "Allow order to trigger/fill outside regular trading hours"},
                "source": {"type": "string", "description": "Provenance tag, e.g. 'scan 2026-04-15 AMD T1'"}
            },
            "required": ["symbol", "action", "quantity", "limit_price"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="list_staged_orders",
        description="List all staged (not yet submitted) orders. Optionally filter by symbol or source prefix.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "source_prefix": {"type": "string"}
            },
            "additionalProperties": False
        }
    ),
    Tool(
        name="confirm_order",
        description=("Submit a staged order to IBKR. Re-validates safety gates. "
                     "Refused on live-trading port if ENABLE_LIVE_TRADING=false."),
        inputSchema={
            "type": "object",
            "properties": {"staged_id": {"type": "string"}},
            "required": ["staged_id"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="cancel_staged_order",
        description="Remove a staged order without submitting it.",
        inputSchema={
            "type": "object",
            "properties": {"staged_id": {"type": "string"}},
            "required": ["staged_id"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_live_orders",
        description="List all currently-open orders on the IBKR account.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
    ),
    Tool(
        name="get_todays_fills",
        description=(
            "List today's executed fills from TWS's execution log. Reliable "
            "across reconnects within the trading day — unlike get_live_orders, "
            "which drops fully-reconciled entries. Use this in post-hoc reports "
            "(e.g. evening scan 'Today's Filled Orders' section) where fills "
            "may have rolled off the live-orders window. Pass an optional "
            "`account` to filter; otherwise returns fills for all accounts on "
            "the connection."
        ),
        inputSchema={
            "type": "object",
            "properties": {"account": {"type": "string"}},
            "additionalProperties": False
        }
    ),
    Tool(
        name="cancel_live_order",
        description="Cancel a live IBKR order by its orderId.",
        inputSchema={
            "type": "object",
            "properties": {"order_id": {"type": "integer"}},
            "required": ["order_id"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_market_quote",
        description="Snapshot quote (last, bid, ask, close) for a symbol.",
        inputSchema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False
        }
    )
]


# Register tools list handler
@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOLS


# Register tool call handler  
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Handle tool calls."""
    try:
        if name == "get_portfolio":
            account = arguments.get("account")
            positions = await ibkr_client.get_portfolio(account)
            return [TextContent(
                type="text",
                text=json.dumps(positions, indent=2)
            )]
            
        elif name == "get_account_summary":
            account = arguments.get("account")
            summary = await ibkr_client.get_account_summary(account)
            return [TextContent(
                type="text", 
                text=json.dumps(summary, indent=2)
            )]
            
        elif name == "switch_account":
            account_id = arguments["account_id"]
            result = await ibkr_client.switch_account(account_id)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        elif name == "get_accounts":
            accounts = await ibkr_client.get_accounts()
            return [TextContent(
                type="text",
                text=json.dumps(accounts, indent=2)
            )]
            
        elif name == "check_shortable_shares":
            symbols = arguments["symbols"]
            account = arguments.get("account")
            try:
                symbol_list = validate_symbols(symbols)
                results = []
                for symbol in symbol_list:
                    shortable_info = await ibkr_client.get_shortable_shares(symbol, account)
                    results.append({
                        "symbol": symbol,
                        "shortable_shares": shortable_info
                    })
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error checking shortable shares: {str(e)}"
                )]
                
        elif name == "get_margin_requirements":
            symbols = arguments["symbols"]
            account = arguments.get("account")
            try:
                symbol_list = validate_symbols(symbols)
                results = []
                for symbol in symbol_list:
                    margin_info = await ibkr_client.get_margin_requirements(symbol, account)
                    results.append({
                        "symbol": symbol,
                        "margin_requirements": margin_info
                    })
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error getting margin requirements: {str(e)}"
                )]
                
        elif name == "short_selling_analysis":
            symbols = arguments["symbols"]
            account = arguments.get("account")
            try:
                symbol_list = validate_symbols(symbols)
                analysis = await ibkr_client.short_selling_analysis(symbol_list, account)
                return [TextContent(
                    type="text",
                    text=json.dumps(analysis, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error performing short selling analysis: {str(e)}"
                )]
                
        elif name == "get_connection_status":
            # Actively probe — don't return stale cached state.
            try:
                await ibkr_client._ensure_connected()
            except Exception:
                pass
            status = {
                "connected": ibkr_client.is_connected(),
                "host": ibkr_client.host,
                "port": ibkr_client.port,
                "client_id": ibkr_client.client_id,
                "current_account": ibkr_client.current_account,
                "available_accounts": ibkr_client.accounts,
                "paper_trading": ibkr_client.is_paper
            }
            return [TextContent(
                type="text",
                text=json.dumps(status, indent=2)
            )]
        
        elif name == "stage_order":
            symbol = validate_symbol(arguments["symbol"])
            action = arguments["action"]
            quantity = int(arguments["quantity"])
            limit_price = float(arguments["limit_price"])
            tif = arguments.get("tif", "DAY")
            outside_rth = bool(arguments.get("outside_rth", False))
            source = arguments.get("source", "")

            v = await _validate_order_inputs(symbol, action, quantity, limit_price,
                                               require_quote=False)
            if not v["ok"]:
                return [TextContent(type="text", text=json.dumps({"staged": False, "error": v["error"]}))]

            order = StagedOrder.new(symbol, action, quantity, limit_price,
                                    tif=tif, source=source, outside_rth=outside_rth)
            staged_store.add(order)
            return [TextContent(type="text", text=json.dumps({
                "staged": True,
                "staged_id": order.id,
                "summary": order.summary(),
                "reference_price": v["reference_price"],
                "drift_pct": v["drift_pct"],
                "reference_source": v.get("reference_source"),
            }, indent=2))]

        elif name == "list_staged_orders":
            orders = staged_store.list(
                symbol=arguments.get("symbol"),
                source_prefix=arguments.get("source_prefix"),
            )
            return [TextContent(type="text",
                                text=json.dumps([asdict(o) for o in orders], indent=2))]

        elif name == "confirm_order":
            staged_id = arguments["staged_id"]
            order = staged_store.get(staged_id)
            if not order:
                return [TextContent(type="text",
                                    text=json.dumps({"submitted": False,
                                                     "error": f"No staged order with id {staged_id}"}))]
            if order.is_expired():
                staged_store.remove(staged_id)
                return [TextContent(type="text",
                                    text=json.dumps({"submitted": False,
                                                     "error": "Staged order expired (>7d old). Re-stage."}))]

            # Live-trading gate.
            if ibkr_client.port in LIVE_PORTS and not settings.enable_live_trading:
                return [TextContent(type="text", text=json.dumps({
                    "submitted": False,
                    "error": (f"Live port {ibkr_client.port} detected but ENABLE_LIVE_TRADING=false. "
                              "Refusing to submit. Set ENABLE_LIVE_TRADING=true in .env to allow."),
                }))]

            # Re-validate against current market (strict — IBKR must be connected)
            v = await _validate_order_inputs(order.symbol, order.action,
                                             order.quantity, order.limit_price,
                                             require_quote=True)
            if not v["ok"]:
                return [TextContent(type="text",
                                    text=json.dumps({"submitted": False, "error": v["error"]}))]

            try:
                result = await ibkr_client.place_limit_order(
                    symbol=order.symbol, action=order.action,
                    quantity=order.quantity, limit_price=order.limit_price,
                    tif=order.tif,
                    outside_rth=order.outside_rth,
                )
            except Exception as e:
                return [TextContent(type="text",
                                    text=json.dumps({"submitted": False,
                                                     "error": f"IBKR rejected order: {e}"}))]

            staged_store.remove(staged_id)
            return [TextContent(type="text", text=json.dumps({
                "submitted": True,
                "staged_id": staged_id,
                "reference_price": v["reference_price"],
                "drift_pct": v["drift_pct"],
                "reference_source": v.get("reference_source"),
                "ibkr": result,
            }, indent=2))]

        elif name == "cancel_staged_order":
            staged_id = arguments["staged_id"]
            removed = staged_store.remove(staged_id)
            return [TextContent(type="text",
                                text=json.dumps({"cancelled": removed, "staged_id": staged_id}))]

        elif name == "get_live_orders":
            trades = await ibkr_client.get_open_trades()
            return [TextContent(type="text", text=json.dumps(trades, indent=2))]

        elif name == "get_todays_fills":
            account = arguments.get("account")
            fills = await ibkr_client.get_todays_fills(account)
            return [TextContent(type="text", text=json.dumps(fills, indent=2))]

        elif name == "cancel_live_order":
            result = await ibkr_client.cancel_order(int(arguments["order_id"]))
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_market_quote":
            quote = await ibkr_client.get_quote(validate_symbol(arguments["symbol"]))
            return [TextContent(type="text", text=json.dumps(quote, indent=2))]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing tool {name}: {str(e)}"
        )]
