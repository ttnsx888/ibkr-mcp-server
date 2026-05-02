"""IBKR Client with advanced trading capabilities."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union
from decimal import Decimal

from ib_async import IB, Stock, LimitOrder, StopOrder, Order, ExecutionFilter, util
from .config import settings
from .utils import rate_limit, retry_on_failure, retry_on_transient, safe_float, safe_int, ValidationError, ConnectionError as IBKRConnectionError


class IBKRClient:
    """Enhanced IBKR client with multi-account and short selling support."""
    
    def __init__(self):
        self.ib: Optional[IB] = None
        self.logger = logging.getLogger(__name__)
        
        # Connection settings
        self.host = settings.ibkr_host
        self.port = settings.ibkr_port
        self.client_id = settings.ibkr_client_id
        self.max_reconnect_attempts = settings.max_reconnect_attempts
        self.reconnect_delay = settings.reconnect_delay
        self.reconnect_attempts = 0
        
        # Account management
        self.accounts: List[str] = []
        self.current_account: Optional[str] = settings.ibkr_default_account
        
        # Connection state
        self._connected = False
        self._connecting = False
    
    @property
    def is_paper(self) -> bool:
        """Check if this is a paper trading connection."""
        return self.port in [7497, 4002]  # Common paper trading ports
    
    async def _ensure_connected(self) -> bool:
        """Ensure IBKR connection is active, reconnect if needed."""
        if self.is_connected():
            return True
        
        try:
            await self.connect()
            return self.is_connected()
        except Exception as e:
            self.logger.error(f"Failed to ensure connection: {e}")
            return False
    
    @retry_on_failure(max_attempts=3)
    async def connect(self) -> bool:
        """Establish connection and discover accounts."""
        if self._connected and self.ib and self.ib.isConnected():
            return True
        
        if self._connecting:
            # Wait for ongoing connection attempt
            while self._connecting:
                await asyncio.sleep(0.1)
            return self._connected
        
        self._connecting = True
        
        try:
            self.ib = IB()
            
            self.logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=10
            )
            
            # Setup event handlers
            self.ib.disconnectedEvent += self._on_disconnect
            self.ib.errorEvent += self._on_error

            # Wait for connection to stabilize
            await asyncio.sleep(2)
            
            # Discover accounts
            self.accounts = self.ib.managedAccounts()
            if self.accounts:
                if not self.current_account or self.current_account not in self.accounts:
                    self.current_account = self.accounts[0]
                
                self.logger.info(f"Connected to IBKR. Accounts: {self.accounts}")
                self.logger.info(f"Current account: {self.current_account}")
            else:
                self.logger.warning("No managed accounts found")
            
            self._connected = True
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            raise IBKRConnectionError(f"Connection failed: {e}")
        finally:
            self._connecting = False
    
    async def disconnect(self):
        """Clean disconnection."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            self.logger.info("IBKR disconnected")
    
    def _on_disconnect(self):
        """Handle disconnection with automatic reconnection."""
        self._connected = False
        self.logger.warning("IBKR disconnected, scheduling reconnection...")
        asyncio.create_task(self._reconnect())
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Centralized error logging."""
        # Don't log certain routine messages as errors
        if errorCode in [2104, 2106, 2158]:  # Market data warnings
            self.logger.debug(f"IBKR Info {errorCode}: {errorString}")
        else:
            self.logger.error(f"IBKR Error {errorCode}: {errorString} (reqId: {reqId})")
    
    async def _reconnect(self):
        """Background reconnection task."""
        try:
            await asyncio.sleep(self.reconnect_delay)
            await self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self.ib is not None and self.ib.isConnected()
    
    @rate_limit(calls_per_second=1.0)
    async def get_portfolio(self, account: Optional[str] = None) -> List[Dict]:
        """Get portfolio positions."""
        try:
            if not await self._ensure_connected():
                raise ConnectionError("Not connected to IBKR")
            
            account = account or self.current_account
            
            positions = await self.ib.reqPositionsAsync()
            
            portfolio = []
            for pos in positions:
                if not account or pos.account == account:
                    portfolio.append(self._serialize_position(pos))
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Portfolio request failed: {e}")
            raise RuntimeError(f"IBKR API error: {str(e)}")
    
    @rate_limit(calls_per_second=1.0)
    async def get_account_summary(self, account: Optional[str] = None) -> List[Dict]:
        """Get account summary."""
        try:
            if not await self._ensure_connected():
                raise ConnectionError("Not connected to IBKR")
            
            account = account or self.current_account or "All"
            
            summary_tags = [
                'TotalCashValue', 'NetLiquidation', 'UnrealizedPnL', 'RealizedPnL',
                'GrossPositionValue', 'BuyingPower', 'EquityWithLoanValue',
                'PreviousDayEquityWithLoanValue', 'FullInitMarginReq', 'FullMaintMarginReq'
            ]
            
            # New ib_async API: accountSummaryAsync() handles the underlying
            # reqAccountSummaryAsync() call and returns the filtered values.
            # Calling the sync accountSummary() deadlocks inside a running loop.
            acct = account if account and account != "All" else ""
            account_values = await self.ib.accountSummaryAsync(acct)
            filtered = [av for av in account_values if av.tag in summary_tags]
            return [self._serialize_account_value(av) for av in filtered]
            
        except Exception as e:
            self.logger.error(f"Account summary request failed: {e}")
            raise RuntimeError(f"IBKR API error: {str(e)}")
    
    @rate_limit(calls_per_second=0.5)
    async def get_shortable_shares(self, symbol: str, account: str = None) -> Dict:
        """Get short selling information for a symbol."""
        try:
            if not await self._ensure_connected():
                raise ConnectionError("Not connected to IBKR")
            
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Qualify the contract
            qualified_contracts = await self.ib.reqContractDetailsAsync(contract)
            if not qualified_contracts:
                return {"error": "Contract not found"}
            
            qualified_contract = qualified_contracts[0].contract
            
            # Request shortable shares
            shortable_shares = await self.ib.reqShortableSharesAsync(qualified_contract)
            
            # Get current market data
            ticker = self.ib.reqMktData(qualified_contract, '', False, False)
            await asyncio.sleep(1.5)  # Wait for market data
            
            result = {
                "symbol": symbol,
                "shortable_shares": shortable_shares if shortable_shares != -1 else "Unlimited",
                "current_price": safe_float(ticker.last or ticker.close),
                "bid": safe_float(ticker.bid),
                "ask": safe_float(ticker.ask),
                "contract_id": qualified_contract.conId
            }
            
            # Clean up ticker
            self.ib.cancelMktData(qualified_contract)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting shortable shares for {symbol}: {e}")
            return {"error": str(e)}

    @retry_on_failure(max_attempts=2)
    async def get_margin_requirements(self, symbol: str, account: str = None) -> Dict:
        """Get margin requirements for a symbol."""
        try:
            if not await self._ensure_connected():
                raise ConnectionError("Not connected to IBKR")
                
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync([contract])
            
            if not contract.conId:
                return {"error": f"Invalid symbol: {symbol}"}
            
            # Get margin requirements - simplified for now
            # Note: IBKR API doesn't provide direct margin requirements
            # This would typically require additional market data subscriptions
            margin_info = {
                "symbol": symbol,
                "contract_id": contract.conId,
                "exchange": contract.exchange,
                "margin_requirement": "Market data subscription required",
                "note": "Use TWS for detailed margin calculations"
            }
            
            return margin_info
            
        except Exception as e:
            self.logger.error(f"Error getting margin info for {symbol}: {e}")
            return {"error": str(e)}

    async def short_selling_analysis(self, symbols: List[str], account: str = None) -> Dict:
        """Complete short selling analysis for multiple symbols."""
        try:
            if not await self._ensure_connected():
                raise ConnectionError("Not connected to IBKR")
            
            analysis = {
                "account": account or self.current_account,
                "symbols_analyzed": symbols,
                "shortable_data": {},
                "margin_data": {},
                "summary": {
                    "total_symbols": len(symbols),
                    "shortable_count": 0,
                    "errors": []
                }
            }
            
            # Get shortable shares data
            for symbol in symbols:
                try:
                    shortable_info = await self.get_shortable_shares(symbol, account)
                    analysis["shortable_data"][symbol] = shortable_info
                    
                    if "error" not in shortable_info:
                        analysis["summary"]["shortable_count"] += 1
                except Exception as e:
                    analysis["summary"]["errors"].append(f"{symbol}: {str(e)}")
            
            # Get margin requirements
            for symbol in symbols:
                try:
                    margin_info = await self.get_margin_requirements(symbol, account)
                    analysis["margin_data"][symbol] = margin_info
                except Exception as e:
                    analysis["summary"]["errors"].append(f"{symbol} margin: {str(e)}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in short selling analysis: {e}")
            return {"error": str(e)}
    
    async def switch_account(self, account_id: str) -> Dict:
        """Switch to a different IBKR account."""
        try:
            if account_id not in self.accounts:
                self.logger.error(f"Account {account_id} not found. Available: {self.accounts}")
                return {
                    "success": False,
                    "message": f"Account {account_id} not found",
                    "current_account": self.current_account,
                    "available_accounts": self.accounts
                }
            
            self.current_account = account_id
            self.logger.info(f"Switched to account: {account_id}")
            
            return {
                "success": True,
                "message": f"Switched to account: {account_id}",
                "current_account": self.current_account,
                "available_accounts": self.accounts
            }
            
        except Exception as e:
            self.logger.error(f"Error switching account: {e}")
            return {"success": False, "error": str(e)}

    async def get_accounts(self) -> Dict[str, Union[str, List[str]]]:
        """Get available accounts information."""
        try:
            if not await self._ensure_connected():
                await self.connect()
            
            return {
                "current_account": self.current_account,
                "available_accounts": self.accounts,
                "connected": self.is_connected(),
                "paper_trading": self.is_paper
            }
            
        except Exception as e:
            self.logger.error(f"Error getting accounts: {e}")
            return {"error": str(e)}
    
    @rate_limit(calls_per_second=1.0)
    async def get_quote(self, symbol: str) -> Dict:
        """Quote for a symbol (last, bid, ask, close).

        Tries live/delayed streaming first. Falls back to historical daily
        bars for paper accounts that lack streaming market data subscription
        (common case — emits error 10089).
        """
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        contract = Stock(symbol.upper(), 'SMART', 'USD')
        qualified = await self.ib.reqContractDetailsAsync(contract)
        if not qualified:
            return {"symbol": symbol, "error": "Contract not found"}
        qualified_contract = qualified[0].contract

        def _valid(v):
            return v is not None and not (isinstance(v, float) and v != v) and v > 0

        # Try live streaming first (mode 1 = default, live).
        async def _stream(mode_label: str) -> tuple:
            ticker = self.ib.reqMktData(qualified_contract, '', False, False)
            for _ in range(6):
                await asyncio.sleep(0.5)
                if any(_valid(v) for v in (ticker.last, ticker.close, ticker.bid, ticker.ask)):
                    break
            try:
                self.ib.cancelMktData(qualified_contract)
            except Exception:
                pass
            return (
                safe_float(ticker.last) if _valid(ticker.last) else 0.0,
                safe_float(ticker.bid) if _valid(ticker.bid) else 0.0,
                safe_float(ticker.ask) if _valid(ticker.ask) else 0.0,
                safe_float(ticker.close) if _valid(ticker.close) else 0.0,
                mode_label,
            )

        try:
            self.ib.reqMarketDataType(1)  # live
        except Exception:
            pass
        last, bid, ask, close, source = await _stream("live")

        # If live returned nothing (no subscription or 10089), try delayed.
        if not any((last, bid, ask, close)):
            try:
                self.ib.reqMarketDataType(3)  # delayed
            except Exception:
                pass
            last, bid, ask, close, source = await _stream("delayed")

        # If live+delayed both empty, try frozen — returns last known quote
        # regardless of session state (intended for closed-market lookups).
        if not any((last, bid, ask, close)):
            try:
                self.ib.reqMarketDataType(2)  # frozen
            except Exception:
                pass
            last, bid, ask, close, source = await _stream("frozen")

        # Fallback: historical daily bars — works on paper without subscription.
        if not any((last, bid, ask, close)):
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    qualified_contract,
                    endDateTime='',
                    durationStr='2 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                )
                if bars:
                    last_bar = bars[-1]
                    last = safe_float(last_bar.close)
                    close = safe_float(last_bar.close)
                    source = f"historical (bar {last_bar.date})"
            except Exception as e:
                self.logger.warning(f"Historical fallback failed for {symbol}: {e}")

        return {
            "symbol": symbol.upper(),
            "last": last,
            "bid": bid,
            "ask": ask,
            "close": close,
            "contract_id": qualified_contract.conId,
            "source": source,
        }

    @rate_limit(calls_per_second=0.2)
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batched quotes for multiple symbols.

        Qualifies all contracts in one `qualifyContractsAsync` call, fans out
        `reqMktData` concurrently (Semaphore(8)), and falls back per batch
        through live → delayed → frozen, then per-symbol historical.
        Returns `{symbol: {last, bid, ask, close, ...} | {error: ...}}`.

        Per-symbol failures don't crash the batch; callers see an `error` key
        on the affected symbol and the rest of the batch returns normally.
        """
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        symbols = [s.upper() for s in symbols]
        results: Dict[str, Dict] = {}

        # Qualify all contracts in one round-trip. Unqualified ones come back
        # with conId == 0 — surface those as per-symbol errors.
        contracts = [Stock(s, 'SMART', 'USD') for s in symbols]
        try:
            await self.ib.qualifyContractsAsync(*contracts)
        except Exception as e:
            self.logger.warning(f"qualifyContractsAsync batch failed: {e}")

        sym_to_contract: Dict[str, object] = {}
        pending: List[str] = []
        for sym, c in zip(symbols, contracts):
            if not getattr(c, "conId", 0):
                results[sym] = {"symbol": sym, "error": "Contract not found"}
            else:
                sym_to_contract[sym] = c
                pending.append(sym)

        if not pending:
            return results

        def _valid(v):
            return v is not None and not (isinstance(v, float) and v != v) and v > 0

        sem = asyncio.Semaphore(8)

        async def _stream_one(sym: str, mode_label: str):
            async with sem:
                contract = sym_to_contract[sym]
                try:
                    ticker = self.ib.reqMktData(contract, '', False, False)
                    for _ in range(6):
                        await asyncio.sleep(0.5)
                        if any(_valid(v) for v in (ticker.last, ticker.close,
                                                   ticker.bid, ticker.ask)):
                            break
                    try:
                        self.ib.cancelMktData(contract)
                    except Exception:
                        pass
                    return (
                        sym,
                        safe_float(ticker.last) if _valid(ticker.last) else 0.0,
                        safe_float(ticker.bid) if _valid(ticker.bid) else 0.0,
                        safe_float(ticker.ask) if _valid(ticker.ask) else 0.0,
                        safe_float(ticker.close) if _valid(ticker.close) else 0.0,
                        mode_label,
                    )
                except Exception as e:
                    return (sym, 0.0, 0.0, 0.0, 0.0, f"error:{e}")

        remaining = list(pending)
        for mode_num, mode_label in [(1, "live"), (3, "delayed"), (2, "frozen")]:
            if not remaining:
                break
            try:
                self.ib.reqMarketDataType(mode_num)
            except Exception:
                pass
            batch = await asyncio.gather(
                *[_stream_one(s, mode_label) for s in remaining],
                return_exceptions=True,
            )
            next_remaining: List[str] = []
            for item in batch:
                if isinstance(item, Exception):
                    continue
                sym, last, bid, ask, close, source = item
                if any((last, bid, ask, close)):
                    contract = sym_to_contract[sym]
                    results[sym] = {
                        "symbol": sym,
                        "last": last, "bid": bid, "ask": ask, "close": close,
                        "contract_id": contract.conId,
                        "source": source,
                    }
                else:
                    next_remaining.append(sym)
            remaining = next_remaining

        # Historical fallback — sequential is fine, this is the rare path.
        for sym in remaining:
            contract = sym_to_contract[sym]
            filled = False
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime='', durationStr='2 D', barSizeSetting='1 day',
                    whatToShow='TRADES', useRTH=True, formatDate=1,
                )
                if bars:
                    last_bar = bars[-1]
                    results[sym] = {
                        "symbol": sym,
                        "last": safe_float(last_bar.close),
                        "bid": 0.0, "ask": 0.0,
                        "close": safe_float(last_bar.close),
                        "contract_id": contract.conId,
                        "source": f"historical (bar {last_bar.date})",
                    }
                    filled = True
            except Exception as e:
                self.logger.warning(f"Historical fallback failed for {sym}: {e}")
            if not filled:
                results[sym] = {
                    "symbol": sym,
                    "last": 0.0, "bid": 0.0, "ask": 0.0, "close": 0.0,
                    "contract_id": contract.conId,
                    "source": "unavailable",
                }

        return results

    @rate_limit(calls_per_second=0.5)
    @retry_on_transient(max_attempts=2, delay=5.0)
    async def place_limit_order(self, symbol: str, action: str, quantity: int,
                                limit_price: float, tif: str = "DAY",
                                outside_rth: bool = False,
                                order_ref: str = "",
                                account: Optional[str] = None) -> Dict:
        """Submit a limit order to IBKR. Does NOT stage — sends immediately.

        Audit M2: wrapped with @retry_on_transient — one retry after a 5s
        backoff if the call raises a transient (network/timeout) error.
        Non-transient errors (validation, business logic, halted symbol)
        propagate immediately — retrying them would just delay the same
        rejection.
        """
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValidationError(f"Invalid action: {action}")

        contract = Stock(symbol.upper(), 'SMART', 'USD')
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified or not contract.conId:
            raise ValidationError(f"Could not qualify contract for {symbol}")

        order = LimitOrder(
            action=action,
            totalQuantity=int(quantity),
            lmtPrice=float(limit_price),
            tif=tif.upper(),
            outsideRth=bool(outside_rth),
            orderRef=str(order_ref or ""),
        )
        if account or self.current_account:
            order.account = account or self.current_account

        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(1.0)  # let IBKR echo initial status

        return {
            "order_id": trade.order.orderId,
            "perm_id": trade.order.permId,
            "symbol": symbol.upper(),
            "action": action,
            "quantity": int(quantity),
            "limit_price": float(limit_price),
            "tif": tif.upper(),
            "outside_rth": bool(outside_rth),
            "status": trade.orderStatus.status,
            "filled": safe_float(trade.orderStatus.filled),
            "remaining": safe_float(trade.orderStatus.remaining),
            "account": order.account,
        }

    @retry_on_transient(max_attempts=2, delay=5.0)
    async def place_stop_order(self, symbol: str, action: str, quantity: int,
                                stop_price: float, tif: str = "GTC",
                                outside_rth: bool = False,
                                order_ref: str = "",
                                oca_group: Optional[str] = None,
                                oca_type: int = 0,
                                parent_id: Optional[int] = None,
                                transmit: bool = True,
                                account: Optional[str] = None) -> Dict:
        """Submit a STP order (`StopOrder`) to IBKR. Used for fail-safe stops
        and BE-stops in the swing-monitor flow. Honors OCA grouping when
        `oca_group` is set."""
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValidationError(f"Invalid action: {action}")
        if stop_price <= 0:
            raise ValidationError("stop_price must be positive")

        contract = Stock(symbol.upper(), 'SMART', 'USD')
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified or not contract.conId:
            raise ValidationError(f"Could not qualify contract for {symbol}")

        order = StopOrder(
            action=action,
            totalQuantity=int(quantity),
            stopPrice=float(stop_price),
            tif=tif.upper(),
            outsideRth=bool(outside_rth),
            orderRef=str(order_ref or ""),
        )
        order.transmit = bool(transmit)
        if oca_group:
            order.ocaGroup = oca_group
            order.ocaType = int(oca_type or 2)   # default REDUCE_WITH_BLOCK
        if parent_id:
            order.parentId = int(parent_id)
        if account or self.current_account:
            order.account = account or self.current_account

        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(1.0)

        return {
            "order_id":    trade.order.orderId,
            "perm_id":     trade.order.permId,
            "parent_id":   int(getattr(trade.order, "parentId", 0) or 0),
            "symbol":      symbol.upper(),
            "action":      action,
            "quantity":    int(quantity),
            "stop_price":  float(stop_price),
            "order_type":  trade.order.orderType,
            "oca_group":   getattr(trade.order, "ocaGroup", "") or None,
            "oca_type":    int(getattr(trade.order, "ocaType", 0) or 0),
            "tif":         tif.upper(),
            "outside_rth": bool(outside_rth),
            "transmit":    bool(transmit),
            "status":      trade.orderStatus.status,
            "filled":      safe_float(trade.orderStatus.filled),
            "remaining":   safe_float(trade.orderStatus.remaining),
            "account":     order.account,
        }

    @retry_on_transient(max_attempts=2, delay=5.0)
    async def place_bracket_order(self, symbol: str, parent_action: str,
                                   parent_quantity: int, parent_limit_price: float,
                                   children: List[Dict],
                                   parent_tif: str = "GTC",
                                   parent_outside_rth: bool = False,
                                   order_ref: str = "",
                                   account: Optional[str] = None) -> Dict:
        """Place a bracket: one parent LMT entry + N OCA-linked SELL children.

        `children` shape:
          [{"order_type": "LMT" | "STP",
            "action": "SELL",
            "quantity": <int>,
            "limit_price": <float>,    # if LMT
            "stop_price":  <float>,    # if STP
            "tif": <str> default "GTC",
            "outside_rth": <bool>,
            "oca_type": <int> default 2 (REDUCE_WITH_BLOCK),
            "tag": <str> optional override to orderRef}, ...]

        IBKR semantics: parent + children are linked via `parentId` so children
        ONLY activate after parent fills. Children share an `ocaGroup`; with
        `ocaType=2` (REDUCE_WITH_BLOCK), when one child fills, the others'
        quantities reduce by the filled amount automatically — exactly the
        scale-out + protective-stop pattern the swing strategy needs.

        Atomicity: parent + intermediate children are placed with
        `transmit=False`; the final child carries `transmit=True`, which tells
        IBKR to commit the entire bracket as one unit.
        """
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")
        parent_action = parent_action.upper()
        if parent_action not in ("BUY", "SELL"):
            raise ValidationError(f"Invalid parent_action: {parent_action}")
        if parent_limit_price <= 0:
            raise ValidationError("parent_limit_price must be positive")
        if not children:
            raise ValidationError("bracket must have at least one child")

        contract = Stock(symbol.upper(), 'SMART', 'USD')
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified or not contract.conId:
            raise ValidationError(f"Could not qualify contract for {symbol}")

        oca_group = f"BRK_{symbol.upper()}_{int(time.time() * 1000)}"

        # Parent: LMT, transmit=False (children will activate it on the last leg)
        parent = LimitOrder(
            action=parent_action,
            totalQuantity=int(parent_quantity),
            lmtPrice=float(parent_limit_price),
            tif=parent_tif.upper(),
            outsideRth=bool(parent_outside_rth),
            orderRef=str(order_ref or ""),
        )
        parent.transmit = False
        if account or self.current_account:
            parent.account = account or self.current_account

        parent_trade = self.ib.placeOrder(contract, parent)
        # Tiny settle so IBKR assigns the orderId we'll reference.
        await asyncio.sleep(0.3)
        parent_order_id = parent_trade.order.orderId
        if not parent_order_id:
            raise IBKRConnectionError("IBKR did not assign orderId to parent")

        child_results = []
        n = len(children)
        try:
            for i, c in enumerate(children):
                is_last = (i == n - 1)
                ot = c.get("order_type", "LMT").upper()
                action = c.get("action", "SELL").upper()
                qty = int(c["quantity"])
                tif = c.get("tif", "GTC").upper()
                outside_rth = bool(c.get("outside_rth", False))
                oca_type = int(c.get("oca_type", 2))
                tag = c.get("tag") or f"{order_ref}_C{i}"

                if ot == "LMT":
                    child = LimitOrder(
                        action=action, totalQuantity=qty,
                        lmtPrice=float(c["limit_price"]),
                        tif=tif, outsideRth=outside_rth,
                        orderRef=str(tag),
                    )
                elif ot == "STP":
                    child = StopOrder(
                        action=action, totalQuantity=qty,
                        stopPrice=float(c["stop_price"]),
                        tif=tif, outsideRth=outside_rth,
                        orderRef=str(tag),
                    )
                else:
                    raise ValidationError(
                        f"bracket child[{i}]: unsupported order_type {ot!r} (LMT or STP only)"
                    )
                child.parentId = parent_order_id
                child.ocaGroup = oca_group
                child.ocaType = oca_type
                child.transmit = is_last
                if account or self.current_account:
                    child.account = account or self.current_account

                trade = self.ib.placeOrder(contract, child)
                child_results.append({
                    "order_id":   trade.order.orderId,
                    "perm_id":    trade.order.permId,
                    "order_type": trade.order.orderType,
                    "action":     action,
                    "quantity":   qty,
                    "limit_price": safe_float(getattr(trade.order, "lmtPrice", 0)),
                    "stop_price":  safe_float(getattr(trade.order, "auxPrice", 0)),
                    "oca_group":   oca_group,
                    "oca_type":    oca_type,
                    "tif":         tif,
                    "outside_rth": outside_rth,
                    "transmit":    is_last,
                    "tag":         tag,
                    "status":      trade.orderStatus.status,
                })
        except Exception:
            # Atomicity: if any child fails, cancel the parent (transmit=False
            # means it never reached the market — cancel just removes the
            # provisional entry from IBKR's queue).
            try:
                self.ib.cancelOrder(parent_trade.order)
                await asyncio.sleep(0.3)
            except Exception:
                pass
            raise

        await asyncio.sleep(0.5)
        return {
            "parent": {
                "order_id":    parent_trade.order.orderId,
                "perm_id":     parent_trade.order.permId,
                "symbol":      symbol.upper(),
                "action":      parent_action,
                "quantity":    int(parent_quantity),
                "limit_price": float(parent_limit_price),
                "order_type":  parent_trade.order.orderType,
                "tif":         parent_tif.upper(),
                "outside_rth": bool(parent_outside_rth),
                "status":      parent_trade.orderStatus.status,
            },
            "children":  child_results,
            "oca_group": oca_group,
            "account":   parent.account,
        }

    @retry_on_transient(max_attempts=2, delay=5.0)
    async def modify_order(self, order_id: int,
                            quantity: Optional[int] = None,
                            limit_price: Optional[float] = None,
                            stop_price: Optional[float] = None) -> Dict:
        """Modify a live order in place (no cancel + restage). Used by the
        swing-monitor's post-T1 BE-stop transition: instead of cancel-then-stage
        (which has a tiny no-coverage gap), bump the failsafe STP's price up
        to entry and reduce qty.

        Only fields that are provided are modified.
        """
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        oid = int(order_id)
        target = None
        for trade in self.ib.trades():
            if trade.order.orderId == oid:
                target = trade
                break
        if target is None:
            return {"order_id": oid, "modified": False, "error": "order not found"}

        original = {
            "quantity":    safe_float(target.order.totalQuantity),
            "limit_price": safe_float(getattr(target.order, "lmtPrice", 0)),
            "stop_price":  safe_float(getattr(target.order, "auxPrice", 0)),
        }
        if quantity is not None:
            target.order.totalQuantity = int(quantity)
        if limit_price is not None:
            target.order.lmtPrice = float(limit_price)
        if stop_price is not None:
            target.order.auxPrice = float(stop_price)
        target.order.transmit = True

        # Re-submit modified order with the same orderId — IBKR replaces it.
        trade = self.ib.placeOrder(target.contract, target.order)
        await asyncio.sleep(0.5)
        return {
            "order_id":   oid,
            "perm_id":    trade.order.permId,
            "symbol":     target.contract.symbol,
            "modified":   True,
            "original":   original,
            "new": {
                "quantity":    safe_float(trade.order.totalQuantity),
                "limit_price": safe_float(getattr(trade.order, "lmtPrice", 0)),
                "stop_price":  safe_float(getattr(trade.order, "auxPrice", 0)),
            },
            "status":     trade.orderStatus.status,
        }

    @rate_limit(calls_per_second=1.0)
    async def get_open_trades(self) -> List[Dict]:
        """Return all currently-open trades."""
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        trades = self.ib.openTrades()
        out = []
        for t in trades:
            order_ref = (getattr(t.order, "orderRef", "") or "").strip()
            # STP / STP_LMT use auxPrice for the stop trigger price.
            stop_price = safe_float(getattr(t.order, "auxPrice", 0))
            ot = t.order.orderType
            out.append({
                "order_id":   t.order.orderId,
                "perm_id":    t.order.permId,
                "parent_id":  int(getattr(t.order, "parentId", 0) or 0),
                "symbol":     t.contract.symbol,
                "action":     t.order.action,
                "quantity":   safe_float(t.order.totalQuantity),
                "limit_price": safe_float(getattr(t.order, "lmtPrice", 0)),
                "stop_price": stop_price if ot in ("STP", "STP LMT", "STP_LMT") else 0.0,
                "order_type": ot,
                "oca_group":  (getattr(t.order, "ocaGroup", "") or "") or None,
                "oca_type":   int(getattr(t.order, "ocaType", 0) or 0),
                "tif":        t.order.tif,
                "outside_rth": bool(getattr(t.order, "outsideRth", False)),
                "transmit":   bool(getattr(t.order, "transmit", True)),
                "status":     t.orderStatus.status,
                "filled":     safe_float(t.orderStatus.filled),
                "remaining":  safe_float(t.orderStatus.remaining),
                "account":    t.order.account,
                "source":     order_ref or None,
                "tag":        order_ref or None,   # alias — most callers use `tag`
            })
        return out

    @rate_limit(calls_per_second=1.0)
    async def cancel_order(self, order_id: int) -> Dict:
        """Cancel a live order by its IBKR orderId."""
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        for trade in self.ib.trades():
            if trade.order.orderId == int(order_id):
                self.ib.cancelOrder(trade.order)
                await asyncio.sleep(0.5)
                return {
                    "order_id": order_id,
                    "symbol": trade.contract.symbol,
                    "status": trade.orderStatus.status,
                    "cancelled": True,
                }
        return {"order_id": order_id, "cancelled": False, "error": "Order not found"}

    @rate_limit(calls_per_second=1.0)
    async def get_todays_fills(self, account: Optional[str] = None) -> List[Dict]:
        """Return today's executed fills from TWS.

        Uses ib.reqExecutionsAsync() which queries TWS's execution history for the
        current trading day. More reliable than filtering get_open_trades() by
        status=="Filled" — those entries roll off the live-orders window once fully
        reconciled, so any post-hoc scan (e.g. evening fund-manager run) can't see
        them. reqExecutions pulls straight from TWS's own day-log and survives
        reconnects within the same trading day.

        Args:
            account: optional account filter. If None, returns fills for all
                     accounts visible on this connection.

        Returns a list of dicts with fields aligned to the scan report's
        `filled_orders[]` schema. `source` is NOT populated here (executions don't
        carry the caller's scan/tier tag); callers enrich by joining on `order_id`
        against their own `submitted_orders[]` record if they need it.
        """
        if not await self._ensure_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        # Empty ExecutionFilter → today's executions, all accounts on the connection.
        filt = ExecutionFilter()
        if account:
            filt.acctCode = account

        fills = await self.ib.reqExecutionsAsync(filt)

        out = []
        for f in fills:
            execution = f.execution
            contract = f.contract
            comm = f.commissionReport

            # Map IBKR side codes ("BOT"/"SLD") to the action verbs the scan
            # report uses so downstream consumers don't need to translate.
            side = execution.side
            action = "BUY" if side == "BOT" else "SELL" if side == "SLD" else side

            out.append({
                "order_id": execution.orderId,
                "perm_id": execution.permId,
                "exec_id": execution.execId,
                "symbol": contract.symbol,
                "action": action,
                "quantity": safe_float(execution.shares),
                "fill_price": safe_float(execution.price),
                "avg_price": safe_float(getattr(execution, "avgPrice", 0)) or safe_float(execution.price),
                "time": execution.time.isoformat() if execution.time else None,
                "commission": safe_float(getattr(comm, "commission", 0)),
                "commission_currency": getattr(comm, "currency", ""),
                "account": execution.acctNumber,
                "exchange": execution.exchange,
            })
        return out

    def _serialize_position(self, position) -> Dict:
        """Convert Position to serializable dict."""
        return {
            "symbol": position.contract.symbol,
            "secType": position.contract.secType,
            "exchange": position.contract.exchange,
            "position": safe_float(position.position),
            "avgCost": safe_float(position.avgCost),
            "marketPrice": safe_float(getattr(position, 'marketPrice', 0)),
            "marketValue": safe_float(getattr(position, 'marketValue', 0)),
            "unrealizedPNL": safe_float(getattr(position, 'unrealizedPNL', 0)),
            "realizedPNL": safe_float(getattr(position, 'realizedPNL', 0)),
            "account": position.account
        }
    
    def _serialize_account_value(self, account_value) -> Dict:
        """Convert AccountValue to serializable dict."""
        return {
            "tag": account_value.tag,
            "value": account_value.value,
            "currency": account_value.currency,
            "account": account_value.account
        }


# Global client instance
ibkr_client = IBKRClient()
