"""Pure, unit-testable option-order safety gates.

No IB/network calls happen here — everything the checks need (positions, open
orders, quote, whatif margin-impact) is fetched by the caller and passed in.
`check_option_order` is run once at stage_option_order time and RE-RUN at
confirm_order time against freshly-fetched state, mirroring the STK stage/
confirm gate pattern in tools.py (_validate_order_inputs / _buy_funds_gate).

Design intent (2026-09-23): options are refused by default unless a rule here
explicitly allows them. In particular:
  - BTO (buy to open — a new long option position) is refused outright.
  - STO calls are refused unless fully covered by owned shares (never naked).
  - BTC/STC are refused if they would over-close a position into the
    opposite sign (accidental long from closing a short, or vice versa).
  - Any order without a usable quote is refused unless the caller explicitly
    opts in via allow_no_quote (then it's a warning, not a refusal).
"""

from typing import Dict, List, Optional, Tuple

VALID_INTENTS = ("BTC", "STC", "STO")
_DONE_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})


def _fnum(v) -> Optional[float]:
    """Float, or None for missing/non-numeric/NaN/IBKR-UNSET-ish values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or abs(f) >= 1e9:  # NaN or IBKR's ~1.8e308 UNSET_DOUBLE sentinel
        return None
    return f


def _match_contract(row: dict, conid) -> bool:
    """True when a position/open-order row refers to the same option contract
    (string comparison — get_portfolio/get_open_trades serialize conid as str,
    tests may pass ints; both directions must match)."""
    row_conid = row.get("conid")
    if row_conid is None or conid is None:
        return False
    return str(row_conid) == str(conid)


def _is_resting(row: dict) -> bool:
    return (row.get("status") or "") not in _DONE_STATUSES


def _row_qty(row: dict) -> float:
    rem = row.get("remaining")
    if rem is None:
        rem = row.get("quantity")
    return _fnum(rem) or 0.0


def _resting_qty(open_orders: List[dict], conid, action: str) -> float:
    """Sum of remaining quantity on resting orders for this conid + action."""
    total = 0.0
    for o in open_orders or []:
        if not _match_contract(o, conid):
            continue
        if (o.get("action") or "").upper() != action.upper():
            continue
        if not _is_resting(o):
            continue
        total += _row_qty(o)
    return total


def _position_qty(positions: List[dict], conid) -> float:
    """Signed position (negative = short) for this conid, 0.0 if not held."""
    for p in positions or []:
        if _match_contract(p, conid):
            return _fnum(p.get("position")) or 0.0
    return 0.0


def _underlying_share_position(positions: List[dict], symbol: str) -> float:
    sym = (symbol or "").upper()
    for p in positions or []:
        if (p.get("secType") or "STK") == "STK" and (p.get("symbol") or "").upper() == sym:
            return _fnum(p.get("position")) or 0.0
    return 0.0


def _resting_stk_sell_shares(open_orders: List[dict], symbol: str) -> float:
    sym = (symbol or "").upper()
    total = 0.0
    for o in open_orders or []:
        if (o.get("secType") or "STK") != "STK":
            continue
        if (o.get("symbol") or "").upper() != sym:
            continue
        if (o.get("action") or "").upper() != "SELL":
            continue
        if not _is_resting(o):
            continue
        total += _row_qty(o)
    return total


def _short_call_contracts(positions: List[dict], symbol: str) -> float:
    """Existing short call *contracts* (positive count) on this underlying."""
    sym = (symbol or "").upper()
    total = 0.0
    for p in positions or []:
        if (p.get("secType") or "") != "OPT":
            continue
        if (p.get("symbol") or "").upper() != sym:
            continue
        if (p.get("right") or "").upper() != "C":
            continue
        pos = _fnum(p.get("position")) or 0.0
        if pos < 0:
            total += -pos
    return total


def _resting_sto_call_contracts(open_orders: List[dict], symbol: str) -> float:
    sym = (symbol or "").upper()
    total = 0.0
    for o in open_orders or []:
        if (o.get("secType") or "") != "OPT":
            continue
        if (o.get("symbol") or "").upper() != sym:
            continue
        if (o.get("right") or "").upper() != "C":
            continue
        if (o.get("action") or "").upper() != "SELL":
            continue
        if not _is_resting(o):
            continue
        total += _row_qty(o)
    return total


def check_option_order(
    intent: str,
    action: str,
    qty: int,
    limit_price: float,
    contract: Dict,
    positions: List[Dict],
    open_orders: List[Dict],
    quote: Optional[Dict],
    whatif: Optional[Dict],
    funds: Optional[Dict],
    allow_no_quote: bool = False,
    max_contracts: int = 10,
) -> Tuple[bool, List[str], List[str]]:
    """Evaluate every safety rule for one option order. Returns
    (ok, reasons, warnings) — `ok` is False iff `reasons` is non-empty.

    `contract` = {conid, right, strike, multiplier, symbol}.
    `positions` = list of serialized positions (client._serialize_position shape).
    `open_orders` = list of serialized open orders (client.get_open_trades shape).
    `quote` = client.get_option_quote(...) result, or None.
    `whatif` = client.whatif_option_order(...) result, or None.
    `funds` = {tag: value} account-summary snapshot (e.g. AvailableFunds).
    """
    reasons: List[str] = []
    warnings: List[str] = []
    funds = funds or {}
    contract = contract or {}

    intent_u = (intent or "").upper()
    action_u = (action or "").upper()
    right = (contract.get("right") or "").upper()
    symbol = contract.get("symbol") or ""
    conid = contract.get("conid")
    multiplier = _fnum(contract.get("multiplier")) or 100.0

    # --- basic shape — bail before intent-specific logic if malformed ---
    if intent_u not in VALID_INTENTS:
        if intent_u == "BTO":
            reasons.append(
                "BTO (buy to open) is refused — opening a new long option "
                "position is not supported.")
        else:
            reasons.append(f"intent must be one of {VALID_INTENTS}, got {intent!r}")
    if action_u not in ("BUY", "SELL"):
        reasons.append(f"action must be BUY or SELL, got {action!r}")

    qty_i = 0
    try:
        qty_i = int(qty)
    except (TypeError, ValueError):
        reasons.append("quantity must be an integer")
    else:
        if qty_i < 1:
            reasons.append("quantity must be >= 1")
        if qty_i > max_contracts:
            reasons.append(f"quantity {qty_i} exceeds max_contracts ({max_contracts})")

    price = _fnum(limit_price)
    if price is None or price <= 0:
        reasons.append("limit_price must be positive")

    if reasons:
        return False, reasons, warnings

    # --- intent/action consistency ---
    if intent_u == "BTC" and action_u != "BUY":
        reasons.append("BTC (buy to close) requires action=BUY")
    if intent_u == "STC" and action_u != "SELL":
        reasons.append("STC (sell to close) requires action=SELL")
    if intent_u == "STO" and action_u != "SELL":
        reasons.append("STO (sell to open) requires action=SELL")

    # --- over-close / coverage checks, by intent ---
    if intent_u == "BTC":
        short_pos = -_position_qty(positions, conid)  # positive when short
        resting_buys = _resting_qty(open_orders, conid, "BUY")
        available = short_pos - resting_buys
        if available < qty_i:
            reasons.append(
                f"BTC {qty_i}x refused: short position {short_pos:g} minus "
                f"{resting_buys:g} already resting in BUY orders leaves only "
                f"{available:g} contracts closeable — would over-close into a "
                f"long position.")

    elif intent_u == "STC":
        long_pos = _position_qty(positions, conid)
        resting_sells = _resting_qty(open_orders, conid, "SELL")
        available = long_pos - resting_sells
        if available < qty_i:
            reasons.append(
                f"STC {qty_i}x refused: long position {long_pos:g} minus "
                f"{resting_sells:g} already resting in SELL orders leaves only "
                f"{available:g} contracts closeable.")

    elif intent_u == "STO":
        if right == "P":
            available_funds = _fnum(funds.get("AvailableFunds"))
            if whatif is None:
                reasons.append("STO put refused: no whatif margin-impact result available.")
            else:
                init_margin_change = _fnum(whatif.get("init_margin_change"))
                if init_margin_change is None:
                    reasons.append(
                        "STO put refused: whatif did not return init_margin_change.")
                elif available_funds is None:
                    reasons.append(
                        "STO put refused: AvailableFunds not available to check margin.")
                elif init_margin_change > available_funds:
                    reasons.append(
                        f"STO put refused: whatif init margin impact "
                        f"${init_margin_change:,.2f} exceeds AvailableFunds "
                        f"${available_funds:,.2f}.")
        elif right == "C":
            shares = _underlying_share_position(positions, symbol)
            resting_stk_sells = _resting_stk_sell_shares(open_orders, symbol)
            short_calls = _short_call_contracts(positions, symbol)
            resting_sto_calls = _resting_sto_call_contracts(open_orders, symbol)
            committed_calls = short_calls + resting_sto_calls
            available_cover = shares - resting_stk_sells - committed_calls * multiplier
            needed = multiplier * qty_i
            if available_cover < needed:
                reasons.append(
                    f"STO call refused (naked-call guard): {shares:g} sh − "
                    f"{resting_stk_sells:g} sh committed to resting STK SELL orders "
                    f"− {committed_calls * multiplier:g} sh already committed to "
                    f"{committed_calls:g} short/resting calls = {available_cover:g} sh "
                    f"available cover, need {needed:g} sh for {qty_i}x calls. "
                    f"Naked calls are never allowed.")
        else:
            reasons.append(f"STO requires right C or P, got {contract.get('right')!r}")

    # --- price sanity vs quote (applies to every intent) ---
    bid = ask = None
    if quote and "error" not in quote:
        bid = _fnum(quote.get("bid"))
        ask = _fnum(quote.get("ask"))
    have_quote = bid is not None and ask is not None and bid > 0 and ask > 0
    if not have_quote:
        if allow_no_quote:
            warnings.append(
                "no usable bid/ask quote — price-sanity check skipped "
                "(allow_no_quote=True).")
        else:
            reasons.append(
                "no usable bid/ask quote — refusing (pass allow_no_quote=True to "
                "override).")
    elif action_u == "BUY":
        ceiling = ask + max(0.05, 0.10 * ask)
        if price > ceiling:
            reasons.append(
                f"limit ${price:.2f} exceeds BUY price-sanity ceiling ${ceiling:.2f} "
                f"(ask ${ask:.2f} + max($0.05, 10%)).")
    elif action_u == "SELL":
        floor = bid - max(0.05, 0.10 * bid)
        if price < floor:
            reasons.append(
                f"limit ${price:.2f} is below SELL price-sanity floor ${floor:.2f} "
                f"(bid ${bid:.2f} − max($0.05, 10%)).")

    return (not reasons), reasons, warnings
