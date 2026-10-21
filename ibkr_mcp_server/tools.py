"""MCP tools for IBKR functionality."""

import importlib.util
import json
import logging
import os
import sys
import time
from typing import Any, Sequence

from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolRequest

from dataclasses import asdict

from .client import ibkr_client
from .config import settings
from .option_gates import check_option_order
from .orders import StagedOrder, staged_store
from .utils import safe_float, validate_symbol, validate_symbols, IBKRError

logger = logging.getLogger(__name__)


LIVE_PORTS = {7496, 4001}
MAX_QUOTE_DRIFT = 0.30  # reject staging/confirming if limit is >30% away from last
                        # MUST match MAX_SUBMIT_DRIFT_PCT in /Users/ttang/Trader/scripts/compute_signals.py
                        # so the scanner pre-filter and this server-side gate agree.

# Resting SELL LMTs above market (e.g. swing T1 profit targets) are
# non-aggressive — they can't cross the book on current liquidity, so the
# fat-finger risk is much lower than for marketable orders. BBTAST_LONG
# swing setups target 3.5R T1 (V3 tuning in
# /Users/ttang/Trader/scripts/setups/bbtast_long.py); when the per-share
# stop is wide (~11%+ of price) the resulting T1 sits ~38%+ above entry
# and the 30% gate blocks every stage/repair attempt (S-R0b on INTU,
# 2026-05-27 — 4 consecutive rejections).
# Scope is narrow on purpose: DCA resting BUYs stay on the 30% gate so
# the scanner pre-filter contract is preserved.
MAX_QUOTE_DRIFT_RESTING_SELL = 0.60

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


# ---------------------------------------------------------------------------
# BUY funds gate (2026-07-31).
#
# IBKR does NOT reject a BUY that exceeds a cash account's funds — it accepts
# the order and parks it 'Inactive', where it can activate in arbitrary
# sequence when cash frees up, or silently evaporate. (2026-07-29 incident:
# the fund-manager scan rested $20.6k of Inactive BUY LMTs against $189 cash
# on live-U25242754 and recorded them all as submitted.) This gate refuses
# the order up front: BUY notional must fit inside the account's funds
# ceiling minus what open BUY orders have already committed.
#
# Mirrored in the scanner as STRATEGY.md §8h (compute_signals.py cash gate);
# this server-side check is the account-level backstop that protects every
# caller, not just the watchlist scan.
# ---------------------------------------------------------------------------

_FUNDS_CACHE: dict[str, tuple[float, dict]] = {}
_FUNDS_CACHE_TTL = 30.0  # seconds — same reasoning as the quote cache


def _fnum(v) -> "float | None":
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # ib_async leaves unset numeric fields at UNSET_DOUBLE (~1.8e308).
    return f if abs(f) < 1e9 else None


async def _funds_snapshot(account: "str | None") -> dict:
    """{tag: value} from get_account_summary, cached per account for TTL secs."""
    key = account or ibkr_client.current_account or "ALL"
    now = time.monotonic()
    entry = _FUNDS_CACHE.get(key)
    if entry is not None and now - entry[0] < _FUNDS_CACHE_TTL:
        return entry[1]
    rows = await ibkr_client.get_account_summary(account)
    snap: dict = {}
    for r in rows:
        tag, ccy = r.get("tag"), (r.get("currency") or "")
        # Multi-currency accounts repeat tags per currency — prefer USD/BASE.
        if tag in snap and ccy not in ("USD", "BASE", ""):
            continue
        snap[tag] = r.get("value")
    _FUNDS_CACHE[key] = (now, snap)
    return snap


def _open_buy_notional(account: "str | None") -> float:
    """Unfilled notional committed to open BUY orders on `account`.

    Includes Inactive parked orders — they consume cash the moment IBKR
    activates them. openTrades() is local client state, no API round-trip.
    Orders with no usable price (MKT) are skipped — rare for resting BUYs.
    """
    total = 0.0
    try:
        for t in ibkr_client.ib.openTrades():
            o = t.order
            if getattr(o, "action", "") != "BUY":
                continue
            if account and getattr(o, "account", "") and o.account != account:
                continue
            rem = _fnum(t.orderStatus.remaining) or _fnum(o.totalQuantity) or 0.0
            px = _fnum(getattr(o, "lmtPrice", None)) or \
                 _fnum(getattr(o, "auxPrice", None)) or 0.0
            if rem > 0 and px > 0:
                total += rem * px
    except Exception:
        pass  # deduction is best-effort; the ceiling itself is still enforced
    return total


# ---------------------------------------------------------------------------
# Margin gate (owner 2026-10-15).
#
# On 2026-10-14 buying power on the margin account U4022128 fell from $97.6k to
# $73.6k after manual buys and nothing stopped new bot entries (risk_guard only
# pages). The rule lives in ONE place, the Trader repo's scripts/live/
# margin_gate.py (line $200k, release $205k, margin account only, latch against
# chop around the line). It is loaded here by FILE PATH so this server needs no
# install-time dependency on that repo; TRADER_DIR overrides the checkout.
#
# Applied to: stock BUYs (_buy_funds_gate, margin path), option sell-to-open
# (_run_option_gates), and BUY modifies that raise notional (modify_live_order).
# NEVER applied to SELLs, stops, buy-to-close, sell-to-close, or cancels.
#
# Never raises. The module is re-exec'd whenever its file changes on disk (mtime/size
# checked on every gate call), so a Trader merge reaches long-lived MCP processes with
# no restart. If the module cannot be loaded the inline fallback below applies the
# higher of the config line and the owner's dashboard override (no latch), so the gate
# does not silently open or loosen.
#
# Burst ledger: resting unfilled orders do not lower BuyingPower until they fill,
# so several entries confirmed back-to-back by ONE session (the 09:15 swing
# scout) would each see the same BuyingPower. confirm_order therefore records
# every stock BUY / STO put it submits on a gated account in _MG_PENDING, and the
# gate subtracts the still-unfilled ones from BuyingPower. Only orders confirmed
# by THIS process count - never the owner's manual ladders or other clients'
# orders (rejected 2026-08-13, see the _buy_funds_gate docstring).
# ---------------------------------------------------------------------------

_MARGIN_GATE_ACCOUNT = "U4022128"
_MARGIN_GATE_LINE_USD = 200000.0   # fallback only; owner 2026-10-17 (was 150k from 10-16, 100k from 10-15, 75k before)
_MARGIN_GATE_FLOOR_USD = 100000.0  # owner-override floor (Trader risk.margin_line_min_usd)
_MARGIN_GATE_RETRY_S = 60.0   # after a failed load, retry no sooner than this
_MG_MODULE = None             # cached margin_gate module
_MG_SIG: "tuple | None" = None  # (mtime_ns, size) of margin_gate.py / common.py when _MG_MODULE was loaded
_MG_FAILED_AT: "float | None" = None
_MG_FAILED_SIG: "tuple | None" = None
# Orders this process confirmed that may still be unfilled:
# {"order_id": int, "impact": float (BP dollars), "account": str, "client_id": int|None}
_MG_PENDING: list = []
_MG_DONE_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})


def _trader_dir() -> str:
    return os.environ.get("TRADER_DIR", "/Users/ttang/Trader")


def _margin_gate_path() -> str:
    return os.path.join(_trader_dir(), "scripts", "live", "margin_gate.py")


def _margin_gate_sig() -> "tuple | None":
    """(mtime_ns, size) of margin_gate.py and its sibling common.py, or None when
    margin_gate.py cannot be stat()ed.  Two stat() calls per gate check: cheap."""
    try:
        p = _margin_gate_path()
        a = os.stat(p)
        try:
            b = os.stat(os.path.join(os.path.dirname(p), "common.py"))
            bs = (b.st_mtime_ns, b.st_size)
        except OSError:
            bs = None
        return (a.st_mtime_ns, a.st_size, bs)
    except OSError:
        return None


def _load_margin_gate():
    """The Trader repo's margin_gate module, or None if it cannot be loaded.

    Self-reloading (2026-10-21, after a merged margin_gate.py left every long-lived MCP
    process on the old rule): the file's mtime/size is checked on every call and the
    module is re-exec'd when it changed, so a Trader merge takes effect with no restart.
    A reload that fails keeps the previous good module (never silently drops to the
    inline fallback because of a half-written file) and retries on the next change or
    after _MARGIN_GATE_RETRY_S."""
    global _MG_MODULE, _MG_SIG, _MG_FAILED_AT, _MG_FAILED_SIG
    sig = _margin_gate_sig()
    if _MG_MODULE is not None and sig == _MG_SIG:
        return _MG_MODULE
    now = time.monotonic()
    if (_MG_FAILED_AT is not None and sig == _MG_FAILED_SIG
            and now - _MG_FAILED_AT < _MARGIN_GATE_RETRY_S):
        return _MG_MODULE            # last good module (None before the first good load)
    path = _margin_gate_path()
    saved_path = list(sys.path)
    saved_common = sys.modules.get("common")
    live_dir = os.path.dirname(path)
    swapped_common = False
    try:
        if _MG_MODULE is not None and saved_common is not None:
            # a changed common.py must be re-imported too; restored below if the reload fails
            cf = getattr(saved_common, "__file__", "") or ""
            if os.path.dirname(os.path.abspath(cf)) == live_dir:
                del sys.modules["common"]
                swapped_common = True
        spec = importlib.util.spec_from_file_location("trader_margin_gate", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build an import spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in ("evaluate", "bp_impact_from_margin"):
            if not callable(getattr(mod, attr, None)):
                raise ImportError(f"margin_gate has no {attr}()")
        if _MG_MODULE is not None:
            logger.info("margin_gate.py changed on disk - reloaded")
        _MG_MODULE, _MG_SIG, _MG_FAILED_AT, _MG_FAILED_SIG = mod, sig, None, None
        return mod
    except Exception as e:  # noqa: BLE001 - a broken gate module must not break orders
        _MG_FAILED_AT, _MG_FAILED_SIG = now, sig
        if swapped_common and saved_common is not None:
            sys.modules["common"] = saved_common
        if _MG_MODULE is not None:
            logger.warning("margin_gate reload failed (%s: %s) - keeping the previously "
                           "loaded rule", type(e).__name__, e)
            return _MG_MODULE
        logger.warning("margin_gate unavailable (%s: %s) - using inline fallback "
                       "(config line or dashboard override, whichever is higher, no latch)",
                       type(e).__name__, e)
        return None
    finally:
        # margin_gate.py puts scripts/live first on sys.path to import common;
        # do not leave that behind in the server process.
        sys.path[:] = saved_path


def _mg_money(x: "float | None") -> str:
    return "unknown" if x is None else f"${x:,.0f}"


def _norm_account(account) -> str:
    a = str(account or "").strip()
    return a.split("-", 1)[1] if a.lower().startswith("live-") else a


def _mg_live_dir() -> str:
    d = os.environ.get("MARGIN_GATE_LIVE_DIR")
    return d if d else os.path.join(_trader_dir(), "data", "live")


def _mg_read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - missing / torn / corrupt -> treated as absent
        return None


def _mg_pos(v) -> "float | None":
    """A finite positive number (bool / NaN / inf / junk -> None)."""
    if isinstance(v, bool):
        return None
    x = _fnum(v)
    return x if (x is not None and x == x and x not in (float("inf"), float("-inf")) and x > 0) else None


def _fallback_config_risk() -> dict:
    """Trader's `risk` config block (config.default.json overlaid by the profile's
    live_agents.json), {} when unreadable.  Never raises."""
    merged: dict = {}
    try:
        base = _trader_dir()
        for rel in (("scripts", "live", "config.default.json"),
                    ("accounts", "live-" + _MARGIN_GATE_ACCOUNT, "live_agents.json")):
            d = _mg_read_json(os.path.join(base, *rel))
            if isinstance(d, dict) and isinstance(d.get("risk"), dict):
                merged.update(d["risk"])
    except Exception:  # noqa: BLE001
        return {}
    return merged


def _fallback_margin_line() -> float:
    """Line for the inline fallback (margin_gate.py unloadable): never looser than the
    config line (risk.buying_power_crit_usd from Trader's config.default.json overlaid by
    accounts/<profile>/live_agents.json; the _MARGIN_GATE_LINE_USD constant when neither is
    readable) nor than a VALID owner dashboard override (data/live/margin_gate_override.json,
    >= the floor).  An override file that exists but is unreadable / below the floor falls
    back to the last-good record (margin_gate_override.last_good.json) so a corrupt file
    cannot loosen the gate.  No latch, no assignment reserve.  Never raises."""
    merged = _fallback_config_risk()
    cfg_line = _mg_pos(merged.get("buying_power_crit_usd"))
    floor = _mg_pos(merged.get("margin_line_min_usd")) or _MARGIN_GATE_FLOOR_USD
    line = cfg_line if cfg_line is not None else _MARGIN_GATE_LINE_USD
    try:
        live = _mg_live_dir()
        ov_path = os.path.join(live, "margin_gate_override.json")
        ov = None
        if os.path.exists(ov_path):
            d = _mg_read_json(ov_path)
            v = _mg_pos(d.get("line_usd")) if isinstance(d, dict) else None
            if v is not None and v >= floor:
                ov = v
            else:                                   # present but unusable: stay strict
                lg = _mg_read_json(os.path.join(live, "margin_gate_override.last_good.json"))
                v = _mg_pos(lg.get("line_usd")) if isinstance(lg, dict) else None
                ov = v if (v is not None and v >= floor) else None
        if ov is not None:
            line = max(line, ov)
    except Exception:  # noqa: BLE001
        pass
    return line


def _inline_margin_gate(account, impact_usd, bp) -> dict:
    """Fallback when margin_gate.py cannot be loaded: config line or owner override,
    whichever is higher; no latch."""
    line = _fallback_margin_line()
    if _norm_account(account) != _MARGIN_GATE_ACCOUNT:
        return {"state": "n/a", "blocked": False, "reason": "", "bp": None}
    b = _fnum(bp)
    impact = max(0.0, _fnum(impact_usd) or 0.0)
    if b is None:
        return {"state": "unknown", "blocked": True, "bp": None,
                "reason": "Buying power is unknown, so new orders that use margin "
                          "are held until it can be read."}
    if b < line:
        return {"state": "blocked", "blocked": True, "bp": b,
                "reason": f"Buying power is {_mg_money(b)}, below the "
                          f"{_mg_money(line)} margin line, so new orders that use "
                          "margin are blocked."}
    if b - impact < line:
        return {"state": "blocked", "blocked": True, "bp": b,
                "reason": f"This order would take buying power from {_mg_money(b)} to "
                          f"{_mg_money(b - impact)}, below the {_mg_money(line)} margin line."}
    return {"state": "clear", "blocked": False, "reason": "", "bp": b}


def _evaluate_margin(account, impact_usd, bp, update_latch: bool = True) -> dict:
    """The shared module's verdict, or the inline fallback's. Never raises."""
    mg = _load_margin_gate()
    if mg is not None:
        try:
            v = mg.evaluate(account, impact_usd, bp=bp, update_latch=update_latch)
            if isinstance(v, dict):
                return v
        except Exception as e:  # noqa: BLE001
            logger.warning("margin_gate.evaluate failed (%s: %s) - using inline fallback",
                           type(e).__name__, e)
    return _inline_margin_gate(account, impact_usd, bp)


def _pending_margin_impact(account) -> float:
    """Buying-power dollars this process has committed on `account` to orders
    that are still working and wholly unfilled. Entries whose order is gone,
    done, or partly filled are pruned (a fill is already in BuyingPower).
    Lookup failure counts as 0 for the affected entry. Never raises."""
    try:
        acct = _norm_account(account)
        if not any(e.get("account") == acct for e in _MG_PENDING):
            return 0.0
        by_id: dict = {}
        for t in ibkr_client.ib.openTrades():
            try:
                by_id.setdefault(int(t.order.orderId), []).append(t)
            except Exception:  # noqa: BLE001
                continue
        total, keep = 0.0, []
        for e in list(_MG_PENDING):
            if e.get("account") != acct:
                keep.append(e)
                continue
            try:
                trade = None
                for t in by_id.get(e["order_id"], []):
                    cid = getattr(t.order, "clientId", None)
                    if (e.get("client_id") is None or not isinstance(cid, int)
                            or cid == e["client_id"]):
                        trade = t
                        break
                if trade is None:
                    continue                      # no longer open: dropped
                status = str(getattr(trade.orderStatus, "status", "") or "")
                filled = _fnum(getattr(trade.orderStatus, "filled", 0)) or 0.0
                if status in _MG_DONE_STATUSES or filled > 0:
                    continue                      # done or filling: BP has it
                total += float(e["impact"])
                keep.append(e)
            except Exception:  # noqa: BLE001 - this entry counts as 0, stays
                keep.append(e)
        _MG_PENDING[:] = keep
        return total
    except Exception as e:  # noqa: BLE001
        logger.warning("margin gate pending lookup failed (%s: %s)", type(e).__name__, e)
        return 0.0


def _margin_gate_applies(account) -> bool:
    mg = _load_margin_gate()
    if mg is not None:
        try:
            return bool(mg.applies_to(account, mg.gate_config()))
        except Exception:  # noqa: BLE001
            pass
    return _norm_account(account) == _MARGIN_GATE_ACCOUNT


def _placed_order_id(result):
    """orderId from a place_* result (bracket results nest it under 'parent')."""
    try:
        parent = result.get("parent")
        src = parent if isinstance(parent, dict) else result
        return int(src["order_id"])
    except Exception:  # noqa: BLE001
        return None


def _record_pending_margin_order(account, order_id, impact_usd) -> None:
    """Remember an order this process just sent so later gate checks subtract it
    until it fills. Only margin-gated accounts are recorded. Never raises."""
    try:
        oid, impact = order_id, _fnum(impact_usd)
        if oid is None or impact is None or impact <= 0:
            return
        if not _margin_gate_applies(account):
            return
        cid = ibkr_client.client_id
        _MG_PENDING.append({"order_id": int(oid), "impact": float(impact),
                            "account": _norm_account(account),
                            "client_id": cid if isinstance(cid, int) else None})
    except Exception as e:  # noqa: BLE001
        logger.warning("margin gate ledger record failed (%s: %s)", type(e).__name__, e)


def _adjust_pending_margin_order(order_id, delta_usd) -> None:
    """A successful modify changed a recorded order's notional by delta_usd."""
    try:
        d = _fnum(delta_usd)
        if d is None:
            return
        for e in _MG_PENDING:
            if e.get("order_id") == int(order_id):
                e["impact"] = max(0.0, float(e["impact"]) + d)
    except Exception:  # noqa: BLE001
        pass


def _margin_gate_refusal(account, impact_usd=0.0, bp=None) -> "str | None":
    """None when the order may proceed, else a plain-English refusal that starts
    with "Margin gate: ". `bp` is the caller's live BuyingPower (None lets the
    shared module fall back to eventd's broker snapshot, else fail closed).

    Two passes: (1) the live reading, which also drives the shared latch;
    (2) when this process has unfilled orders on the account, the same check
    on BuyingPower minus those orders (latch NOT updated, so a hypothetical
    number never trips the shared latch). Never raises."""
    verdict = _evaluate_margin(account, impact_usd, bp)
    if verdict.get("blocked"):
        reason = str(verdict.get("reason") or
                     "Buying power is too low for new orders that use margin.")
        return "Margin gate: " + reason
    if verdict.get("state") == "n/a":
        return None
    live_bp = _fnum(verdict.get("bp"))
    pending = _pending_margin_impact(account)
    if pending > 0 and live_bp is not None:
        v2 = _evaluate_margin(account, impact_usd, live_bp - pending, update_latch=False)
        if v2.get("blocked"):
            reason = str(v2.get("reason") or
                         "Buying power is too low for new orders that use margin.")
            return ("Margin gate: " + reason.rstrip(".") +
                    f" (counting {_mg_money(pending)} of orders already sent this "
                    "session that have not filled yet).")
    return None


def _margin_gate_option_impact(whatif, bp, available_funds) -> float:
    """Buying-power impact of an option sell-to-open from its what-if initial
    margin change. 0.0 when the what-if is unavailable (the line/latch still
    applies to the order)."""
    imc = _fnum(whatif.get("init_margin_change")) if isinstance(whatif, dict) else None
    if imc is None:
        return 0.0
    mg = _load_margin_gate()
    if mg is not None:
        try:
            v = mg.bp_impact_from_margin(imc, bp, available_funds)
            if v is not None:
                return max(0.0, float(v))
        except Exception as e:  # noqa: BLE001
            logger.warning("margin_gate.bp_impact_from_margin failed (%s: %s)",
                           type(e).__name__, e)
    b, af = _fnum(bp), _fnum(available_funds)
    ratio = (b / af) if (b is not None and af is not None and b > 0 and af > 0) else 4.0
    return max(0.0, imc) * ratio


def _configured_cash_accounts() -> frozenset:
    """Normalised ids from settings.ibkr_cash_accounts (IBKR_CASH_ACCOUNTS)."""
    raw = getattr(settings, "ibkr_cash_accounts", "") or ""
    return frozenset(a for a in (_norm_account(p) for p in str(raw).split(",")) if a)


def _is_cash_account(account, acct_type: str) -> bool:
    """True when the §8h cumulative open-BUY deduction applies to `account`.

    IBKR's AccountType tag describes ownership structure (INDIVIDUAL, JOINT,
    IRA...), not cash vs margin: both live accounts read INDIVIDUAL (all 50
    snapshots since 2026-07-31), so from 07-31 to 10-15 the tag never matched
    "CASH" and the cash branch never ran. The account id in the configured cash
    list is the authority; AccountType == "CASH" is kept as a second positive
    signal. An account not in the list that reads INDIVIDUAL stays on the
    margin path (2026-08-13 fix).
    """
    if str(acct_type or "").upper() == "CASH":
        return True
    return _norm_account(account) in _configured_cash_accounts()


async def _buy_funds_gate(action: str, quantity: int, price: float,
                          strict: bool, margin_gate: bool = True) -> dict:
    """Refuse a BUY whose notional exceeds funds headroom.

    CASH accounts (settings.ibkr_cash_accounts, or AccountType == "CASH" —
    see _is_cash_account): headroom = AvailableFunds (fallback TotalCashValue)
    − total open BUY notional. Cash is consumed dollar-for-dollar as
    resting BUYs fill, and IBKR parks over-committed cash-account BUYs
    Inactive instead of rejecting them (§8h), so the worst-case
    all-fills sum is the right check.

    Margin accounts: this order's notional is checked against
    BuyingPower (fallback AvailableFunds) alone — open BUY notional is
    NOT deducted. IBKR enforces margin per-fill natively on margin
    accounts, so summing every resting BUY (e.g. operator DCA ladders)
    only produces false refusals (2026-08-12 MSFT/AVGO swing incident).
    A non-blocking warning is attached when the all-fills sum would
    exceed the ceiling.

    strict=False (stage time): fail-open with a warning when IBKR is
    offline or the summary fetch fails — confirm re-validates.
    strict=True (confirm time): those failures refuse the order, same
    contract as the quote-drift gate.

    Margin gate (2026-10-15): on the margin path, after the checks above,
    a BUY that would use margin while BuyingPower is at/below the line
    (or would push it below) is refused with "Margin gate: <reason>".
    Cash-account path is untouched. margin_gate=False skips it (protective
    STP orders are never gated).
    """
    if action.upper() != "BUY" or price <= 0 or quantity <= 0:
        return {"ok": True}
    if not ibkr_client.is_connected():
        if strict:
            return {"ok": False, "error": "IBKR not connected — cannot validate funds"}
        return {"ok": True,
                "warning": "IBKR offline — funds check skipped, will re-validate at confirm time"}
    try:
        snap = await _funds_snapshot(None)
    except Exception as e:
        if strict:
            return {"ok": False, "error": f"funds lookup failed: {e}"}
        return {"ok": True,
                "warning": f"funds lookup failed ({e}) — will re-validate at confirm time"}

    account = ibkr_client.current_account
    is_cash = _is_cash_account(account, str(snap.get("AccountType", "")))
    if is_cash:
        candidates = [("AvailableFunds", snap.get("AvailableFunds")),
                      ("TotalCashValue", snap.get("TotalCashValue"))]
    else:
        candidates = [("BuyingPower", snap.get("BuyingPower")),
                      ("AvailableFunds", snap.get("AvailableFunds"))]
    ceiling, ceiling_tag = None, None
    for tag, raw in candidates:
        ceiling = _fnum(raw)
        if ceiling is not None:
            ceiling_tag = tag
            break
    if ceiling is None:
        if strict:
            return {"ok": False, "error": "could not obtain funds figure for account"}
        return {"ok": True,
                "warning": "no funds figure available — will re-validate at confirm time"}

    notional = quantity * price
    open_buy = _open_buy_notional(account)
    if is_cash:
        headroom = ceiling - open_buy
        if notional > headroom + 0.01:
            return {"ok": False, "error": (
                f"BUY notional ${notional:,.2f} exceeds available funds: "
                f"{ceiling_tag} ${ceiling:,.2f} − ${open_buy:,.2f} committed to open "
                f"BUY orders = ${headroom:,.2f} headroom "
                f"(account {account or 'current'}, CASH). Refusing.")}
        return {"ok": True, "funds_headroom_after": round(headroom - notional, 2)}

    # Margin: IBKR enforces margin per-fill; gate only the incremental order.
    if notional > ceiling + 0.01:
        return {"ok": False, "error": (
            f"BUY notional ${notional:,.2f} exceeds {ceiling_tag} "
            f"${ceiling:,.2f} (account {account or 'current'}, margin). Refusing.")}
    if margin_gate:
        refusal = _margin_gate_refusal(account, notional,
                                       bp=_fnum(snap.get("BuyingPower")))
        if refusal:
            return {"ok": False, "error": refusal}
    result = {"ok": True, "funds_headroom_after": round(ceiling - notional, 2)}
    if open_buy and notional + open_buy > ceiling:
        result["warning"] = (
            f"all-fills exposure ${notional + open_buy:,.2f} (this order plus "
            f"${open_buy:,.2f} resting BUYs) exceeds {ceiling_tag} ${ceiling:,.2f}; "
            f"not blocking — IBKR margin checks apply at fill time")
    return result


async def _modify_margin_refusal(order_id: int, qty, lmt,
                                 out: "dict | None" = None) -> "str | None":
    """Margin gate for modify_live_order. Looks up the live order and gates only
    the INCREASE in unfilled notional of a stock BUY (qty up and/or limit up):
    impact = new unfilled notional - old unfilled notional. Decreases, SELLs,
    option and stop orders are never gated. Returns a refusal string or None.
    Never raises (a failed lookup falls through - modify_order reports it).
    out (optional) receives {"delta": signed notional change} for stock BUY LMTs
    so a successful modify can adjust the burst ledger."""
    try:
        try:
            await ibkr_client._ensure_connected()
        except Exception:  # noqa: BLE001 - modify_order will surface a real outage
            pass
        target = None
        for t in ibkr_client.ib.trades():
            if t.order.orderId == int(order_id):
                target = t
                break
        if target is None:
            return None
        o, c = target.order, target.contract
        if str(getattr(o, "action", "")).upper() != "BUY":
            return None
        if str(getattr(c, "secType", "STK") or "STK").upper() != "STK":
            return None
        if str(getattr(o, "orderType", "")).upper() not in ("LMT", "STP LMT"):
            return None
        old_qty = _fnum(getattr(o, "totalQuantity", None))
        old_px = _fnum(getattr(o, "lmtPrice", None))
        if old_qty is None or old_px is None or old_qty <= 0 or old_px <= 0:
            return None
        filled = _fnum(getattr(target.orderStatus, "filled", 0)) or 0.0
        new_qty = float(int(qty)) if qty is not None else old_qty
        new_px = float(lmt) if lmt is not None else old_px
        old_notional = max(0.0, old_qty - filled) * old_px
        new_notional = max(0.0, new_qty - filled) * new_px
        increase = new_notional - old_notional
        if out is not None:
            out["delta"] = increase
        if increase <= 0:
            return None
        account = getattr(o, "account", "") or ibkr_client.current_account
        bp = None
        if account == ibkr_client.current_account:
            try:
                bp = _fnum((await _funds_snapshot(None)).get("BuyingPower"))
            except Exception:  # noqa: BLE001 - None -> broker snapshot or fail closed
                bp = None
        return _margin_gate_refusal(account, increase, bp=bp)
    except Exception as e:  # noqa: BLE001
        logger.warning("modify margin gate lookup failed (%s: %s)", type(e).__name__, e)
        return None


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
    is_resting_sell_above = (action == "SELL" and limit_price > ref)
    effective_max = MAX_QUOTE_DRIFT_RESTING_SELL if is_resting_sell_above else MAX_QUOTE_DRIFT
    if drift > effective_max:
        return {"ok": False,
                "error": f"limit ${limit_price:.2f} is {drift*100:.1f}% from last ${ref:.2f} "
                         f"(max {effective_max*100:.0f}%, source={source}). Refusing."}
    return {"ok": True, "reference_price": ref, "drift_pct": round(drift * 100, 2),
            "reference_source": source}


def _option_contract_spec(arguments: dict) -> dict:
    """Build qualify_option()/get_option_quote() kwargs from tool arguments:
    conid (preferred, if present) OR symbol+expiry+strike+right. Raises
    ValueError with a caller-friendly message when neither is fully given."""
    conid = arguments.get("conid")
    if conid:
        return {"conid": int(conid)}
    symbol = arguments.get("symbol")
    expiry = arguments.get("expiry")
    strike = arguments.get("strike")
    right = arguments.get("right")
    if not (symbol and expiry and strike is not None and right):
        raise ValueError("Must provide conid, or symbol+expiry+strike+right")
    return {
        "symbol": validate_symbol(symbol),
        "expiry": str(expiry),
        "strike": float(strike),
        "right": right,
    }


def _contract_dict(contract) -> dict:
    """The {conid, right, strike, multiplier, symbol} shape option_gates expects,
    built from a qualified ib_async contract."""
    return {
        "conid": contract.conId,
        "right": getattr(contract, "right", None) or None,
        "strike": safe_float(getattr(contract, "strike", 0)) or None,
        "multiplier": getattr(contract, "multiplier", None) or 100,
        "symbol": contract.symbol,
    }


async def _run_option_gates(intent: str, action: str, quantity: int, limit_price: float,
                            contract, allow_no_quote: bool,
                            impact_out: "dict | None" = None):
    """Fetch fresh positions/open-orders/quote/whatif state and run
    check_option_order. Shared by stage_option_order and the confirm_order
    OPT dispatch so both paths validate against the exact same rules.

    impact_out (optional): receives {"impact": <BP dollars>} for a margin-gated
    sell-to-open put, so confirm_order can record it in the burst ledger."""
    contract_dict = _contract_dict(contract)
    positions = await ibkr_client.get_portfolio()
    open_orders = await ibkr_client.get_open_trades()
    try:
        quote = await ibkr_client.get_option_quote(conid=contract.conId)
    except Exception as e:
        quote = {"error": str(e)}
    whatif = None
    if intent.upper() == "STO" and (contract_dict.get("right") or "").upper() == "P":
        try:
            whatif = await ibkr_client.whatif_option_order(
                contract, "SELL", quantity, limit_price)
        except Exception as e:
            whatif = {"error": str(e)}
    try:
        funds = await _funds_snapshot(None)
    except Exception:
        funds = {}
    ok, reasons, warnings = check_option_order(
        intent=intent, action=action, qty=quantity, limit_price=limit_price,
        contract=contract_dict, positions=positions, open_orders=open_orders,
        quote=quote, whatif=whatif, funds=funds, allow_no_quote=allow_no_quote,
    )
    # Margin gate (2026-10-15): only a SELL that OPENS a short PUT adds margin.
    # BTC / STC are never gated; an STO call that passed check_option_order is
    # covered (naked calls are refused there) and uses no margin.
    if (ok and intent.upper() == "STO" and action.upper() == "SELL"
            and (contract_dict.get("right") or "").upper() == "P"):
        bp = _fnum(funds.get("BuyingPower"))
        impact = _margin_gate_option_impact(
            whatif, bp, _fnum(funds.get("AvailableFunds")))
        if impact_out is not None:
            impact_out["impact"] = impact
        refusal = _margin_gate_refusal(ibkr_client.current_account, impact, bp=bp)
        if refusal:
            return False, [refusal], warnings
    return ok, reasons, warnings


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
                     "Applies MAX_ORDER_SIZE, quote-drift, and BUY funds-headroom safety gates. Returns a staged_id for confirm_order. "
                     "Honors OCA grouping when `oca_group` is provided — siblings sharing the same group "
                     "auto-cancel each other on first fill (used by /swing-scout to stage multiple sibling "
                     "BUY LMTs that must not co-fill on opening gap-throughs). "
                     "Pass tif='OPG' to stage a limit-on-open for the next session's opening auction — "
                     "required for end-of-day exits staged after the 16:00 close, which IBKR otherwise "
                     "parks Inactive and drops overnight."),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "action": {"type": "string", "enum": ["BUY", "SELL"]},
                "quantity": {"type": "integer", "minimum": 1},
                "limit_price": {"type": "number", "exclusiveMinimum": 0},
                "tif": {"type": "string", "enum": ["DAY", "GTC", "IOC", "FOK", "OPG"], "default": "DAY",
                        "description": ("Time in force. OPG = limit-on-open: the order participates ONLY "
                                        "in the next session's opening auction and is cancelled if unfilled. "
                                        "Use OPG for end-of-day exits staged after the 16:00 close — a "
                                        "marketable LMT with DAY/GTC is parked Inactive by IBKR and "
                                        "evaporates overnight. OPG forces outside_rth=False.")},
                "outside_rth": {"type": "boolean", "default": False,
                                 "description": ("Allow order to trigger/fill outside regular trading hours. "
                                                 "Ignored (forced False) when tif=OPG — IBKR rejects "
                                                 "outsideRth on opening-auction orders.")},
                "oca_group": {"type": "string",
                                "description": "OCA tag — orders sharing this group cancel/reduce on first fill."},
                "oca_type":  {"type": "integer", "enum": [0, 1, 2, 3], "default": 1,
                                "description": ("0=none, 1=cancel-with-block (default for entry-side siblings — "
                                                 "one fills, the rest are cancelled), 2=reduce-with-block, 3=reduce-no-block.")},
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
            "may have rolled off the live-orders window. Each fill is enriched "
            "with the parent order's orderRef under the `tag` / `order_ref` / "
            "`source` keys (matching `get_live_orders`), so callers can merge "
            "the two responses by tag to reconcile filled tagged orders. "
            "Pass an optional `account` to filter; otherwise returns fills for "
            "all accounts on the connection."
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
    ),
    Tool(
        name="get_market_quotes",
        description=(
            "Batched snapshot quotes for multiple symbols in one call. "
            "Qualifies contracts together and streams reqMktData concurrently, "
            "which is much faster than calling get_market_quote in a loop. "
            "Returns a dict keyed by symbol; per-symbol failures surface as "
            "{\"error\": \"...\"} without crashing the rest of the batch."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "string",
                    "description": "Comma-separated list of symbols (max 50)"
                }
            },
            "required": ["symbols"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="stage_stop_order",
        description=(
            "Validate and stage a STOP (STP) order for later approval. "
            "Returns staged_id for confirm_order. Used for fail-safe protective "
            "stops, BE-stops, and trailing stops. Honors OCA grouping when "
            "`oca_group` is provided. Drift safety gate uses stop_price as the "
            "reference."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":      {"type": "string"},
                "action":      {"type": "string", "enum": ["BUY", "SELL"]},
                "quantity":    {"type": "integer", "minimum": 1},
                "stop_price":  {"type": "number", "exclusiveMinimum": 0},
                "tif":         {"type": "string", "enum": ["DAY", "GTC", "IOC", "FOK"], "default": "GTC",
                                 "description": ("Time in force. OPG is NOT available here — IBKR only "
                                                 "accepts opening-auction TIFs on LMT/MKT order types. "
                                                 "For an opening-auction exit use stage_order with tif='OPG'.")},
                "outside_rth": {"type": "boolean", "default": False},
                "oca_group":   {"type": "string",
                                 "description": "OCA tag — orders sharing this group reduce/cancel each other on fill."},
                "oca_type":    {"type": "integer", "enum": [0, 1, 2, 3], "default": 0,
                                 "description": "0=none, 1=cancel-with-block, 2=reduce-with-block (default for brackets), 3=reduce-no-block"},
                "source":      {"type": "string"}
            },
            "required": ["symbol", "action", "quantity", "stop_price"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="stage_bracket_order",
        description=(
            "Validate and stage a BRACKET order: parent BUY LMT entry plus "
            "1+ OCA-linked SELL children (each LMT or STP). Children share "
            "an OCA group with `oca_type=2` (REDUCE_WITH_BLOCK) by default — "
            "when one fills, the others' quantities reduce automatically. "
            "Returns the parent's staged_id (children carry parent_staged_id "
            "linkage). `confirm_order` on the parent submits the entire "
            "bracket atomically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":              {"type": "string"},
                "parent_action":       {"type": "string", "enum": ["BUY", "SELL"], "default": "BUY"},
                "parent_quantity":     {"type": "integer", "minimum": 1},
                "parent_limit_price":  {"type": "number", "exclusiveMinimum": 0},
                "parent_tif":          {"type": "string", "enum": ["DAY", "GTC"], "default": "GTC"},
                "parent_outside_rth":  {"type": "boolean", "default": False},
                "children": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "order_type":  {"type": "string", "enum": ["LMT", "STP"]},
                            "action":      {"type": "string", "enum": ["BUY", "SELL"], "default": "SELL"},
                            "quantity":    {"type": "integer", "minimum": 1},
                            "limit_price": {"type": "number", "exclusiveMinimum": 0},
                            "stop_price":  {"type": "number", "exclusiveMinimum": 0},
                            "tif":         {"type": "string", "enum": ["DAY", "GTC"], "default": "GTC"},
                            "outside_rth": {"type": "boolean", "default": False},
                            "oca_type":    {"type": "integer", "enum": [1, 2, 3], "default": 2},
                            "tag":         {"type": "string"}
                        },
                        "required": ["order_type", "quantity"],
                        "additionalProperties": False
                    }
                },
                "source": {"type": "string"}
            },
            "required": ["symbol", "parent_quantity", "parent_limit_price", "children"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="modify_live_order",
        description=(
            "Modify a live IBKR order in place: change qty, limit_price, "
            "and/or stop_price without cancel-then-restage (avoids the "
            "no-coverage gap that arises when a protective order is removed "
            "before its replacement is acknowledged). Pass only the fields "
            "you want to change."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "order_id":    {"type": "integer"},
                "quantity":    {"type": "integer", "minimum": 1},
                "limit_price": {"type": "number", "exclusiveMinimum": 0},
                "stop_price":  {"type": "number", "exclusiveMinimum": 0}
            },
            "required": ["order_id"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_option_chain",
        description=(
            "Get the option chain (expirations, strikes, trading class, "
            "multiplier) for a symbol's underlying, preferring the SMART-"
            "routed row."
        ),
        inputSchema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False
        }
    ),
    Tool(
        name="get_option_quote",
        description=(
            "Snapshot option quote: bid/ask/last/mid plus Greeks "
            "(delta/gamma/theta/vega/iv) and underlying price. Identify the "
            "contract by `conid`, or by symbol+expiry+strike+right. Never "
            "fails just because market data is unavailable (no OPRA "
            "subscription is a normal case) — returns source='none' with any "
            "observed IBKR error codes (e.g. 354/10089/10090/10167)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "conid":  {"type": "integer",
                            "description": "IBKR contract id — takes precedence over symbol/expiry/strike/right."},
                "symbol": {"type": "string"},
                "expiry": {"type": "string", "description": "YYYYMMDD"},
                "strike": {"type": "number"},
                "right":  {"type": "string", "enum": ["C", "P", "CALL", "PUT"]}
            },
            "additionalProperties": False
        }
    ),
    Tool(
        name="stage_option_order",
        description=(
            "Validate and stage an OPTION order for later approval. Does NOT "
            "submit to IBKR. Identify the contract by `conid`, or by "
            "symbol+expiry+strike+right. `intent` classifies the trade: BTC "
            "(buy to close a short), STC (sell to close a long), or STO "
            "(sell to open — put or COVERED call only; naked calls and "
            "BTO/opening-long are refused). Runs option_gates.check_option_order "
            "against current positions/open orders/quote/whatif margin impact; "
            "re-run at confirm_order time against fresh state."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "conid":         {"type": "integer"},
                "symbol":        {"type": "string"},
                "expiry":        {"type": "string", "description": "YYYYMMDD"},
                "strike":        {"type": "number"},
                "right":         {"type": "string", "enum": ["C", "P", "CALL", "PUT"]},
                "intent":        {"type": "string", "enum": ["BTC", "STC", "STO"]},
                "action":        {"type": "string", "enum": ["BUY", "SELL"]},
                "quantity":      {"type": "integer", "minimum": 1,
                                    "description": "Number of contracts."},
                "limit_price":   {"type": "number", "exclusiveMinimum": 0},
                "tif":           {"type": "string", "enum": ["DAY", "GTC"], "default": "DAY"},
                "source":        {"type": "string", "description": "Provenance tag, e.g. 'WHEEL_BTC_...'"},
                "order_ref":     {"type": "string", "description": "Alias for `source`."},
                "allow_no_quote": {"type": "boolean", "default": False,
                                    "description": "Skip the price-sanity check (as a warning, not a refusal) when no usable bid/ask is available."}
            },
            "required": ["intent", "action", "quantity", "limit_price"],
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
                "paper_trading": ibkr_client.is_paper,
                # Diagnostics (2026-09-08 PLTR incident): last 20 IBKR
                # errorEvents across all requests/orders, so a rejection
                # reason is visible without tailing the log file.
                "recent_ibkr_errors": ibkr_client.get_recent_errors(20),
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
            oca_group = arguments.get("oca_group")
            oca_type = int(arguments.get("oca_type", 1))   # default = CANCEL_WITH_BLOCK for entry-side siblings

            v = await _validate_order_inputs(symbol, action, quantity, limit_price,
                                               require_quote=False)
            if not v["ok"]:
                return [TextContent(type="text", text=json.dumps({"staged": False, "error": v["error"]}))]

            fg = await _buy_funds_gate(action, quantity, limit_price, strict=False)
            if not fg["ok"]:
                return [TextContent(type="text", text=json.dumps({"staged": False, "error": fg["error"]}))]

            try:
                order = StagedOrder.new(symbol, action, quantity, limit_price,
                                        tif=tif, source=source, outside_rth=outside_rth,
                                        oca_group=oca_group, oca_type=oca_type)
            except ValueError as e:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False, "error": str(e)}))]
            staged_store.add(order)
            # OPG coerces outside_rth off (IBKR rejects outsideRth on auction
            # orders). Surface it so the caller's log shows what was actually
            # staged rather than what it asked for.
            tif_note = None
            if outside_rth and not order.outside_rth:
                tif_note = (f"tif={order.tif}: outside_rth forced False "
                            "(IBKR rejects outsideRth on opening-auction orders)")
            return [TextContent(type="text", text=json.dumps({
                "staged": True,
                "staged_id": order.id,
                "summary": order.summary(),
                "tif": order.tif,
                "outside_rth": order.outside_rth,
                "tif_note": tif_note,
                "reference_price": v["reference_price"],
                "drift_pct": v["drift_pct"],
                "reference_source": v.get("reference_source"),
                "funds_warning": fg.get("warning"),
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
            # Audit I3: confirming a bracket child solo would orphan the parent
            # and siblings. Reject — caller must confirm the parent's staged_id,
            # which submits the entire bracket atomically via place_bracket_order.
            if order.parent_staged_id:
                return [TextContent(type="text", text=json.dumps({
                    "submitted": False,
                    "error": (f"staged_id {staged_id} is a bracket child of "
                              f"{order.parent_staged_id}. Confirm the parent "
                              "instead — the entire bracket submits atomically."),
                    "parent_staged_id": order.parent_staged_id,
                }))]

            # Live-trading gate.
            if ibkr_client.port in LIVE_PORTS and not settings.enable_live_trading:
                return [TextContent(type="text", text=json.dumps({
                    "submitted": False,
                    "error": (f"Live port {ibkr_client.port} detected but ENABLE_LIVE_TRADING=false. "
                              "Refusing to submit. Set ENABLE_LIVE_TRADING=true in .env to allow."),
                }))]

            # OPT dispatch (2026-09-23). Option orders skip the stock-only
            # validators below (_validate_order_inputs' quote-drift check and
            # _buy_funds_gate assume shares/notional) — option_gates covers
            # price sanity and margin impact (via whatif) on its own terms.
            # Re-run against FRESH state (positions/open orders/quote/whatif
            # can all have moved since stage time), then place directly.
            if getattr(order, "sec_type", "STK") == "OPT":
                try:
                    contract = await ibkr_client.qualify_option(conid=order.conid)
                except Exception as e:
                    return [TextContent(type="text", text=json.dumps({
                        "submitted": False,
                        "error": f"could not qualify option contract: {e}"}))]

                gate_impact: dict = {}
                ok, reasons, warnings = await _run_option_gates(
                    intent=order.intent or "", action=order.action,
                    quantity=order.quantity, limit_price=order.limit_price,
                    contract=contract, allow_no_quote=False,
                    impact_out=gate_impact,
                )
                if not ok:
                    return [TextContent(type="text", text=json.dumps({
                        "submitted": False,
                        "error": "; ".join(reasons),
                        "reasons": reasons,
                        "warnings": warnings,
                    }))]

                try:
                    result = await ibkr_client.place_option_limit_order(
                        contract=contract, action=order.action,
                        quantity=order.quantity, limit_price=order.limit_price,
                        tif=order.tif, order_ref=order.source,
                    )
                except Exception as e:
                    return [TextContent(type="text", text=json.dumps({
                        "submitted": False,
                        "error": f"IBKR rejected order: {e}"}))]

                # Burst ledger: a sold put reserves margin until it fills.
                if "impact" in gate_impact:
                    _record_pending_margin_order(
                        ibkr_client.current_account, _placed_order_id(result),
                        gate_impact["impact"])

                staged_store.remove(staged_id)
                return [TextContent(type="text", text=json.dumps({
                    "submitted": True,
                    "staged_id": staged_id,
                    "warnings": warnings,
                    "ibkr": result,
                }, indent=2, default=str))]

            # --- Stock path (unchanged below) ---

            # Re-validate against current market (strict — IBKR must be connected).
            # For STP/STP_LMT orders, validate drift against stop_price (the trigger),
            # since limit_price is 0 for pure STP.
            ref_price = order.limit_price if order.order_type in ("LMT", "STP_LMT") else \
                        (order.stop_price or 0)
            if ref_price > 0:
                v = await _validate_order_inputs(order.symbol, order.action,
                                                 order.quantity, ref_price,
                                                 require_quote=True)
                if not v["ok"]:
                    return [TextContent(type="text",
                                        text=json.dumps({"submitted": False, "error": v["error"]}))]
            else:
                v = {"reference_price": None, "drift_pct": None, "reference_source": None}

            # Funds gate — strict at confirm time (same contract as the drift
            # gate above). For brackets only the parent consumes cash; the
            # SELL children pass through _buy_funds_gate untouched.
            fg = await _buy_funds_gate(order.action, order.quantity,
                                       ref_price if ref_price > 0 else 0.0,
                                       strict=True,
                                       margin_gate=(order.order_type != "STP"))
            if not fg["ok"]:
                return [TextContent(type="text",
                                    text=json.dumps({"submitted": False, "error": fg["error"]}))]

            # Bracket parent? Submit the whole bracket atomically via place_bracket_order.
            children = staged_store.children_of(staged_id)
            try:
                if children:
                    child_specs = []
                    for c in children:
                        spec = {
                            "order_type":  c.order_type,
                            "action":      c.action,
                            "quantity":    c.quantity,
                            "tif":         c.tif,
                            "outside_rth": c.outside_rth,
                            "oca_type":    c.oca_type or 2,
                            "tag":         c.source,
                        }
                        if c.order_type == "LMT":
                            spec["limit_price"] = c.limit_price
                        elif c.order_type == "STP":
                            spec["stop_price"] = c.stop_price
                        else:
                            return [TextContent(type="text", text=json.dumps({
                                "submitted": False,
                                "error": f"Unsupported child order_type for bracket: {c.order_type}",
                            }))]
                        child_specs.append(spec)
                    result = await ibkr_client.place_bracket_order(
                        symbol=order.symbol,
                        parent_action=order.action,
                        parent_quantity=order.quantity,
                        parent_limit_price=order.limit_price,
                        children=child_specs,
                        parent_tif=order.tif,
                        parent_outside_rth=order.outside_rth,
                        order_ref=order.source,
                    )
                elif order.order_type == "LMT":
                    result = await ibkr_client.place_limit_order(
                        symbol=order.symbol, action=order.action,
                        quantity=order.quantity, limit_price=order.limit_price,
                        tif=order.tif, outside_rth=order.outside_rth,
                        order_ref=order.source,
                        oca_group=order.oca_group, oca_type=order.oca_type,
                    )
                elif order.order_type == "STP":
                    result = await ibkr_client.place_stop_order(
                        symbol=order.symbol, action=order.action,
                        quantity=order.quantity, stop_price=order.stop_price,
                        tif=order.tif, outside_rth=order.outside_rth,
                        order_ref=order.source,
                        oca_group=order.oca_group, oca_type=order.oca_type,
                    )
                else:
                    return [TextContent(type="text", text=json.dumps({
                        "submitted": False,
                        "error": f"Unsupported order_type at confirm: {order.order_type}",
                    }))]
            except Exception as e:
                return [TextContent(type="text",
                                    text=json.dumps({"submitted": False,
                                                     "error": f"IBKR rejected order: {e}"}))]

            # Burst ledger: remember a margin BUY just sent (bracket: the parent
            # only) so the next confirms in this session see it until it fills.
            if (order.action.upper() == "BUY" and order.order_type != "STP"
                    and ref_price > 0):
                _record_pending_margin_order(
                    ibkr_client.current_account, _placed_order_id(result),
                    order.quantity * ref_price)

            # Cleanup local staged store: remove parent + bracket children together.
            if children:
                staged_store.remove_bracket(staged_id)
            else:
                staged_store.remove(staged_id)
            return [TextContent(type="text", text=json.dumps({
                "submitted": True,
                "staged_id": staged_id,
                "reference_price": v.get("reference_price"),
                "drift_pct": v.get("drift_pct"),
                "reference_source": v.get("reference_source"),
                "bracket": bool(children),
                "ibkr": result,
            }, indent=2, default=str))]

        elif name == "stage_stop_order":
            symbol = validate_symbol(arguments["symbol"])
            action = arguments["action"]
            quantity = int(arguments["quantity"])
            stop_price = float(arguments["stop_price"])
            tif = arguments.get("tif", "GTC")
            outside_rth = bool(arguments.get("outside_rth", False))
            source = arguments.get("source", "")
            oca_group = arguments.get("oca_group")
            oca_type = int(arguments.get("oca_type", 0))

            v = await _validate_order_inputs(symbol, action, quantity, stop_price,
                                              require_quote=False)
            if not v["ok"]:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False, "error": v["error"]}))]

            # Stops are never margin-gated (margin_gate=False).
            fg = await _buy_funds_gate(action, quantity, stop_price, strict=False,
                                       margin_gate=False)
            if not fg["ok"]:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False, "error": fg["error"]}))]

            try:
                order = StagedOrder.new(symbol, action, quantity,
                                         order_type="STP", stop_price=stop_price,
                                         tif=tif, source=source,
                                         outside_rth=outside_rth,
                                         oca_group=oca_group, oca_type=oca_type)
            except ValueError as e:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False, "error": str(e)}))]
            staged_store.add(order)
            return [TextContent(type="text", text=json.dumps({
                "staged":           True,
                "staged_id":        order.id,
                "summary":          order.summary(),
                "reference_price":  v.get("reference_price"),
                "drift_pct":        v.get("drift_pct"),
                "reference_source": v.get("reference_source"),
            }, indent=2))]

        elif name == "stage_bracket_order":
            symbol = validate_symbol(arguments["symbol"])
            parent_action = arguments.get("parent_action", "BUY")
            parent_qty = int(arguments["parent_quantity"])
            parent_lmt = float(arguments["parent_limit_price"])
            parent_tif = arguments.get("parent_tif", "GTC")
            parent_outside_rth = bool(arguments.get("parent_outside_rth", False))
            children = arguments["children"]
            source = arguments.get("source", "")

            # Validate parent against drift gate.
            v_parent = await _validate_order_inputs(symbol, parent_action, parent_qty,
                                                     parent_lmt, require_quote=False)
            if not v_parent["ok"]:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False,
                                                     "error": f"parent: {v_parent['error']}"}))]

            # Funds gate on the parent (children are exits — no cash needed).
            fg_parent = await _buy_funds_gate(parent_action, parent_qty, parent_lmt,
                                              strict=False)
            if not fg_parent["ok"]:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False,
                                                     "error": f"parent: {fg_parent['error']}"}))]

            # Validate each child up-front. Collect specs; nothing hits the
            # store until all validations pass.
            child_specs = []
            for i, c in enumerate(children):
                ot = c.get("order_type", "LMT").upper()
                action = c.get("action", "SELL").upper()
                qty = int(c["quantity"])
                if ot == "LMT":
                    lmt = float(c.get("limit_price", 0))
                    if lmt <= 0:
                        return [TextContent(type="text", text=json.dumps({
                            "staged": False,
                            "error": f"child[{i}] (LMT): limit_price > 0 required"}))]
                    vc = await _validate_order_inputs(symbol, action, qty, lmt,
                                                       require_quote=False)
                elif ot == "STP":
                    stp = float(c.get("stop_price", 0))
                    if stp <= 0:
                        return [TextContent(type="text", text=json.dumps({
                            "staged": False,
                            "error": f"child[{i}] (STP): stop_price > 0 required"}))]
                    vc = await _validate_order_inputs(symbol, action, qty, stp,
                                                       require_quote=False)
                else:
                    return [TextContent(type="text", text=json.dumps({
                        "staged": False,
                        "error": f"child[{i}]: unsupported order_type {ot!r}"}))]
                if not vc["ok"]:
                    return [TextContent(type="text", text=json.dumps({
                        "staged": False,
                        "error": f"child[{i}]: {vc['error']}"}))]
                child_specs.append({**c, "order_type": ot, "action": action})

            # Stage parent first so we have its id for child linkage. Use a
            # provisional OCA group name reflecting the parent's id.
            try:
                parent = StagedOrder.new(symbol, parent_action, parent_qty,
                                          limit_price=parent_lmt,
                                          order_type="LMT",
                                          tif=parent_tif, source=source,
                                          outside_rth=parent_outside_rth,
                                          transmit_last=False)
            except ValueError as e:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False,
                                                     "error": f"parent: {e}"}))]
            staged_store.add(parent)
            oca_group = f"BRK_{symbol.upper()}_{parent.id}"

            # Stage children, all linked to the parent. A child that fails
            # model-level validation must not leave a half-built bracket in
            # the store — roll the parent back out.
            child_ids = []
            n = len(child_specs)
            for i, c in enumerate(child_specs):
                ot = c["order_type"]
                try:
                    ch = StagedOrder.new(
                        symbol,
                        c.get("action", "SELL"),
                        int(c["quantity"]),
                        limit_price=float(c.get("limit_price", 0) or 0),
                        order_type=ot,
                        stop_price=float(c.get("stop_price", 0)) if ot == "STP" else None,
                        tif=c.get("tif", "GTC"),
                        source=c.get("tag") or f"{source}_C{i}",
                        outside_rth=bool(c.get("outside_rth", False)),
                        oca_group=oca_group,
                        oca_type=int(c.get("oca_type", 2)),
                        parent_staged_id=parent.id,
                        transmit_last=(i == n - 1),
                    )
                except ValueError as e:
                    staged_store.remove_bracket(parent.id)
                    return [TextContent(type="text",
                                        text=json.dumps({"staged": False,
                                                         "error": f"child[{i}]: {e}"}))]
                staged_store.add(ch)
                child_ids.append(ch.id)

            return [TextContent(type="text", text=json.dumps({
                "staged":           True,
                "staged_id":        parent.id,                      # parent id; confirm_order on this submits the bracket
                "child_staged_ids": child_ids,
                "oca_group":        oca_group,
                "summary": {
                    "parent":   parent.summary(),
                    "children": [staged_store.get(cid).summary() for cid in child_ids],
                },
                "reference_price":  v_parent.get("reference_price"),
                "drift_pct":        v_parent.get("drift_pct"),
                "reference_source": v_parent.get("reference_source"),
            }, indent=2))]

        elif name == "modify_live_order":
            order_id = int(arguments["order_id"])
            qty = arguments.get("quantity")
            lmt = arguments.get("limit_price")
            stp = arguments.get("stop_price")
            if qty is None and lmt is None and stp is None:
                return [TextContent(type="text", text=json.dumps({
                    "modified": False,
                    "error": "must provide at least one of: quantity, limit_price, stop_price"}))]

            # Live-trading gate (modify is a real submit).
            if ibkr_client.port in LIVE_PORTS and not settings.enable_live_trading:
                return [TextContent(type="text", text=json.dumps({
                    "modified": False,
                    "error": (f"Live port {ibkr_client.port} detected but ENABLE_LIVE_TRADING=false. "
                              "Refusing to modify."),
                }))]

            # Margin gate (2026-10-15): only a BUY modify that RAISES notional
            # is gated; decreases, SELLs, stops and options pass untouched.
            modify_gate: dict = {}
            refusal = await _modify_margin_refusal(order_id, qty, lmt, out=modify_gate)
            if refusal:
                return [TextContent(type="text", text=json.dumps({
                    "modified": False, "error": refusal}))]

            try:
                result = await ibkr_client.modify_order(
                    order_id=order_id,
                    quantity=int(qty) if qty is not None else None,
                    limit_price=float(lmt) if lmt is not None else None,
                    stop_price=float(stp) if stp is not None else None,
                )
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "modified": False, "error": f"IBKR modify failed: {e}"}))]
            if isinstance(result, dict) and result.get("modified") and "delta" in modify_gate:
                _adjust_pending_margin_order(order_id, modify_gate["delta"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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

        elif name == "get_market_quotes":
            try:
                symbol_list = validate_symbols(arguments["symbols"])
            except ValueError as e:
                return [TextContent(type="text",
                                    text=json.dumps({"error": str(e)}))]
            quotes = await ibkr_client.get_quotes(symbol_list)
            return [TextContent(type="text", text=json.dumps(quotes, indent=2))]

        elif name == "get_option_chain":
            symbol = validate_symbol(arguments["symbol"])
            chain = await ibkr_client.get_option_chain(symbol)
            return [TextContent(type="text", text=json.dumps(chain, indent=2))]

        elif name == "get_option_quote":
            try:
                spec = _option_contract_spec(arguments)
            except ValueError as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
            quote = await ibkr_client.get_option_quote(**spec)
            return [TextContent(type="text", text=json.dumps(quote, indent=2))]

        elif name == "stage_option_order":
            try:
                spec = _option_contract_spec(arguments)
            except ValueError as e:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False, "error": str(e)}))]

            intent = (arguments.get("intent") or "").upper()
            action = (arguments.get("action") or "").upper()
            quantity = int(arguments["quantity"])
            limit_price = float(arguments["limit_price"])
            tif = (arguments.get("tif") or "DAY").upper()
            source = arguments.get("source") or arguments.get("order_ref") or ""
            allow_no_quote = bool(arguments.get("allow_no_quote", False))

            try:
                contract = await ibkr_client.qualify_option(**spec)
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "staged": False,
                    "error": f"could not qualify option contract: {e}"}))]

            ok, reasons, warnings = await _run_option_gates(
                intent=intent, action=action, quantity=quantity,
                limit_price=limit_price, contract=contract,
                allow_no_quote=allow_no_quote,
            )
            if not ok:
                return [TextContent(type="text", text=json.dumps({
                    "staged": False,
                    "error": "; ".join(reasons),
                    "reasons": reasons,
                    "warnings": warnings,
                }))]

            try:
                order = StagedOrder.new(
                    symbol=contract.symbol, action=action, quantity=quantity,
                    limit_price=limit_price, tif=tif, source=source,
                    order_type="LMT",
                    sec_type="OPT",
                    conid=contract.conId,
                    expiry=getattr(contract, "lastTradeDateOrContractMonth", None) or None,
                    strike=safe_float(getattr(contract, "strike", 0)) or None,
                    right=getattr(contract, "right", None) or None,
                    multiplier=getattr(contract, "multiplier", None) or None,
                    intent=intent,
                )
            except ValueError as e:
                return [TextContent(type="text",
                                    text=json.dumps({"staged": False, "error": str(e)}))]
            staged_store.add(order)
            return [TextContent(type="text", text=json.dumps({
                "staged":    True,
                "staged_id": order.id,
                "summary":   order.summary(),
                "warnings":  warnings,
            }, indent=2))]

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
