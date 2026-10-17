"""Tests for the margin gate wiring in tools.py (owner 2026-10-15).

On 2026-10-14 buying power on the margin account U4022128 fell from $97.6k to
$73.6k after manual buys and nothing stopped new bot entries. The shared rule
lives in the Trader repo (scripts/live/margin_gate.py, loaded here by file
path). These tests cover the MCP-side wiring:

  * stock BUY on the margin account  (_buy_funds_gate, margin path)
  * option sell-to-open              (_run_option_gates via stage_option_order)
  * BUY modifies that raise notional (modify_live_order)
  * the inline fallback when margin_gate.py cannot be loaded
  * never gating the cash account, SELLs, stops, BTC/STC, or decreases

The real module is used where available (its latch file is redirected to a tmp
dir); TRADER_DIR points at it. Tests that need the real module skip if it is
not on disk.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ibkr_mcp_server import tools
from ibkr_mcp_server.orders import StagedOrderStore
from ibkr_mcp_server.tools import _buy_funds_gate, call_tool

MARGIN_ACCT = "U4022128"
CASH_ACCT = "U25242754"

_TRADER_CANDIDATES = (
    os.environ.get("MARGIN_GATE_TEST_TRADER_DIR", ""),   # e.g. a Trader worktree
    "/Users/ttang/Trader",
)


def _real_trader_dir():
    for c in _TRADER_CANDIDATES:
        if c and (Path(c) / "scripts" / "live" / "margin_gate.py").is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch, tmp_path):
    """Every test starts with no cached gate module, funds or quote snapshot, and
    the shared module's files (broker_state / latch) point at tmp_path."""
    monkeypatch.setenv("MARGIN_GATE_LIVE_DIR", str(tmp_path))
    # Rule tests use the 2026-10-14 numbers against a $75k line; the shipped
    # line is $200k (owner 2026-10-17) - see test_shipped_line_is_200k.
    monkeypatch.setattr(tools, "_MARGIN_GATE_LINE_USD", 75000.0)
    monkeypatch.setattr(tools, "_MG_MODULE", None)
    monkeypatch.setattr(tools, "_MG_FAILED_AT", None)
    monkeypatch.setattr(tools, "_FUNDS_CACHE", {})
    monkeypatch.setattr(tools, "_QUOTE_CACHE", {})
    monkeypatch.setattr(tools, "_MG_PENDING", [])


@pytest.fixture
def real_gate(tmp_path, monkeypatch):
    """Load the real shared margin_gate module with its latch dir in tmp_path."""
    d = _real_trader_dir()
    if d is None:
        pytest.skip("Trader scripts/live/margin_gate.py not found")
    monkeypatch.setenv("TRADER_DIR", d)
    mod = tools._load_margin_gate()
    assert mod is not None, "real margin_gate failed to load"
    monkeypatch.setattr(mod.C, "LIVE_DIR", tmp_path)
    monkeypatch.setenv("MARGIN_GATE_LIVE_DIR", str(tmp_path))  # never the live latch
    orig = mod.gate_config
    monkeypatch.setattr(mod, "gate_config", lambda *a, **k: {
        **orig(*a, **k), "buying_power_crit_usd": 75000, "buying_power_clear_usd": 5000})
    return mod


@pytest.fixture
def no_gate_module(tmp_path, monkeypatch):
    """TRADER_DIR with no margin_gate.py -> the inline fallback is used."""
    monkeypatch.setenv("TRADER_DIR", str(tmp_path / "nowhere"))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    store = StagedOrderStore(path=tmp_path / "staged.json")
    monkeypatch.setattr(tools, "staged_store", store)
    return store


def _snapshot(monkeypatch, tags: dict):
    async def fake_snapshot(account):
        return tags
    monkeypatch.setattr(tools, "_funds_snapshot", fake_snapshot)


def _client(monkeypatch, account=MARGIN_ACCT, connected=True):
    client = MagicMock()
    client.is_connected.return_value = connected
    client.current_account = account
    client.port = 7497
    client.ib.openTrades.return_value = []
    client._ensure_connected = AsyncMock(return_value=True)
    monkeypatch.setattr(tools, "ibkr_client", client)
    return client


def _margin_tags(bp, af=30000):
    return {"AccountType": "INDIVIDUAL", "BuyingPower": str(bp),
            "AvailableFunds": str(af)}


def _summary_rows(bp, af=30000):
    return [{"tag": "BuyingPower", "value": str(bp), "currency": "USD"},
            {"tag": "AvailableFunds", "value": str(af), "currency": "USD"},
            {"tag": "AccountType", "value": "INDIVIDUAL", "currency": ""}]


# ---------------------------------------------------------------------------
# stock BUY  (_buy_funds_gate, margin path)
# ---------------------------------------------------------------------------

class TestStockBuy:
    async def test_blocked_below_line(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(73557))
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert not r["ok"]
        assert r["error"].startswith("Margin gate: ")
        assert "$73,557" in r["error"] and "$75,000" in r["error"]
        assert "below the" in r["error"] and "blocked" in r["error"]

    async def test_blocked_at_confirm_time_too(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(73557))
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=True)
        assert not r["ok"] and r["error"].startswith("Margin gate: ")

    async def test_order_that_would_cross_the_line_blocked(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(80000))
        r = await _buy_funds_gate("BUY", 78, 100.0, strict=False)   # $7,800
        assert not r["ok"]
        assert r["error"].startswith("Margin gate: ")
        assert "$80,000" in r["error"] and "$72,200" in r["error"]
        assert "$75,000" in r["error"]

    async def test_clear_above_line(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(100000))
        r = await _buy_funds_gate("BUY", 100, 100.0, strict=False)  # $10,000
        assert r["ok"]
        assert r["funds_headroom_after"] == 90000.0

    async def test_latch_holds_until_release_line(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(73557))                 # trips the latch
        assert not (await _buy_funds_gate("BUY", 1, 100.0, strict=False))["ok"]
        _snapshot(monkeypatch, _margin_tags(77000))                 # above line, under $80k
        r = await _buy_funds_gate("BUY", 1, 100.0, strict=False)
        assert not r["ok"] and "release line" in r["error"]
        _snapshot(monkeypatch, _margin_tags(82000))                 # released
        assert (await _buy_funds_gate("BUY", 1, 100.0, strict=False))["ok"]

    async def test_unknown_buying_power_fails_closed(self, monkeypatch, real_gate):
        """Snapshot with no BuyingPower and no broker snapshot -> held."""
        _client(monkeypatch)
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL", "AvailableFunds": "50000"})
        r = await _buy_funds_gate("BUY", 1, 100.0, strict=False)
        assert not r["ok"]
        assert r["error"].startswith("Margin gate: ")
        assert "unknown" in r["error"]

    async def test_cash_account_unaffected(self, monkeypatch, real_gate):
        """U25242754 keeps its own cash gates; the margin gate says n/a."""
        _client(monkeypatch, account=CASH_ACCT)
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL",
                                "BuyingPower": "1000", "AvailableFunds": "50000"})
        # BP is far below the margin line, yet the cash account is not gated.
        r = await _buy_funds_gate("BUY", 5, 100.0, strict=False)
        assert r["ok"], r
        # CASH branch is still the original one (2026-07-29 incident wording).
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "189.21"})
        r = await _buy_funds_gate("BUY", 12, 181.82, strict=False)
        assert not r["ok"]
        assert "exceeds available funds" in r["error"]
        assert not r["error"].startswith("Margin gate")

    async def test_sell_never_gated(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(50000))
        assert (await _buy_funds_gate("SELL", 100, 100.0, strict=True))["ok"]

    async def test_stop_order_not_gated(self, monkeypatch, real_gate):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(50000))
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False, margin_gate=False)
        assert r["ok"]

    async def test_stage_order_refusal_shape(self, monkeypatch, real_gate, staged):
        client = _client(monkeypatch)
        client.get_quote = AsyncMock(return_value={"last": 100.0, "source": "live"})
        client.get_account_summary = AsyncMock(return_value=_summary_rows(73557))
        res = await call_tool("stage_order", {
            "symbol": "MU", "action": "BUY", "quantity": 10, "limit_price": 100.0,
            "source": "TEST"})
        body = json.loads(res[0].text)
        assert body["staged"] is False
        assert body["error"].startswith("Margin gate: Buying power is $73,557")
        assert staged.list() == []

    async def test_stage_stop_order_never_margin_gated(self, monkeypatch, real_gate, staged):
        client = _client(monkeypatch)
        client.get_quote = AsyncMock(return_value={"last": 100.0, "source": "live"})
        client.get_account_summary = AsyncMock(return_value=_summary_rows(73557))
        res = await call_tool("stage_stop_order", {
            "symbol": "MU", "action": "SELL", "quantity": 10, "stop_price": 95.0,
            "source": "SWING_CAT_STOP_TEST"})
        body = json.loads(res[0].text)
        assert body["staged"] is True, body


# ---------------------------------------------------------------------------
# option sell-to-open  (stage_option_order -> _run_option_gates)
# ---------------------------------------------------------------------------

def _contract(conid=111, symbol="NVDA", right="P"):
    return SimpleNamespace(conId=conid, symbol=symbol, right=right, strike=20.0,
                           lastTradeDateOrContractMonth="20261016", multiplier="100",
                           localSymbol="NVDA  261016P00020000")


def _opt_client(monkeypatch, *, bp, af=30000, contract=None, positions=None,
                whatif=None, account=MARGIN_ACCT):
    client = _client(monkeypatch, account=account)
    client.qualify_option = AsyncMock(return_value=contract or _contract())
    client.get_portfolio = AsyncMock(return_value=positions or [])
    client.get_open_trades = AsyncMock(return_value=[])
    client.get_option_quote = AsyncMock(return_value={"bid": 1.00, "ask": 1.10})
    client.whatif_option_order = AsyncMock(return_value=whatif)
    client.get_account_summary = AsyncMock(return_value=_summary_rows(bp, af))
    return client


def _opt_args(intent, action, conid=111):
    return {"conid": conid, "intent": intent, "action": action, "quantity": 1,
            "limit_price": 1.00, "source": "WHEEL_TEST"}


class TestOptionSellToOpen:
    async def test_sto_put_blocked_below_line(self, monkeypatch, real_gate, staged):
        _opt_client(monkeypatch, bp=73557, whatif={"init_margin_change": 500.0})
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL"))
        body = json.loads(res[0].text)
        assert body["staged"] is False
        assert body["error"].startswith("Margin gate: Buying power is $73,557")
        assert "$75,000" in body["error"]
        assert body["reasons"] == [body["error"]]
        assert staged.list() == []

    async def test_sto_put_that_would_cross_blocked(self, monkeypatch, real_gate, staged):
        # init margin $2,000 x (BP/AF = 80,000/24,000 = 3.33) = $6,667 impact
        _opt_client(monkeypatch, bp=80000, af=24000,
                    whatif={"init_margin_change": 2000.0})
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL"))
        body = json.loads(res[0].text)
        assert body["staged"] is False
        assert body["error"].startswith("Margin gate: This order would take buying power")
        assert "$75,000" in body["error"]

    async def test_sto_put_clear_above_line(self, monkeypatch, real_gate, staged):
        _opt_client(monkeypatch, bp=150000, af=45000,
                    whatif={"init_margin_change": 500.0})
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL"))
        body = json.loads(res[0].text)
        assert body["staged"] is True, body

    async def test_confirm_path_also_gated(self, monkeypatch, real_gate, staged):
        """confirm_order re-runs the gates against fresh state."""
        from ibkr_mcp_server.orders import StagedOrder
        o = StagedOrder.new("NVDA", "SELL", 1, limit_price=1.00, source="WHEEL_TEST",
                            sec_type="OPT", conid=111, expiry="20261016", strike=20.0,
                            right="P", multiplier="100", intent="STO")
        staged.add(o)
        client = _opt_client(monkeypatch, bp=73557,
                             whatif={"init_margin_change": 500.0})
        client.place_option_limit_order = AsyncMock()
        res = await call_tool("confirm_order", {"staged_id": o.id})
        body = json.loads(res[0].text)
        assert body["submitted"] is False
        assert body["error"].startswith("Margin gate: ")
        client.place_option_limit_order.assert_not_awaited()
        assert staged.get(o.id) is not None   # still staged

    async def test_covered_call_sto_not_gated_below_line(
            self, monkeypatch, real_gate, staged):
        # A covered call uses no margin (naked calls are refused by option_gates).
        shares = [{"secType": "STK", "symbol": "NVDA", "position": 200}]
        _opt_client(monkeypatch, bp=73557, contract=_contract(conid=222, right="C"),
                    positions=shares)
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL", conid=222))
        assert json.loads(res[0].text)["staged"] is True

    async def test_covered_call_sto_clear_above_line(self, monkeypatch, real_gate, staged):
        shares = [{"secType": "STK", "symbol": "NVDA", "position": 200}]
        _opt_client(monkeypatch, bp=120000, contract=_contract(conid=222, right="C"),
                    positions=shares)
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL", conid=222))
        assert json.loads(res[0].text)["staged"] is True

    async def test_btc_not_gated(self, monkeypatch, real_gate, staged):
        short_put = [{"conid": "111", "secType": "OPT", "symbol": "NVDA",
                      "right": "P", "position": -1}]
        _opt_client(monkeypatch, bp=50000, positions=short_put)
        res = await call_tool("stage_option_order", _opt_args("BTC", "BUY"))
        body = json.loads(res[0].text)
        assert body["staged"] is True, body

    async def test_stc_not_gated(self, monkeypatch, real_gate, staged):
        long_put = [{"conid": "111", "secType": "OPT", "symbol": "NVDA",
                     "right": "P", "position": 1}]
        _opt_client(monkeypatch, bp=50000, positions=long_put)
        res = await call_tool("stage_option_order", _opt_args("STC", "SELL"))
        body = json.loads(res[0].text)
        assert body["staged"] is True, body

    async def test_cash_account_sto_not_gated(self, monkeypatch, real_gate, staged):
        _opt_client(monkeypatch, bp=1000, account=CASH_ACCT,
                    whatif={"init_margin_change": 500.0})
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL"))
        assert json.loads(res[0].text)["staged"] is True

    def test_option_impact_zero_when_whatif_unavailable(self, real_gate):
        assert tools._margin_gate_option_impact(None, 80000, 24000) == 0.0
        assert tools._margin_gate_option_impact({"error": "x"}, 80000, 24000) == 0.0
        assert tools._margin_gate_option_impact(
            {"init_margin_change": None}, 80000, 24000) == 0.0

    def test_option_impact_scaled_by_bp_over_af(self, real_gate):
        got = tools._margin_gate_option_impact(
            {"init_margin_change": 1000.0}, 90000, 30000)
        assert got == pytest.approx(3000.0)
        # ratio unknown -> the module's 4x fallback
        assert tools._margin_gate_option_impact(
            {"init_margin_change": 1000.0}, None, None) == pytest.approx(4000.0)


# ---------------------------------------------------------------------------
# modify_live_order
# ---------------------------------------------------------------------------

def _trade(order_id=77, action="BUY", qty=10, lmt=100.0, order_type="LMT",
           sec_type="STK", account=MARGIN_ACCT, filled=0, aux=0.0):
    order = SimpleNamespace(orderId=order_id, action=action, totalQuantity=qty,
                            lmtPrice=lmt, auxPrice=aux, orderType=order_type,
                            account=account)
    return SimpleNamespace(order=order, contract=SimpleNamespace(secType=sec_type),
                           orderStatus=SimpleNamespace(filled=filled))


def _modify_client(monkeypatch, trade, bp):
    client = _client(monkeypatch)
    client.ib.trades.return_value = [trade]
    client.modify_order = AsyncMock(return_value={"order_id": trade.order.orderId,
                                                  "modified": True})
    client.get_account_summary = AsyncMock(return_value=_summary_rows(bp))
    return client


async def _modify(**args):
    res = await call_tool("modify_live_order", {"order_id": 77, **args})
    return json.loads(res[0].text)


class TestModifyLiveOrder:
    async def test_quantity_increase_blocked_below_line(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(qty=10, lmt=100.0), bp=73557)
        body = await _modify(quantity=20)
        assert body["modified"] is False
        assert body["error"].startswith("Margin gate: Buying power is $73,557")
        client.modify_order.assert_not_awaited()

    async def test_limit_increase_that_would_cross_blocked(self, monkeypatch, real_gate):
        # 700 sh @ $100 -> @ $111 adds $7,700; $80,000 - $7,700 = $72,300 < line
        client = _modify_client(monkeypatch, _trade(qty=700, lmt=100.0), bp=80000)
        body = await _modify(limit_price=111.0)
        assert body["modified"] is False
        assert body["error"].startswith("Margin gate: This order would take buying power")
        assert "$72,300" in body["error"]
        client.modify_order.assert_not_awaited()

    async def test_increase_allowed_when_clear(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(qty=10, lmt=100.0), bp=150000)
        body = await _modify(quantity=20)
        assert body["modified"] is True
        client.modify_order.assert_awaited_once()

    async def test_decrease_never_gated(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(qty=10, lmt=100.0), bp=50000)
        assert (await _modify(quantity=5))["modified"] is True
        assert (await _modify(limit_price=90.0))["modified"] is True
        assert (await _modify(quantity=5, limit_price=90.0))["modified"] is True
        assert client.modify_order.await_count == 3

    async def test_qty_up_but_price_down_net_decrease_allowed(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(qty=10, lmt=100.0), bp=50000)
        body = await _modify(quantity=11, limit_price=80.0)   # $880 < $1,000
        assert body["modified"] is True
        client.modify_order.assert_awaited_once()

    async def test_sell_modification_never_gated(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(action="SELL", qty=10, lmt=100.0),
                                bp=50000)
        assert (await _modify(quantity=50))["modified"] is True
        client.modify_order.assert_awaited_once()

    async def test_stop_order_modification_never_gated(self, monkeypatch, real_gate):
        client = _modify_client(
            monkeypatch,
            _trade(action="SELL", order_type="STP", lmt=0.0, aux=90.0), bp=50000)
        assert (await _modify(stop_price=95.0))["modified"] is True
        client.modify_order.assert_awaited_once()

    async def test_option_buy_modification_never_gated(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(qty=1, lmt=1.0, sec_type="OPT"),
                                bp=50000)
        assert (await _modify(quantity=5))["modified"] is True
        client.modify_order.assert_awaited_once()

    async def test_cash_account_order_not_gated(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(account=CASH_ACCT), bp=1000)
        client.current_account = CASH_ACCT
        assert (await _modify(quantity=20))["modified"] is True

    async def test_partial_fill_only_counts_unfilled(self, monkeypatch, real_gate):
        # 700 total, 400 filled, @ $100 -> unfilled $30,000; price to $101
        # adds only 300 x $1 = $300, so BP $75,200 -> $74,900 is below the line
        client = _modify_client(monkeypatch, _trade(qty=700, lmt=100.0, filled=400),
                                bp=75200)
        body = await _modify(limit_price=101.0)
        assert body["modified"] is False
        assert "$74,900" in body["error"]
        client.modify_order.assert_not_awaited()

    async def test_unknown_order_falls_through_to_modify_order(self, monkeypatch, real_gate):
        client = _modify_client(monkeypatch, _trade(order_id=1), bp=50000)
        client.modify_order = AsyncMock(return_value={"order_id": 77, "modified": False,
                                                      "error": "order not found"})
        body = await _modify(quantity=20)
        assert body["error"] == "order not found"


# ---------------------------------------------------------------------------
# fallback when margin_gate.py cannot be loaded
# ---------------------------------------------------------------------------

class TestFallbackWhenModuleMissing:
    async def test_missing_module_still_blocks_below_line(self, monkeypatch, no_gate_module):
        assert tools._load_margin_gate() is None
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(73557))
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert not r["ok"]
        assert r["error"] == ("Margin gate: Buying power is $73,557, below the $75,000 "
                              "margin line, so new orders that use margin are blocked.")

    async def test_missing_module_blocks_a_crossing_order(self, monkeypatch, no_gate_module):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(80000))
        r = await _buy_funds_gate("BUY", 78, 100.0, strict=False)
        assert not r["ok"] and "$72,200" in r["error"]

    async def test_missing_module_allows_clear_order(self, monkeypatch, no_gate_module):
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(100000))
        assert (await _buy_funds_gate("BUY", 10, 100.0, strict=False))["ok"]

    async def test_missing_module_blocks_unknown_buying_power(self, monkeypatch, no_gate_module):
        _client(monkeypatch)
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL", "AvailableFunds": "50000"})
        r = await _buy_funds_gate("BUY", 1, 100.0, strict=False)
        assert not r["ok"] and "unknown" in r["error"]

    async def test_missing_module_cash_account_unaffected(self, monkeypatch, no_gate_module):
        _client(monkeypatch, account=CASH_ACCT)
        _snapshot(monkeypatch, _margin_tags(1000))
        assert (await _buy_funds_gate("BUY", 5, 100.0, strict=False))["ok"]

    async def test_missing_module_blocks_modify_increase(self, monkeypatch, no_gate_module):
        client = _modify_client(monkeypatch, _trade(qty=10, lmt=100.0), bp=73557)
        body = await _modify(quantity=20)
        assert body["modified"] is False and body["error"].startswith("Margin gate: ")
        client.modify_order.assert_not_awaited()

    async def test_missing_module_blocks_sto_put(self, monkeypatch, no_gate_module, staged):
        _opt_client(monkeypatch, bp=73557, whatif={"init_margin_change": 500.0})
        res = await call_tool("stage_option_order", _opt_args("STO", "SELL"))
        body = json.loads(res[0].text)
        assert body["staged"] is False and body["error"].startswith("Margin gate: ")

    def test_missing_module_option_impact_inline(self, no_gate_module):
        got = tools._margin_gate_option_impact({"init_margin_change": 1000.0}, 90000, 30000)
        assert got == pytest.approx(3000.0)
        assert tools._margin_gate_option_impact(None, 90000, 30000) == 0.0

    async def test_import_error_in_module_falls_back(self, monkeypatch, tmp_path):
        d = tmp_path / "trader"
        (d / "scripts" / "live").mkdir(parents=True)
        (d / "scripts" / "live" / "margin_gate.py").write_text("raise RuntimeError('boom')\n")
        monkeypatch.setenv("TRADER_DIR", str(d))
        assert tools._load_margin_gate() is None
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(73557))
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert not r["ok"] and r["error"].startswith("Margin gate: ")

    async def test_evaluate_raising_falls_back(self, monkeypatch, real_gate):
        def boom(*a, **k):
            raise RuntimeError("evaluate exploded")
        monkeypatch.setattr(real_gate, "evaluate", boom)
        _client(monkeypatch)
        _snapshot(monkeypatch, _margin_tags(73557))
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert not r["ok"] and r["error"].startswith("Margin gate: ")
        _snapshot(monkeypatch, _margin_tags(100000))
        assert (await _buy_funds_gate("BUY", 10, 100.0, strict=False))["ok"]

    def test_failed_load_not_retried_inside_window(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRADER_DIR", str(tmp_path / "nowhere"))
        assert tools._load_margin_gate() is None
        # file appears, but the retry window has not elapsed yet
        d = _real_trader_dir()
        if d is None:
            pytest.skip("Trader margin_gate.py not found")
        monkeypatch.setenv("TRADER_DIR", d)
        assert tools._load_margin_gate() is None
        monkeypatch.setattr(tools, "_MG_FAILED_AT", None)   # window elapsed
        assert tools._load_margin_gate() is not None


class TestModuleLoading:
    def test_load_is_cached_and_leaves_sys_path_alone(self, monkeypatch):
        d = _real_trader_dir()
        if d is None:
            pytest.skip("Trader margin_gate.py not found")
        monkeypatch.setenv("TRADER_DIR", d)
        before = list(sys.path)
        first = tools._load_margin_gate()
        assert first is not None and tools._load_margin_gate() is first
        assert sys.path == before


# ---------------------------------------------------------------------------
# burst ledger: several confirms in one MCP process (the 09:15 swing scout)
# ---------------------------------------------------------------------------

_DONE = ("Filled", "Cancelled", "ApiCancelled", "Inactive")


class _Book:
    """Fake broker order book behind ib.openTrades()/ib.trades()/place_*()."""

    def __init__(self, client, account=MARGIN_ACCT, client_id=1):
        self.trades = []
        self.next_id = 100
        self.account = account
        self.client_id = client_id
        client.client_id = client_id
        client.ib.openTrades.side_effect = lambda: [
            t for t in self.trades if t.orderStatus.status not in _DONE]
        client.ib.trades.side_effect = lambda: list(self.trades)
        client.place_limit_order = AsyncMock(side_effect=self._place_limit)
        client.place_option_limit_order = AsyncMock(side_effect=self._place_opt)
        client.modify_order = AsyncMock(side_effect=self._modify)
        client.get_quote = AsyncMock(return_value={"last": 100.0, "source": "live"})

    def _add(self, action, qty, px, sec_type, order_id=None, client_id=None):
        oid = order_id or self.next_id
        self.next_id = max(self.next_id, oid) + 1
        t = _trade(order_id=oid, action=action, qty=qty, lmt=px, sec_type=sec_type,
                   account=self.account)
        t.order.clientId = client_id if client_id is not None else self.client_id
        t.orderStatus.status = "Submitted"
        self.trades.append(t)
        return t

    async def _place_limit(self, symbol, action, quantity, limit_price, **kw):
        t = self._add(action, quantity, limit_price, "STK")
        return {"order_id": t.order.orderId, "status": "Submitted"}

    async def _place_opt(self, contract, action, quantity, limit_price, **kw):
        t = self._add(action, quantity, limit_price, "OPT")
        return {"order_id": t.order.orderId, "status": "Submitted"}

    async def _modify(self, order_id, quantity=None, limit_price=None, stop_price=None):
        for t in self.trades:
            if t.order.orderId == order_id:
                if quantity is not None:
                    t.order.totalQuantity = quantity
                if limit_price is not None:
                    t.order.lmtPrice = limit_price
        return {"order_id": order_id, "modified": True}

    def order(self, order_id):
        return next(t for t in self.trades if t.order.orderId == order_id)

    def fill(self, order_id):
        t = self.order(order_id)
        t.orderStatus.status, t.orderStatus.filled = "Filled", t.order.totalQuantity

    def partial_fill(self, order_id, n=10):
        self.order(order_id).orderStatus.filled = n

    def cancel(self, order_id):
        self.order(order_id).orderStatus.status = "Cancelled"


def _burst_client(monkeypatch, bp=95000, account=MARGIN_ACCT):
    client = _client(monkeypatch, account=account)
    client.get_account_summary = AsyncMock(return_value=_summary_rows(bp))
    return client, _Book(client, account=account)


async def _stage_buy(qty=80, px=100.0):
    res = await call_tool("stage_order", {"symbol": "MU", "action": "BUY",
                                          "quantity": qty, "limit_price": px,
                                          "source": "SWING_TEST"})
    return json.loads(res[0].text)


async def _confirm(staged_id):
    res = await call_tool("confirm_order", {"staged_id": staged_id})
    return json.loads(res[0].text)


_THIRD_REFUSAL = (
    "Margin gate: This order would take buying power from $79,000 to $71,000, "
    "below the $75,000 margin line (counting $16,000 of orders already sent this "
    "session that have not filled yet).")


class TestBurstLedger:
    async def test_three_confirms_third_refused(self, monkeypatch, real_gate, staged):
        """BP $95k, three $8k entries staged up front: $87k pass, $79k pass,
        third would land at $71k -> refused at confirm."""
        client, book = _burst_client(monkeypatch)
        ids = [(await _stage_buy())["staged_id"] for _ in range(3)]
        assert (await _confirm(ids[0]))["submitted"] is True
        assert (await _confirm(ids[1]))["submitted"] is True
        body = await _confirm(ids[2])
        assert body["submitted"] is False
        assert body["error"] == _THIRD_REFUSAL
        assert client.place_limit_order.await_count == 2
        assert staged.get(ids[2]) is not None          # still staged, not consumed
        assert [e["impact"] for e in tools._MG_PENDING] == [8000.0, 8000.0]

    async def test_stage_time_check_counts_pending_too(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        for _ in range(2):
            assert (await _confirm((await _stage_buy())["staged_id"]))["submitted"]
        body = await _stage_buy()
        assert body["staged"] is False
        assert body["error"] == _THIRD_REFUSAL

    async def test_single_order_unaffected_with_empty_ledger(self, monkeypatch, real_gate, staged):
        _burst_client(monkeypatch)
        assert (await _confirm((await _stage_buy())["staged_id"]))["submitted"]

    @pytest.mark.parametrize("how", ["filled", "partial", "cancelled", "gone"])
    async def test_first_order_done_no_longer_counts(self, monkeypatch, real_gate, staged, how):
        client, book = _burst_client(monkeypatch)
        ids = [(await _stage_buy())["staged_id"] for _ in range(3)]
        r1 = await _confirm(ids[0])
        assert (await _confirm(ids[1]))["submitted"]
        first = r1["ibkr"]["order_id"]
        {"filled": book.fill, "partial": book.partial_fill, "cancelled": book.cancel,
         "gone": lambda oid: book.trades.remove(book.order(oid))}[how](first)
        # only the second order (8k) is still counted: 95 - 8 - 8 = 79k -> pass
        body = await _confirm(ids[2])
        assert body["submitted"] is True, body
        assert [e["impact"] for e in tools._MG_PENDING] == [8000.0, 8000.0]
        assert first not in [e["order_id"] for e in tools._MG_PENDING]

    async def test_done_order_is_pruned_from_ledger(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        r1 = await _confirm((await _stage_buy())["staged_id"])
        assert len(tools._MG_PENDING) == 1
        book.fill(r1["ibkr"]["order_id"])
        assert tools._pending_margin_impact(MARGIN_ACCT) == 0.0
        assert tools._MG_PENDING == []

    async def test_other_orders_in_the_book_do_not_count(self, monkeypatch, real_gate, staged):
        """Owner's manual ladder (not confirmed by this process) is ignored."""
        client, book = _burst_client(monkeypatch)
        book._add("BUY", 1000, 200.0, "STK", order_id=900, client_id=49)  # $200k ladder
        body = await _confirm((await _stage_buy())["staged_id"])
        assert body["submitted"] is True
        assert tools._pending_margin_impact(MARGIN_ACCT) == 8000.0

    async def test_other_client_same_order_id_not_mistaken(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        r1 = await _confirm((await _stage_buy())["staged_id"])
        oid = r1["ibkr"]["order_id"]
        # our order vanishes; a different client reuses the same numeric id
        book.trades.remove(book.order(oid))
        book._add("BUY", 1000, 200.0, "STK", order_id=oid, client_id=49)
        assert tools._pending_margin_impact(MARGIN_ACCT) == 0.0

    async def test_lookup_failure_counts_as_zero_and_never_raises(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        for _ in range(2):
            assert (await _confirm((await _stage_buy())["staged_id"]))["submitted"]
        client.ib.openTrades.side_effect = RuntimeError("TWS hiccup")
        assert tools._pending_margin_impact(MARGIN_ACCT) == 0.0
        r = await _buy_funds_gate("BUY", 80, 100.0, strict=False)
        assert r["ok"], r                                   # pending treated as 0
        assert len(tools._MG_PENDING) == 2                   # entries kept
        client.ib.openTrades.side_effect = lambda: list(book.trades)
        r = await _buy_funds_gate("BUY", 80, 100.0, strict=False)
        assert not r["ok"] and "counting $16,000" in r["error"]

    async def test_pending_adjusted_pass_does_not_trip_shared_latch(
            self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        for _ in range(2):
            assert (await _confirm((await _stage_buy())["staged_id"]))["submitted"]
        # a big order: adjusted BP is well under the line
        r = await _buy_funds_gate("BUY", 80, 100.0, strict=False)
        assert not r["ok"] and "counting $16,000" in r["error"]
        latch = real_gate.C.read_json(real_gate.latch_path())
        assert latch[MARGIN_ACCT]["tripped"] is False       # real BP is $95k

    async def test_live_below_line_refusal_has_no_pending_note(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch, bp=73557)
        body = await _stage_buy()
        assert body["staged"] is False
        assert "counting" not in body["error"]

    async def test_cash_account_unaffected(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch, account=CASH_ACCT)
        ids = [(await _stage_buy())["staged_id"] for _ in range(3)]
        for sid in ids:
            assert (await _confirm(sid))["submitted"] is True
        assert tools._MG_PENDING == []                      # never recorded

    async def test_margin_ledger_does_not_bleed_into_cash_account(
            self, monkeypatch, real_gate):
        client, book = _burst_client(monkeypatch, account=CASH_ACCT)
        book._add("BUY", 80, 100.0, "STK", order_id=5)
        tools._MG_PENDING.append({"order_id": 5, "impact": 90000.0,
                                  "account": MARGIN_ACCT, "client_id": 1})
        _snapshot(monkeypatch, _margin_tags(95000))
        assert (await _buy_funds_gate("BUY", 80, 100.0, strict=False))["ok"]

    async def test_stop_and_sell_confirms_are_not_recorded(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        client.place_stop_order = AsyncMock(return_value={"order_id": 7})
        res = await call_tool("stage_stop_order", {
            "symbol": "MU", "action": "SELL", "quantity": 10, "stop_price": 95.0,
            "source": "SWING_CAT_STOP_TEST"})
        sid = json.loads(res[0].text)["staged_id"]
        assert (await _confirm(sid))["submitted"] is True
        res = await call_tool("stage_order", {"symbol": "MU", "action": "SELL",
                                              "quantity": 10, "limit_price": 100.0})
        assert (await _confirm(json.loads(res[0].text)["staged_id"]))["submitted"] is True
        assert tools._MG_PENDING == []

    async def test_bracket_parent_recorded_children_not(self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        client.place_bracket_order = AsyncMock(return_value={
            "parent": {"order_id": 321, "status": "Submitted"}, "children": []})
        book._add("BUY", 80, 100.0, "STK", order_id=321)
        res = await call_tool("stage_bracket_order", {
            "symbol": "MU", "parent_quantity": 80, "parent_limit_price": 100.0,
            "children": [{"order_type": "STP", "quantity": 80, "stop_price": 90.0}],
            "source": "SWING_TEST"})
        sid = json.loads(res[0].text)["staged_id"]
        assert (await _confirm(sid))["submitted"] is True
        assert [(e["order_id"], e["impact"]) for e in tools._MG_PENDING] == [(321, 8000.0)]

    async def test_sto_puts_and_stock_buys_share_the_ledger(self, monkeypatch, real_gate, staged):
        # BP $100k, AF $30k -> ratio 3.33; init margin $3,000 -> $10,000 impact
        client = _opt_client(monkeypatch, bp=100000, af=30000,
                             whatif={"init_margin_change": 3000.0})
        book = _Book(client)
        ids = []
        for _ in range(3):
            res = await call_tool("stage_option_order", _opt_args("STO", "SELL"))
            ids.append(json.loads(res[0].text)["staged_id"])
        assert (await _confirm(ids[0]))["submitted"] is True      # 100 -> 90
        assert (await _confirm(ids[1]))["submitted"] is True      # 90 -> 80
        body = await _confirm(ids[2])                              # 80 -> 70
        assert body["submitted"] is False
        assert body["error"].startswith("Margin gate: This order would take buying power "
                                        "from $80,000 to $70,000")
        assert "counting $20,000 of orders already sent this session" in body["error"]
        assert [e["impact"] for e in tools._MG_PENDING] == [pytest.approx(10000.0)] * 2
        # a stock BUY now sees the two pending puts as well
        client.get_account_summary = AsyncMock(return_value=_summary_rows(100000, 30000))
        r = await _buy_funds_gate("BUY", 80, 100.0, strict=False)  # 80 - 8 = 72k
        assert not r["ok"] and "counting $20,000" in r["error"]

    async def test_modify_increase_of_recorded_order_updates_ledger(
            self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        r1 = await _confirm((await _stage_buy())["staged_id"])      # $8k, id 100
        oid = r1["ibkr"]["order_id"]
        res = await call_tool("modify_live_order", {"order_id": oid, "quantity": 160})
        assert json.loads(res[0].text)["modified"] is True          # +$8k: 87 - 8 = 79k
        assert tools._MG_PENDING[0]["impact"] == 16000.0
        body = await _stage_buy()                                   # 95-16-8 = 71k
        assert body["staged"] is False and "counting $16,000" in body["error"]
        res = await call_tool("modify_live_order", {"order_id": oid, "quantity": 80})
        assert json.loads(res[0].text)["modified"] is True
        assert tools._MG_PENDING[0]["impact"] == 8000.0
        assert (await _stage_buy())["staged"] is True

    async def test_modify_increase_counts_other_pending_orders(
            self, monkeypatch, real_gate, staged):
        client, book = _burst_client(monkeypatch)
        for _ in range(2):
            assert (await _confirm((await _stage_buy())["staged_id"]))["submitted"]
        # a different resting order of ours, not in the ledger, goes up by $8k:
        other = book._add("BUY", 80, 100.0, "STK", order_id=555, client_id=49)
        res = await call_tool("modify_live_order", {"order_id": 555, "quantity": 160})
        body = json.loads(res[0].text)
        assert body["modified"] is False
        assert "counting $16,000" in body["error"]
        assert other.order.totalQuantity == 80

    async def test_fallback_gate_also_subtracts_pending(self, monkeypatch, no_gate_module, staged):
        client, book = _burst_client(monkeypatch)
        ids = [(await _stage_buy())["staged_id"] for _ in range(3)]
        assert (await _confirm(ids[0]))["submitted"] is True
        assert (await _confirm(ids[1]))["submitted"] is True
        body = await _confirm(ids[2])
        assert body["submitted"] is False
        assert body["error"] == _THIRD_REFUSAL


def test_shipped_line_is_200k():
    """Owner: line $75k -> $100k (2026-10-15) -> $150k (10-16) -> $200k (10-17), release +$5k."""
    src = open(tools.__file__).read()
    assert "_MARGIN_GATE_LINE_USD = 200000.0" in src
    d = _real_trader_dir()
    if d is None:
        pytest.skip("Trader margin_gate.py not found")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mg_shipped", Path(d) / "scripts" / "live" / "margin_gate.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    g = mod.gate_config()
    assert g["buying_power_crit_usd"] == 200000 and g["buying_power_clear_usd"] == 5000
