"""Tests for MCP server tools and endpoints."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ibkr_mcp_server import tools
from ibkr_mcp_server.tools import server, call_tool
from ibkr_mcp_server.orders import StagedOrder, StagedOrderStore


def test_server_creation():
    """Test server can be created."""
    assert server is not None


# Additional tests to be implemented


# ---------------------------------------------------------------------------
# Option tool dispatch (2026-09-23): get_option_chain / get_option_quote /
# stage_option_order, and the confirm_order OPT dispatch branch.
#
# ibkr_client is fully mocked (no IB calls) — these test the tools.py wiring
# (argument mapping, gate invocation, staged-order shape, live-trading gate),
# not client.py's IBKR-facing methods (covered in test_client.py) or
# option_gates' rules themselves (covered in test_option_gates.py).
# ---------------------------------------------------------------------------


def _fake_contract(conid=111, symbol="NVDA", right="P", strike=20.0,
                    expiry="20261016", multiplier="100",
                    local_symbol="NVDA  261016P00020000"):
    return SimpleNamespace(conId=conid, symbol=symbol, right=right, strike=strike,
                            lastTradeDateOrContractMonth=expiry, multiplier=multiplier,
                            localSymbol=local_symbol)


def _mock_client(monkeypatch, *, contract=None, positions=None, open_orders=None,
                  quote=None, whatif=None, account_summary=None, port=7497,
                  place_result=None):
    client = MagicMock()
    client.port = port
    client.current_account = "U1"
    client.is_connected.return_value = True
    client.qualify_option = AsyncMock(return_value=contract)
    client.get_portfolio = AsyncMock(return_value=positions or [])
    client.get_open_trades = AsyncMock(return_value=open_orders or [])
    client.get_option_quote = AsyncMock(
        return_value=quote if quote is not None else {"bid": 1.00, "ask": 1.10})
    client.whatif_option_order = AsyncMock(return_value=whatif)
    client.get_account_summary = AsyncMock(return_value=account_summary or [])
    client.get_option_chain = AsyncMock(return_value={"symbol": "NVDA"})
    client.place_option_limit_order = AsyncMock(
        return_value=place_result or {"status": "Submitted", "order_id": 1})
    monkeypatch.setattr(tools, "ibkr_client", client)
    return client


@pytest.fixture
def isolated_staged_store(tmp_path, monkeypatch):
    """A staged-order store isolated to a tmp file, so option-tool tests never
    touch the real ~/ibkr-mcp-server/.staged_orders.json (which other tests
    in this suite, e.g. test_pr_a_extensions.py, use directly)."""
    store = StagedOrderStore(path=tmp_path / "staged.json")
    monkeypatch.setattr(tools, "staged_store", store)
    return store


class TestGetOptionChainDispatch:
    async def test_dispatch(self, monkeypatch):
        client = _mock_client(monkeypatch)
        client.get_option_chain = AsyncMock(
            return_value={"symbol": "NVDA", "expirations": ["20261016"]})
        res = await call_tool("get_option_chain", {"symbol": "nvda"})
        body = json.loads(res[0].text)
        assert body["symbol"] == "NVDA"
        client.get_option_chain.assert_awaited_once_with("NVDA")


class TestGetOptionQuoteDispatch:
    async def test_by_conid(self, monkeypatch):
        client = _mock_client(monkeypatch)
        client.get_option_quote = AsyncMock(
            return_value={"bid": 1.0, "ask": 1.1, "source": "live"})
        res = await call_tool("get_option_quote", {"conid": 111})
        body = json.loads(res[0].text)
        assert body["source"] == "live"
        client.get_option_quote.assert_awaited_once_with(conid=111)

    async def test_by_spec(self, monkeypatch):
        client = _mock_client(monkeypatch)
        res = await call_tool("get_option_quote", {
            "symbol": "nvda", "expiry": "20261016", "strike": 20.0, "right": "P"})
        body = json.loads(res[0].text)
        assert "bid" in body
        client.get_option_quote.assert_awaited_once_with(
            symbol="NVDA", expiry="20261016", strike=20.0, right="P")

    async def test_missing_spec_returns_error_without_calling_client(self, monkeypatch):
        client = _mock_client(monkeypatch)
        res = await call_tool("get_option_quote", {"symbol": "NVDA"})  # no expiry/strike/right
        body = json.loads(res[0].text)
        assert "error" in body
        client.get_option_quote.assert_not_awaited()


class TestStageOptionOrder:
    async def test_sto_put_happy_path(self, monkeypatch, isolated_staged_store):
        contract = _fake_contract(right="P", conid=111)
        client = _mock_client(
            monkeypatch, contract=contract,
            quote={"bid": 1.00, "ask": 1.10},
            whatif={"init_margin_change": 500.0},
            account_summary=[{"tag": "AvailableFunds", "value": "50000", "currency": "USD"}],
        )
        args = {"conid": 111, "intent": "STO", "action": "SELL", "quantity": 1,
                "limit_price": 1.00, "source": "WHEEL_STO_TEST"}
        res = await call_tool("stage_option_order", args)
        body = json.loads(res[0].text)
        assert body["staged"] is True, body
        staged = isolated_staged_store.get(body["staged_id"])
        assert staged.sec_type == "OPT"
        assert staged.intent == "STO"
        assert staged.conid == 111
        assert staged.right == "P"
        client.qualify_option.assert_awaited_once_with(conid=111)

    async def test_by_symbol_spec(self, monkeypatch, isolated_staged_store):
        contract = _fake_contract(right="P", conid=111)
        _mock_client(
            monkeypatch, contract=contract,
            quote={"bid": 1.00, "ask": 1.10},
            whatif={"init_margin_change": 500.0},
            account_summary=[{"tag": "AvailableFunds", "value": "50000", "currency": "USD"}],
        )
        args = {"symbol": "nvda", "expiry": "20261016", "strike": 20.0, "right": "put",
                "intent": "STO", "action": "SELL", "quantity": 1, "limit_price": 1.00,
                "order_ref": "WHEEL_STO_TEST"}
        res = await call_tool("stage_option_order", args)
        body = json.loads(res[0].text)
        assert body["staged"] is True, body

    async def test_naked_call_refused_not_staged(self, monkeypatch, isolated_staged_store):
        contract = _fake_contract(right="C", conid=222)
        _mock_client(monkeypatch, contract=contract, positions=[])  # no underlying shares
        args = {"conid": 222, "intent": "STO", "action": "SELL", "quantity": 1,
                "limit_price": 1.00, "source": "WHEEL_STO_CALL"}
        res = await call_tool("stage_option_order", args)
        body = json.loads(res[0].text)
        assert body["staged"] is False
        assert "naked-call guard" in body["error"]
        assert len(isolated_staged_store.list()) == 0

    async def test_bto_refused_not_staged(self, monkeypatch, isolated_staged_store):
        contract = _fake_contract(right="C", conid=222)
        _mock_client(monkeypatch, contract=contract)
        args = {"conid": 222, "intent": "BTO", "action": "BUY", "quantity": 1,
                "limit_price": 1.00, "source": "test"}
        res = await call_tool("stage_option_order", args)
        body = json.loads(res[0].text)
        assert body["staged"] is False
        assert "BTO" in body["error"]
        assert len(isolated_staged_store.list()) == 0

    async def test_missing_contract_spec_returns_error(self, monkeypatch, isolated_staged_store):
        client = _mock_client(monkeypatch)
        args = {"intent": "STO", "action": "SELL", "quantity": 1, "limit_price": 1.0}
        res = await call_tool("stage_option_order", args)
        body = json.loads(res[0].text)
        assert body["staged"] is False
        client.qualify_option.assert_not_awaited()

    async def test_qualify_failure_returns_error(self, monkeypatch, isolated_staged_store):
        client = _mock_client(monkeypatch)
        client.qualify_option = AsyncMock(side_effect=RuntimeError("no match"))
        args = {"conid": 999, "intent": "STO", "action": "SELL", "quantity": 1,
                "limit_price": 1.0}
        res = await call_tool("stage_option_order", args)
        body = json.loads(res[0].text)
        assert body["staged"] is False
        assert "could not qualify" in body["error"]


class TestConfirmOrderOptDispatch:
    def _staged_opt_order(self, store):
        o = StagedOrder.new(
            "NVDA", "SELL", 1, limit_price=1.00, source="WHEEL_STO_TEST",
            sec_type="OPT", conid=111, expiry="20261016", strike=20.0,
            right="P", multiplier="100", intent="STO",
        )
        store.add(o)
        return o

    async def test_confirm_submits_and_removes_from_store(self, monkeypatch, isolated_staged_store):
        o = self._staged_opt_order(isolated_staged_store)
        contract = _fake_contract(right="P", conid=111)
        client = _mock_client(
            monkeypatch, contract=contract,
            quote={"bid": 1.0, "ask": 1.1},
            whatif={"init_margin_change": 500.0},
            account_summary=[{"tag": "AvailableFunds", "value": "50000", "currency": "USD"}],
            place_result={"status": "Submitted", "order_id": 42},
        )
        res = await call_tool("confirm_order", {"staged_id": o.id})
        body = json.loads(res[0].text)
        assert body["submitted"] is True, body
        assert isolated_staged_store.get(o.id) is None
        client.place_option_limit_order.assert_awaited_once()
        _, kwargs = client.place_option_limit_order.call_args
        assert kwargs["action"] == "SELL"
        assert kwargs["quantity"] == 1
        assert kwargs["contract"] is contract

    async def test_confirm_live_trading_gate_blocks(self, monkeypatch, isolated_staged_store):
        o = self._staged_opt_order(isolated_staged_store)
        contract = _fake_contract(right="P", conid=111)
        client = _mock_client(monkeypatch, contract=contract, port=7496)  # LIVE_PORTS
        monkeypatch.setattr(tools.settings, "enable_live_trading", False)

        res = await call_tool("confirm_order", {"staged_id": o.id})
        body = json.loads(res[0].text)
        assert body["submitted"] is False
        assert "ENABLE_LIVE_TRADING" in body["error"]
        client.qualify_option.assert_not_awaited()
        assert isolated_staged_store.get(o.id) is not None  # still staged, not consumed

    async def test_confirm_gate_refusal_keeps_order_staged(self, monkeypatch, isolated_staged_store):
        """STO put with no whatif result is refused by option_gates — confirm
        must surface the reason, NOT submit, and NOT remove the staged order
        (matching the STK path's behavior on a failed re-validation)."""
        o = self._staged_opt_order(isolated_staged_store)
        contract = _fake_contract(right="P", conid=111)
        client = _mock_client(
            monkeypatch, contract=contract,
            quote={"bid": 1.0, "ask": 1.1},
            whatif=None,
        )
        res = await call_tool("confirm_order", {"staged_id": o.id})
        body = json.loads(res[0].text)
        assert body["submitted"] is False
        assert "no whatif" in body["error"]
        assert isolated_staged_store.get(o.id) is not None
        client.place_option_limit_order.assert_not_awaited()

    async def test_confirm_qualify_failure_returns_error(self, monkeypatch, isolated_staged_store):
        o = self._staged_opt_order(isolated_staged_store)
        client = _mock_client(monkeypatch)
        client.qualify_option = AsyncMock(side_effect=RuntimeError("contract vanished"))
        res = await call_tool("confirm_order", {"staged_id": o.id})
        body = json.loads(res[0].text)
        assert body["submitted"] is False
        assert "could not qualify" in body["error"]
        assert isolated_staged_store.get(o.id) is not None


class TestStockPathUnaffectedByOptDispatch:
    """Guard: an STK staged order must never touch the OPT branch — sec_type
    defaults to STK on every existing StagedOrder.new() call site."""

    async def test_stock_confirm_never_calls_option_methods(self, monkeypatch, isolated_staged_store):
        # Use a symbol no other test module pre-seeds into tools._QUOTE_CACHE
        # (test_pr_a_extensions.py stubs SPY/QQQ), and clear any stale entry
        # left by a prior test run in this same process.
        tools._QUOTE_CACHE.pop("ZOPT", None)
        o = StagedOrder.new("ZOPT", "BUY", 1, limit_price=480.0, source="t")
        isolated_staged_store.add(o)
        client = _mock_client(
            monkeypatch,
            account_summary=[{"tag": "BuyingPower", "value": "100000", "currency": "USD"}],
        )
        client.get_quote = AsyncMock(return_value={
            "symbol": "ZOPT", "last": 480.0, "bid": 479.9, "ask": 480.1,
            "close": 480.0, "source": "test"})
        client.place_limit_order = AsyncMock(return_value={"status": "Submitted"})

        res = await call_tool("confirm_order", {"staged_id": o.id})
        body = json.loads(res[0].text)
        assert body["submitted"] is True, body
        client.qualify_option.assert_not_awaited()
        client.place_option_limit_order.assert_not_awaited()
