"""Tests for the IBKR error-diagnostics feature (2026-09-08 incident).

Incident: the Trader swing bot staged `SELL LMT 86 PLTR @ 169.91 tif=OPG`
after the close; `confirm_order` returned status "Cancelled" twice
(order_ids 8304, 8311) with no indication of why. `IBKRClient._on_error`
captured `errorEvent` text via `self.logger.error(...)` but never attached
it anywhere a caller could see, and nothing was written to a log file.

These tests are hermetic: no TWS connection. They exercise the bounded
error rings on `IBKRClient` directly, and the additive `ibkr_errors` /
`last_error` keys on place_limit_order / place_stop_order /
place_bracket_order / modify_order / cancel_order.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ibkr_mcp_server.client import IBKRClient


# ── error ring plumbing ──────────────────────────────────────────────────

def test_on_error_records_into_per_order_and_global_rings():
    client = IBKRClient()
    client._on_error(42, 201, "Order rejected - reason: no such order", None)

    per_order = client._recent_errors[42]
    assert len(per_order) == 1
    assert per_order[0]["code"] == 201
    assert per_order[0]["message"] == "Order rejected - reason: no such order"
    assert "ts" in per_order[0]

    assert len(client._error_ring) == 1
    assert client._error_ring[0]["code"] == 201
    assert client._error_ring[0]["req_id"] == 42


def test_unrelated_order_id_is_not_polluted():
    client = IBKRClient()
    client._on_error(111, 202, "Order Cancelled - Reason: operator", None)

    assert client._recent_errors.get(111)
    assert client._recent_errors.get(222, []) == []
    assert client._order_errors(222) == []


def test_per_order_ring_is_bounded():
    client = IBKRClient()
    limit = client._PER_ORDER_ERROR_RING_SIZE
    for i in range(limit + 15):
        client._on_error(9, 999, f"noise {i}", None)
    assert len(client._recent_errors[9]) == limit
    # Oldest entries fall off — the ring keeps the most recent ones.
    assert client._recent_errors[9][-1]["message"] == f"noise {limit + 14}"


def test_global_ring_is_bounded_across_many_order_ids():
    client = IBKRClient()
    limit = client._GLOBAL_ERROR_RING_SIZE
    for i in range(limit + 20):
        client._on_error(i, 1100 + i, f"err {i}", None)
    assert len(client._error_ring) == limit


def test_get_recent_errors_returns_last_n_oldest_first():
    client = IBKRClient()
    for i in range(5):
        client._on_error(i, 100 + i, f"e{i}", None)
    last3 = client.get_recent_errors(3)
    assert [e["code"] for e in last3] == [102, 103, 104]


def test_routine_codes_do_not_skip_ring_recording():
    """Routine market-data-warning codes are logged at debug (not error) but
    must still be captured — a caller diagnosing a placement shouldn't lose
    a low-severity code just because it's not noisy in the log."""
    client = IBKRClient()
    client._on_error(7, 2104, "Market data farm connection is OK", None)
    assert client._order_errors(7) == [
        {"ts": pytest.approx(client._recent_errors[7][0]["ts"]), "code": 2104,
         "message": "Market data farm connection is OK"}
    ]


# ── placement layer (no TWS) ─────────────────────────────────────────────

def _mock_client(order_id: int, status: str = "Filled", fire_error: dict = None):
    """Build a hermetic IBKRClient whose placeOrder assigns `order_id` and
    the given terminal `status`, optionally firing a synchronous errorEvent
    for that order_id as part of the placeOrder call (modeling IBKR
    rejecting the order before place_limit_order/place_stop_order even
    returns from its first sleep)."""
    client = IBKRClient()
    client.ib = MagicMock()
    client.ib.isConnected.return_value = True
    client._connected = True
    client.current_account = "U1"

    async def _qualify(contract):
        contract.conId = 111222
        return [contract]

    def _place(contract, order):
        order.orderId = order_id
        order.permId = order_id * 10
        if fire_error:
            client._on_error(order_id, fire_error["code"], fire_error["message"], contract)
        return SimpleNamespace(
            order=order,
            orderStatus=SimpleNamespace(status=status, filled=0,
                                        remaining=order.totalQuantity),
        )

    client.ib.qualifyContractsAsync = _qualify
    client.ib.placeOrder = _place
    return client


async def test_place_limit_order_attaches_ibkr_errors_for_its_own_order_id():
    client = _mock_client(order_id=8304, status="Cancelled",
                           fire_error={"code": 201,
                                       "message": "Order rejected - reason: incident repro"})
    result = await client.place_limit_order(
        symbol="PLTR", action="SELL", quantity=86, limit_price=169.91, tif="OPG",
    )
    assert result["status"] == "Cancelled"
    assert result["ibkr_errors"] == [{"code": 201,
                                       "message": "Order rejected - reason: incident repro",
                                       "ts": pytest.approx(result["ibkr_errors"][0]["ts"])}]
    assert result["last_error"] == "201: Order rejected - reason: incident repro"


async def test_place_limit_order_no_error_gives_empty_list_and_none():
    client = _mock_client(order_id=1, status="Filled")
    result = await client.place_limit_order(
        symbol="AAPL", action="BUY", quantity=10, limit_price=200.0,
    )
    assert result["ibkr_errors"] == []
    assert result["last_error"] is None


async def test_error_for_a_different_order_id_does_not_leak_in():
    client = _mock_client(order_id=55, status="Cancelled")
    # Pre-seed an error for an unrelated order — must never appear on 55's result.
    client._on_error(999, 201, "unrelated rejection", None)
    result = await client.place_limit_order(
        symbol="MSFT", action="SELL", quantity=5, limit_price=400.0,
    )
    assert result["ibkr_errors"] == []
    assert result["last_error"] is None


async def test_place_stop_order_attaches_ibkr_errors():
    client = _mock_client(order_id=321, status="Inactive",
                           fire_error={"code": 10349, "message": "Stop order rejected"})
    result = await client.place_stop_order(
        symbol="QQQ", action="SELL", quantity=20, stop_price=410.0,
    )
    assert result["status"] == "Inactive"
    assert result["last_error"] == "10349: Stop order rejected"
    assert len(result["ibkr_errors"]) == 1


async def test_extra_settle_wait_captures_a_late_arriving_error(monkeypatch):
    """If the error hasn't arrived by the time the initial 1.0s echo sleep
    returns, place_limit_order gives it one more brief window (<=1.5s) —
    simulate the errorEvent landing *during* that extra wait rather than
    faking it away, and assert no unbounded/long sleep is used."""
    order_id = 8311
    client = _mock_client(order_id=order_id, status="Cancelled", fire_error=None)

    sleep_calls = []
    import ibkr_mcp_server.client as client_mod
    real_sleep = client_mod.asyncio.sleep

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        if duration >= 1.5:
            # Model the errorEvent landing while we wait for it.
            client._on_error(order_id, 202, "Order Cancelled - Reason: late", None)
            return
        # Keep the harmless short/initial sleeps but don't slow the suite.
        return

    monkeypatch.setattr(client_mod.asyncio, "sleep", fake_sleep)

    result = await client.place_limit_order(
        symbol="PLTR", action="SELL", quantity=86, limit_price=169.91,
    )
    assert result["status"] == "Cancelled"
    assert result["last_error"] == "202: Order Cancelled - Reason: late"
    # The extra settle wait was actually exercised (bounded, not indefinite).
    assert any(d == pytest.approx(1.5) for d in sleep_calls)
    assert all(d <= 1.5 for d in sleep_calls)


async def test_place_bracket_order_attaches_errors_to_parent_and_children_independently():
    client = IBKRClient()
    client.ib = MagicMock()
    client.ib.isConnected.return_value = True
    client._connected = True
    client.current_account = "U1"

    async def _qualify(contract):
        contract.conId = 555
        return [contract]

    placed_ids = iter([700, 701, 702])  # parent, child0, child1

    def _place(contract, order):
        oid = next(placed_ids)
        order.orderId = oid
        order.permId = oid * 10
        status = "Submitted"
        if oid == 701:
            status = "Cancelled"
            client._on_error(701, 201, "child leg rejected", None)
        return SimpleNamespace(
            order=order,
            orderStatus=SimpleNamespace(status=status, filled=0,
                                        remaining=order.totalQuantity),
        )

    client.ib.qualifyContractsAsync = _qualify
    client.ib.placeOrder = _place
    client.ib.cancelOrder = MagicMock()

    result = await client.place_bracket_order(
        symbol="TSM", parent_action="BUY", parent_quantity=100,
        parent_limit_price=250.0,
        children=[
            {"order_type": "STP", "action": "SELL", "quantity": 100, "stop_price": 240.0},
            {"order_type": "LMT", "action": "SELL", "quantity": 100, "limit_price": 260.0},
        ],
        order_ref="TEST_BRACKET",
    )

    assert result["parent"]["ibkr_errors"] == []
    assert result["parent"]["last_error"] is None

    child0, child1 = result["children"]
    assert child0["order_id"] == 701
    assert child0["last_error"] == "201: child leg rejected"
    assert child1["order_id"] == 702
    assert child1["ibkr_errors"] == []
    assert child1["last_error"] is None


async def test_modify_order_attaches_ibkr_errors():
    client = IBKRClient()
    client.ib = MagicMock()
    client.ib.isConnected.return_value = True
    client._connected = True

    order = SimpleNamespace(orderId=88, permId=880, totalQuantity=10, lmtPrice=100.0,
                            auxPrice=0.0, transmit=False)
    contract = SimpleNamespace(symbol="IREN")
    existing_trade = SimpleNamespace(order=order, contract=contract)
    client.ib.trades.return_value = [existing_trade]

    def _place(contract, order):
        client._on_error(88, 201, "modify rejected", None)
        return SimpleNamespace(
            order=order,
            orderStatus=SimpleNamespace(status="Cancelled", filled=0, remaining=order.totalQuantity),
        )
    client.ib.placeOrder = _place

    result = await client.modify_order(order_id=88, limit_price=101.0)
    assert result["modified"] is True
    assert result["status"] == "Cancelled"
    assert result["last_error"] == "201: modify rejected"


async def test_cancel_order_does_not_pay_extra_settle_wait(monkeypatch):
    """A deliberate cancel_order() ending in status "Cancelled" is the
    expected, successful outcome — it must NOT trigger the ~1.5s extra
    settle wait that place_* uses to catch a synchronous reject."""
    client = IBKRClient()
    client.ib = MagicMock()
    client.ib.isConnected.return_value = True
    client._connected = True

    order = SimpleNamespace(orderId=5, totalQuantity=10)
    contract = SimpleNamespace(symbol="AAPL")
    status = SimpleNamespace(status="Cancelled", filled=0, remaining=10)
    trade = SimpleNamespace(order=order, contract=contract, orderStatus=status)
    client.ib.trades.return_value = [trade]
    client.ib.cancelOrder = MagicMock()

    import ibkr_mcp_server.client as client_mod
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fake_sleep)

    result = await client.cancel_order(order_id=5)
    assert result["cancelled"] is True
    assert result["status"] == "Cancelled"
    assert result["ibkr_errors"] == []
    assert result["last_error"] is None
    # Only the existing 0.5s post-cancel settle — no additional 1.5s wait.
    assert sleep_calls == [0.5]


async def test_get_connection_status_exposes_recent_errors(monkeypatch):
    """get_connection_status must surface the global error ring so an
    operator can see recent IBKR rejections without tailing the log file.

    Uses `await` rather than `asyncio.run()` — asyncio.run() tears down and
    clears the process's "current" event loop on exit, which breaks the
    legacy `asyncio.get_event_loop()` pattern other test modules
    (test_pr_a_extensions.py) rely on when the suite runs as a whole."""
    from ibkr_mcp_server import tools

    client = MagicMock()
    client.is_connected.return_value = True
    client.host = "127.0.0.1"
    client.port = 7496
    client.client_id = 1
    client.current_account = "U1"
    client.accounts = ["U1"]
    client.is_paper = False

    async def _ensure_connected():
        return True
    client._ensure_connected = _ensure_connected

    expected = [{"ts": 1.0, "code": 201, "message": "rejected", "req_id": 8304, "symbol": "PLTR"}]
    client.get_recent_errors.return_value = expected

    monkeypatch.setattr(tools, "ibkr_client", client)

    result = await tools.call_tool("get_connection_status", {})
    import json
    payload = json.loads(result[0].text)
    assert payload["recent_ibkr_errors"] == expected
    client.get_recent_errors.assert_called_once_with(20)
