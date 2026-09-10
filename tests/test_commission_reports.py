"""get_todays_fills must wait (bounded) for IBKR's CommissionReport.

IBKR delivers an execution in two messages: execDetails, then a separate
CommissionReport carrying `commission` AND `realizedPNL` (the broker's own
FIFO-matched per-execution P&L). reqExecutionsAsync() returns at
execDetailsEnd, so reading the report immediately yields the zero-valued
default — which is why 382/385 rows in the live ledger had commission 0.0.
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ibkr_mcp_server import client as client_mod


UNSET_DOUBLE = 1.7976931348623157e308


def _execution(exec_id, symbol, order_ref=""):
    ex = MagicMock()
    ex.execId = exec_id
    ex.orderId = 1001
    ex.permId = 9001
    ex.side = "BOT"
    ex.shares = 10
    ex.price = 100.0
    ex.avgPrice = 100.0
    ex.time = datetime(2026, 9, 9, 10, 30, 0)
    ex.acctNumber = "U4022128"
    ex.exchange = "SMART"
    ex.orderRef = order_ref
    return ex


class _Report:
    """Stand-in for ib_async CommissionReport (default = never delivered)."""

    def __init__(self, exec_id="", commission=0.0, currency="", realized=0.0):
        self.execId = exec_id
        self.commission = commission
        self.currency = currency
        self.realizedPNL = realized


class _Fill:
    def __init__(self, execution, symbol, report=None):
        self.execution = execution
        self.contract = MagicMock()
        self.contract.symbol = symbol
        self.commissionReport = report or _Report()


class _FakeEvent:
    """Minimal ib_async Event stand-in supporting += / -=."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def __isub__(self, fn):
        if fn in self.handlers:
            self.handlers.remove(fn)
        return self


@pytest.fixture
def fills_client(ibkr_client_mock):
    ib = ibkr_client_mock.ib
    ib.trades.return_value = []
    ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[])
    ib.reqCompletedOrdersAsync = AsyncMock(return_value=[])
    ib.commissionReportEvent = _FakeEvent()
    return ibkr_client_mock


@pytest.mark.asyncio
async def test_commission_report_arrives_on_second_poll(fills_client):
    """The report lands after reqExecutions returns; the row must pick it up."""
    ib = fills_client.ib

    execution = _execution("e-late", "PLTR", order_ref="SWING_EMA_PULLBACK_1")
    returned_fill = _Fill(execution, "PLTR")          # empty report at return
    registry_fill = _Fill(execution, "PLTR")          # ib.fills() copy, updated in place

    ib.reqExecutionsAsync = AsyncMock(return_value=[returned_fill])

    state = {"polls": 0}

    def _fills():
        state["polls"] += 1
        if state["polls"] >= 2:
            # 2nd poll: TWS delivered the report; ib_async mutates in place.
            registry_fill.commissionReport = _Report(
                exec_id="e-late", commission=1.07, currency="USD", realized=214.55
            )
        return [registry_fill]

    ib.fills = _fills

    rows = await fills_client.get_todays_fills(commission_wait_s=2.5)

    assert len(rows) == 1
    row = rows[0]
    assert row["commission"] == 1.07
    assert row["commission_currency"] == "USD"
    assert row["realized_pnl_broker"] == 214.55
    assert row["commission_report_received"] is True
    # Existing keys untouched.
    assert row["exec_id"] == "e-late"
    assert row["symbol"] == "PLTR"
    assert row["action"] == "BUY"
    assert row["tag"] == "SWING_EMA_PULLBACK_1"
    assert state["polls"] >= 2


@pytest.mark.asyncio
async def test_missing_commission_report_times_out_within_bound(fills_client):
    """A report that never arrives must not hang the swing tick."""
    ib = fills_client.ib

    execution = _execution("e-never", "TSM")
    fill = _Fill(execution, "TSM")
    ib.reqExecutionsAsync = AsyncMock(return_value=[fill])
    ib.fills = lambda: [fill]

    wait_s = 0.6
    t0 = time.monotonic()
    rows = await fills_client.get_todays_fills(commission_wait_s=wait_s)
    elapsed = time.monotonic() - t0

    assert elapsed < wait_s + 1.5, f"waited {elapsed:.2f}s, bound was {wait_s}s"
    assert elapsed >= wait_s * 0.5
    row = rows[0]
    assert row["commission_report_received"] is False
    assert row["realized_pnl_broker"] is None
    assert row["commission"] == 0.0
    assert row["commission_currency"] == ""


@pytest.mark.asyncio
async def test_already_settled_report_skips_the_wait(fills_client):
    """A fill that already carries its report costs no settle window."""
    ib = fills_client.ib

    execution = _execution("e-fast", "NVDA")
    fill = _Fill(
        execution,
        "NVDA",
        report=_Report(exec_id="e-fast", commission=2.5, currency="USD", realized=-30.0),
    )
    ib.reqExecutionsAsync = AsyncMock(return_value=[fill])
    ib.fills = lambda: [fill]

    t0 = time.monotonic()
    rows = await fills_client.get_todays_fills(commission_wait_s=5.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0
    assert rows[0]["commission"] == 2.5
    assert rows[0]["realized_pnl_broker"] == -30.0
    assert rows[0]["commission_report_received"] is True


@pytest.mark.asyncio
async def test_unset_double_realized_pnl_maps_to_none(fills_client):
    """ib_async's UNSET sentinel must never leak into the ledger."""
    ib = fills_client.ib

    execution = _execution("e-unset", "AMD")
    fill = _Fill(
        execution,
        "AMD",
        report=_Report(
            exec_id="e-unset", commission=UNSET_DOUBLE, currency="USD", realized=UNSET_DOUBLE
        ),
    )
    ib.reqExecutionsAsync = AsyncMock(return_value=[fill])
    ib.fills = lambda: [fill]

    rows = await fills_client.get_todays_fills(commission_wait_s=0.0)

    assert rows[0]["realized_pnl_broker"] is None
    assert rows[0]["commission"] == 0.0
    assert rows[0]["commission_report_received"] is True


def test_clean_double_helper():
    assert client_mod._clean_double(None) is None
    assert client_mod._clean_double(UNSET_DOUBLE) is None
    assert client_mod._clean_double(0.0) == 0.0
    assert client_mod._clean_double("1.5") == 1.5
    assert client_mod._clean_double("abc") is None
