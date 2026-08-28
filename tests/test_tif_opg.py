"""Tests for TIF plumbing, in particular TIF=OPG (limit-on-open).

Regression for the 2026-08-24 TSM/QQQ incident: the swing-monitor engine
converts end-of-day exits to an opening-auction order (V4.7, after the
2026-06-25 INTC/META loss), but the MCP staging schema only advertised
DAY/GTC/IOC/FOK. The caller silently fell back to a GTC LMT, which IBKR
parked 'Inactive' after the close and dropped overnight — exactly the
failure V4.7 was written to prevent.

These tests are hermetic: no TWS connection, and nothing touches the real
staged-orders file (StagedOrder.new does not write to the store).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ibkr_mcp_server.client import IBKRClient
from ibkr_mcp_server.orders import (
    OPG_ORDER_TYPES, VALID_TIFS, StagedOrder,
)
from ibkr_mcp_server.tools import TOOLS
from ibkr_mcp_server.utils import ValidationError


# ── schema ───────────────────────────────────────────────────────────────

def _tool(name):
    return next(t for t in TOOLS if t.name == name)


def test_stage_order_schema_advertises_opg():
    enum = _tool("stage_order").inputSchema["properties"]["tif"]["enum"]
    assert "OPG" in enum, f"stage_order tif enum missing OPG: {enum}"
    # Existing values must survive — callers already pass DAY/GTC.
    assert {"DAY", "GTC", "IOC", "FOK"}.issubset(set(enum))


def test_stage_order_schema_default_unchanged():
    assert _tool("stage_order").inputSchema["properties"]["tif"]["default"] == "DAY"
    assert _tool("stage_stop_order").inputSchema["properties"]["tif"]["default"] == "GTC"


def test_stage_stop_order_schema_excludes_opg():
    """IBKR accepts opening-auction TIFs on LMT/MKT only."""
    enum = _tool("stage_stop_order").inputSchema["properties"]["tif"]["enum"]
    assert "OPG" not in enum


# ── staged-order model ───────────────────────────────────────────────────

def test_default_tif_is_day():
    o = StagedOrder.new("SPY", "SELL", 10, limit_price=480.0)
    assert o.tif == "DAY"
    assert o.outside_rth is False


def test_tif_is_normalised_to_upper():
    o = StagedOrder.new("SPY", "SELL", 10, limit_price=480.0, tif="opg")
    assert o.tif == "OPG"


def test_unknown_tif_rejected():
    with pytest.raises(ValueError) as e:
        StagedOrder.new("SPY", "SELL", 10, limit_price=480.0, tif="GTD")
    assert "tif must be one of" in str(e.value)
    assert "GTD" in str(e.value)


def test_all_valid_tifs_accepted_on_lmt():
    for tif in VALID_TIFS:
        o = StagedOrder.new("SPY", "SELL", 10, limit_price=480.0, tif=tif)
        assert o.tif == tif


def test_opg_forces_outside_rth_false():
    """Swing exits pass outside_rth=True by default; OPG must coerce it off
    rather than raise, or the exit is dropped entirely."""
    o = StagedOrder.new("SPY", "SELL", 10, limit_price=480.0,
                        tif="OPG", outside_rth=True, source="SWING_EOD")
    assert o.tif == "OPG"
    assert o.outside_rth is False
    assert "OPG" in o.summary()
    assert "outsideRTH" not in o.summary()


def test_non_opg_preserves_outside_rth():
    o = StagedOrder.new("SPY", "SELL", 10, limit_price=480.0,
                        tif="GTC", outside_rth=True)
    assert o.outside_rth is True


def test_opg_rejected_on_stop_order():
    with pytest.raises(ValueError) as e:
        StagedOrder.new("SPY", "SELL", 10, order_type="STP", stop_price=470.0,
                        tif="OPG")
    assert "OPG" in str(e.value)
    assert str(OPG_ORDER_TYPES) in str(e.value) or "LMT" in str(e.value)


def test_opg_round_trips_through_the_store_record():
    """asdict/StagedOrder(**d) is how the store persists and reloads —
    tif must survive it (list_staged_orders serialises the same way)."""
    from dataclasses import asdict
    o = StagedOrder.new("QQQ", "SELL", 25, limit_price=410.0, tif="OPG",
                        source="SWING_EOD_EXIT")
    d = asdict(o)
    assert d["tif"] == "OPG"
    assert StagedOrder(**d).tif == "OPG"


# ── placement layer (no TWS) ─────────────────────────────────────────────

def _mock_client(placed: dict):
    client = IBKRClient()
    client.ib = MagicMock()
    client.ib.isConnected.return_value = True
    client._connected = True
    client.current_account = "U1"

    async def _qualify(contract):
        contract.conId = 111222
        return [contract]

    def _place(contract, order):
        placed["contract"] = contract
        placed["order"] = order
        return SimpleNamespace(
            order=order,
            orderStatus=SimpleNamespace(status="PreSubmitted", filled=0,
                                        remaining=order.totalQuantity),
        )

    client.ib.qualifyContractsAsync = _qualify
    client.ib.placeOrder = _place
    return client


async def test_place_limit_order_propagates_opg_to_the_ib_order():
    placed = {}
    client = _mock_client(placed)
    result = await client.place_limit_order(
        symbol="TSM", action="SELL", quantity=100, limit_price=250.0,
        tif="OPG", outside_rth=True, order_ref="SWING_EOD_EXIT",
    )
    order = placed["order"]
    assert order.tif == "OPG"
    assert order.outsideRth is False       # forced off for the auction order
    assert order.orderType == "LMT"
    assert order.lmtPrice == 250.0
    assert result["tif"] == "OPG"
    assert result["outside_rth"] is False


async def test_place_limit_order_default_tif_still_day():
    placed = {}
    client = _mock_client(placed)
    result = await client.place_limit_order(
        symbol="TSM", action="BUY", quantity=10, limit_price=250.0,
    )
    assert placed["order"].tif == "DAY"
    assert result["tif"] == "DAY"


async def test_place_stop_order_rejects_opg():
    placed = {}
    client = _mock_client(placed)
    with pytest.raises(ValidationError):
        await client.place_stop_order(
            symbol="TSM", action="SELL", quantity=100, stop_price=240.0,
            tif="OPG",
        )
    assert "order" not in placed
