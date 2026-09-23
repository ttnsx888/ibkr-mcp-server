"""Tests for ibkr_mcp_server.orders — OPTION field additions (2026-09-23) and
backward-compat loading of staged-orders.json files written before those
fields existed.
"""

import json

from ibkr_mcp_server.orders import StagedOrder, StagedOrderStore


class TestStagedOrderOptionFields:
    def test_stock_order_defaults_sec_type_stk(self):
        o = StagedOrder.new("AAPL", "BUY", 10, limit_price=230.0, source="test")
        assert o.sec_type == "STK"
        assert o.conid is None
        assert o.intent is None

    def test_option_order_carries_contract_fields(self):
        o = StagedOrder.new(
            "NVDA", "SELL", 1, limit_price=3.50, source="WHEEL_STO_1",
            sec_type="OPT", conid=778899, expiry="20261016", strike=120.0,
            right="p", multiplier="100", intent="sto",
        )
        assert o.sec_type == "OPT"
        assert o.conid == 778899
        assert o.expiry == "20261016"
        assert o.strike == 120.0
        assert o.right == "P"          # normalized uppercase
        assert o.multiplier == "100"
        assert o.intent == "STO"       # normalized uppercase

    def test_option_summary_shows_contract(self):
        o = StagedOrder.new(
            "NVDA", "SELL", 2, limit_price=3.50, source="WHEEL_STO_1",
            sec_type="OPT", conid=778899, expiry="20261016", strike=120.0,
            right="P", multiplier="100", intent="STO",
        )
        s = o.summary()
        assert "NVDA" in s
        assert "20261016" in s
        assert "120" in s
        assert "P" in s
        assert "[STO]" in s
        assert "2x" in s

    def test_stock_summary_unchanged_format(self):
        o = StagedOrder.new("AAPL", "BUY", 10, limit_price=230.0, source="test")
        s = o.summary()
        assert s == f"BUY 10 AAPL @ $230.00 (DAY) — test"


class TestBackwardCompatLoad:
    def test_load_old_json_without_option_fields(self, tmp_path):
        """A staged_orders.json written before the OPT fields existed has no
        sec_type/conid/expiry/strike/right/multiplier/intent keys at all —
        loading it must not raise and must default sec_type to STK."""
        old_row = {
            "id": "abc12345",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 10,
            "limit_price": 230.0,
            "tif": "DAY",
            "source": "legacy scan",
            "created_at": "2026-01-01T00:00:00",
            "expires_at": "2099-01-01T00:00:00",
            "outside_rth": False,
            "order_type": "LMT",
            "stop_price": None,
            "oca_group": None,
            "oca_type": 0,
            "parent_staged_id": None,
            "transmit_last": True,
            # No sec_type / conid / expiry / strike / right / multiplier / intent.
        }
        path = tmp_path / "staged.json"
        path.write_text(json.dumps({"abc12345": old_row}))

        store = StagedOrderStore(path=path)
        loaded = store.get("abc12345")
        assert loaded is not None
        assert loaded.symbol == "AAPL"
        assert loaded.sec_type == "STK"
        assert loaded.conid is None
        assert loaded.intent is None
        # Summary must not blow up on the legacy row either.
        assert "AAPL" in loaded.summary()

    def test_round_trip_save_and_load_option_order(self, tmp_path):
        path = tmp_path / "staged.json"
        store = StagedOrderStore(path=path)
        o = StagedOrder.new(
            "NVDA", "SELL", 1, limit_price=3.50, source="WHEEL_STO_1",
            sec_type="OPT", conid=778899, expiry="20261016", strike=120.0,
            right="P", multiplier="100", intent="STO",
        )
        store.add(o)

        reloaded_store = StagedOrderStore(path=path)
        reloaded = reloaded_store.get(o.id)
        assert reloaded.sec_type == "OPT"
        assert reloaded.conid == 778899
        assert reloaded.strike == 120.0
        assert reloaded.right == "P"
        assert reloaded.intent == "STO"
