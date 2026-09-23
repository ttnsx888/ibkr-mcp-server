"""Tests for ibkr_mcp_server.option_gates.check_option_order.

Pure-function gate — no IB calls, no event loop needed. Every rule from the
option-support spec (2026-09-23) gets a direct regression test here.
"""

from ibkr_mcp_server.option_gates import check_option_order


CONTRACT_PUT = {"conid": 111, "right": "P", "strike": 20.0, "multiplier": "100", "symbol": "NVDA"}
CONTRACT_CALL = {"conid": 222, "right": "C", "strike": 25.0, "multiplier": "100", "symbol": "NVDA"}

QUOTE_OK = {"bid": 1.00, "ask": 1.10}
FUNDS_OK = {"AvailableFunds": "50000"}


def _pos(conid=None, secType="OPT", symbol="NVDA", right=None, strike=None,
         multiplier="100", position=0.0):
    return {"conid": conid, "secType": secType, "symbol": symbol, "right": right,
            "strike": strike, "multiplier": multiplier, "position": position}


def _order(conid=None, secType="OPT", symbol="NVDA", right=None, action="SELL",
           remaining=1, status="Submitted"):
    return {"conid": conid, "secType": secType, "symbol": symbol, "right": right,
            "action": action, "remaining": remaining, "status": status}


class TestBasicShape:
    def test_bto_refused(self):
        ok, reasons, _ = check_option_order(
            "BTO", "BUY", 1, 1.0, CONTRACT_CALL, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("BTO" in r for r in reasons)

    def test_unknown_intent_refused(self):
        ok, reasons, _ = check_option_order(
            "HODL", "BUY", 1, 1.0, CONTRACT_CALL, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("intent must be one of" in r for r in reasons)

    def test_bad_action_refused(self):
        ok, reasons, _ = check_option_order(
            "BTC", "SHORT", 1, 1.0, CONTRACT_CALL, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("action must be BUY or SELL" in r for r in reasons)

    def test_qty_below_one_refused(self):
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 0, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("quantity must be >= 1" in r for r in reasons)

    def test_qty_over_max_refused(self):
        pos = [_pos(conid=111, right="P", position=-20)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 11, 1.0, CONTRACT_PUT, pos, [], QUOTE_OK, None, FUNDS_OK,
            max_contracts=10)
        assert not ok
        assert any("exceeds max_contracts" in r for r in reasons)

    def test_negative_or_zero_price_refused(self):
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 0.0, CONTRACT_PUT, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("limit_price must be positive" in r for r in reasons)

    def test_intent_action_mismatch_refused(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "SELL", 1, 1.0, CONTRACT_PUT, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("BTC" in r and "BUY" in r for r in reasons)


class TestBTC:
    def test_over_close_refused(self):
        """Short 5, trying to BTC 6 — over-close into a long position."""
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 6, 1.0, CONTRACT_PUT, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("over-close" in r for r in reasons)

    def test_resting_buys_reduce_available(self):
        """Short 5, with 3 already resting in a BUY order — only 2 more closeable."""
        pos = [_pos(conid=111, right="P", position=-5)]
        orders = [_order(conid=111, right="P", action="BUY", remaining=3)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 3, 1.0, CONTRACT_PUT, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("over-close" in r for r in reasons)

        ok2, _, _ = check_option_order(
            "BTC", "BUY", 2, 1.0, CONTRACT_PUT, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert ok2

    def test_exact_close_allowed(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 5, 1.0, CONTRACT_PUT, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons

    def test_no_position_refused(self):
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("over-close" in r for r in reasons)


class TestSTC:
    def test_over_close_refused(self):
        pos = [_pos(conid=222, right="C", position=5)]
        ok, reasons, _ = check_option_order(
            "STC", "SELL", 6, 1.0, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("contracts closeable" in r for r in reasons)

    def test_resting_sells_reduce_available(self):
        pos = [_pos(conid=222, right="C", position=5)]
        orders = [_order(conid=222, right="C", action="SELL", remaining=4)]
        ok, reasons, _ = check_option_order(
            "STC", "SELL", 2, 1.0, CONTRACT_CALL, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert not ok

        ok2, _, _ = check_option_order(
            "STC", "SELL", 1, 1.0, CONTRACT_CALL, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert ok2

    def test_exact_close_allowed(self):
        pos = [_pos(conid=222, right="C", position=5)]
        ok, reasons, _ = check_option_order(
            "STC", "SELL", 5, 1.0, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons


class TestSTOPut:
    def test_no_whatif_refused(self):
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("no whatif" in r for r in reasons)

    def test_margin_within_funds_allowed(self):
        whatif = {"init_margin_change": 2000.0}
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, whatif, FUNDS_OK)
        assert ok, reasons

    def test_margin_exceeds_funds_refused(self):
        whatif = {"init_margin_change": 60000.0}
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, whatif, FUNDS_OK)
        assert not ok
        assert any("exceeds AvailableFunds" in r for r in reasons)

    def test_missing_available_funds_refused(self):
        whatif = {"init_margin_change": 100.0}
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, whatif, {})
        assert not ok
        assert any("AvailableFunds not available" in r for r in reasons)

    def test_whatif_missing_field_refused(self):
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_PUT, [], [], QUOTE_OK, {}, FUNDS_OK)
        assert not ok
        assert any("did not return init_margin_change" in r for r in reasons)


class TestSTOCallCovered:
    def test_naked_call_refused_no_shares(self):
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_CALL, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("naked-call guard" in r for r in reasons)

    def test_covered_call_allowed(self):
        """400 shares owned, 0 committed elsewhere, 1x100-multiplier call → covered."""
        pos = [_pos(secType="STK", right=None, strike=None, multiplier=None, position=400)]
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons

    def test_covered_call_math_with_resting_stk_sells(self):
        """The spec's exact example: 400 sh with 400 sh committed to resting STK
        SELL LMTs → 0 sh available cover → refuse ANY call."""
        pos = [_pos(secType="STK", right=None, strike=None, multiplier=None, position=400)]
        orders = [_order(conid=None, secType="STK", right=None, action="SELL",
                         remaining=400)]
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_CALL, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("naked-call guard" in r for r in reasons)

    def test_existing_short_calls_reduce_cover(self):
        """400 sh owned, already short 3 calls (300 sh committed) → only 1
        more call coverable, refuse the 2nd."""
        pos = [
            _pos(secType="STK", right=None, strike=None, multiplier=None, position=400),
            _pos(conid=999, secType="OPT", right="C", strike=30.0, multiplier="100",
                 position=-3),
        ]
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons

        ok2, reasons2, _ = check_option_order(
            "STO", "SELL", 2, 1.0, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert not ok2
        assert any("naked-call guard" in r for r in reasons2)

    def test_resting_sto_calls_reduce_cover(self):
        """200 sh owned, 1 call already resting-STO (100 sh committed) → only
        1 more coverable."""
        pos = [_pos(secType="STK", right=None, strike=None, multiplier=None, position=200)]
        orders = [_order(conid=888, secType="OPT", right="C", action="SELL",
                         remaining=1)]
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, CONTRACT_CALL, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons

        ok2, reasons2, _ = check_option_order(
            "STO", "SELL", 2, 1.0, CONTRACT_CALL, pos, orders, QUOTE_OK, None, FUNDS_OK)
        assert not ok2

    def test_sto_bad_right_refused(self):
        bad_contract = {**CONTRACT_CALL, "right": "X"}
        ok, reasons, _ = check_option_order(
            "STO", "SELL", 1, 1.0, bad_contract, [], [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("STO requires right C or P" in r for r in reasons)


class TestPriceSanity:
    def test_buy_within_ceiling_allowed(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 1.15, CONTRACT_PUT, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons

    def test_buy_above_ceiling_refused(self):
        """ask=1.10 → ceiling = 1.10 + max(0.05, 0.11) = 1.21."""
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 1.30, CONTRACT_PUT, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("price-sanity ceiling" in r for r in reasons)

    def test_sell_within_floor_allowed(self):
        pos = [_pos(conid=222, right="C", position=5)]
        ok, reasons, _ = check_option_order(
            "STC", "SELL", 1, 0.95, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert ok, reasons

    def test_sell_below_floor_refused(self):
        """bid=1.00 → floor = 1.00 - max(0.05, 0.10) = 0.90."""
        pos = [_pos(conid=222, right="C", position=5)]
        ok, reasons, _ = check_option_order(
            "STC", "SELL", 1, 0.80, CONTRACT_CALL, pos, [], QUOTE_OK, None, FUNDS_OK)
        assert not ok
        assert any("price-sanity floor" in r for r in reasons)

    def test_low_priced_option_uses_nickel_floor_not_percent(self):
        """bid=0.10 → 10% = 0.01, so the $0.05 floor dominates: floor = 0.05.
        0.04 is below that floor (refused); 0.06 clears it (allowed) — proves
        the floor isn't the (wrong) 10%-only figure of 0.09."""
        quote = {"bid": 0.10, "ask": 0.15}
        pos = [_pos(conid=222, right="C", position=5)]
        ok, reasons, _ = check_option_order(
            "STC", "SELL", 1, 0.04, CONTRACT_CALL, pos, [], quote, None, FUNDS_OK)
        assert not ok
        assert any("0.05" in r for r in reasons)

        ok2, reasons2, _ = check_option_order(
            "STC", "SELL", 1, 0.06, CONTRACT_CALL, pos, [], quote, None, FUNDS_OK)
        assert ok2, reasons2


class TestNoQuote:
    def test_missing_quote_refused_by_default(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 1.0, CONTRACT_PUT, pos, [], None, None, FUNDS_OK)
        assert not ok
        assert any("no usable bid/ask quote" in r for r in reasons)

    def test_quote_error_refused_by_default(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 1.0, CONTRACT_PUT, pos, [],
            {"error": "no subscription"}, None, FUNDS_OK)
        assert not ok

    def test_allow_no_quote_downgrades_to_warning(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, warnings = check_option_order(
            "BTC", "BUY", 1, 1.0, CONTRACT_PUT, pos, [], None, None, FUNDS_OK,
            allow_no_quote=True)
        assert ok, reasons
        assert any("price-sanity check skipped" in w for w in warnings)

    def test_one_sided_quote_treated_as_missing(self):
        pos = [_pos(conid=111, right="P", position=-5)]
        ok, reasons, _ = check_option_order(
            "BTC", "BUY", 1, 1.0, CONTRACT_PUT, pos, [], {"bid": 1.0, "ask": None},
            None, FUNDS_OK)
        assert not ok
        assert any("no usable bid/ask quote" in r for r in reasons)
