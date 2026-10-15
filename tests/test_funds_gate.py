"""Tests for the BUY funds gate (_buy_funds_gate, 2026-07-31).

Regression for the 2026-07-29 incident: on a cash account IBKR accepts a
BUY LMT exceeding available funds and parks it 'Inactive' instead of
rejecting it — the fund-manager scan rested $20.6k of Inactive BUYs
against $189 cash on live-U25242754. The gate must refuse such orders at
stage/confirm time.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ibkr_mcp_server import tools
from ibkr_mcp_server.config import Settings
from ibkr_mcp_server.tools import (_buy_funds_gate, _is_cash_account,
                                   _open_buy_notional)


def _snapshot(monkeypatch, tags: dict):
    async def fake_snapshot(account):
        return tags
    monkeypatch.setattr(tools, "_funds_snapshot", fake_snapshot)


def _client(monkeypatch, connected=True, open_trades=(), account="U1"):
    client = MagicMock()
    client.is_connected.return_value = connected
    client.current_account = account
    client.ib.openTrades.return_value = list(open_trades)
    monkeypatch.setattr(tools, "ibkr_client", client)
    return client


def _trade(action="BUY", remaining=10, lmt=100.0, aux=None, account="U1"):
    order = SimpleNamespace(action=action, account=account,
                            totalQuantity=remaining, lmtPrice=lmt,
                            auxPrice=aux)
    status = SimpleNamespace(remaining=remaining)
    return SimpleNamespace(order=order, orderStatus=status)


class TestBuyFundsGate:
    async def test_sell_orders_always_pass(self, monkeypatch):
        _client(monkeypatch, connected=False)
        assert (await _buy_funds_gate("SELL", 100, 50.0, strict=True))["ok"]

    async def test_cash_account_rejects_over_available_funds(self, monkeypatch):
        """The incident case: $189 cash, $2,182 BUY → refuse."""
        _client(monkeypatch)
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "189.21"})
        r = await _buy_funds_gate("BUY", 12, 181.82, strict=False)
        assert not r["ok"]
        assert "exceeds available funds" in r["error"]
        assert "AvailableFunds" in r["error"]

    async def test_cash_account_accepts_within_funds(self, monkeypatch):
        _client(monkeypatch)
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "10000"})
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert r["ok"]
        assert r["funds_headroom_after"] == 9000.0

    async def test_open_buy_orders_reduce_headroom(self, monkeypatch):
        """Resting (incl. Inactive) BUYs consume the pool cumulatively."""
        _client(monkeypatch, open_trades=[
            _trade(remaining=30, lmt=200.0),          # $6,000 committed
            _trade(action="SELL", remaining=5, lmt=300.0),  # SELLs ignored
        ])
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "7000"})
        # headroom = 7000 − 6000 = 1000 → $1,500 BUY refused
        r = await _buy_funds_gate("BUY", 10, 150.0, strict=False)
        assert not r["ok"]
        # …but a $900 BUY fits
        r2 = await _buy_funds_gate("BUY", 6, 150.0, strict=False)
        assert r2["ok"]

    async def test_margin_account_uses_buying_power(self, monkeypatch):
        _client(monkeypatch)
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL",
                                "AvailableFunds": "5000",
                                "BuyingPower": "20000"})
        # 100 × $150 = $15,000 > AvailableFunds but < BuyingPower → pass
        r = await _buy_funds_gate("BUY", 100, 150.0, strict=False)
        assert r["ok"]

    async def test_margin_resting_buys_do_not_block(self, monkeypatch):
        """2026-08-12 incident: operator DCA ladders ($199k resting BUYs)
        must not starve a small swing BUY on a margin account — IBKR
        enforces margin per-fill natively. A non-blocking warning notes
        the all-fills exposure."""
        _client(monkeypatch, open_trades=[
            _trade(remaining=400, lmt=207.5),   # ~$83k NVDA-style ladder
            _trade(remaining=280, lmt=415.0),   # ~$116k more
        ])
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL",
                                "AvailableFunds": "56679",
                                "BuyingPower": "200001"})
        r = await _buy_funds_gate("BUY", 16, 485.54, strict=False)
        assert r["ok"]
        assert "warning" in r and "all-fills exposure" in r["warning"]

    async def test_margin_order_over_buying_power_refused(self, monkeypatch):
        _client(monkeypatch)
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL",
                                "AvailableFunds": "5000",
                                "BuyingPower": "20000"})
        r = await _buy_funds_gate("BUY", 100, 250.0, strict=False)  # $25k > BP
        assert not r["ok"]
        assert "BuyingPower" in r["error"]

    async def test_offline_fails_open_at_stage_closed_at_confirm(self, monkeypatch):
        _client(monkeypatch, connected=False)
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert r["ok"] and "warning" in r
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=True)
        assert not r["ok"]

    async def test_summary_failure_fails_open_at_stage_closed_at_confirm(self, monkeypatch):
        _client(monkeypatch)

        async def boom(account):
            raise RuntimeError("IBKR API error")
        monkeypatch.setattr(tools, "_funds_snapshot", boom)
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=False)
        assert r["ok"] and "warning" in r
        r = await _buy_funds_gate("BUY", 10, 100.0, strict=True)
        assert not r["ok"]

    async def test_stop_orders_use_aux_price(self, monkeypatch):
        _client(monkeypatch, open_trades=[
            _trade(remaining=10, lmt=None, aux=90.0),  # BUY STP $900
        ])
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "1000"})
        r = await _buy_funds_gate("BUY", 5, 100.0, strict=False)  # $500 > $100 left
        assert not r["ok"]

    async def test_other_account_orders_ignored(self, monkeypatch):
        _client(monkeypatch, open_trades=[
            _trade(remaining=100, lmt=500.0, account="U2"),  # different account
        ])
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "1000"})
        r = await _buy_funds_gate("BUY", 5, 100.0, strict=False)
        assert r["ok"]

    async def test_unset_double_prices_skipped(self, monkeypatch):
        """ib_async leaves unset lmtPrice at UNSET_DOUBLE — must not count."""
        client = _client(monkeypatch, open_trades=[
            _trade(remaining=10, lmt=1.7976931348623157e+308, aux=None),
        ])
        assert _open_buy_notional(client.current_account) == 0.0


# Real tags IBKR returned for the cash account live-U25242754 (accounts/
# live-U25242754/snapshots/latest.json, 2026-10-15). AccountType is INDIVIDUAL,
# NOT "CASH": the AccountType tag describes ownership structure, so a gate
# keyed on it never fired on this account between 2026-07-31 and 2026-10-15.
CASH_ACCT = "U25242754"
CASH_ACCT_TAGS = {"AccountType": "INDIVIDUAL", "AvailableFunds": "252.9",
                  "BuyingPower": "252.9", "TotalCashValue": "252.9",
                  "SettledCash": "252.9", "NetLiquidation": "252.9",
                  "EquityWithLoanValue": "252.9", "FullInitMarginReq": "0.0"}


class TestCashAccountDetection:
    """The §8h cumulative open-BUY deduction keys on the account id, not on
    AccountType (which reads INDIVIDUAL for cash and margin accounts alike)."""

    async def test_incident_case_with_real_individual_tag(self, monkeypatch):
        """$253 cash, $2,182 BUY: refused as a cash-account overspend."""
        _client(monkeypatch, account=CASH_ACCT)
        _snapshot(monkeypatch, CASH_ACCT_TAGS)
        r = await _buy_funds_gate("BUY", 12, 181.82, strict=False)
        assert not r["ok"]
        assert "exceeds available funds" in r["error"]
        assert "CASH" in r["error"]

    async def test_open_buys_deducted_despite_individual_tag(self, monkeypatch):
        """The gap: a BUY that fits AvailableFunds alone but not what open
        BUYs already committed must be refused on the configured cash account."""
        _client(monkeypatch, account=CASH_ACCT, open_trades=[
            _trade(remaining=30, lmt=200.0, account=CASH_ACCT),  # $6,000
        ])
        _snapshot(monkeypatch, {**CASH_ACCT_TAGS, "AvailableFunds": "7000",
                                "BuyingPower": "7000"})
        r = await _buy_funds_gate("BUY", 10, 150.0, strict=False)  # $1,500 > $1,000
        assert not r["ok"]
        assert "committed to open BUY orders" in r["error"]
        r2 = await _buy_funds_gate("BUY", 6, 150.0, strict=False)  # $900 fits
        assert r2["ok"] and r2["funds_headroom_after"] == 100.0

    async def test_live_prefixed_account_id_matches(self, monkeypatch):
        _client(monkeypatch, account="live-" + CASH_ACCT, open_trades=[
            _trade(remaining=30, lmt=200.0, account="live-" + CASH_ACCT),
        ])
        _snapshot(monkeypatch, {**CASH_ACCT_TAGS, "AvailableFunds": "7000"})
        assert not (await _buy_funds_gate("BUY", 10, 150.0, strict=False))["ok"]

    async def test_unlisted_individual_account_stays_margin(self, monkeypatch):
        """2026-08-13 regression guard: an INDIVIDUAL account that is not in
        the cash list never gets the cumulative deduction."""
        _client(monkeypatch, account="U7777777", open_trades=[
            _trade(remaining=30, lmt=200.0, account="U7777777"),
        ])
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL",
                                "AvailableFunds": "7000", "BuyingPower": "28000"})
        r = await _buy_funds_gate("BUY", 10, 150.0, strict=False)
        assert r["ok"]
        assert "funds_headroom_after" in r and "CASH" not in str(r)

    async def test_account_type_cash_still_detected_without_listing(self, monkeypatch):
        _client(monkeypatch, account="U7777777")
        _snapshot(monkeypatch, {"AccountType": "CASH", "AvailableFunds": "189.21"})
        r = await _buy_funds_gate("BUY", 12, 181.82, strict=False)
        assert not r["ok"] and "exceeds available funds" in r["error"]

    async def test_setting_overrides_the_cash_list(self, monkeypatch):
        monkeypatch.setattr(tools.settings, "ibkr_cash_accounts", "U9999999")
        _client(monkeypatch, account="U9999999", open_trades=[
            _trade(remaining=30, lmt=200.0, account="U9999999"),
        ])
        _snapshot(monkeypatch, {"AccountType": "INDIVIDUAL", "AvailableFunds": "7000",
                                "BuyingPower": "28000"})
        assert not (await _buy_funds_gate("BUY", 10, 150.0, strict=False))["ok"]
        # ...and the default account is no longer treated as cash
        assert not _is_cash_account(CASH_ACCT, "INDIVIDUAL")

    async def test_empty_setting_means_no_listed_cash_accounts(self, monkeypatch):
        monkeypatch.setattr(tools.settings, "ibkr_cash_accounts", "")
        assert not _is_cash_account(CASH_ACCT, "INDIVIDUAL")
        assert _is_cash_account(CASH_ACCT, "CASH")

    def test_is_cash_account_edge_inputs(self):
        assert _is_cash_account(CASH_ACCT, "INDIVIDUAL")
        assert _is_cash_account(f" live-{CASH_ACCT} ", "")
        assert not _is_cash_account(None, "INDIVIDUAL")
        assert not _is_cash_account("All", "INDIVIDUAL")
        assert not _is_cash_account("U4022128", "INDIVIDUAL")  # the margin account

    def test_setting_parses_comma_list_from_env(self, monkeypatch):
        monkeypatch.setenv("IBKR_CASH_ACCOUNTS", "U1, live-U2 ,,")
        monkeypatch.setattr(tools, "settings", Settings())
        assert tools._configured_cash_accounts() == frozenset({"U1", "U2"})

    def test_default_setting_is_the_cash_account(self):
        assert Settings().ibkr_cash_accounts == CASH_ACCT
