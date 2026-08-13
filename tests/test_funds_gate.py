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
from ibkr_mcp_server.tools import _buy_funds_gate, _open_buy_notional


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
