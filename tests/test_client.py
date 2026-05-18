"""Tests for IBKR client functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ibkr_mcp_server.client import IBKRClient
from ibkr_mcp_server.utils import ConnectionError as IBKRConnectionError


class TestIBKRClient:
    """Test IBKR client functionality."""
    
    def test_account_switching(self, ibkr_client_mock):
        """Test account switching functionality."""
        # Test valid account switch
        assert ibkr_client_mock.switch_account('DU7654321') is True
        assert ibkr_client_mock.current_account == 'DU7654321'
        
        # Test invalid account switch
        assert ibkr_client_mock.switch_account('INVALID') is False
        assert ibkr_client_mock.current_account == 'DU7654321'  # Should remain unchanged
    
    def test_get_accounts(self, ibkr_client_mock):
        """Test getting account information."""
        accounts = ibkr_client_mock.get_accounts()
        assert accounts['current_account'] == 'DU1234567'
        assert 'DU1234567' in accounts['available_accounts']
        assert 'DU7654321' in accounts['available_accounts']
    
    def test_is_connected(self, ibkr_client_mock):
        """Test connection status check."""
        # Mock the ib.isConnected method properly
        ibkr_client_mock.ib.isConnected.return_value = True
        assert ibkr_client_mock.is_connected() is True
        
        # Test disconnected state
        ibkr_client_mock._connected = False
        assert ibkr_client_mock.is_connected() is False
    
    @pytest.mark.asyncio
    async def test_get_portfolio_not_connected(self):
        """Test portfolio request when not connected."""
        client = IBKRClient()
        client._connected = False

        with pytest.raises(IBKRConnectionError):
            await client.get_portfolio()

    @pytest.mark.asyncio
    async def test_get_quotes_per_symbol_errors(self, ibkr_client_mock):
        """Unqualified symbols surface as per-symbol errors without crashing the batch."""
        # qualifyContractsAsync leaves conId=0 on unqualified contracts; simulate
        # that for "BADSYM" while populating a conId for "AMD".
        async def fake_qualify(*contracts):
            for c in contracts:
                if c.symbol == "AMD":
                    c.conId = 12345
            return list(contracts)

        ibkr_client_mock.ib.qualifyContractsAsync = fake_qualify

        # Ticker with empty fields so the batch stream path returns 0s, driving
        # AMD into the historical-fallback branch. Give that branch no bars →
        # AMD lands in "unavailable". The point is to verify the return shape.
        ticker = MagicMock()
        ticker.last = None
        ticker.bid = None
        ticker.ask = None
        ticker.close = None
        ibkr_client_mock.ib.reqMktData.return_value = ticker
        ibkr_client_mock.ib.reqMarketDataType = MagicMock()
        ibkr_client_mock.ib.cancelMktData = MagicMock()
        ibkr_client_mock.ib.reqHistoricalDataAsync = AsyncMock(return_value=[])

        results = await ibkr_client_mock.get_quotes(["AMD", "BADSYM"])

        assert set(results.keys()) == {"AMD", "BADSYM"}
        assert results["BADSYM"] == {"symbol": "BADSYM", "error": "Contract not found"}
        assert results["AMD"]["symbol"] == "AMD"
        assert results["AMD"]["source"] == "unavailable"

    @pytest.mark.asyncio
    async def test_get_todays_fills_includes_order_ref(self, ibkr_client_mock):
        """Fills carry the parent order's orderRef as `tag`/`order_ref`/`source`.

        Three cases, exercising the cascade:
          1. SPY  — `Execution.orderRef` populated directly on execDetails.
          2. QQQ  — execution.orderRef empty; falls back to permId map from trades.
          3. NVDA — execution.orderRef empty AND permId not in trades map; falls
                    back to orderId map.
        Mirrors the 2026-05-18 SPY/QQQ/NVDA incident where the orders had rolled
        off `get_live_orders` but `get_todays_fills` returned no tag, breaking
        the swing-monitor reconciliation merge.
        """
        from datetime import datetime

        def _exec(*, exec_id, order_id, perm_id, symbol, side, qty, price,
                  order_ref=""):
            execution = MagicMock()
            execution.execId = exec_id
            execution.orderId = order_id
            execution.permId = perm_id
            execution.side = side
            execution.shares = qty
            execution.price = price
            execution.avgPrice = price
            execution.time = datetime(2026, 5, 18, 10, 33, 0)
            execution.acctNumber = "U4022128"
            execution.exchange = "SMART"
            execution.orderRef = order_ref

            contract = MagicMock()
            contract.symbol = symbol

            comm = MagicMock()
            comm.commission = 1.0
            comm.currency = "USD"

            fill = MagicMock()
            fill.execution = execution
            fill.contract = contract
            fill.commissionReport = comm
            return fill

        # Build the trades-cache map: SPY trade present (but won't be needed —
        # exec.orderRef is set), QQQ uses permId match, NVDA uses orderId match.
        def _trade(*, order_id, perm_id, order_ref):
            order = MagicMock()
            order.orderId = order_id
            order.permId = perm_id
            order.orderRef = order_ref
            tr = MagicMock()
            tr.order = order
            return tr

        ibkr_client_mock.ib.trades.return_value = [
            _trade(order_id=1001, perm_id=9001, order_ref="SWING_BBTAST_LONG_001"),
            _trade(order_id=1002, perm_id=9002, order_ref="SWING_EMA_PULLBACK_002"),
            _trade(order_id=1003, perm_id=0,    order_ref="SWING_FIB_GOLDEN_003"),
        ]

        ibkr_client_mock.ib.reqExecutionsAsync = AsyncMock(return_value=[
            _exec(exec_id="e1", order_id=1001, perm_id=9001, symbol="SPY",
                  side="BOT", qty=10, price=585.5,
                  order_ref="SWING_BBTAST_LONG_001"),
            _exec(exec_id="e2", order_id=1002, perm_id=9002, symbol="QQQ",
                  side="BOT", qty=8, price=505.25, order_ref=""),
            _exec(exec_id="e3", order_id=1003, perm_id=0, symbol="NVDA",
                  side="BOT", qty=5, price=142.10, order_ref=""),
        ])

        fills = await ibkr_client_mock.get_todays_fills()

        by_symbol = {f["symbol"]: f for f in fills}
        assert by_symbol["SPY"]["tag"] == "SWING_BBTAST_LONG_001"
        assert by_symbol["SPY"]["order_ref"] == "SWING_BBTAST_LONG_001"
        assert by_symbol["SPY"]["source"] == "SWING_BBTAST_LONG_001"
        assert by_symbol["QQQ"]["tag"] == "SWING_EMA_PULLBACK_002"
        assert by_symbol["NVDA"]["tag"] == "SWING_FIB_GOLDEN_003"

    @pytest.mark.asyncio
    async def test_get_todays_fills_untagged_returns_none(self, ibkr_client_mock):
        """Fill with no matching tag in any source → tag/order_ref/source = None."""
        from datetime import datetime

        execution = MagicMock()
        execution.execId = "x1"
        execution.orderId = 5555
        execution.permId = 7777
        execution.side = "SLD"
        execution.shares = 100
        execution.price = 50.0
        execution.avgPrice = 50.0
        execution.time = datetime(2026, 5, 18, 14, 0, 0)
        execution.acctNumber = "U4022128"
        execution.exchange = "SMART"
        execution.orderRef = ""

        contract = MagicMock()
        contract.symbol = "AAPL"

        comm = MagicMock()
        comm.commission = 1.0
        comm.currency = "USD"

        fill = MagicMock()
        fill.execution = execution
        fill.contract = contract
        fill.commissionReport = comm

        ibkr_client_mock.ib.trades.return_value = []
        ibkr_client_mock.ib.reqExecutionsAsync = AsyncMock(return_value=[fill])

        fills = await ibkr_client_mock.get_todays_fills()

        assert fills[0]["symbol"] == "AAPL"
        assert fills[0]["action"] == "SELL"
        assert fills[0]["tag"] is None
        assert fills[0]["order_ref"] is None
        assert fills[0]["source"] is None
