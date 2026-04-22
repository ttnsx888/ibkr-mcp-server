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
