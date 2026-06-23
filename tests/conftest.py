"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import MagicMock

from ibkr_mcp_server.client import IBKRClient


@pytest.fixture(autouse=True)
def _isolate_order_ref_cache(tmp_path, monkeypatch):
    """Point the placement-time orderRef cache (order_ref_cache.py) at a
    per-test tmp dir so tests never read or write the developer's real
    ~/.ibkr-mcp-server cache (keeps get_todays_fills tier-4 hermetic)."""
    monkeypatch.setenv("IBKR_ORDER_REF_CACHE_DIR", str(tmp_path / "order_refs"))


@pytest.fixture
def mock_ib():
    """Mock IB object for testing."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.managedAccounts.return_value = ['DU1234567', 'DU7654321']
    return ib


@pytest.fixture
def ibkr_client_mock(mock_ib):
    """Mock IBKR client for testing."""
    client = IBKRClient()
    client.ib = mock_ib
    client._connected = True
    client.accounts = ['DU1234567', 'DU7654321']
    client.current_account = 'DU1234567'
    return client
