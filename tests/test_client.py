"""Tests for IBKR client functionality."""

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ibkr_mcp_server.client import IBKRClient


class TestIBKRClient:
    """Test IBKR client functionality."""

    @pytest.mark.asyncio
    async def test_account_switching(self, ibkr_client_mock):
        """Test account switching functionality."""
        # Test valid account switch
        result = await ibkr_client_mock.switch_account('DU7654321')
        assert result['success'] is True
        assert ibkr_client_mock.current_account == 'DU7654321'

        # Test invalid account switch
        result = await ibkr_client_mock.switch_account('INVALID')
        assert result['success'] is False
        assert ibkr_client_mock.current_account == 'DU7654321'  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_get_accounts(self, ibkr_client_mock):
        """Test getting account information."""
        accounts = await ibkr_client_mock.get_accounts()
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
        # Avoid real reconnect attempts (with retries) against TWS.
        client._ensure_connected = AsyncMock(return_value=False)

        with pytest.raises(RuntimeError, match="Not connected to IBKR"):
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
    async def test_get_todays_fills_warms_trades_cache(self, ibkr_client_mock):
        """get_todays_fills must request open + completed orders before reading
        the trades cache. Each Claude tick spawns a fresh MCP process so
        `ib.trades()` starts empty; without warming, the orderRef fallback
        misses any order placed by a prior process. Regression guard for the
        2026-05-18 SPY/QQQ/NVDA naked-position incident.
        """
        ibkr_client_mock.ib.reqAllOpenOrdersAsync = AsyncMock(return_value=None)
        ibkr_client_mock.ib.reqCompletedOrdersAsync = AsyncMock(return_value=None)
        ibkr_client_mock.ib.trades.return_value = []
        ibkr_client_mock.ib.reqExecutionsAsync = AsyncMock(return_value=[])

        await ibkr_client_mock.get_todays_fills()

        ibkr_client_mock.ib.reqAllOpenOrdersAsync.assert_awaited_once()
        ibkr_client_mock.ib.reqCompletedOrdersAsync.assert_awaited_once_with(
            apiOnly=False)

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

    @pytest.mark.asyncio
    async def test_get_todays_fills_tier4_recovers_tag_from_cache(self, ibkr_client_mock):
        """Tier-4 fallback: a fill whose execution has no orderRef and is absent
        from the trades cache (different client_id, rolled off open-orders) still
        recovers its tag from the placement-time order_ref_cache by perm_id.

        Direct regression for the 2026-06-23 incident: swing_manual_exit_now.py
        (client_id 47) closed AMD; the SELL filled + rolled off; the swing-monitor
        tick (client_id 1) read it with tag=None, so Step 1.7's
        pending_close_filled map could not detect it (s_manual_amd_reconcile_no_tag).
        """
        from datetime import datetime
        from ibkr_mcp_server import order_ref_cache

        # The placing process (client 47) recorded the tag at stage time.
        order_ref_cache.record(perm_id=8888, order_id=6666,
                               order_ref="SWING_MANUAL_100_2026-06-23_0909",
                               account="U4022128")

        execution = MagicMock()
        execution.execId = "m1"
        execution.orderId = 6666
        execution.permId = 8888
        execution.side = "SLD"
        execution.shares = 28
        execution.price = 522.05
        execution.avgPrice = 522.05
        execution.time = datetime(2026, 6, 23, 14, 25, 0)
        execution.acctNumber = "U4022128"
        execution.exchange = "SMART"
        execution.orderRef = ""                       # cross-client: empty on execDetails

        contract = MagicMock()
        contract.symbol = "AMD"

        comm = MagicMock()
        comm.commission = 1.0
        comm.currency = "USD"

        fill = MagicMock()
        fill.execution = execution
        fill.contract = contract
        fill.commissionReport = comm

        ibkr_client_mock.ib.trades.return_value = []  # tiers 2/3 miss (rolled off)
        ibkr_client_mock.ib.reqExecutionsAsync = AsyncMock(return_value=[fill])

        fills = await ibkr_client_mock.get_todays_fills()

        assert fills[0]["symbol"] == "AMD"
        assert fills[0]["action"] == "SELL"
        assert fills[0]["tag"] == "SWING_MANUAL_100_2026-06-23_0909"
        assert fills[0]["order_ref"] == "SWING_MANUAL_100_2026-06-23_0909"
        assert fills[0]["source"] == "SWING_MANUAL_100_2026-06-23_0909"

    @pytest.mark.asyncio
    async def test_get_todays_fills_tier4_no_orderid_bleed(self, ibkr_client_mock):
        """A fill with permId=0 must NOT pick up a foreign order's tag via a
        colliding (client-scoped) orderId. Tier-4 keys on permId only, so the
        result is None — never a wrong tag. Guards the H1 footgun."""
        from datetime import datetime
        from ibkr_mcp_server import order_ref_cache

        # A different order (perm 999) happens to share orderId=3 (client-scoped).
        order_ref_cache.record(perm_id=999, order_id=3, order_ref="SWING_FOREIGN_TAG")

        execution = MagicMock()
        execution.execId = "z1"
        execution.orderId = 3                          # collides with the cached row
        execution.permId = 0                           # unresolved permId on this exec
        execution.side = "SLD"
        execution.shares = 10
        execution.price = 100.0
        execution.avgPrice = 100.0
        execution.time = datetime(2026, 6, 23, 15, 0, 0)
        execution.acctNumber = "U4022128"
        execution.exchange = "SMART"
        execution.orderRef = ""

        contract = MagicMock()
        contract.symbol = "FOO"
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

        assert fills[0]["symbol"] == "FOO"
        assert fills[0]["tag"] is None                 # NOT "SWING_FOREIGN_TAG"

    @pytest.mark.asyncio
    async def test_get_todays_fills_stk_vs_opt_contract_fields(self, ibkr_client_mock):
        """2026-09-22: option fills (e.g. short NVDA puts) must carry secType
        "OPT" plus right/strike/expiry/multiplier/conid/local_symbol so they
        can be separated from stock fills instead of leaking into swing perf
        as 1-share stock sells. A plain STK fill gets secType "STK" and None
        for the option-only fields."""
        from datetime import datetime

        def _exec(*, exec_id, order_id, perm_id, symbol, side, qty, price):
            execution = MagicMock()
            execution.execId = exec_id
            execution.orderId = order_id
            execution.permId = perm_id
            execution.side = side
            execution.shares = qty
            execution.price = price
            execution.avgPrice = price
            execution.time = datetime(2026, 9, 22, 10, 0, 0)
            execution.acctNumber = "U4022128"
            execution.exchange = "SMART"
            execution.orderRef = ""
            return execution

        comm = MagicMock()
        comm.commission = 1.0
        comm.currency = "USD"

        # STK contract — mimics ib_insync's Stock(), whose option-only fields
        # (right/strike/multiplier) come back as empty strings, not None.
        stk_contract = MagicMock()
        stk_contract.symbol = "AAPL"
        stk_contract.secType = "STK"
        stk_contract.conId = 265598
        stk_contract.localSymbol = "AAPL"
        stk_contract.right = ""
        stk_contract.strike = 0.0
        stk_contract.lastTradeDateOrContractMonth = ""
        stk_contract.multiplier = ""

        stk_fill = MagicMock()
        stk_fill.execution = _exec(exec_id="s1", order_id=1, perm_id=1,
                                    symbol="AAPL", side="BOT", qty=100, price=230.0)
        stk_fill.contract = stk_contract
        stk_fill.commissionReport = comm

        # OPT contract — short NVDA put.
        opt_contract = MagicMock()
        opt_contract.symbol = "NVDA"
        opt_contract.secType = "OPT"
        opt_contract.conId = 778899
        opt_contract.localSymbol = "NVDA  261016P00120000"
        opt_contract.right = "P"
        opt_contract.strike = 120.0
        opt_contract.lastTradeDateOrContractMonth = "20261016"
        opt_contract.multiplier = "100"

        opt_fill = MagicMock()
        opt_fill.execution = _exec(exec_id="o1", order_id=2, perm_id=2,
                                    symbol="NVDA", side="SLD", qty=1, price=3.5)
        opt_fill.contract = opt_contract
        opt_fill.commissionReport = comm

        ibkr_client_mock.ib.trades.return_value = []
        ibkr_client_mock.ib.reqExecutionsAsync = AsyncMock(
            return_value=[stk_fill, opt_fill])

        fills = await ibkr_client_mock.get_todays_fills()
        by_symbol = {f["symbol"]: f for f in fills}

        stk = by_symbol["AAPL"]
        assert stk["secType"] == "STK"
        assert stk["conid"] == "265598"
        assert stk["local_symbol"] == "AAPL"
        assert stk["right"] is None
        assert stk["strike"] is None
        assert stk["expiry"] is None
        assert stk["multiplier"] is None

        opt = by_symbol["NVDA"]
        assert opt["secType"] == "OPT"
        assert opt["conid"] == "778899"
        assert opt["local_symbol"] == "NVDA  261016P00120000"
        assert opt["right"] == "P"
        assert opt["strike"] == 120.0
        assert opt["expiry"] == "20261016"
        assert opt["multiplier"] == "100"

    def test_serialize_position_stk_vs_opt_contract_fields(self, ibkr_client_mock):
        """_serialize_position mirrors get_todays_fills' option-vs-stock
        contract fields (2026-09-22) — same rationale, positions side."""
        stk_contract = MagicMock()
        stk_contract.symbol = "AAPL"
        stk_contract.secType = "STK"
        stk_contract.conId = 265598
        stk_contract.localSymbol = "AAPL"
        stk_contract.right = ""
        stk_contract.strike = 0.0
        stk_contract.lastTradeDateOrContractMonth = ""
        stk_contract.multiplier = ""
        stk_contract.exchange = "SMART"

        stk_position = MagicMock()
        stk_position.contract = stk_contract
        stk_position.position = 100
        stk_position.avgCost = 230.0
        stk_position.account = "U4022128"

        opt_contract = MagicMock()
        opt_contract.symbol = "NVDA"
        opt_contract.secType = "OPT"
        opt_contract.conId = 778899
        opt_contract.localSymbol = "NVDA  261016P00120000"
        opt_contract.right = "P"
        opt_contract.strike = 120.0
        opt_contract.lastTradeDateOrContractMonth = "20261016"
        opt_contract.multiplier = "100"
        opt_contract.exchange = "SMART"

        opt_position = MagicMock()
        opt_position.contract = opt_contract
        opt_position.position = -1
        opt_position.avgCost = 350.0
        opt_position.account = "U4022128"

        stk = ibkr_client_mock._serialize_position(stk_position)
        assert stk["secType"] == "STK"
        assert stk["conid"] == "265598"
        assert stk["right"] is None
        assert stk["strike"] is None
        assert stk["multiplier"] is None

        opt = ibkr_client_mock._serialize_position(opt_position)
        assert opt["secType"] == "OPT"
        assert opt["conid"] == "778899"
        assert opt["right"] == "P"
        assert opt["strike"] == 120.0
        assert opt["multiplier"] == "100"


class TestOptionSupport:
    """2026-09-23: qualify_option / get_option_chain / get_option_quote /
    place_option_limit_order / whatif_option_order, plus the additive OPT
    fields on get_open_trades and the modify/cancel contract-reuse guarantee.
    """

    def _contract_details(self, *, conId=778899, secType="OPT", symbol="NVDA",
                           right="P", strike=120.0, expiry="20261016",
                           multiplier="100", exchange="SMART", currency="USD",
                           localSymbol="NVDA  261016P00120000"):
        contract = MagicMock()
        contract.conId = conId
        contract.secType = secType
        contract.symbol = symbol
        contract.right = right
        contract.strike = strike
        contract.lastTradeDateOrContractMonth = expiry
        contract.multiplier = multiplier
        contract.exchange = exchange
        contract.currency = currency
        contract.localSymbol = localSymbol
        cd = MagicMock()
        cd.contract = contract
        return cd

    # -- qualify_option -----------------------------------------------

    @pytest.mark.asyncio
    async def test_qualify_option_by_conid(self, ibkr_client_mock):
        cd = self._contract_details()
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[cd])

        contract = await ibkr_client_mock.qualify_option(conid=778899)
        assert contract.conId == 778899
        assert contract.secType == "OPT"

    @pytest.mark.asyncio
    async def test_qualify_option_by_spec(self, ibkr_client_mock):
        cd = self._contract_details()
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[cd])

        contract = await ibkr_client_mock.qualify_option(
            symbol="nvda", expiry="20261016", strike=120.0, right="put")
        assert contract.conId == 778899
        assert contract.right == "P"

    @pytest.mark.asyncio
    async def test_qualify_option_no_match_raises(self, ibkr_client_mock):
        from ibkr_mcp_server.utils import ValidationError
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[])
        with pytest.raises(ValidationError):
            await ibkr_client_mock.qualify_option(conid=999999)

    @pytest.mark.asyncio
    async def test_qualify_option_ambiguous_raises(self, ibkr_client_mock):
        from ibkr_mcp_server.utils import ValidationError
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(
            return_value=[self._contract_details(), self._contract_details()])
        with pytest.raises(ValidationError, match="Ambiguous"):
            await ibkr_client_mock.qualify_option(
                symbol="NVDA", expiry="20261016", strike=120.0, right="P")

    @pytest.mark.asyncio
    async def test_qualify_option_missing_spec_raises(self, ibkr_client_mock):
        from ibkr_mcp_server.utils import ValidationError
        with pytest.raises(ValidationError):
            await ibkr_client_mock.qualify_option(symbol="NVDA")

    @pytest.mark.asyncio
    async def test_qualify_option_non_option_result_raises(self, ibkr_client_mock):
        from ibkr_mcp_server.utils import ValidationError
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(
            return_value=[self._contract_details(secType="STK")])
        with pytest.raises(ValidationError):
            await ibkr_client_mock.qualify_option(conid=778899)

    # -- get_option_chain -----------------------------------------------

    @pytest.mark.asyncio
    async def test_get_option_chain(self, ibkr_client_mock):
        stock_cd = MagicMock()
        stock_contract = MagicMock()
        stock_contract.conId = 4815
        stock_cd.contract = stock_contract
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[stock_cd])

        smart_chain = SimpleNamespace(
            exchange="SMART", underlyingConId=4815, tradingClass="NVDA",
            multiplier="100", expirations={"20261016", "20261120"},
            strikes={100.0, 110.0, 120.0})
        other_chain = SimpleNamespace(
            exchange="CBOE", underlyingConId=4815, tradingClass="NVDA",
            multiplier="100", expirations={"20261016"}, strikes={100.0})
        ibkr_client_mock.ib.reqSecDefOptParamsAsync = AsyncMock(
            return_value=[other_chain, smart_chain])

        chain = await ibkr_client_mock.get_option_chain("nvda")
        assert chain["symbol"] == "NVDA"
        assert chain["expirations"] == ["20261016", "20261120"]
        assert chain["strikes"] == [100.0, 110.0, 120.0]
        assert chain["trading_class"] == "NVDA"
        assert chain["multiplier"] == "100"

    @pytest.mark.asyncio
    async def test_get_option_chain_no_underlying(self, ibkr_client_mock):
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[])
        chain = await ibkr_client_mock.get_option_chain("BADSYM")
        assert "error" in chain

    # -- get_option_quote -----------------------------------------------

    @pytest.mark.asyncio
    async def test_get_option_quote_returns_bid_ask_and_greeks(self, ibkr_client_mock):
        cd = self._contract_details()
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[cd])

        greeks = SimpleNamespace(delta=-0.32, gamma=0.01, theta=-0.05, vega=0.08,
                                 impliedVol=0.45, undPrice=131.2)
        ticker = MagicMock()
        ticker.bid = 3.40
        ticker.ask = 3.60
        ticker.last = 3.50
        ticker.modelGreeks = greeks
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.lastGreeks = None
        ibkr_client_mock.ib.reqMktData.return_value = ticker
        ibkr_client_mock.ib.reqMarketDataType = MagicMock()
        ibkr_client_mock.ib.cancelMktData = MagicMock()

        quote = await ibkr_client_mock.get_option_quote(conid=778899)
        assert quote["bid"] == 3.40
        assert quote["ask"] == 3.60
        assert quote["mid"] == 3.50
        assert quote["delta"] == -0.32
        assert quote["source"] == "live"
        assert quote["errors"] == []

    @pytest.mark.asyncio
    async def test_get_option_quote_no_data_returns_source_none_not_raise(
            self, ibkr_client_mock, monkeypatch):
        # Both the live and delayed snapshot windows poll for ~4s each with no
        # data ever arriving — fast-forward the sleeps (via a fake that still
        # fires the errorEvent mid-window, matching how a real 10089 arrives
        # asynchronously while the snapshot is polling) so this test doesn't
        # burn ~8s of wall clock.
        import ibkr_mcp_server.client as client_module

        cd = self._contract_details()
        ibkr_client_mock.ib.reqContractDetailsAsync = AsyncMock(return_value=[cd])

        fired = {"done": False}

        async def fake_sleep(*args, **kwargs):
            if not fired["done"]:
                fired["done"] = True
                # Simulate IBKR emitting "no OPRA subscription" (10089) while
                # the snapshot's poll window is open.
                ibkr_client_mock._on_error(
                    0, 10089, "Requested market data is not subscribed", cd.contract)

        monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

        ticker = MagicMock()
        ticker.bid = None
        ticker.ask = None
        ticker.last = None
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.lastGreeks = None
        ibkr_client_mock.ib.reqMktData.return_value = ticker
        ibkr_client_mock.ib.reqMarketDataType = MagicMock()
        ibkr_client_mock.ib.cancelMktData = MagicMock()

        quote = await ibkr_client_mock.get_option_quote(conid=778899)
        assert quote["source"] == "none"
        assert quote["bid"] is None
        assert 10089 in quote["errors"]

    @pytest.mark.asyncio
    async def test_get_option_quote_bad_spec_returns_error_dict_not_raise(self, ibkr_client_mock):
        quote = await ibkr_client_mock.get_option_quote(symbol="NVDA")  # missing expiry/strike/right
        assert "error" in quote

    # -- whatif_option_order ----------------------------------------------

    @pytest.mark.asyncio
    async def test_whatif_option_order(self, ibkr_client_mock):
        cd = self._contract_details()
        contract = cd.contract
        state = SimpleNamespace(initMarginChange=1500.0, maintMarginChange=1200.0,
                                equityWithLoanAfter=98000.0, commission=0.65)
        ibkr_client_mock.ib.whatIfOrderAsync = AsyncMock(return_value=state)

        result = await ibkr_client_mock.whatif_option_order(contract, "SELL", 1, 3.50)
        assert result["init_margin_change"] == 1500.0
        assert result["maint_margin_change"] == 1200.0
        assert result["equity_with_loan_after"] == 98000.0
        assert result["commission_est"] == 0.65

    @pytest.mark.asyncio
    async def test_whatif_option_order_sets_tif_and_account(self, ibkr_client_mock):
        # IBKR returns an empty OrderState for a what-if with no TIF (10349
        # preset warning) or, on a multi-account login, no account.
        cd = self._contract_details()
        ibkr_client_mock.current_account = "U4022128"
        state = SimpleNamespace(initMarginChange="369.24", maintMarginChange="369.25",
                                equityWithLoanAfter="379238.0", commission=1.7976931348623157e308)
        ibkr_client_mock.ib.whatIfOrderAsync = AsyncMock(return_value=state)

        result = await ibkr_client_mock.whatif_option_order(cd.contract, "SELL", 1, 3.70)
        sent_order = ibkr_client_mock.ib.whatIfOrderAsync.call_args[0][1]
        assert sent_order.tif == "DAY"
        assert sent_order.account == "U4022128"
        assert result["init_margin_change"] == 369.24      # string from TWS parsed
        assert result["commission_est"] is None             # UNSET sentinel dropped

    @pytest.mark.asyncio
    async def test_whatif_option_order_failure_returns_none_fields_not_raise(self, ibkr_client_mock):
        cd = self._contract_details()
        ibkr_client_mock.ib.whatIfOrderAsync = AsyncMock(side_effect=RuntimeError("boom"))

        result = await ibkr_client_mock.whatif_option_order(cd.contract, "SELL", 1, 3.50)
        assert result["init_margin_change"] is None
        assert "error" in result

    # -- place_option_limit_order ------------------------------------------

    @pytest.mark.asyncio
    async def test_place_option_limit_order_outside_rth_always_false(self, ibkr_client_mock):
        cd = self._contract_details()
        contract = cd.contract

        order = SimpleNamespace(orderId=501, permId=90501, action="SELL",
                                totalQuantity=1, lmtPrice=3.50, tif="DAY",
                                outsideRth=False, orderRef="WHEEL_STO_1", account="U1")
        status = SimpleNamespace(status="Submitted", filled=0, remaining=1)
        trade = SimpleNamespace(order=order, orderStatus=status)
        ibkr_client_mock.ib.placeOrder.return_value = trade

        result = await ibkr_client_mock.place_option_limit_order(
            contract=contract, action="sell", quantity=1, limit_price=3.499,
            tif="day", order_ref="WHEEL_STO_1")

        assert result["outside_rth"] is False
        assert result["limit_price"] == 3.50   # rounded to 2dp
        assert result["conid"] == 778899
        assert result["right"] == "P"
        placed_order = ibkr_client_mock.ib.placeOrder.call_args[0][1]
        assert placed_order.outsideRth is False

    @pytest.mark.asyncio
    async def test_place_option_limit_order_rejects_bad_tif(self, ibkr_client_mock):
        from ibkr_mcp_server.utils import ValidationError
        cd = self._contract_details()
        with pytest.raises(ValidationError):
            await ibkr_client_mock.place_option_limit_order(
                contract=cd.contract, action="SELL", quantity=1,
                limit_price=3.50, tif="OPG")

    @pytest.mark.asyncio
    async def test_place_option_limit_order_requires_qualified_contract(self, ibkr_client_mock):
        from ibkr_mcp_server.utils import ValidationError
        unqualified = MagicMock()
        unqualified.conId = 0
        with pytest.raises(ValidationError):
            await ibkr_client_mock.place_option_limit_order(
                contract=unqualified, action="SELL", quantity=1, limit_price=3.50)

    # -- get_open_trades: additive OPT fields --------------------------------

    @pytest.mark.asyncio
    async def test_get_open_trades_opt_additive_fields(self, ibkr_client_mock):
        opt_order = SimpleNamespace(orderId=1, permId=1, parentId=0, action="SELL",
                                    totalQuantity=1, lmtPrice=3.5, auxPrice=None,
                                    orderType="LMT", ocaGroup="", ocaType=0, tif="DAY",
                                    outsideRth=False, transmit=True, account="U1",
                                    orderRef="TAG")
        opt_status = SimpleNamespace(status="Submitted", filled=0, remaining=1)
        opt_contract = self._contract_details().contract
        opt_trade = SimpleNamespace(order=opt_order, orderStatus=opt_status,
                                    contract=opt_contract)

        stk_order = SimpleNamespace(orderId=2, permId=2, parentId=0, action="BUY",
                                    totalQuantity=10, lmtPrice=230.0, auxPrice=None,
                                    orderType="LMT", ocaGroup="", ocaType=0, tif="DAY",
                                    outsideRth=False, transmit=True, account="U1",
                                    orderRef="")
        stk_status = SimpleNamespace(status="Submitted", filled=0, remaining=10)
        stk_contract = MagicMock()
        stk_contract.symbol = "AAPL"
        stk_contract.secType = "STK"
        stk_contract.conId = 265598
        stk_contract.right = ""
        stk_contract.strike = 0.0
        stk_contract.lastTradeDateOrContractMonth = ""
        stk_contract.multiplier = ""
        stk_trade = SimpleNamespace(order=stk_order, orderStatus=stk_status,
                                    contract=stk_contract)

        ibkr_client_mock.ib.openTrades.return_value = [opt_trade, stk_trade]

        trades = await ibkr_client_mock.get_open_trades()
        by_id = {t["order_id"]: t for t in trades}

        opt = by_id[1]
        assert opt["secType"] == "OPT"
        assert opt["conid"] == "778899"
        assert opt["right"] == "P"
        assert opt["strike"] == 120.0
        assert opt["expiry"] == "20261016"
        assert opt["multiplier"] == "100"

        stk = by_id[2]
        assert stk["secType"] == "STK"
        assert stk["right"] is None
        assert stk["strike"] is None
        assert stk["expiry"] is None
        assert stk["multiplier"] is None

    # -- modify/cancel reuse trade.contract (never rebuild Stock()) --------

    @pytest.mark.asyncio
    async def test_modify_order_reuses_opt_trade_contract(self, ibkr_client_mock):
        """modify_order must resubmit with the SAME contract object the trade
        already carries — never a freshly-built Stock(symbol), which would be
        wrong (and un-placeable) for an OPT trade."""
        cd = self._contract_details()
        opt_contract = cd.contract
        order = SimpleNamespace(orderId=77, permId=907, totalQuantity=1,
                                lmtPrice=3.50, auxPrice=None, account="U1",
                                transmit=False)
        status = SimpleNamespace(status="Submitted", filled=0, remaining=1)
        trade = SimpleNamespace(order=order, orderStatus=status, contract=opt_contract)
        ibkr_client_mock.ib.trades.return_value = [trade]
        ibkr_client_mock.ib.placeOrder.return_value = trade

        result = await ibkr_client_mock.modify_order(77, limit_price=3.75)

        assert result["modified"] is True
        placed_contract = ibkr_client_mock.ib.placeOrder.call_args[0][0]
        assert placed_contract is opt_contract
        assert placed_contract.secType == "OPT"

    @pytest.mark.asyncio
    async def test_cancel_order_reuses_opt_trade_no_rebuild(self, ibkr_client_mock):
        """cancel_order must cancel the trade's own order — it never touches
        contract construction at all, so an OPT trade cancels exactly like a
        STK trade."""
        cd = self._contract_details()
        opt_contract = cd.contract
        order = SimpleNamespace(orderId=88, permId=908)
        status = SimpleNamespace(status="Submitted", filled=0, remaining=1)
        trade = SimpleNamespace(order=order, orderStatus=status, contract=opt_contract)
        ibkr_client_mock.ib.trades.return_value = [trade]

        result = await ibkr_client_mock.cancel_order(88)

        assert result["cancelled"] is True
        assert result["symbol"] == "NVDA"
        ibkr_client_mock.ib.cancelOrder.assert_called_once_with(order)
