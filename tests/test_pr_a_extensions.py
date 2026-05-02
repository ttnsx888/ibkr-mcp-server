"""
PR-A smoke tests — verifies the bracket / STP / modify extensions to
ibkr-mcp-server load cleanly and the staging store round-trips correctly.

Does NOT exercise IBKR-side placement (that needs paper TWS running).
Run with the venv python:
    /Users/ttang/ibkr-mcp-server/venv/bin/python tests/test_pr_a_extensions.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ibkr_mcp_server.orders import (
    StagedOrder, VALID_ORDER_TYPES, staged_store, STAGED_ORDERS_PATH,
)
from ibkr_mcp_server.tools import TOOLS, server, call_tool
from ibkr_mcp_server import tools as tools_module


# ── Hygiene: nuke the staged-orders file so we start fresh ────────────────
if STAGED_ORDERS_PATH.exists():
    STAGED_ORDERS_PATH.unlink()
staged_store._orders = {}


# ── Quote-cache stub: avoid coupling tests to live IBKR / current prices ──
# The drift-gate validator (tools._validate_order_inputs) calls
# _cached_get_quote → ibkr_client.get_quote, which on a connected MCP
# server returns real market data. That makes test fixtures brittle
# (any hard-coded LMT > 30% from current price gets rejected).
# Instead, pre-seed the validator's per-symbol cache with stub quotes.
# Tests can then use any LMT prices that are within 30% of the stub last.
import time as _time

def _stub_quote(symbol: str, last: float):
    """Inject (now, fake-quote) into the validator's quote cache for `symbol`."""
    tools_module._QUOTE_CACHE[symbol.upper()] = (
        _time.monotonic(),
        {"symbol": symbol.upper(), "last": last, "bid": last - 0.05,
          "ask": last + 0.05, "close": last, "source": "test_stub"},
    )

# Stub all symbols our tests touch.
for _sym, _last in (("SPY", 480.0), ("QQQ", 410.0)):
    _stub_quote(_sym, _last)


def test_tool_registry_has_new_tools():
    names = {t.name for t in TOOLS}
    for new in ("stage_stop_order", "stage_bracket_order", "modify_live_order"):
        assert new in names, f"{new} not registered. registered: {sorted(names)}"
    print(f"  ✓ tool registry: {len(names)} total, all 3 PR-A tools present")


def test_staged_order_lmt_compat():
    o = StagedOrder.new("SPY", "BUY", 100, limit_price=480.0, source="t")
    assert o.order_type == "LMT"
    assert o.stop_price is None
    assert o.parent_staged_id is None
    print(f"  ✓ LMT compat: {o.summary()}")


def test_staged_order_stp_validates():
    try:
        StagedOrder.new("SPY", "SELL", 100, order_type="STP", source="t")
        raise AssertionError("expected ValueError for missing stop_price")
    except ValueError as e:
        assert "stop_price required" in str(e)
    o = StagedOrder.new("SPY", "SELL", 100, order_type="STP", stop_price=475.0,
                         tif="GTC", outside_rth=True, source="failsafe",
                         oca_group="G1", oca_type=2, parent_staged_id="abc12345")
    assert o.order_type == "STP"
    assert o.stop_price == 475.0
    assert o.parent_staged_id == "abc12345"
    print(f"  ✓ STP validation + fields: {o.summary()}")


def test_staged_order_moc():
    o = StagedOrder.new("SPY", "SELL", 100, order_type="MOC", source="time-stop")
    assert o.order_type == "MOC"
    assert o.limit_price == 0.0
    print(f"  ✓ MOC stages with no price: {o.summary()}")


def _async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_stage_bracket_via_dispatch():
    """End-to-end: dispatch stage_bracket_order via the MCP call_tool path,
    confirm we get a parent staged_id + 2 child ids, and store reflects the
    parent_staged_id linkage so confirm_order would submit the bracket."""
    # Pre-condition: empty store
    assert len(staged_store._orders) == 0, f"store dirty: {staged_store._orders}"

    # Prices chosen relative to the SPY stub last=$480 (see top-of-file).
    args = {
        "symbol": "SPY",
        "parent_action": "BUY",
        "parent_quantity": 20,
        "parent_limit_price": 480.00,
        "parent_tif": "GTC",
        "parent_outside_rth": False,
        "children": [
            {"order_type": "LMT", "action": "SELL", "quantity": 10,
             "limit_price": 488.00, "tif": "GTC", "tag": "SWING_T1"},
            {"order_type": "STP", "action": "SELL", "quantity": 20,
             "stop_price":  476.00, "tif": "GTC", "outside_rth": True,
             "tag": "SWING_FAILSAFE_STOP"},
        ],
        "source": "swing-scout 2026-05-02 SPY",
    }
    result = _async(call_tool("stage_bracket_order", args))
    body = json.loads(result[0].text)
    assert body.get("staged") is True, f"bracket stage failed: {body}"
    parent_id = body["staged_id"]
    child_ids = body["child_staged_ids"]
    assert len(child_ids) == 2

    # Verify store linkage
    parent = staged_store.get(parent_id)
    assert parent is not None
    assert parent.order_type == "LMT"
    assert parent.action == "BUY"
    children = staged_store.children_of(parent_id)
    assert len(children) == 2
    types = {c.order_type for c in children}
    assert types == {"LMT", "STP"}
    # OCA group propagation
    assert all(c.oca_group == body["oca_group"] for c in children)
    assert all(c.oca_type == 2 for c in children)
    # Last child has transmit_last=True (the only commit-trigger leg)
    transmit_flags = [c.transmit_last for c in children]
    assert transmit_flags[-1] is True
    assert all(not f for f in transmit_flags[:-1])
    print(f"  ✓ stage_bracket_order: parent={parent_id} children={child_ids} oca={body['oca_group']}")


def test_stage_stop_via_dispatch():
    # Stub-relative price (SPY last=$480).
    args = {
        "symbol": "SPY",
        "action": "SELL",
        "quantity": 10,
        "stop_price": 478.00,
        "tif": "GTC",
        "outside_rth": True,
        "source": "ad-hoc BE-stop",
    }
    result = _async(call_tool("stage_stop_order", args))
    body = json.loads(result[0].text)
    assert body.get("staged") is True, f"STP stage failed: {body}"
    sid = body["staged_id"]
    o = staged_store.get(sid)
    assert o.order_type == "STP"
    assert o.stop_price == 478.00
    assert o.action == "SELL"
    assert o.outside_rth is True
    print(f"  ✓ stage_stop_order: {o.summary()}")


def test_bracket_remove_clears_all():
    """remove_bracket on the parent id removes parent + all children."""
    # Stub-relative prices (QQQ last=$410).
    args = {
        "symbol": "QQQ", "parent_action": "BUY",
        "parent_quantity": 5, "parent_limit_price": 410.0,
        "children": [
            {"order_type": "LMT", "action": "SELL", "quantity": 2, "limit_price": 415.0},
            {"order_type": "STP", "action": "SELL", "quantity": 5, "stop_price": 405.0},
        ],
        "source": "test",
    }
    result = _async(call_tool("stage_bracket_order", args))
    body = json.loads(result[0].text)
    parent_id = body["staged_id"]
    assert len(staged_store.children_of(parent_id)) == 2
    removed = staged_store.remove_bracket(parent_id)
    assert removed == 3   # parent + 2 children
    assert staged_store.get(parent_id) is None
    assert len(staged_store.children_of(parent_id)) == 0
    print(f"  ✓ remove_bracket: cleared parent + 2 children atomically")


def test_validation_rejects_negative_stop():
    args = {"symbol": "SPY", "action": "SELL", "quantity": 10, "stop_price": -1.0}
    try:
        result = _async(call_tool("stage_stop_order", args))
        body = json.loads(result[0].text)
        # JSONSchema may catch this OR our validator may. Either path is fine —
        # we just need "staged: False" or some error.
        assert body.get("staged") is False or "error" in body
    except (ValueError, KeyError):
        pass  # JSONSchema rejection at MCP layer
    print(f"  ✓ negative stop_price rejected")


def test_modify_live_order_requires_a_field():
    args = {"order_id": 12345}
    result = _async(call_tool("modify_live_order", args))
    body = json.loads(result[0].text)
    assert body.get("modified") is False
    assert "must provide at least one" in body.get("error", "")
    print(f"  ✓ modify_live_order rejects empty modify request")


def test_confirm_order_rejects_bracket_child():
    """Audit I3: confirming a bracket child's staged_id alone would orphan
    the parent + siblings. confirm_order must reject with a helpful error."""
    # Stage a bracket; pick a child id
    # Stub-relative prices (SPY last=$480).
    args = {
        "symbol": "SPY", "parent_action": "BUY",
        "parent_quantity": 5, "parent_limit_price": 480.0,
        "children": [
            {"order_type": "LMT", "action": "SELL", "quantity": 5,
             "limit_price": 488.0, "tif": "GTC"},
            {"order_type": "STP", "action": "SELL", "quantity": 5,
             "stop_price": 472.0, "tif": "GTC", "outside_rth": True},
        ],
        "source": "I3_TEST",
    }
    res = _async(call_tool("stage_bracket_order", args))
    body = json.loads(res[0].text)
    assert body["staged"] is True
    parent_id = body["staged_id"]
    child_ids = body["child_staged_ids"]
    assert len(child_ids) == 2

    # Try to confirm a child alone — must be rejected.
    res2 = _async(call_tool("confirm_order", {"staged_id": child_ids[0]}))
    body2 = json.loads(res2[0].text)
    assert body2.get("submitted") is False
    assert body2.get("parent_staged_id") == parent_id
    err = body2.get("error", "")
    assert "bracket child" in err.lower() and parent_id in err

    # Cleanup
    staged_store.remove_bracket(parent_id)
    print(f"  ✓ I3: confirm_order on bracket child rejected with parent_staged_id={parent_id}")


# ── runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_tool_registry_has_new_tools,
        test_staged_order_lmt_compat,
        test_staged_order_stp_validates,
        test_staged_order_moc,
        test_stage_bracket_via_dispatch,
        test_stage_stop_via_dispatch,
        test_bracket_remove_clears_all,
        test_validation_rejects_negative_stop,
        test_modify_live_order_requires_a_field,
        test_confirm_order_rejects_bracket_child,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"all {len(tests)} passed")
