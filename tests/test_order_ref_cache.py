"""Tests for the placement-time orderRef cache (order_ref_cache.py).

Regression cover for the 2026-06-23 manual-exit null-tag incident: a SELL placed
by swing_manual_exit_now.py (client_id 47) filled and rolled off open-orders, so
the swing-monitor tick (client_id 1) read the fill with no tag. The cache lets
any later process recover the tag by perm_id.

Lookup is keyed on perm_id ONLY (globally unique, safe cross-client). order_id is
persisted for forensics but never indexed — it is client-scoped and would
collide across clients, risking a wrong tag.

The conftest `_isolate_order_ref_cache` autouse fixture points the cache dir at a
per-test tmp path, so these run hermetically with no real disk side effects.
"""

import json
import os

from ibkr_mcp_server import order_ref_cache


def test_record_and_lookup_by_perm_id():
    order_ref_cache.record(perm_id=9001, order_id=1001,
                           order_ref="SWING_MANUAL_100_2026-06-23_0909",
                           account="U4022128")
    assert order_ref_cache.lookup(perm_id=9001) == "SWING_MANUAL_100_2026-06-23_0909"


def test_load_map_indexes_by_perm_id_only():
    order_ref_cache.record(9002, 1002, "SWING_CAT_STOP_EMA_PULLBACK_x")
    perm_map = order_ref_cache.load_map()
    assert perm_map[9002] == "SWING_CAT_STOP_EMA_PULLBACK_x"
    # order_id is NOT a lookup key.
    assert 1002 not in perm_map


def test_order_id_collision_does_not_bleed_tag():
    """H1 regression: two orders from different clients colliding on order_id
    (orderId is client-scoped) must each resolve to their OWN tag by perm_id —
    no cross-bleed, and order_id is never a usable key."""
    order_ref_cache.record(perm_id=111, order_id=3, order_ref="SWING_MANUAL_100_a")  # client 47
    order_ref_cache.record(perm_id=222, order_id=3, order_ref="SWING_BBTAST_LONG_b")  # client 1
    assert order_ref_cache.lookup(perm_id=111) == "SWING_MANUAL_100_a"
    assert order_ref_cache.lookup(perm_id=222) == "SWING_BBTAST_LONG_b"
    # The colliding order_id is not exposed as a key at all.
    assert order_ref_cache.lookup(perm_id=3) == ""


def test_empty_tag_is_noop():
    order_ref_cache.record(perm_id=500, order_id=600, order_ref="")
    order_ref_cache.record(perm_id=500, order_id=600, order_ref=None)
    assert order_ref_cache.lookup(perm_id=500) == ""
    assert order_ref_cache.load_map() == {}


def test_no_key_is_noop():
    # No perm_id and no order_id → nothing usable → not written.
    order_ref_cache.record(perm_id=0, order_id=0, order_ref="SWING_ORPHAN")
    assert order_ref_cache.load_map() == {}


def test_perm_id_zero_at_record_is_unrecoverable_but_safe():
    # If permId were 0 at placement (does not happen in practice after the 1s
    # settle), the row is keyed only by the forensic order_id and is not
    # recoverable via lookup — but it can never attach a WRONG tag either.
    order_ref_cache.record(perm_id=0, order_id=42, order_ref="SWING_T1_x")
    assert order_ref_cache.load_map() == {}
    assert order_ref_cache.lookup(perm_id=42) == ""


def test_miss_returns_empty_string():
    assert order_ref_cache.lookup(perm_id=123456) == ""


def test_later_write_wins_on_collision():
    order_ref_cache.record(perm_id=7, order_id=7, order_ref="OLD")
    order_ref_cache.record(perm_id=7, order_id=7, order_ref="NEW")
    assert order_ref_cache.lookup(perm_id=7) == "NEW"


def test_tolerates_torn_final_line():
    # Simulate a partially written (torn) last line — load_map must skip it, not crash.
    order_ref_cache.record(perm_id=10, order_id=20, order_ref="GOOD")
    path = order_ref_cache._today_path()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"perm_id": 11, "order_id": 21, "order_ref": "TRUNC')  # no newline, no close brace
    perm_map = order_ref_cache.load_map()
    assert perm_map.get(10) == "GOOD"
    assert 11 not in perm_map


def test_record_writes_single_jsonl_line_with_order_id_forensics():
    order_ref_cache.record(perm_id=33, order_id=44, order_ref="SWING_T1_x")
    path = order_ref_cache._today_path()
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["perm_id"] == 33 and rec["order_id"] == 44   # order_id retained for forensics
    assert rec["order_ref"] == "SWING_T1_x"
