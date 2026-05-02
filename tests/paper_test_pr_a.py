"""
PR-A paper-trading verification.

Exercises the new IBKRClient methods (place_bracket_order, place_stop_order,
modify_order) against the IBKR paper account. Designed to be SAFE:

  • Uses 1 share of SPY (smallest viable test order).
  • Parent BUY LMT is set 30% below last price → won't fill.
  • Child T1 SELL LMT is well above market.
  • Child failsafe SELL STP is well below market.
  • Cancels everything at the end, even if assertions fail.

Run:
    /Users/ttang/ibkr-mcp-server/venv/bin/python \\
        /Users/ttang/ibkr-mcp-server/tests/paper_test_pr_a.py

Requires: TWS paper running on port 7497 with API enabled.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ibkr_mcp_server.client import ibkr_client


SYMBOL = "SPY"
PAPER_PORT = 7497
TEST_QTY = 1


async def _safe_cancel(order_id, label):
    if not order_id:
        return
    try:
        r = await ibkr_client.cancel_order(int(order_id))
        print(f"    cancel {label} oid={order_id}: {r.get('cancelled')} status={r.get('status')}")
    except Exception as e:
        print(f"    cancel {label} oid={order_id} FAILED: {e}")


async def _await_state(predicate, timeout=10.0, every=0.5):
    """Poll until predicate() is truthy or timeout."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        await asyncio.sleep(every)
        if predicate():
            return True
    return False


async def main():
    print("=" * 78)
    print("PR-A paper test — bracket, STP, modify against paper IBKR")
    print("=" * 78)

    # 1. Connect
    print("\n[1] Connect to paper TWS @ 127.0.0.1:7497")
    ok = await ibkr_client.connect()
    assert ok, "connect failed"
    status = {
        "connected": ibkr_client.is_connected(),
        "port": ibkr_client.port,
        "current_account": ibkr_client.current_account,
        "available_accounts": ibkr_client.accounts,
        "is_paper": ibkr_client.is_paper,
    }
    print(json.dumps(status, indent=2))
    assert status["is_paper"] is True, f"NOT a paper account! refusing. {status}"

    # 2. Quote SPY for sane price selection
    print(f"\n[2] Get quote for {SYMBOL}")
    q = await ibkr_client.get_quote(SYMBOL)
    print(json.dumps(q, indent=2))
    last = q.get("last") or q.get("close") or 0
    if last <= 0:
        bid, ask = q.get("bid", 0), q.get("ask", 0)
        last = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
    assert last > 0, f"no usable price for {SYMBOL}: {q}"

    # Pick prices so nothing fills:
    #   parent BUY LMT  = last * 0.70   (30% below)
    #   child T1 LMT   = last * 1.30   (30% above)
    #   child failsafe = last * 0.65   (35% below — catastrophic level)
    parent_lmt = round(last * 0.70, 2)
    t1_lmt     = round(last * 1.30, 2)
    failsafe   = round(last * 0.65, 2)
    print(f"\n  prices: last=${last:.2f}  parent=${parent_lmt}  t1=${t1_lmt}  failsafe=${failsafe}")

    placed_ids: list[int] = []
    parent_order_id: int | None = None

    try:
        # 3. Place bracket
        print("\n[3] place_bracket_order (parent BUY LMT + 2 OCA SELL children)")
        bracket = await ibkr_client.place_bracket_order(
            symbol=SYMBOL,
            parent_action="BUY",
            parent_quantity=TEST_QTY,
            parent_limit_price=parent_lmt,
            children=[
                {"order_type": "LMT", "action": "SELL", "quantity": TEST_QTY,
                 "limit_price": t1_lmt, "tif": "GTC", "tag": "TEST_T1"},
                {"order_type": "STP", "action": "SELL", "quantity": TEST_QTY,
                 "stop_price": failsafe, "tif": "GTC", "outside_rth": True,
                 "oca_type": 2, "tag": "TEST_FAILSAFE"},
            ],
            parent_tif="GTC",
            order_ref="PR-A_PAPER_TEST",
        )
        print(json.dumps(bracket, indent=2, default=str))
        parent_order_id = bracket["parent"]["order_id"]
        placed_ids = [parent_order_id] + [c["order_id"] for c in bracket["children"]]
        oca_group = bracket["oca_group"]
        assert len(bracket["children"]) == 2
        print(f"  ✓ bracket placed: parent={parent_order_id} oca={oca_group} children={[c['order_id'] for c in bracket['children']]}")

        # 4. Verify via get_open_trades
        print("\n[4] get_open_trades — confirm parent + children + OCA + parent_id wiring")
        await asyncio.sleep(2.0)
        trades = await ibkr_client.get_open_trades()
        ours = [t for t in trades if t["order_id"] in placed_ids]
        print(f"  found {len(ours)}/3 of our orders in open_trades")
        for t in ours:
            print(f"    oid={t['order_id']:>8} parent_id={t['parent_id']:>8} "
                  f"{t['action']} {int(t['quantity'])} {t['symbol']} "
                  f"{t['order_type']:>4} lmt=${t['limit_price']:.2f} stop=${t['stop_price']:.2f} "
                  f"oca={t['oca_group']} type={t['oca_type']} status={t['status']}")
        # Find the children — they should have parent_id == parent_order_id
        children_seen = [t for t in ours if t["parent_id"] == parent_order_id]
        assert len(children_seen) == 2, f"expected 2 children with parent_id={parent_order_id}, got {len(children_seen)}"
        # Both children share the same OCA group
        ocas = {t["oca_group"] for t in children_seen}
        assert len(ocas) == 1 and oca_group in ocas, f"OCA mismatch: {ocas} vs {oca_group}"
        # Children types: one LMT, one STP
        ctypes = sorted(t["order_type"] for t in children_seen)
        assert ctypes == ["LMT", "STP"], f"child types: {ctypes}"
        # STP child has stop_price set; LMT child has limit_price set
        for t in children_seen:
            if t["order_type"] == "STP":
                assert abs(t["stop_price"] - failsafe) < 0.01, f"failsafe stop_price drift: {t}"
            elif t["order_type"] == "LMT":
                assert abs(t["limit_price"] - t1_lmt) < 0.01, f"T1 limit_price drift: {t}"
        print("  ✓ parent_id linkage, OCA grouping, prices all match")

        # 5. Test modify_order — bump the failsafe STP price
        print("\n[5] modify_order — bump failsafe stop_price by $0.10 (no fill, just modify)")
        failsafe_oid = next(t["order_id"] for t in children_seen if t["order_type"] == "STP")
        new_stop = round(failsafe + 0.10, 2)
        modify_result = await ibkr_client.modify_order(order_id=failsafe_oid, stop_price=new_stop)
        print(json.dumps(modify_result, indent=2, default=str))
        assert modify_result.get("modified") is True, f"modify failed: {modify_result}"
        await asyncio.sleep(1.5)
        trades_after = await ibkr_client.get_open_trades()
        modified_seen = next((t for t in trades_after if t["order_id"] == failsafe_oid), None)
        assert modified_seen is not None, "modified order disappeared"
        assert abs(modified_seen["stop_price"] - new_stop) < 0.01, \
            f"modify didn't take effect: stop_price still {modified_seen['stop_price']}"
        print(f"  ✓ failsafe stop modified: ${failsafe} → ${modified_seen['stop_price']}")

        # 6. Standalone STP via place_stop_order (separate test — not bracket-linked)
        print("\n[6] place_stop_order — standalone SELL STP, no OCA, won't fill")
        # Note: standalone SELL STP requires a position to exist OR short-selling enabled.
        # On paper, IBKR usually accepts it. We'll place it well below market and cancel.
        stp_result = await ibkr_client.place_stop_order(
            symbol=SYMBOL, action="SELL", quantity=TEST_QTY,
            stop_price=round(last * 0.60, 2),
            tif="GTC", outside_rth=True,
            order_ref="PR-A_PAPER_STANDALONE_STP",
        )
        print(json.dumps(stp_result, indent=2, default=str))
        standalone_stp_oid = stp_result.get("order_id")
        if standalone_stp_oid:
            placed_ids.append(standalone_stp_oid)
            await asyncio.sleep(1.0)
            trades3 = await ibkr_client.get_open_trades()
            stp_seen = next((t for t in trades3 if t["order_id"] == standalone_stp_oid), None)
            if stp_seen:
                print(f"  ✓ standalone STP visible: {stp_seen['action']} {int(stp_seen['quantity'])} "
                      f"{stp_seen['symbol']} STP ${stp_seen['stop_price']:.2f} status={stp_seen['status']}")
            else:
                print(f"  ⚠ standalone STP not visible in open_trades (may be paper-rejected — that's OK)")
        else:
            print(f"  ⚠ standalone STP didn't get an order_id — paper may reject naked SELL STP")

        print("\n" + "=" * 78)
        print("✓ ALL VERIFICATIONS PASSED")
        print("=" * 78)

    finally:
        # 7. Always cancel everything we placed
        print("\n[7] Cleanup — cancel all placed orders")
        for oid in placed_ids:
            await _safe_cancel(oid, label=f"oid={oid}")
        await asyncio.sleep(1.0)

        # 8. Disconnect
        print("\n[8] Disconnect")
        await ibkr_client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
