"""Verify modify_order works on a NON-OCA order (the OCA test failed because
IBKR rejects revisions to OCA-grouped orders — error 10326). We need to
confirm the modify_order primitive itself is sound."""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ibkr_mcp_server.client import ibkr_client


async def main():
    print("=" * 78)
    print("modify_order — non-OCA standalone order")
    print("=" * 78)
    await ibkr_client.connect()
    assert ibkr_client.is_paper

    q = await ibkr_client.get_quote("SPY")
    last = q.get("last") or q.get("close")
    print(f"\nSPY ref price: ${last:.2f}")

    # Place a standalone BUY LMT 30% below market (no OCA, no parent).
    initial_lmt = round(last * 0.70, 2)
    print(f"\n[1] place standalone BUY LMT @ ${initial_lmt} (won't fill)")
    res = await ibkr_client.place_limit_order(
        symbol="SPY", action="BUY", quantity=1, limit_price=initial_lmt,
        tif="GTC", order_ref="MODIFY_TEST_NON_OCA",
    )
    print(json.dumps(res, indent=2, default=str))
    oid = res["order_id"]

    try:
        # Modify limit_price up by $0.50
        new_lmt = round(initial_lmt + 0.50, 2)
        print(f"\n[2] modify_order: limit_price ${initial_lmt} → ${new_lmt}")
        m = await ibkr_client.modify_order(order_id=oid, limit_price=new_lmt)
        print(json.dumps(m, indent=2, default=str))
        assert m.get("modified") is True

        await asyncio.sleep(2.0)
        trades = await ibkr_client.get_open_trades()
        seen = next((t for t in trades if t["order_id"] == oid), None)
        if seen is None:
            print(f"  ⚠ order disappeared from get_open_trades — checking status")
            # Maybe it filled or got cancelled?
            return
        print(f"  current state: lmt=${seen['limit_price']} status={seen['status']}")
        assert abs(seen["limit_price"] - new_lmt) < 0.01, \
            f"modify didn't take: expected {new_lmt}, got {seen['limit_price']}"

        # Modify quantity 1 → 2
        print(f"\n[3] modify_order: quantity 1 → 2")
        m2 = await ibkr_client.modify_order(order_id=oid, quantity=2)
        print(json.dumps(m2, indent=2, default=str))
        await asyncio.sleep(2.0)
        trades2 = await ibkr_client.get_open_trades()
        seen2 = next((t for t in trades2 if t["order_id"] == oid), None)
        if seen2:
            print(f"  current state: qty={seen2['quantity']} lmt=${seen2['limit_price']} status={seen2['status']}")

        print("\n✓ modify_order works on non-OCA orders")

    finally:
        print(f"\n[cleanup] cancel oid={oid}")
        try:
            await ibkr_client.cancel_order(oid)
        except Exception as e:
            print(f"  cancel failed: {e}")
        await ibkr_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
