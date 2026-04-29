# Upgrade — orderRef tagging (2026-04-28)

## What changed

Three places now plumb the `source` provenance tag through to IBKR's order
log via `Order.orderRef`, so the scanner's per-tier identifier round-trips
across `confirm_order` → IB → `get_live_orders`:

- `client.py` `place_limit_order` — accepts `order_ref: str = ""` and sets
  `LimitOrder(orderRef=...)`.
- `tools.py` `confirm_order` branch — passes `order_ref=order.source` when
  invoking `place_limit_order`.
- `client.py` `get_open_trades` — reads `t.order.orderRef` and exposes it
  as `source` in the returned dict (None if blank).

Prior to this change, `place_limit_order` ignored the `source` argument
entirely and the in-memory tag in `staged_store` was discarded the moment
`confirm_order` removed the staged entry. As a result, every confirmed
order arrived in IBKR with an empty `orderRef`, and the next scanner read
of `get_live_orders` saw no `source` field — flagging the order as a
"legacy" untagged one and leaving it alone forever, even when the DCA
ladder shifted and the tier should have been cancelled and resubmitted.

## Recovery for pre-fix orders

Orders that were placed by the scanner before this fix are stored in IBKR
with an empty `orderRef`. They cannot be retroactively tagged — IB does
not let clients modify a working order's `orderRef` after submission.
Two ways to recover:

1. **Manual one-shot cancel** (recommended). In TWS, cancel every
   scanner-placed working order. The next `/fund-manager` scan will
   re-submit them via the new code path with proper tags. Identify them
   by source description in TWS or by matching `(symbol, side, qty,
   limit)` against the most recent scan report.

2. **Adopt-by-shape** (not implemented). A scanner-side fallback could
   match `source`-less orders against the desired ladder by
   `(symbol, side, qty, price ±1%)` and adopt them as the tier. This is
   risky — it could touch orders the user placed manually at the same
   price and tier — so it is intentionally not implemented. If the
   number of pre-fix orders is small, prefer option 1.

After the next scan post-cancel, every live scanner order carries an
`orderRef` of the form `scan YYYY-MM-DD SYMBOL T<n> @<price>x<qty>` and
the legacy branch in `skills/watchlist-scan/order-submission.md §3x-i.b`
stops firing.
