"""Persistent placement-time orderRef cache (2026-06-23).

Why this exists
---------------
`get_todays_fills` enriches each execution with its parent order's `orderRef`
(the tag swing-monitor reconciles on). It tries three tiers:

  1. `Execution.orderRef` straight from `reqExecutions`
  2. a `permId -> orderRef` map from `ib.trades()`
  3. an `orderId -> orderRef` map from `ib.trades()`

All three are populated from the *currently connected* session. They fail for a
fill whose order was placed by a **different client_id** that has since
disconnected AND whose order has **rolled off** the open-orders window:

  * `Execution.orderRef` is empty cross-client (a known IBKR quirk — the field
    is only delivered to the placing client's `execDetails`),
  * `reqAllOpenOrders` no longer lists it (it filled),
  * `reqCompletedOrders` does not reliably carry `orderRef` for a now-gone
    foreign client.

The 2026-06-23 incident: `swing_manual_exit_now.py` (client_id 47) manually
closed AMD; the SELL filled and rolled off; the 14:30 swing-monitor tick
(client_id 1) read the fill with `tag=None`, so Step 1.7's `pending_close_filled`
map could not detect it and the close fell back to position-diff reconciliation
(`s_manual_amd_reconcile_no_tag`).

Fix
---
Every order placement records `(perm_id, order_id) -> order_ref` to a small
append-only per-day JSONL on disk. `get_todays_fills` consults it as a tier-4
fallback. Because `permId` is globally unique across clients and is captured at
placement by the very process that owns the tag, the tag becomes recoverable by
ANY later process/client regardless of roll-off — restoring true tag-based
detection.

Design notes
------------
* Append-only: each `record()` is a single `os.write()` to an `O_APPEND` fd.
  POSIX makes a small append atomic across processes, so the four swing client
  processes (MCP=1, manual=47, watchdog=48, dashboard=42) can write
  concurrently without a lock.
* Pure observability: `record()` runs AFTER the order is already placed and
  never raises — a cache failure can never affect order placement.
* `load_map()` globs the (pruned, tiny) cache dir once so a fills loop does a
  single read, not one per execution.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from glob import glob
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Per-day files are pruned after this many days. The cache only needs to cover
# the current trading day; a few days of slack absorbs timezone / overnight
# boundaries and weekends cheaply (a handful of tiny lines per day).
RETENTION_DAYS = 5

_FILE_PREFIX = "order_refs_"
_FILE_SUFFIX = ".jsonl"


def _cache_dir() -> str:
    """Resolve the cache directory (env override → ~/.ibkr-mcp-server/order_refs)."""
    override = os.environ.get("IBKR_ORDER_REF_CACHE_DIR")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".ibkr-mcp-server", "order_refs")


def _today_path() -> str:
    return os.path.join(_cache_dir(), f"{_FILE_PREFIX}{datetime.now():%Y%m%d}{_FILE_SUFFIX}")


def _coerce_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record(perm_id, order_id, order_ref: str,
           account: Optional[str] = None) -> None:
    """Append a `(perm_id, order_id) -> order_ref` row to today's cache file.

    Best-effort and silent: never raises, so an order-placement call site can
    invoke it bare. No-ops when there is no tag to cache or no usable key.
    """
    try:
        ref = (str(order_ref) if order_ref is not None else "").strip()
        if not ref:
            return                                    # nothing to cache
        pid = _coerce_int(perm_id)
        oid = _coerce_int(order_id)
        if pid == 0 and oid == 0:
            return                                    # no key to look it up by

        # order_id is persisted for forensic correlation only — it is NOT used
        # as a lookup key (it is client-scoped and would collide across the
        # MCP/manual/watchdog/dashboard clients). permId is the lookup key.
        rec = {
            "perm_id":   pid,
            "order_id":  oid,
            "order_ref": ref,
            "account":   account or None,
            "ts":        datetime.now().isoformat(timespec="seconds"),
        }
        line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")

        path = _today_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # One short (<512B) write to an O_APPEND fd: atomic in practice on
        # macOS/Linux, so the swing client processes can append concurrently
        # without a lock. The real safety net for any partial/interleaved line
        # is the reader — load_map() tolerates torn and garbage lines.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception as exc:                          # noqa: BLE001 — never block placement
        logger.debug("order_ref_cache.record skipped: %s: %s",
                     type(exc).__name__, exc)


def load_map() -> Dict[int, str]:
    """Return `perm_id -> order_ref` from all cache files.

    Keyed on `permId` only: it is globally unique across clients, so it is the
    one safe cross-client key. `order_id` is deliberately NOT indexed — it is
    client-scoped (each client mints low orderIds), so a global order_id index
    would collide across the MCP/manual/watchdog/dashboard clients and could
    attach a *wrong* tag to a fill (worse than no tag). Correctness does NOT
    depend on `_today_path()` matching the fill's date — this globs every
    retained file, so a write/read straddling local midnight still resolves.

    Reads every (pruned, tiny) per-day file once. Later rows win on collision.
    Never raises — returns whatever could be parsed.
    """
    _prune_old_files()
    perm_map: Dict[int, str] = {}
    try:
        pattern = os.path.join(_cache_dir(), f"{_FILE_PREFIX}*{_FILE_SUFFIX}")
        for fpath in sorted(glob(pattern)):
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except json.JSONDecodeError:
                            continue                  # tolerate a torn final line
                        ref = (rec.get("order_ref") or "").strip()
                        if not ref:
                            continue
                        pid = _coerce_int(rec.get("perm_id"))
                        if pid:
                            perm_map[pid] = ref
            except OSError:
                continue
    except Exception as exc:                          # noqa: BLE001
        logger.debug("order_ref_cache.load_map failed: %s: %s",
                     type(exc).__name__, exc)
    return perm_map


def lookup(perm_id=None) -> str:
    """Single-key lookup by `perm_id` (the only safe cross-client key).
    Returns "" on miss."""
    pid = _coerce_int(perm_id)
    if pid:
        return load_map().get(pid, "")
    return ""


def _prune_old_files() -> None:
    """Delete cache files older than RETENTION_DAYS (best-effort)."""
    try:
        cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y%m%d")
        pattern = os.path.join(_cache_dir(), f"{_FILE_PREFIX}*{_FILE_SUFFIX}")
        for fpath in glob(pattern):
            base = os.path.basename(fpath)
            stamp = base[len(_FILE_PREFIX):-len(_FILE_SUFFIX)]
            if len(stamp) == 8 and stamp.isdigit() and stamp < cutoff:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception:                                 # noqa: BLE001
        pass
