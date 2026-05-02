"""Staged order store.

Staged orders are validated and persisted locally but NEVER sent to IBKR until
confirm_order is called. Gives the user a review step between scan-time staging
and actual submission.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional


logger = logging.getLogger(__name__)

STAGED_ORDERS_PATH = Path.home() / "ibkr-mcp-server" / ".staged_orders.json"
STAGED_TTL_DAYS = 7


VALID_ORDER_TYPES = ("LMT", "STP", "STP_LMT", "MKT", "MOC", "MOO")


@dataclass
class StagedOrder:
    id: str
    symbol: str
    action: str                       # BUY | SELL
    quantity: int
    limit_price: float                # 0.0 for STP/MKT/MOC/MOO (only LMT/STP_LMT use this)
    tif: str
    source: str
    created_at: str                   # ISO 8601
    expires_at: str                   # ISO 8601
    outside_rth: bool = False
    # 2026-05-02: bracket / OCA / non-LMT support (audit H5)
    order_type: str = "LMT"           # one of VALID_ORDER_TYPES
    stop_price: Optional[float] = None     # required for STP, STP_LMT
    oca_group: Optional[str] = None        # OCA tag — children share it
    oca_type: int = 0                       # 0=none, 1=cancel, 2=reduce_with_block, 3=reduce_no_block
    parent_staged_id: Optional[str] = None  # if set, this is a bracket child of that parent
    transmit_last: bool = True              # for bracket legs — only the final leg should be True

    @classmethod
    def new(cls, symbol: str, action: str, quantity: int,
            limit_price: float = 0.0,
            tif: str = "DAY", source: str = "",
            outside_rth: bool = False,
            order_type: str = "LMT",
            stop_price: Optional[float] = None,
            oca_group: Optional[str] = None,
            oca_type: int = 0,
            parent_staged_id: Optional[str] = None,
            transmit_last: bool = True) -> "StagedOrder":
        now = datetime.utcnow()
        ot = order_type.upper()
        if ot not in VALID_ORDER_TYPES:
            raise ValueError(f"order_type must be one of {VALID_ORDER_TYPES}, got {ot!r}")
        if ot in ("STP", "STP_LMT") and stop_price is None:
            raise ValueError(f"stop_price required for order_type={ot}")
        if ot in ("LMT", "STP_LMT") and limit_price <= 0:
            raise ValueError(f"limit_price required (>0) for order_type={ot}")
        return cls(
            id=str(uuid.uuid4())[:8],
            symbol=symbol.upper(),
            action=action.upper(),
            quantity=int(quantity),
            limit_price=float(limit_price or 0.0),
            tif=tif.upper(),
            source=source,
            created_at=now.isoformat(timespec="seconds"),
            expires_at=(now + timedelta(days=STAGED_TTL_DAYS)).isoformat(timespec="seconds"),
            outside_rth=bool(outside_rth),
            order_type=ot,
            stop_price=float(stop_price) if stop_price is not None else None,
            oca_group=oca_group,
            oca_type=int(oca_type),
            parent_staged_id=parent_staged_id,
            transmit_last=bool(transmit_last),
        )

    def is_expired(self) -> bool:
        return datetime.fromisoformat(self.expires_at) < datetime.utcnow()

    def summary(self) -> str:
        rth = " outsideRTH" if self.outside_rth else ""
        ot = self.order_type
        if ot == "LMT":
            px = f"@ ${self.limit_price:.2f}"
        elif ot == "STP":
            px = f"STP ${self.stop_price:.2f}"
        elif ot == "STP_LMT":
            px = f"STP_LMT trigger ${self.stop_price:.2f} lmt ${self.limit_price:.2f}"
        elif ot in ("MKT", "MOC", "MOO"):
            px = ot
        else:
            px = ot
        oca = f" oca={self.oca_group}" if self.oca_group else ""
        bracket = f" parent={self.parent_staged_id}" if self.parent_staged_id else ""
        return (f"{self.action} {self.quantity} {self.symbol} "
                f"{px} ({self.tif}{rth}){oca}{bracket} — {self.source}")


class StagedOrderStore:
    """File-backed staged-order store. Single-user, no concurrency contention expected."""

    def __init__(self, path: Path = STAGED_ORDERS_PATH):
        self.path = path
        self._lock = Lock()
        self._orders: dict[str, StagedOrder] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            self._orders = {k: StagedOrder(**v) for k, v in raw.items()}
        except Exception as e:
            logger.warning(f"Failed to load staged orders from {self.path}: {e}")
            self._orders = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._orders.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, order: StagedOrder) -> str:
        with self._lock:
            self._orders[order.id] = order
            self._save()
        logger.info(f"Staged: {order.summary()} (id={order.id})")
        return order.id

    def get(self, staged_id: str) -> Optional[StagedOrder]:
        return self._orders.get(staged_id)

    def list(self, symbol: Optional[str] = None,
             source_prefix: Optional[str] = None) -> list[StagedOrder]:
        self.prune_expired()
        out = list(self._orders.values())
        if symbol:
            sym = symbol.upper()
            out = [o for o in out if o.symbol == sym]
        if source_prefix:
            out = [o for o in out if o.source.startswith(source_prefix)]
        out.sort(key=lambda o: o.created_at)
        return out

    def remove(self, staged_id: str) -> bool:
        with self._lock:
            if staged_id in self._orders:
                del self._orders[staged_id]
                self._save()
                return True
        return False

    def children_of(self, parent_staged_id: str) -> list[StagedOrder]:
        """All staged children whose parent_staged_id matches. Used by
        confirm_bracket_order to submit the entire bracket atomically."""
        return sorted(
            (o for o in self._orders.values() if o.parent_staged_id == parent_staged_id),
            key=lambda o: o.created_at,
        )

    def remove_bracket(self, parent_staged_id: str) -> int:
        """Remove parent + all its children from the store. Returns count."""
        with self._lock:
            ids = [parent_staged_id] + [
                o.id for o in self._orders.values() if o.parent_staged_id == parent_staged_id
            ]
            removed = 0
            for sid in ids:
                if sid in self._orders:
                    del self._orders[sid]
                    removed += 1
            if removed:
                self._save()
        return removed

    def prune_expired(self) -> int:
        with self._lock:
            expired = [k for k, v in self._orders.items() if v.is_expired()]
            for k in expired:
                del self._orders[k]
            if expired:
                self._save()
        return len(expired)


# Global store
staged_store = StagedOrderStore()
