"""Staged order store.

Staged orders are validated and persisted locally but NEVER sent to IBKR until
confirm_order is called. Gives the user a review step between scan-time staging
and actual submission.
"""

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


@dataclass
class StagedOrder:
    id: str
    symbol: str
    action: str              # BUY | SELL
    quantity: int
    limit_price: float
    tif: str
    source: str
    created_at: str          # ISO 8601
    expires_at: str          # ISO 8601

    @classmethod
    def new(cls, symbol: str, action: str, quantity: int, limit_price: float,
            tif: str = "DAY", source: str = "") -> "StagedOrder":
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4())[:8],
            symbol=symbol.upper(),
            action=action.upper(),
            quantity=int(quantity),
            limit_price=float(limit_price),
            tif=tif.upper(),
            source=source,
            created_at=now.isoformat(timespec="seconds"),
            expires_at=(now + timedelta(days=STAGED_TTL_DAYS)).isoformat(timespec="seconds"),
        )

    def is_expired(self) -> bool:
        return datetime.fromisoformat(self.expires_at) < datetime.utcnow()

    def summary(self) -> str:
        return (f"{self.action} {self.quantity} {self.symbol} "
                f"@ ${self.limit_price:.2f} ({self.tif}) — {self.source}")


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
