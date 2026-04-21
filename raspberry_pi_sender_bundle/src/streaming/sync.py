"""GPS-to-video timestamp synchronization buffer."""

from datetime import datetime, timezone
from typing import List, Optional

from src.database.models import GPSFix
from src.streaming.packet import parse_iso_ts


class GPSSyncBuffer:
    """Stores recent GPS fixes and resolves nearest fix for a frame timestamp."""

    def __init__(self, max_size: int = 512):
        self.max_size = max(16, max_size)
        self._items: List[GPSFix] = []

    def add_fix(self, fix: GPSFix) -> None:
        self._items.append(fix)
        if len(self._items) > self.max_size:
            self._items = self._items[-self.max_size :]

    def get_closest(self, frame_ts_utc: str, max_age_seconds: float = 2.0) -> Optional[GPSFix]:
        target = parse_iso_ts(frame_ts_utc)
        if target is None or not self._items:
            return None

        best_fix = None
        best_delta = None

        for fix in self._items:
            ts = fix.timestamp
            if ts.tzinfo is None:
                continue
            delta = abs((ts - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_fix = fix

        if best_fix is None:
            return None

        if best_delta is not None and best_delta > max_age_seconds:
            return None

        return best_fix

    @staticmethod
    def parse_fix(payload: dict) -> GPSFix:
        ts = parse_iso_ts(str(payload.get("timestamp", "")))
        if ts is None:
            ts = datetime.now(timezone.utc)

        return GPSFix(
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
            satellites=payload.get("satellites"),
            heading_deg=payload.get("heading_deg"),
            speed_mps=payload.get("speed_mps"),
            fix=bool(payload.get("fix", False)),
            source=str(payload.get("source", "stream")),
            timestamp=ts,
        )
