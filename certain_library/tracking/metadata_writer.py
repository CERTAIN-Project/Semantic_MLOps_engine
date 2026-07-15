"""Append-only JSONL storage for mirrored MLflow events."""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

try:
    import fcntl
except ImportError:  # Windows fallback
    fcntl = None


EVENT_FILES = {
    "run": "runs.jsonl",
    "param": "params.jsonl",
    "metric": "metrics.jsonl",
    "tag": "tags.jsonl",
    "artifact": "artifacts.jsonl",
    "resource": "resources.jsonl",
    "input": "inputs.jsonl",
}


def timestamp_ms() -> int:
    """Return the current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


class MetadataWriter:
    """Write mirror events to append-only JSONL files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self.sequence_path = self.events_dir / ".event_sequence"
        self._thread_lock = threading.RLock()

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append an event and return the complete stored record."""
        if event_type not in EVENT_FILES:
            raise ValueError("Unsupported event type: {!r}".format(event_type))

        with self._thread_lock:
            self.events_dir.mkdir(parents=True, exist_ok=True)

            event = {
                "event_id": self._next_event_id(),
                "event_type": event_type,
                "recorded_at": timestamp_ms(),
            }
            event.update(payload)

            event_path = self.events_dir / EVENT_FILES[event_type]

            with event_path.open("a", encoding="utf-8") as handle:
                self._lock_file(handle)
                try:
                    handle.write(
                        json.dumps(
                            event,
                            default=str,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    self._unlock_file(handle)

            return event

    def _next_event_id(self) -> int:
        """Return the next mirror-wide event ID."""
        self.sequence_path.touch(exist_ok=True)

        with self.sequence_path.open("r+", encoding="utf-8") as handle:
            self._lock_file(handle)
            try:
                current = handle.read().strip()
                event_id = int(current or "0") + 1

                handle.seek(0)
                handle.write(str(event_id))
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())

                return event_id
            finally:
                self._unlock_file(handle)

    @staticmethod
    def _lock_file(handle: Any) -> None:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_file(handle: Any) -> None:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
