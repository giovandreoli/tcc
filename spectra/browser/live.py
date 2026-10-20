"""In-process registry of running browser sessions, for the therapist's live monitor.

The panel and the patient page run in the same local Streamlit process, so the therapist
can follow progress without any network relay. Only counters are exposed: no frames.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LiveStatus:
    finished: bool
    saved: bool
    error_key: str | None
    exercise: str
    reps: int
    target: int
    total_reps: int
    frames: int
    hand_visible: bool
    elapsed_seconds: float
    limit_seconds: float
    exercise_index: int
    exercise_total: int

    @property
    def time_fraction(self) -> float:
        if self.limit_seconds <= 0:
            return 0.0
        return min(1.0, self.elapsed_seconds / self.limit_seconds)


class _Reporter(Protocol):
    def status(self) -> LiveStatus: ...


_lock = threading.Lock()
_sessions: weakref.WeakValueDictionary[str, _Reporter] = weakref.WeakValueDictionary()


def register(session_id: str, reporter: _Reporter) -> None:
    with _lock:
        _sessions[session_id] = reporter


def live_status(session_id: str) -> LiveStatus | None:
    with _lock:
        reporter = _sessions.get(session_id)
    return reporter.status() if reporter is not None else None
