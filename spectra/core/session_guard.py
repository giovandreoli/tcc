"""Session safety: duration limit and periodic rest prompts.

Rehabilitation sessions must not run unbounded, and patients need breaks. The guard is
pure time arithmetic so it can be unit tested without a clock or a camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class SafetyPrompt(Enum):
    """Values double as i18n keys under ``safety.``."""

    NONE = "none"
    REST = "rest"
    TIME_LIMIT = "time_limit"


@dataclass
class SessionGuard:
    """Tracks elapsed session time, emitting a rest prompt and a hard stop."""

    limit_minutes: int = 20
    rest_every_minutes: int = 5
    started_at: float | None = None
    _rests_emitted: int = 0
    _limit_emitted: bool = False

    def start(self, now: float | None = None) -> None:
        self.started_at = time.time() if now is None else now
        self._rests_emitted = 0
        self._limit_emitted = False

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        return (time.time() if now is None else now) - self.started_at

    @property
    def limit_seconds(self) -> float:
        return self.limit_minutes * 60.0

    def remaining(self, now: float | None = None) -> float:
        return max(0.0, self.limit_seconds - self.elapsed(now))

    def check(self, now: float | None = None) -> SafetyPrompt:
        """Return a prompt at most once per occurrence, so the UI does not nag."""
        if self.started_at is None:
            return SafetyPrompt.NONE
        elapsed = self.elapsed(now)
        if elapsed >= self.limit_seconds:
            if self._limit_emitted:
                return SafetyPrompt.NONE
            self._limit_emitted = True
            return SafetyPrompt.TIME_LIMIT
        if self.rest_every_minutes <= 0:
            return SafetyPrompt.NONE
        due = int(elapsed // (self.rest_every_minutes * 60.0))
        if due > self._rests_emitted:
            self._rests_emitted = due
            return SafetyPrompt.REST
        return SafetyPrompt.NONE

    @property
    def expired(self) -> bool:
        return self._limit_emitted
