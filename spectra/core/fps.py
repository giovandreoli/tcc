"""Frames-per-second counter for the HUD."""

from __future__ import annotations

import time


class FpsCounter:
    """Rolling FPS estimate, refreshed once per second."""

    def __init__(self, window_seconds: float = 1.0) -> None:
        self.window_seconds = window_seconds
        self.value = 0.0
        self._frames = 0
        self._started_at: float | None = None

    def tick(self, now: float | None = None) -> float:
        current = time.time() if now is None else now
        if self._started_at is None:
            self._started_at = current
        self._frames += 1
        elapsed = current - self._started_at
        if elapsed >= self.window_seconds:
            self.value = self._frames / elapsed
            self._frames = 0
            self._started_at = current
        return self.value
