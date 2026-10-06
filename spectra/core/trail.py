"""Fading pointer trail rendered behind the fingertip."""

from __future__ import annotations

import collections
import time

import cv2
import numpy as np

DEFAULT_LENGTH = 18
FADE_SECONDS = 0.45

Point = tuple[int, int]
ColorBGR = tuple[int, int, int]


class Trail:
    """A short, time-faded history of fingertip positions."""

    def __init__(self, max_length: int = DEFAULT_LENGTH, fade_seconds: float = FADE_SECONDS):
        self._points: collections.deque[tuple[Point, ColorBGR | None, float]] = collections.deque(
            maxlen=max_length
        )
        self._fade_seconds = fade_seconds

    def add(self, point: Point, color: ColorBGR | None, now: float | None = None) -> None:
        self._points.append((point, color, time.time() if now is None else now))

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    def draw(self, frame: np.ndarray, now: float | None = None) -> None:
        current = time.time() if now is None else now
        for point, color, created in list(self._points):
            age = current - created
            if age > self._fade_seconds:
                continue
            alpha = 1.0 - age / self._fade_seconds
            radius = max(2, int(7 * alpha))
            base = color if color else (180, 180, 180)
            faded = tuple(int(channel * alpha) for channel in base)
            overlay = frame.copy()
            cv2.circle(overlay, point, radius, faded, -1)
            weight = alpha * 0.65
            cv2.addWeighted(overlay, weight, frame, 1 - weight, 0, frame)
