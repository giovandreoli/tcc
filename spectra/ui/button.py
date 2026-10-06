"""Dwell-activated button: hovering the index fingertip for a while selects it."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from spectra.ui.text import draw_text, text_size

DEFAULT_HOVER_SECONDS = 1.2
HIT_MARGIN = 12

ColorBGR = tuple[int, int, int]
Point = tuple[int, int]


class Button:
    """A rectangular button selected by dwelling on it with the pointer."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        label: str,
        background: ColorBGR = (50, 50, 50),
        foreground: ColorBGR = (255, 255, 255),
        value: Any = None,
        hover_seconds: float = DEFAULT_HOVER_SECONDS,
    ) -> None:
        self.rect = (x, y, width, height)
        self.label = label
        self.background = background
        self.foreground = foreground
        self.value = value
        self.hover_seconds = hover_seconds
        self.hover_start: float | None = None
        self.progress = 0.0
        self.hovering = False

    def contains(self, point: Point | None) -> bool:
        if point is None:
            return False
        x, y, width, height = self.rect
        return (
            x - HIT_MARGIN <= point[0] <= x + width + HIT_MARGIN
            and y - HIT_MARGIN <= point[1] <= y + height + HIT_MARGIN
        )

    def reset(self) -> None:
        self.hover_start = None
        self.progress = 0.0
        self.hovering = False

    def update_hover(self, pointing: bool, now: float | None = None) -> bool:
        """Advance the dwell timer; returns ``True`` on the frame it is selected."""
        if not pointing:
            self.reset()
            return False
        current = time.time() if now is None else now
        if self.hover_start is None:
            self.hover_start = current
        self.hovering = True
        elapsed = current - self.hover_start
        self.progress = min(1.0, elapsed / self.hover_seconds)
        if elapsed >= self.hover_seconds:
            self.reset()
            return True
        return False

    def draw(self, frame: np.ndarray) -> None:
        import cv2

        x, y, width, height = self.rect
        fill = self.background
        if self.hovering:
            fill = tuple(min(255, channel + 40) for channel in self.background)
        cv2.rectangle(frame, (x, y), (x + width, y + height), fill, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 2)
        if self.progress > 0:
            cv2.rectangle(
                frame,
                (x, y + height - 6),
                (x + int(width * self.progress), y + height),
                (0, 220, 255),
                -1,
            )
        scale = 0.52
        label_width, label_height = text_size(self.label, scale)
        draw_text(
            frame,
            self.label,
            (x + (width - label_width) // 2, y + (height + label_height) // 2 - 3),
            scale,
            self.foreground,
        )
