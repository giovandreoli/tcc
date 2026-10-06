"""Paintable canvas with a bounded undo history."""

from __future__ import annotations

import collections
from pathlib import Path

import cv2
import numpy as np

ERASER_COLOR_BGR = (255, 255, 255)
ERASER_SIZE = 50
DEFAULT_BRUSH_SIZE = 10
DEFAULT_UNDO_HISTORY = 15

Point = tuple[int, int]
ColorBGR = tuple[int, int, int]


class DrawingCanvas:
    """A white BGR surface the patient paints on, with undo support."""

    def __init__(self, width: int, height: int, undo_history: int = DEFAULT_UNDO_HISTORY) -> None:
        self.width = width
        self.height = height
        self.image = np.full((height, width, 3), 255, dtype=np.uint8)
        self._history: collections.deque[np.ndarray] = collections.deque(maxlen=undo_history)

    def begin_stroke(self) -> None:
        """Snapshot the current surface so the next stroke can be undone."""
        self._history.append(self.image.copy())

    def undo(self) -> bool:
        """Restore the previous snapshot; ``False`` when the history is empty."""
        if not self._history:
            return False
        self.image = self._history.pop()
        return True

    def draw(
        self,
        start: Point,
        end: Point,
        color_bgr: ColorBGR | None,
        size: int = DEFAULT_BRUSH_SIZE,
    ) -> None:
        """Draw a segment; ``color_bgr=None`` means erase."""
        color = ERASER_COLOR_BGR if color_bgr is None else color_bgr
        width = ERASER_SIZE if color_bgr is None else size
        cv2.line(self.image, start, end, color, width)
        cv2.circle(self.image, end, max(1, width // 2), color, -1)

    def clear(self) -> None:
        self._history.append(self.image.copy())
        self.image[:] = 255

    def save(self, path: Path) -> Path:
        """Write the canvas as a PNG, creating the parent directory if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.image):
            raise OSError(f"could not write image to {path}")
        return path

    @property
    def can_undo(self) -> bool:
        return bool(self._history)
