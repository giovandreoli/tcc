"""Small reusable HUD widgets."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from spectra.detection.landmarks import HAND_CONNECTIONS, HandLandmarks
from spectra.i18n import t
from spectra.ui.text import draw_text

ColorBGR = tuple[int, int, int]

FINGER_SHORT_KEYS = (
    "finger.short.thumb",
    "finger.short.index",
    "finger.short.middle",
    "finger.short.ring",
    "finger.short.pinky",
)


def draw_status_bar(frame: np.ndarray, title: str, message: str | None = None) -> None:
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 44), (18, 18, 25), -1)
    draw_text(frame, title, (14, 28), 0.78, (240, 240, 240), 2)
    if message:
        draw_text(frame, message, (14, 44), 0.45, (185, 185, 185))


def draw_finger_hud(frame: np.ndarray, states: Sequence[bool], x0: int, y0: int) -> None:
    """Five dots showing which fingers the detector currently sees extended."""
    for i, (key, extended) in enumerate(zip(FINGER_SHORT_KEYS, states, strict=False)):
        cx = x0 + i * 28
        fill = (0, 200, 80) if extended else (75, 75, 75)
        cv2.circle(frame, (cx, y0), 11, fill, -1)
        cv2.circle(frame, (cx, y0), 11, (190, 190, 190), 1)
        draw_text(frame, t(key), (cx - 6, y0 + 4), 0.33)


def draw_hand_landmarks(
    frame: np.ndarray, landmarks: HandLandmarks, color: ColorBGR = (0, 220, 255)
) -> None:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, color, -1)


def draw_progress_bar(
    frame: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    progress: float,
    fill: ColorBGR = (0, 220, 255),
    background: ColorBGR = (50, 50, 50),
) -> None:
    x, y = top_left
    width, height = size
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + width, y + height), background, -1)
    cv2.rectangle(frame, (x, y), (x + int(width * progress), y + height), fill, -1)


def dim_frame(frame: np.ndarray, color: ColorBGR, strength: float) -> None:
    """Blend a flat colour over the whole frame, in place."""
    overlay = np.empty_like(frame)
    overlay[:] = color
    cv2.addWeighted(overlay, strength, frame, 1.0 - strength, 0, frame)
