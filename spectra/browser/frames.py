"""Parsing of landmark batches sent by the patient's browser.

The browser runs the hand detector locally and sends only 21 normalised points per frame,
never pixels. Because the page can be reached from the internet, every batch is treated
as untrusted input: sizes, types and ranges are checked before anything reaches the
exercise engine, and a malformed batch is rejected as a whole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from spectra.detection.hand_detector import DetectedHand
from spectra.detection.landmarks import LANDMARK_COUNT, Landmark

MAX_FRAMES_PER_BATCH = 120
HAND_LABELS = ("Left", "Right")
#: Normalised coordinates may fall slightly outside the image when a hand is cropped.
COORDINATE_RANGE = (-1.0, 2.0)


class BatchError(ValueError):
    """A batch that does not follow the expected shape. Carries no payload data."""


@dataclass(frozen=True)
class BrowserFrame:
    """One analysed frame: the browser's timestamp and the hand found, if any."""

    at_ms: float
    hand: DetectedHand | None


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchError("expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise BatchError("expected a finite number")
    return number


def _frame(entry: Any) -> BrowserFrame:
    if not isinstance(entry, list) or len(entry) != 4:
        raise BatchError("malformed frame")
    at_ms = _number(entry[0])
    if at_ms < 0:
        raise BatchError("negative timestamp")
    label, score, coordinates = entry[1], entry[2], entry[3]
    if coordinates is None:
        return BrowserFrame(at_ms, None)
    if label not in HAND_LABELS:
        raise BatchError("unknown hand label")
    confidence = _number(score)
    if not 0.0 <= confidence <= 1.0:
        raise BatchError("score out of range")
    if not isinstance(coordinates, list) or len(coordinates) != LANDMARK_COUNT * 3:
        raise BatchError("wrong number of coordinates")
    values = [_number(value) for value in coordinates]
    low, high = COORDINATE_RANGE
    if any(not low <= value <= high for value in values):
        raise BatchError("coordinate out of range")
    landmarks = tuple(
        Landmark(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)
    )
    return BrowserFrame(at_ms, DetectedHand(landmarks=landmarks, label=label, score=confidence))


def parse_batch(frames: Any) -> list[BrowserFrame]:
    """Validate a batch and return its frames in timestamp order."""
    if not isinstance(frames, list) or len(frames) > MAX_FRAMES_PER_BATCH:
        raise BatchError("malformed batch")
    parsed = [_frame(entry) for entry in frames]
    return sorted(parsed, key=lambda frame: frame.at_ms)
