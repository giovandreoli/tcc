"""Shared fixtures: synthetic hands so tests never need a camera or MediaPipe."""

from __future__ import annotations

import math
import time

import pytest

from spectra.config import AppConfig
from spectra.detection.hand_detector import DetectedHand, DetectionResult
from spectra.detection.landmarks import LANDMARK_COUNT, Landmark
from spectra.i18n import set_locale

# Horizontal position of each finger's MCP joint, thumb excluded.
_FINGER_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}

# (dx, y) offsets for MCP, PIP, DIP and TIP relative to the finger's MCP column.
_EXTENDED = ((0.00, 0.65), (0.00, 0.55), (0.00, 0.48), (0.00, 0.42))
_CURLED = ((0.00, 0.65), (0.00, 0.57), (-0.03, 0.59), (-0.01, 0.64))

_THUMB_EXTENDED = ((0.42, 0.82), (0.36, 0.76), (0.30, 0.72), (0.26, 0.70))
_THUMB_CURLED = ((0.42, 0.82), (0.40, 0.76), (0.44, 0.72), (0.47, 0.70))


def make_hand(
    thumb: bool = False,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    label: str = "Right",
) -> list[Landmark]:
    """Build 21 anatomically plausible landmarks for the requested finger pattern."""
    points: list[tuple[float, float]] = [(0.50, 0.90)]  # wrist
    points.extend(_THUMB_EXTENDED if thumb else _THUMB_CURLED)
    for name, extended in (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)):
        x = _FINGER_X[name]
        joints = _EXTENDED if extended else _CURLED
        points.extend((x + dx, y) for dx, y in joints)

    if label == "Left":
        points = [(1.0 - x, y) for x, y in points]

    landmarks = [Landmark(x, y, 0.0) for x, y in points]
    assert len(landmarks) == LANDMARK_COUNT
    return landmarks


def make_detection(
    *patterns: tuple[bool, bool, bool, bool, bool], label: str = "Right"
) -> DetectionResult:
    """Wrap one or more finger patterns into a :class:`DetectionResult`."""
    hands = tuple(
        DetectedHand(landmarks=make_hand(*pattern, label=label), label=label, score=0.9)
        for pattern in patterns
    )
    return DetectionResult(hands)


def pinching_hand(distance: float = 0.01, label: str = "Right") -> list[Landmark]:
    """A hand whose thumb tip sits ``distance`` (normalised) from the index tip."""
    landmarks = make_hand(thumb=True, index=True, label=label)
    index_tip = landmarks[8]
    landmarks[4] = Landmark(index_tip.x + distance, index_tip.y, 0.0)
    return landmarks


def rotate_hand(
    landmarks: list[Landmark], degrees: float, pivot: tuple[float, float] = (0.5, 0.5)
) -> list[Landmark]:
    """Rotate a hand in the image plane, to prove angle features are orientation free."""
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    px, py = pivot
    rotated = []
    for point in landmarks:
        dx, dy = point.x - px, point.y - py
        rotated.append(Landmark(px + dx * cos - dy * sin, py + dx * sin + dy * cos, point.z))
    return rotated


@pytest.fixture(autouse=True)
def _default_locale():
    set_locale("pt_BR")


@pytest.fixture
def config(tmp_path) -> AppConfig:
    cfg = AppConfig(data_dir=tmp_path / "Spectra")
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def empty_detection() -> DetectionResult:
    return DetectionResult()


class FakeClock:
    """A controllable stand-in for ``time.time``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    """Freeze ``time.time`` so hold times and dwell timers are deterministic."""
    fake = FakeClock()
    monkeypatch.setattr(time, "time", fake)
    return fake
