"""Touchless "click": the Euclidean distance between the thumb tip and the index fingertip.

The distance is divided by the palm size (wrist to middle knuckle), so the same threshold
works close to or far from the camera. Two thresholds give hysteresis: the fingers must come
closer than ``on`` to press and separate beyond ``off`` to release, so landmark jitter at the
boundary never produces a burst of clicks. A click is the press edge, with a short refractory
period.

Thresholds can be **learned from the user**: a few frames with the fingers apart and a few
pinching are enough to place both thresholds inside that person's own range, which matters
for patients with limited thumb opposition.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from spectra.detection.landmarks import HandLandmarks
from spectra.gestures.features import pinch_ratio

DEFAULT_ON = 0.35
DEFAULT_OFF = 0.45
REFRACTORY_SECONDS = 0.25
#: Fractions of the user's open-to-pinched range where the thresholds are placed.
ON_FRACTION = 0.35
OFF_FRACTION = 0.55
#: Below this separation the user cannot produce a distinct pinch; keep the defaults.
MIN_SPAN = 0.12


@dataclass(frozen=True)
class PinchThresholds:
    """``on < off``: lower ratios mean the fingertips are closer."""

    on: float = DEFAULT_ON
    off: float = DEFAULT_OFF
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.on < self.off:
            raise ValueError("pinch thresholds need 0 < on < off")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PinchThresholds:
        return cls(float(data["on"]), float(data["off"]), bool(data.get("calibrated", True)))


def calibrate_pinch(
    apart_samples: list[HandLandmarks], pinch_samples: list[HandLandmarks]
) -> PinchThresholds:
    """Place both thresholds inside the user's own open-to-pinched range."""
    if not apart_samples or not pinch_samples:
        return PinchThresholds()
    apart = statistics.median(pinch_ratio(sample) for sample in apart_samples)
    pinched = statistics.median(pinch_ratio(sample) for sample in pinch_samples)
    span = apart - pinched
    if span < MIN_SPAN:
        return PinchThresholds()
    return PinchThresholds(
        on=pinched + ON_FRACTION * span, off=pinched + OFF_FRACTION * span, calibrated=True
    )


class PinchClick:
    """Turns the thumb-index distance into press state and click events."""

    def __init__(self, thresholds: PinchThresholds | None = None) -> None:
        self.thresholds = thresholds or PinchThresholds()
        self.pressed = False
        self.ratio: float | None = None
        self._last_click: float | None = None
        self.clicks = 0

    def reset(self) -> None:
        self.pressed = False
        self.ratio = None

    def update(self, landmarks: HandLandmarks | None, at: float) -> bool:
        """Account for one frame; returns ``True`` on the frame a click happens."""
        if landmarks is None:
            self.reset()
            return False
        self.ratio = pinch_ratio(landmarks)
        if self.pressed:
            if self.ratio > self.thresholds.off:
                self.pressed = False
            return False
        if self.ratio > self.thresholds.on:
            return False
        self.pressed = True
        if self._last_click is not None and at - self._last_click < REFRACTORY_SECONDS:
            return False
        self._last_click = at
        self.clicks += 1
        return True
