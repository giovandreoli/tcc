"""Choosing a drawing colour by touching the thumb to a specific finger.

Each finger is bound to a colour (the binding is configurable). Touching the thumb to that
fingertip and holding briefly selects the colour; the thumb must move away before another
selection. The thumb-index contact is left out by default because it is the "click".

Contacts use the same rule as the exercises: palm-normalised distance below a threshold **and**
the target being the nearest fingertip, so a touch between two fingers selects only one.
"""

from __future__ import annotations

from collections.abc import Mapping

from spectra.detection.landmarks import HandLandmarks
from spectra.exercises.detectors import (
    CONTACT_THRESHOLDS,
    FINGER_ORDER,
    contact_ratio,
    in_contact,
)

DEFAULT_FINGER_COLORS: dict[str, str] = {
    "middle": "color.red",
    "ring": "color.green",
    "pinky": "color.blue",
}
HOLD_SECONDS = 0.3


class FingerColorSelector:
    """Emits the finger whose colour was chosen, once per touch."""

    def __init__(
        self,
        bindings: Mapping[str, str] | None = None,
        hold_seconds: float = HOLD_SECONDS,
        tolerance: str = "normal",
    ) -> None:
        self.bindings = dict(DEFAULT_FINGER_COLORS if bindings is None else bindings)
        unknown = set(self.bindings) - set(FINGER_ORDER)
        if unknown:
            raise ValueError(f"unknown fingers: {sorted(unknown)}")
        self.hold_seconds = hold_seconds
        self.on, self.off = CONTACT_THRESHOLDS[tolerance]
        self.candidate: str | None = None
        self.since: float | None = None
        self.latched: str | None = None

    def reset(self) -> None:
        self.candidate = None
        self.since = None
        self.latched = None

    def progress(self, at: float) -> float:
        if self.candidate is None or self.since is None or self.hold_seconds <= 0:
            return 0.0
        return min(1.0, (at - self.since) / self.hold_seconds)

    def update(self, landmarks: HandLandmarks | None, at: float) -> str | None:
        """Returns the selected finger on the frame its colour is chosen."""
        if landmarks is None:
            self.reset()
            return None
        if self.latched is not None:
            if contact_ratio(landmarks, self.latched) > self.off:
                self.latched = None
            return None
        touching = next(
            (finger for finger in self.bindings if in_contact(landmarks, finger, self.on)), None
        )
        if touching != self.candidate:
            self.candidate = touching
            self.since = at if touching else None
            return None
        if touching is None or self.since is None:
            return None
        if at - self.since + 1e-9 >= self.hold_seconds:
            self.latched = touching
            self.candidate = None
            self.since = None
            return touching
        return None
