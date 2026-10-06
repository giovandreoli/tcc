"""Per-patient calibration and the hysteretic finger-state estimator.

Each finger has its own *signal*: an interphalangeal angle in degrees for index to
pinky, and the palm-normalised abduction ratio for the thumb. In both cases a higher
value means "more extended", so one threshold type covers all five.

Hysteresis uses two thresholds: a finger must cross ``on`` to be considered extended
and fall back below ``off`` to be released. The gap stops the flag from flickering
when the patient holds a borderline posture.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import asdict, dataclass, field

from spectra.detection.landmarks import HandLandmarks
from spectra.gestures.features import (
    FingerStates,
    finger_extension_angles,
    thumb_abduction_ratio,
)

#: Minimum open/closed separation for a calibrated finger to be trusted.
MIN_SEPARATION = {"thumb": 0.12, "finger": 25.0}

#: Fraction of the open/closed range used for the two thresholds.
ON_FRACTION = 0.60
OFF_FRACTION = 0.40


@dataclass(frozen=True)
class Hysteresis:
    """A pair of thresholds; ``on`` must be greater than ``off``."""

    on: float
    off: float

    def __post_init__(self) -> None:
        if self.on <= self.off:
            raise ValueError(f"hysteresis needs on > off, got on={self.on} off={self.off}")

    def apply(self, value: float, previously_on: bool) -> bool:
        return value > self.off if previously_on else value >= self.on


DEFAULT_THUMB = Hysteresis(on=0.45, off=0.35)
DEFAULT_FINGER = Hysteresis(on=150.0, off=120.0)


@dataclass
class CalibrationProfile:
    """Per-finger thresholds, in thumb-to-pinky order."""

    thresholds: tuple[Hysteresis, ...] = field(
        default_factory=lambda: (DEFAULT_THUMB,) + (DEFAULT_FINGER,) * 4
    )
    #: Fingers that fell back to defaults because open and closed looked alike.
    uncalibrated: tuple[str, ...] = ()
    samples_used: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self.samples_used > 0

    def to_dict(self) -> dict:
        return {
            "thresholds": [asdict(h) for h in self.thresholds],
            "uncalibrated": list(self.uncalibrated),
            "samples_used": self.samples_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationProfile:
        thresholds = tuple(Hysteresis(**entry) for entry in data["thresholds"])
        return cls(
            thresholds=thresholds,
            uncalibrated=tuple(data.get("uncalibrated", ())),
            samples_used=int(data.get("samples_used", 0)),
        )


def finger_signals(landmarks: HandLandmarks) -> tuple[float, ...]:
    """The five comparable extension signals: thumb ratio then four joint angles."""
    angles = finger_extension_angles(landmarks)
    return (thumb_abduction_ratio(landmarks), *angles[1:])


def build_profile(
    open_samples: list[HandLandmarks], closed_samples: list[HandLandmarks]
) -> CalibrationProfile:
    """Derive thresholds from an open-hand and a closed-hand recording."""
    if not open_samples or not closed_samples:
        return CalibrationProfile()

    open_signals = [finger_signals(sample) for sample in open_samples]
    closed_signals = [finger_signals(sample) for sample in closed_samples]
    names = ("thumb", "index", "middle", "ring", "pinky")

    thresholds: list[Hysteresis] = []
    uncalibrated: list[str] = []
    for i, name in enumerate(names):
        open_value = statistics.median(signal[i] for signal in open_signals)
        closed_value = statistics.median(signal[i] for signal in closed_signals)
        span = open_value - closed_value
        minimum = MIN_SEPARATION["thumb"] if i == 0 else MIN_SEPARATION["finger"]
        if span < minimum:
            uncalibrated.append(name)
            thresholds.append(DEFAULT_THUMB if i == 0 else DEFAULT_FINGER)
            continue
        thresholds.append(
            Hysteresis(
                on=closed_value + ON_FRACTION * span,
                off=closed_value + OFF_FRACTION * span,
            )
        )
    return CalibrationProfile(
        thresholds=tuple(thresholds),
        uncalibrated=tuple(uncalibrated),
        samples_used=len(open_samples) + len(closed_samples),
    )


class FingerStateEstimator:
    """Turns landmarks into stable finger flags using hysteresis plus a majority vote."""

    def __init__(self, profile: CalibrationProfile | None = None, smoothing_frames: int = 3):
        self.profile = profile or CalibrationProfile()
        self.smoothing_frames = max(1, smoothing_frames)
        self._states = FingerStates(False, False, False, False, False)
        self._history: deque[tuple[bool, ...]] = deque(maxlen=self.smoothing_frames)

    @property
    def states(self) -> FingerStates:
        return self._states

    def reset(self) -> None:
        self._states = FingerStates(False, False, False, False, False)
        self._history.clear()

    def update(self, landmarks: HandLandmarks) -> FingerStates:
        signals = finger_signals(landmarks)
        raw = tuple(
            threshold.apply(value, previous)
            for threshold, value, previous in zip(
                self.profile.thresholds, signals, self._states, strict=True
            )
        )
        self._history.append(raw)
        votes = len(self._history)
        majority = tuple(sum(frame[i] for frame in self._history) * 2 > votes for i in range(5))
        self._states = FingerStates(*majority)
        return self._states


def finger_states(
    landmarks: HandLandmarks,
    hand_label: str = "Right",
    profile: CalibrationProfile | None = None,
) -> FingerStates:
    """Stateless finger reading, for callers that do not keep an estimator around."""
    thresholds = (profile or CalibrationProfile()).thresholds
    signals = finger_signals(landmarks)
    return FingerStates(
        *(
            threshold.apply(value, previously_on=False)
            for threshold, value in zip(thresholds, signals, strict=True)
        )
    )
