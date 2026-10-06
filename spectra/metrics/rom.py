"""Range of motion.

ROM is reported per finger as the minimum, maximum and span of the **flexion angle**
(0 degrees = straight finger). The wrist block is a separate, explicitly flagged estimate:
MediaPipe Hands has no forearm landmarks, so there is no true wrist angle to measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spectra.gestures.features import FINGER_NAMES
from spectra.metrics.recorder import FrameSample


@dataclass(frozen=True)
class FingerRom:
    """Flexion range of one finger over a window, in degrees."""

    finger: str
    minimum: float
    maximum: float
    samples: int

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def to_dict(self) -> dict:
        return {
            "finger": self.finger,
            "min": self.minimum,
            "max": self.maximum,
            "span": self.span,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class WristEstimate:
    """Palm-derived wrist proxy. **Always an estimate**, never goniometry."""

    rotation_min: float
    rotation_max: float
    openness_min: float
    openness_max: float
    samples: int

    #: Rendered next to the value everywhere it appears.
    estimate = True

    @property
    def rotation_span(self) -> float:
        return self.rotation_max - self.rotation_min

    @property
    def openness_span(self) -> float:
        return self.openness_max - self.openness_min

    def to_dict(self) -> dict:
        return {
            "estimate": True,
            "rotation_min": self.rotation_min,
            "rotation_max": self.rotation_max,
            "rotation_span": self.rotation_span,
            "openness_min": self.openness_min,
            "openness_max": self.openness_max,
            "openness_span": self.openness_span,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class OppositionRom:
    """Closest thumb-to-fingertip distance reached, palm-normalised (lower is better)."""

    finger: str
    minimum: float
    samples: int

    def to_dict(self) -> dict:
        return {"finger": self.finger, "min_distance": self.minimum, "samples": self.samples}


def finger_rom(samples: Sequence[FrameSample]) -> tuple[FingerRom, ...]:
    """Per-finger flexion range over ``samples``, thumb to pinky."""
    if not samples:
        return tuple(FingerRom(name, 0.0, 0.0, 0) for name in FINGER_NAMES)
    return tuple(
        FingerRom(
            finger=name,
            minimum=min(sample.flexion[index] for sample in samples),
            maximum=max(sample.flexion[index] for sample in samples),
            samples=len(samples),
        )
        for index, name in enumerate(FINGER_NAMES)
    )


def opposition_rom(samples: Sequence[FrameSample]) -> tuple[OppositionRom, ...]:
    """Best thumb opposition reached against each finger."""
    targets = FINGER_NAMES[1:]
    if not samples:
        return tuple(OppositionRom(name, 0.0, 0) for name in targets)
    return tuple(
        OppositionRom(
            finger=name,
            minimum=min(sample.opposition[index] for sample in samples),
            samples=len(samples),
        )
        for index, name in enumerate(targets)
    )


def wrist_estimate(samples: Sequence[FrameSample]) -> WristEstimate:
    """Palm orientation range. Approximate: see ``docs/metrics.md``."""
    if not samples:
        return WristEstimate(0.0, 0.0, 0.0, 0.0, 0)
    rotations = [sample.palm_rotation for sample in samples]
    opennesses = [sample.palm_openness for sample in samples]
    return WristEstimate(
        rotation_min=min(rotations),
        rotation_max=max(rotations),
        openness_min=min(opennesses),
        openness_max=max(opennesses),
        samples=len(samples),
    )
