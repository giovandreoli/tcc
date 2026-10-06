"""Derived-feature recorder.

**No video or image of the patient is ever stored.** Each frame is reduced to a handful of
numbers (joint angles, opposition distances, palm orientation) and those are kept at a
reduced rate, 10 Hz by default.

The fingertip **trajectory is kept at the full frame rate** in a separate, much lighter
buffer (three floats per frame). Tremor lives in the 4-12 Hz band, and by Nyquist a 10 Hz
recording could not see it at all; 30 fps gives a 15 Hz ceiling, which covers the band.

Tracking quality is counted over *every* frame, not only stored ones, so the report can tell
a therapist that a session was too poorly tracked to be trusted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from spectra.detection.landmarks import INDEX_TIP, HandLandmarks
from spectra.gestures.features import (
    finger_flexion_angles,
    opposition_ratios,
    palm_openness,
    palm_rotation,
    palm_size,
    thumb_abduction_ratio,
)
from spectra.metrics.tracking_quality import (
    MIN_RELIABLE_CONFIDENCE,
    MIN_RELIABLE_PRESENCE,
    TrackingQuality,
)

DEFAULT_RATE_HZ = 10.0

__all__ = [
    "DEFAULT_RATE_HZ",
    "MIN_RELIABLE_CONFIDENCE",
    "MIN_RELIABLE_PRESENCE",
    "FrameSample",
    "SessionRecorder",
    "TrackingQuality",
    "TrajectoryPoint",
]


@dataclass(frozen=True)
class FrameSample:
    """One stored frame, already reduced to derived features."""

    at: float
    flexion: tuple[float, float, float, float, float]
    opposition: tuple[float, float, float, float]
    thumb_abduction: float
    index_tip: tuple[float, float]
    palm_size: float
    #: Wrist proxies. Both are **estimates**; see ``docs/metrics.md``.
    palm_rotation: float = 0.0
    palm_openness: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "at": self.at,
            "flexion": list(self.flexion),
            "opposition": list(self.opposition),
            "thumb_abduction": self.thumb_abduction,
            "index_tip": list(self.index_tip),
            "palm_size": self.palm_size,
            "palm_rotation": self.palm_rotation,
            "palm_openness": self.palm_openness,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TrajectoryPoint:
    """One full-rate fingertip position, for the tremor analysis."""

    at: float
    x: float
    y: float


@dataclass
class SessionRecorder:
    """Collects :class:`FrameSample` at a fixed rate plus a full-rate trajectory."""

    rate_hz: float = DEFAULT_RATE_HZ
    samples: list[FrameSample] = field(default_factory=list)
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    frames_total: int = 0
    frames_with_hand: int = 0
    _confidence_sum: float = 0.0
    _last_stored_at: float | None = None

    @property
    def interval(self) -> float:
        return 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0

    def observe(
        self,
        landmarks: HandLandmarks | None,
        confidence: float = 0.0,
        now: float | None = None,
    ) -> FrameSample | None:
        """Account for one frame; returns the stored sample, if this frame was kept."""
        current = time.time() if now is None else now
        self.frames_total += 1
        if landmarks is None:
            return None
        self.frames_with_hand += 1
        self._confidence_sum += confidence

        tip = landmarks[INDEX_TIP]
        self.trajectory.append(TrajectoryPoint(current, tip.x, tip.y))

        if self._last_stored_at is not None and current - self._last_stored_at < self.interval:
            return None
        self._last_stored_at = current
        sample = FrameSample(
            at=current,
            flexion=finger_flexion_angles(landmarks),  # type: ignore[arg-type]
            opposition=opposition_ratios(landmarks),  # type: ignore[arg-type]
            thumb_abduction=thumb_abduction_ratio(landmarks),
            index_tip=(tip.x, tip.y),
            palm_size=palm_size(landmarks),
            palm_rotation=palm_rotation(landmarks),
            palm_openness=palm_openness(landmarks),
            confidence=confidence,
        )
        self.samples.append(sample)
        return sample

    def quality(self) -> TrackingQuality:
        mean_confidence = (
            self._confidence_sum / self.frames_with_hand if self.frames_with_hand else 0.0
        )
        return TrackingQuality(
            frames_total=self.frames_total,
            frames_with_hand=self.frames_with_hand,
            mean_confidence=mean_confidence,
        )

    def reset(self) -> None:
        self.samples.clear()
        self.trajectory.clear()
        self.frames_total = 0
        self.frames_with_hand = 0
        self._confidence_sum = 0.0
        self._last_stored_at = None

    def window(self, start: float, end: float) -> list[FrameSample]:
        """Samples recorded within ``[start, end]``, for per-exercise summaries."""
        return [sample for sample in self.samples if start <= sample.at <= end]

    def trajectory_window(self, start: float, end: float) -> list[TrajectoryPoint]:
        return [point for point in self.trajectory if start <= point.at <= end]
