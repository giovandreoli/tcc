"""Fatigue: decline of movement amplitude and pace over a session.

Fatigue is measured as a *relative* decline, comparing the first third of a window with
the last third. Relative numbers survive the fact that two patients have very different
absolute ranges, and thirds (rather than first-versus-last repetition) average out the
noise of any single movement.

A positive decline means the patient got worse over the window; a negative one means they
warmed up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spectra.gestures.features import FINGER_NAMES
from spectra.metrics.recorder import FrameSample

#: Below this many samples (or repetitions) the thirds are too small to compare.
MIN_SAMPLES = 6
MIN_REPS = 6


@dataclass(frozen=True)
class FatigueMetrics:
    """Relative decline in amplitude and pace across a window."""

    amplitude_start: float
    amplitude_end: float
    amplitude_decline: float
    pace_start: float
    pace_end: float
    pace_decline: float
    measurable: bool

    def to_dict(self) -> dict:
        return {
            "amplitude_start": self.amplitude_start,
            "amplitude_end": self.amplitude_end,
            "amplitude_decline": self.amplitude_decline,
            "pace_start": self.pace_start,
            "pace_end": self.pace_end,
            "pace_decline": self.pace_decline,
            "measurable": self.measurable,
        }


def _thirds(items: Sequence) -> tuple[Sequence, Sequence]:
    third = max(1, len(items) // 3)
    return items[:third], items[-third:]


def _mean_amplitude(samples: Sequence[FrameSample]) -> float:
    """Mean per-finger flexion span across the four fingers (the thumb is noisier)."""
    if not samples:
        return 0.0
    spans = []
    for index in range(1, len(FINGER_NAMES)):
        values = [sample.flexion[index] for sample in samples]
        spans.append(max(values) - min(values))
    return sum(spans) / len(spans)


def _relative_decline(start: float, end: float) -> float:
    """Symmetric relative change in ``[-1, 1]``; positive means the patient declined.

    Dividing by ``max(start, end)`` rather than by ``start`` keeps the value bounded and
    stays defined when the patient started from no measurable movement at all.
    """
    scale = max(start, end)
    if scale <= 0:
        return 0.0
    return (start - end) / scale


def amplitude_fatigue(samples: Sequence[FrameSample]) -> tuple[float, float, float, bool]:
    """``(start, end, decline, measurable)`` for movement amplitude."""
    if len(samples) < MIN_SAMPLES:
        return 0.0, 0.0, 0.0, False
    first, last = _thirds(samples)
    start, end = _mean_amplitude(first), _mean_amplitude(last)
    return start, end, _relative_decline(start, end), True


def pace_fatigue(rep_times: Sequence[float]) -> tuple[float, float, float, bool]:
    """``(start, end, decline, measurable)`` for repetitions per minute."""
    if len(rep_times) < MIN_REPS:
        return 0.0, 0.0, 0.0, False
    ordered = sorted(rep_times)
    first, last = _thirds(ordered)
    start, end = _rate_per_minute(first), _rate_per_minute(last)
    return start, end, _relative_decline(start, end), True


def _rate_per_minute(times: Sequence[float]) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span * 60.0 if span > 0 else 0.0


def fatigue_metrics(
    samples: Sequence[FrameSample], rep_times: Sequence[float] = ()
) -> FatigueMetrics:
    """Combine amplitude and pace decline into one summary."""
    a_start, a_end, a_decline, a_ok = amplitude_fatigue(samples)
    p_start, p_end, p_decline, p_ok = pace_fatigue(rep_times)
    return FatigueMetrics(
        amplitude_start=a_start,
        amplitude_end=a_end,
        amplitude_decline=a_decline,
        pace_start=p_start,
        pace_end=p_end,
        pace_decline=p_decline,
        measurable=a_ok or p_ok,
    )
