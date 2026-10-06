"""Movement smoothness and tremor.

Two complementary views of the fingertip trajectory:

* **Jerk** (the third derivative of position). A smooth, well-controlled movement has low
  jerk; hesitation and correction raise it. Reported normalised by movement duration and
  amplitude so it can be compared across sessions.
* **Spectral energy in the 4-12 Hz band**, the classic physiological/pathological tremor
  band. The trajectory is recorded at the full frame rate precisely so this band stays
  below Nyquist; at 30 fps the usable ceiling is 15 Hz.

Both are computed from landmark trajectories, never from video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spectra.metrics.recorder import TrajectoryPoint

TREMOR_BAND_HZ = (4.0, 12.0)

#: Below this many samples the spectrum is too short to mean anything.
MIN_SPECTRUM_SAMPLES = 32

#: Minimum sample rate for the tremor band to be below Nyquist.
MIN_RATE_FOR_TREMOR_HZ = 2 * TREMOR_BAND_HZ[1]


@dataclass(frozen=True)
class TremorMetrics:
    """Smoothness summary for one window of fingertip motion."""

    samples: int
    sample_rate: float
    mean_speed: float
    mean_jerk: float
    band_power_ratio: float
    dominant_frequency: float
    #: ``False`` when the recording was too short or too slow to resolve the band.
    band_measurable: bool

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "sample_rate": self.sample_rate,
            "mean_speed": self.mean_speed,
            "mean_jerk": self.mean_jerk,
            "band_power_ratio": self.band_power_ratio,
            "dominant_frequency": self.dominant_frequency,
            "band_measurable": self.band_measurable,
        }


def _empty() -> TremorMetrics:
    return TremorMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


def estimate_sample_rate(points: Sequence[TrajectoryPoint]) -> float:
    if len(points) < 2:
        return 0.0
    span = points[-1].at - points[0].at
    return (len(points) - 1) / span if span > 0 else 0.0


def tremor_metrics(points: Sequence[TrajectoryPoint]) -> TremorMetrics:
    """Speed, jerk and 4-12 Hz band power of a fingertip trajectory."""
    if len(points) < 4:
        return _empty()

    times = np.array([p.at for p in points], dtype=float)
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)

    rate = estimate_sample_rate(points)
    if rate <= 0 or times.size < 4:
        return _empty()

    dt = 1.0 / rate
    velocity = np.gradient(np.stack([xs, ys]), dt, axis=1)
    acceleration = np.gradient(velocity, dt, axis=1)
    jerk = np.gradient(acceleration, dt, axis=1)

    speed = np.linalg.norm(velocity, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=0)

    band_ratio, dominant, measurable = _band_power(xs, ys, rate)
    return TremorMetrics(
        samples=len(points),
        sample_rate=rate,
        mean_speed=float(speed.mean()),
        mean_jerk=float(jerk_magnitude.mean()),
        band_power_ratio=band_ratio,
        dominant_frequency=dominant,
        band_measurable=measurable,
    )


def _band_power(xs: np.ndarray, ys: np.ndarray, rate: float) -> tuple[float, float, bool]:
    """Fraction of spectral power inside the tremor band, and the dominant frequency."""
    if len(xs) < MIN_SPECTRUM_SAMPLES or rate < MIN_RATE_FOR_TREMOR_HZ:
        return 0.0, 0.0, False

    # Remove the voluntary movement: only the residual oscillation is tremor.
    signal = np.stack([xs - xs.mean(), ys - ys.mean()])
    window = np.hanning(signal.shape[1])
    spectrum = np.abs(np.fft.rfft(signal * window, axis=1)) ** 2
    power = spectrum.sum(axis=0)
    frequencies = np.fft.rfftfreq(signal.shape[1], d=1.0 / rate)

    # Drop the DC bin: it carries no oscillation information.
    power[0] = 0.0
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0, True

    low, high = TREMOR_BAND_HZ
    band = (frequencies >= low) & (frequencies <= high)
    return (
        float(power[band].sum() / total),
        float(frequencies[int(np.argmax(power))]),
        True,
    )
