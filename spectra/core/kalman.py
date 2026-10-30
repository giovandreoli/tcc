"""Kalman filtering of the pointer trajectory.

Camera-based hand tracking jitters by a few pixels from frame to frame even when the hand is
still. The exponential moving average used originally removes jitter only by lagging behind
real motion. A Kalman filter with a **constant-velocity model** predicts where the fingertip
should be from its estimated velocity and corrects that prediction with each new
measurement, weighting both by their uncertainty. The result follows fast strokes with
less lag and holds still points steadier.

The two image axes are independent (diagonal measurement noise, separable process noise),
so the filter runs as two 2-state filters in plain Python: no NumPy and no matrix library.

State per axis: ``[position, velocity]``. With frame interval ``dt``::

    F = [[1, dt], [0, 1]]                       state transition
    Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]  white-noise acceleration, q = sigma_a^2
    H = [1, 0]                                  only the position is measured
    R = sigma_m^2                               measurement noise

``dt`` comes from the frame timestamps, so a dropped frame is handled correctly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

Point = tuple[int, int]

#: Measurement noise (pixels, one standard deviation). Chosen with the evaluation sweep in
#: docs/learning.md: with 900 px/s² it keeps jitter near the exponential average while
#: cutting its lag-induced error roughly threefold.
DEFAULT_MEASUREMENT_STD = 8.0
#: Process noise: standard deviation of the hand's acceleration, in pixels per second².
DEFAULT_ACCELERATION_STD = 900.0
DEFAULT_DT = 1.0 / 30.0
#: A gap longer than this restarts the filter instead of extrapolating across it.
MAX_GAP_SECONDS = 0.5


@dataclass
class AxisKalman:
    """One-dimensional constant-velocity Kalman filter."""

    measurement_var: float
    acceleration_var: float
    position: float = 0.0
    velocity: float = 0.0
    p00: float = 0.0
    p01: float = 0.0
    p11: float = 0.0
    initialized: bool = False

    def reset(self, position: float) -> None:
        self.position = position
        self.velocity = 0.0
        # Unknown velocity: large initial uncertainty, so the first frames trust measurements.
        self.p00 = self.measurement_var
        self.p01 = 0.0
        self.p11 = 1e6
        self.initialized = True

    def predict(self, dt: float) -> None:
        q = self.acceleration_var
        self.position += self.velocity * dt
        p00 = self.p00 + dt * (2 * self.p01 + dt * self.p11) + q * dt**4 / 4
        p01 = self.p01 + dt * self.p11 + q * dt**3 / 2
        p11 = self.p11 + q * dt**2
        self.p00, self.p01, self.p11 = p00, p01, p11

    def correct(self, measurement: float) -> None:
        innovation = measurement - self.position
        s = self.p00 + self.measurement_var
        k0, k1 = self.p00 / s, self.p01 / s
        self.position += k0 * innovation
        self.velocity += k1 * innovation
        p00 = (1 - k0) * self.p00
        p01 = (1 - k0) * self.p01
        p11 = self.p11 - k1 * self.p01
        self.p00, self.p01, self.p11 = p00, p01, p11

    @property
    def gain(self) -> float:
        """Current position gain: 0 trusts the model, 1 trusts the measurement."""
        return self.p00 / (self.p00 + self.measurement_var)


class KalmanPointFilter:
    """Smooths a 2-D point stream; drop-in replacement for the exponential smoother."""

    def __init__(
        self,
        measurement_std: float = DEFAULT_MEASUREMENT_STD,
        acceleration_std: float = DEFAULT_ACCELERATION_STD,
    ) -> None:
        if measurement_std <= 0 or acceleration_std <= 0:
            raise ValueError("noise parameters must be positive")
        self.measurement_std = measurement_std
        self.acceleration_std = acceleration_std
        self._axes = self._new_axes()
        self._last_at: float | None = None
        self.value: Point | None = None

    def _new_axes(self) -> tuple[AxisKalman, AxisKalman]:
        args = (self.measurement_std**2, self.acceleration_std**2)
        return AxisKalman(*args), AxisKalman(*args)

    def reset(self) -> None:
        self._axes = self._new_axes()
        self._last_at = None
        self.value = None

    def update_float(
        self, point: tuple[float, float] | None, now: float | None = None
    ) -> tuple[float, float] | None:
        """Filter one measurement; ``None`` (no pointer) resets the filter."""
        if point is None:
            self.reset()
            return None
        current = time.time() if now is None else now
        x_axis, y_axis = self._axes
        gap = None if self._last_at is None else current - self._last_at
        if gap is None or not x_axis.initialized or gap > MAX_GAP_SECONDS:
            x_axis.reset(point[0])
            y_axis.reset(point[1])
        else:
            dt = gap if gap > 0 else DEFAULT_DT
            for axis, measurement in ((x_axis, point[0]), (y_axis, point[1])):
                axis.predict(dt)
                axis.correct(measurement)
        self._last_at = current
        return (x_axis.position, y_axis.position)

    def update(self, point: Point | None, now: float | None = None) -> Point | None:
        filtered = self.update_float(point, now)
        self.value = None if filtered is None else (round(filtered[0]), round(filtered[1]))
        return self.value

    @property
    def velocity(self) -> tuple[float, float]:
        return (self._axes[0].velocity, self._axes[1].velocity)


class ExponentialPointFilter:
    """The original exponential moving average, with the same interface as the Kalman filter."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: tuple[float, float] | None = None
        self.value: Point | None = None

    def reset(self) -> None:
        self._value = None
        self.value = None

    def update_float(
        self, point: tuple[float, float] | None, now: float | None = None
    ) -> tuple[float, float] | None:
        if point is None:
            self.reset()
            return None
        if self._value is None:
            self._value = (float(point[0]), float(point[1]))
        else:
            a = self.alpha
            self._value = (
                (1 - a) * self._value[0] + a * point[0],
                (1 - a) * self._value[1] + a * point[1],
            )
        return self._value

    def update(self, point: Point | None, now: float | None = None) -> Point | None:
        filtered = self.update_float(point, now)
        self.value = None if filtered is None else (round(filtered[0]), round(filtered[1]))
        return self.value


class RawPointFilter:
    """No filtering: the baseline for evaluations and user tests."""

    value: Point | None = None

    def reset(self) -> None:
        self.value = None

    def update_float(self, point, now=None):
        return None if point is None else (float(point[0]), float(point[1]))

    def update(self, point: Point | None, now: float | None = None) -> Point | None:
        self.value = point
        return point


FILTERS = ("kalman", "exponential", "raw")


def make_point_filter(
    kind: str = "kalman",
    *,
    alpha: float = 0.35,
    measurement_std: float = DEFAULT_MEASUREMENT_STD,
    acceleration_std: float = DEFAULT_ACCELERATION_STD,
):
    """Build the pointer filter named ``kind`` (``kalman``, ``exponential`` or ``raw``)."""
    if kind == "kalman":
        return KalmanPointFilter(measurement_std, acceleration_std)
    if kind == "exponential":
        return ExponentialPointFilter(alpha)
    if kind == "raw":
        return RawPointFilter()
    raise ValueError(f"unknown pointer filter: {kind}")
