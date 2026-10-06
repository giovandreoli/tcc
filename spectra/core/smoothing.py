"""Exponential smoothing for noisy pointer positions."""

from __future__ import annotations

DEFAULT_ALPHA = 0.35

Point = tuple[int, int]


def smooth_point(previous: Point | None, current: Point, alpha: float = DEFAULT_ALPHA) -> Point:
    """Blend ``current`` into ``previous`` with an exponential moving average."""
    if previous is None:
        return current
    return (
        int((1.0 - alpha) * previous[0] + alpha * current[0]),
        int((1.0 - alpha) * previous[1] + alpha * current[1]),
    )


class ExponentialSmoother:
    """Stateful variant of :func:`smooth_point`."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.value: Point | None = None

    def update(self, point: Point | None) -> Point | None:
        if point is None:
            self.value = None
            return None
        self.value = smooth_point(self.value, point, self.alpha)
        return self.value

    def reset(self) -> None:
        self.value = None
