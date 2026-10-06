"""Scoring for a guided-drawing attempt.

The scorer consumes pointer samples in normalised coordinates and produces the accuracy
metrics the therapist report needs: mean and maximum deviation from the target, how much
of the path was covered, how long it took, and a single 0-100 score.

Deviations are reported in normalised units; multiply by the frame height for pixels, or
see ``docs/metrics.md`` for the interpretation and its limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectra.guided.shapes import Difficulty, TargetPath

Point = tuple[float, float]

#: Weights of the final score. Completion is worth more than precision: finishing the
#: movement matters more clinically than tracing it beautifully.
COMPLETION_WEIGHT = 0.6
ACCURACY_WEIGHT = 0.4

#: A deviation of one tolerance width scores zero on the accuracy component.
ACCURACY_ZERO_AT_TOLERANCES = 2.0


@dataclass(frozen=True)
class TraceResult:
    """Immutable summary of one guided-drawing attempt."""

    shape_key: str
    difficulty: Difficulty
    samples: int
    mean_deviation: float
    max_deviation: float
    inside_ratio: float
    completion: float
    duration: float
    score: float

    @property
    def is_empty(self) -> bool:
        return self.samples == 0

    def to_dict(self) -> dict:
        return {
            "shape": self.shape_key,
            "difficulty": self.difficulty.value,
            "samples": self.samples,
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "inside_ratio": self.inside_ratio,
            "completion": self.completion,
            "duration": self.duration,
            "score": self.score,
        }


@dataclass
class TraceScorer:
    """Accumulates pointer samples against a :class:`TargetPath`."""

    path: TargetPath
    #: A waypoint counts as covered once the pointer passes within the corridor.
    _covered: set[int] = field(default_factory=set)
    _deviations: list[float] = field(default_factory=list)
    _inside: int = 0
    _first_at: float | None = None
    _last_at: float = 0.0

    @property
    def samples(self) -> int:
        return len(self._deviations)

    @property
    def duration(self) -> float:
        if self._first_at is None:
            return 0.0
        return max(0.0, self._last_at - self._first_at)

    @property
    def completion(self) -> float:
        """Fraction of the target waypoints the pointer has covered."""
        if not self.path.points:
            return 0.0
        return len(self._covered) / len(self.path.points)

    def add(self, point: Point, now: float) -> float:
        """Record one pointer sample; returns its deviation from the path."""
        deviation = self.path.distance_to(point)
        self._deviations.append(deviation)
        if self._first_at is None:
            self._first_at = now
        self._last_at = max(self._last_at, now)
        if deviation <= self.path.tolerance:
            self._inside += 1
            self._covered.add(self.path.nearest_index(point))
        return deviation

    def reset(self) -> None:
        self._covered.clear()
        self._deviations.clear()
        self._inside = 0
        self._first_at = None
        self._last_at = 0.0

    def result(self) -> TraceResult:
        if not self._deviations:
            return TraceResult(
                shape_key=self.path.shape_key,
                difficulty=self.path.difficulty,
                samples=0,
                mean_deviation=0.0,
                max_deviation=0.0,
                inside_ratio=0.0,
                completion=0.0,
                duration=0.0,
                score=0.0,
            )
        mean_deviation = sum(self._deviations) / len(self._deviations)
        completion = self.completion
        accuracy = _accuracy_component(mean_deviation, self.path.tolerance)
        return TraceResult(
            shape_key=self.path.shape_key,
            difficulty=self.path.difficulty,
            samples=len(self._deviations),
            mean_deviation=mean_deviation,
            max_deviation=max(self._deviations),
            inside_ratio=self._inside / len(self._deviations),
            completion=completion,
            duration=self.duration,
            score=round(100.0 * (COMPLETION_WEIGHT * completion + ACCURACY_WEIGHT * accuracy), 1),
        )


def _accuracy_component(mean_deviation: float, tolerance: float) -> float:
    """1.0 on the line, falling linearly to 0.0 at ``ACCURACY_ZERO_AT_TOLERANCES``."""
    if tolerance <= 0:
        return 0.0
    limit = tolerance * ACCURACY_ZERO_AT_TOLERANCES
    return max(0.0, min(1.0, 1.0 - mean_deviation / limit))
