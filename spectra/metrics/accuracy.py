"""Accuracy: aggregation of guided-drawing attempts and repetition counting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spectra.guided.scoring import TraceResult


@dataclass(frozen=True)
class AccuracySummary:
    """Aggregate of every guided-drawing attempt in a session."""

    attempts: int
    best_score: float
    mean_score: float
    mean_deviation: float
    max_deviation: float
    mean_completion: float
    total_duration: float

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "mean_completion": self.mean_completion,
            "total_duration": self.total_duration,
        }


@dataclass(frozen=True)
class RepetitionSummary:
    """Repetition count and pace for one exercise run."""

    exercise: str
    reps: int
    target: int
    duration: float

    @property
    def completion(self) -> float:
        return min(1.0, self.reps / self.target) if self.target else 0.0

    @property
    def pace_rpm(self) -> float:
        return self.reps / self.duration * 60.0 if self.duration > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "target": self.target,
            "duration": self.duration,
            "completion": self.completion,
            "pace_rpm": self.pace_rpm,
        }


def summarise_traces(results: Sequence[TraceResult]) -> AccuracySummary:
    """Aggregate guided-drawing attempts; empty attempts are ignored."""
    scored = [result for result in results if not result.is_empty]
    if not scored:
        return AccuracySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = len(scored)
    return AccuracySummary(
        attempts=count,
        best_score=max(result.score for result in scored),
        mean_score=sum(result.score for result in scored) / count,
        mean_deviation=sum(result.mean_deviation for result in scored) / count,
        max_deviation=max(result.max_deviation for result in scored),
        mean_completion=sum(result.completion for result in scored) / count,
        total_duration=sum(result.duration for result in scored),
    )
