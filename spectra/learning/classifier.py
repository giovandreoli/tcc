"""A k-nearest-neighbour classifier that learns gestures from the user's own examples.

Why k-NN instead of a neural network: a therapist records a few seconds of each gesture
(tens of examples), training must take milliseconds on any laptop, and the result must be
explainable in a thesis. k-NN needs no training loop, adapts immediately when examples are
added, and its decisions can be traced to the stored examples.

Two safeguards make it suitable for an interface:

* **Rejection.** A pose far from every example is reported as unknown instead of being
  forced into the nearest class. The threshold is learned from the training data itself
  (the distribution of nearest-neighbour distances inside each class).
* **Honest evaluation.** Accuracy is measured with leave-one-out cross-validation: each
  example is classified by a model trained on all the others, and the confusion matrix
  shows which gestures are mistaken for which.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from spectra.learning.features import FEATURE_SIZE

MODEL_VERSION = 1
MIN_SAMPLES_PER_LABEL = 5
MIN_LABELS = 2
DEFAULT_K = 3
#: Rejection threshold = this percentile of in-class neighbour distances times a margin.
REJECT_PERCENTILE = 0.95
REJECT_MARGIN = 1.5
LABEL_MAX = 40

Vector = tuple[float, ...]


class TrainingError(ValueError):
    """Carries an i18n key."""


def distance(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    distance: float


@dataclass
class Evaluation:
    """Leave-one-out results."""

    labels: list[str]
    accuracy: float
    rejected: int
    confusion: dict[str, dict[str, int]]
    per_label: dict[str, float] = field(default_factory=dict)


class GestureClassifier:
    """Distance-weighted k-NN over normalised pose features, with rejection."""

    def __init__(self, k: int = DEFAULT_K) -> None:
        self.k = k
        self.samples: list[tuple[str, Vector]] = []
        self.threshold: float | None = None

    # --------------------------------------------------------------- training
    @property
    def labels(self) -> list[str]:
        return sorted({label for label, _ in self.samples})

    def counts(self) -> dict[str, int]:
        return dict(Counter(label for label, _ in self.samples))

    def add(self, label: str, features: Sequence[float]) -> None:
        label = label.strip()
        if not label or len(label) > LABEL_MAX:
            raise TrainingError("learning.error.label")
        if len(features) != FEATURE_SIZE:
            raise TrainingError("learning.error.features")
        self.samples.append((label, tuple(float(value) for value in features)))
        self.threshold = None

    def remove_label(self, label: str) -> None:
        self.samples = [sample for sample in self.samples if sample[0] != label]
        self.threshold = None

    def validate(self) -> None:
        counts = self.counts()
        if len(counts) < MIN_LABELS:
            raise TrainingError("learning.error.labels")
        if min(counts.values()) < MIN_SAMPLES_PER_LABEL:
            raise TrainingError("learning.error.samples")

    def fit(self) -> GestureClassifier:
        """Learn the rejection threshold; raises :class:`TrainingError` if data is too thin."""
        self.validate()
        nearest_same = []
        for index, (label, vector) in enumerate(self.samples):
            distances = [
                distance(vector, other)
                for position, (other_label, other) in enumerate(self.samples)
                if position != index and other_label == label
            ]
            nearest_same.append(min(distances))
        self.threshold = max(_percentile(nearest_same, REJECT_PERCENTILE) * REJECT_MARGIN, 1e-6)
        return self

    @property
    def trained(self) -> bool:
        return self.threshold is not None

    # ------------------------------------------------------------- inference
    def _vote(
        self, vector: Vector, pool: Sequence[tuple[str, Vector]], threshold: float | None
    ) -> Prediction:
        neighbours = sorted((distance(vector, other), label) for label, other in pool)[: self.k]
        if not neighbours:
            return Prediction(None, 0.0, math.inf)
        nearest = neighbours[0][0]
        if threshold is not None and nearest > threshold:
            return Prediction(None, 0.0, nearest)
        weights: Counter[str] = Counter()
        for value, label in neighbours:
            weights[label] += 1.0 / (value + 1e-6)
        label, weight = weights.most_common(1)[0]
        return Prediction(label, weight / sum(weights.values()), nearest)

    def predict(self, features: Sequence[float]) -> Prediction:
        if not self.trained:
            raise TrainingError("learning.error.untrained")
        return self._vote(tuple(features), self.samples, self.threshold)

    def evaluate(self) -> Evaluation:
        """Leave-one-out cross-validation (unknown predictions count as errors)."""
        self.validate()
        labels = self.labels
        confusion = {actual: dict.fromkeys([*labels, "?"], 0) for actual in labels}
        correct = rejected = 0
        for index, (actual, vector) in enumerate(self.samples):
            pool = self.samples[:index] + self.samples[index + 1 :]
            predicted = self._vote(vector, pool, self.threshold).label
            confusion[actual][predicted or "?"] += 1
            correct += predicted == actual
            rejected += predicted is None
        totals = self.counts()
        return Evaluation(
            labels=labels,
            accuracy=correct / len(self.samples),
            rejected=rejected,
            confusion=confusion,
            per_label={label: confusion[label][label] / totals[label] for label in labels},
        )

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "k": self.k,
            "threshold": self.threshold,
            "samples": [
                {"label": label, "features": list(vector)} for label, vector in self.samples
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> GestureClassifier:
        if data.get("version") != MODEL_VERSION:
            raise TrainingError("learning.error.version")
        model = cls(int(data.get("k", DEFAULT_K)))
        for sample in data.get("samples", []):
            model.add(sample["label"], sample["features"])
        model.threshold = data.get("threshold")
        return model


class GestureRecognizer:
    """Turns per-frame predictions into stable gesture events (majority vote + hold)."""

    def __init__(self, model: GestureClassifier, window: int = 7, min_votes: int = 5) -> None:
        self.model = model
        self.window = window
        self.min_votes = min_votes
        self._recent: list[str | None] = []
        self.current: str | None = None

    def reset(self) -> None:
        self._recent.clear()
        self.current = None

    def update(self, features: Sequence[float] | None) -> str | None:
        """Returns a label on the frame it becomes stable (an onset event), else ``None``."""
        label = None if features is None else self.model.predict(features).label
        self._recent = [*self._recent, label][-self.window :]
        winner, votes = Counter(self._recent).most_common(1)[0]
        stable = winner if votes >= self.min_votes else None
        if stable == self.current:
            return None
        self.current = stable
        return stable
