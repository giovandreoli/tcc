"""Learning: gesture classifier trained on the user's own examples, and the gesture path."""

from spectra.learning.classifier import (
    GestureClassifier,
    GestureRecognizer,
    Prediction,
    TrainingError,
)
from spectra.learning.features import FEATURE_SIZE, pose_features
from spectra.learning.tutorial import LESSON_KEYS, TutorialEngine, TutorialStatus

__all__ = [
    "FEATURE_SIZE",
    "LESSON_KEYS",
    "GestureClassifier",
    "GestureRecognizer",
    "Prediction",
    "TrainingError",
    "TutorialEngine",
    "TutorialStatus",
    "pose_features",
]
