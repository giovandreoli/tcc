"""Thin wrapper around the MediaPipe HandLandmarker.

MediaPipe is imported lazily so that the rest of the package (and the whole test
suite) can be imported on machines without the vision stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spectra.detection.landmarks import HandLandmarks
from spectra.detection.model_store import ensure_model

DEFAULT_HAND_LABEL = "Right"


@dataclass(frozen=True)
class DetectedHand:
    """One detected hand: its landmarks, handedness label and detection score."""

    landmarks: HandLandmarks
    label: str = DEFAULT_HAND_LABEL
    score: float = 0.0


@dataclass(frozen=True)
class DetectionResult:
    """Result of a single frame detection."""

    hands: tuple[DetectedHand, ...] = field(default_factory=tuple)

    @property
    def primary(self) -> DetectedHand | None:
        return self.hands[0] if self.hands else None

    def for_label(self, label: str | None) -> DetectedHand | None:
        """Return the hand matching ``label``; with no label, the first hand."""
        if label is None:
            return self.primary
        for hand in self.hands:
            if hand.label == label:
                return hand
        return None


def handedness_label(entry: object) -> str:
    """Extract a handedness label tolerantly, defaulting to ``Right``."""
    if entry is None:
        return DEFAULT_HAND_LABEL
    if isinstance(entry, (list, tuple)):
        if not entry:
            return DEFAULT_HAND_LABEL
        entry = entry[0]
    return getattr(entry, "category_name", DEFAULT_HAND_LABEL) or DEFAULT_HAND_LABEL


def handedness_score(entry: object) -> float:
    if isinstance(entry, (list, tuple)) and entry:
        entry = entry[0]
    return float(getattr(entry, "score", 0.0) or 0.0)


class HandDetector:
    """Detects hand landmarks on BGR frames using MediaPipe Tasks."""

    def __init__(
        self,
        model_path: Path,
        detection_size: tuple[int, int] = (640, 360),
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._model_path = ensure_model(model_path)
        self._detection_size = detection_size
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._model_path)),
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        import mediapipe as mp

        small = cv2.resize(frame_bgr, self._detection_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)

        hands_landmarks = result.hand_landmarks or []
        handedness = result.handedness or []
        hands = []
        for i, landmarks in enumerate(hands_landmarks):
            entry = handedness[i] if i < len(handedness) else None
            hands.append(
                DetectedHand(
                    landmarks=landmarks,
                    label=handedness_label(entry),
                    score=handedness_score(entry),
                )
            )
        return DetectionResult(tuple(hands))

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
