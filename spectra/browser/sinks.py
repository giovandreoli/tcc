"""Landmark consumers for the browser camera other than exercise sessions.

The camera component (:mod:`spectra.browser.video`) hands validated landmark batches to any
object with ``process_batch``, ``runner`` and ``finish``. These sinks power the learning
features: the gesture learning path, recording examples for the classifier, testing the
classifier live, and calibrating the pinch click. None of them stores images; the classifier
keeps normalised pose features only.
"""

from __future__ import annotations

import threading
import time

from spectra.browser.frames import BrowserFrame
from spectra.detection.hand_detector import DetectedHand
from spectra.detection.landmarks import HandLandmarks
from spectra.gestures.calibration import CalibrationProfile
from spectra.gestures.click import PinchThresholds, calibrate_pinch
from spectra.gestures.features import pinch_ratio
from spectra.learning.classifier import GestureClassifier, Prediction
from spectra.learning.features import pose_features
from spectra.learning.tutorial import TutorialEngine, TutorialStatus

SAMPLE_INTERVAL_SECONDS = 0.1


class FrameClock:
    """Maps the browser's millisecond clock onto wall time, strictly increasing.

    The offset is anchored on the **newest** frame of the first batch, which has just been
    captured, so the earlier frames of that batch keep their real spacing instead of being
    squeezed together at "now".
    """

    def __init__(self) -> None:
        self._offset: float | None = None
        self._last = 0.0

    def anchor(self, newest_ms: float) -> None:
        if self._offset is None:
            self._offset = time.time() - newest_ms / 1000.0

    def __call__(self, at_ms: float) -> float:
        now = time.time()
        if self._offset is None:
            self._offset = now - at_ms / 1000.0
        moment = max(min(self._offset + at_ms / 1000.0, now), self._last + 0.001)
        self._last = moment
        return moment


class _Sink:
    """Shared plumbing: locking, hand filtering and the ``runner``/``finish`` protocol."""

    def __init__(self, hand_label: str | None = None) -> None:
        self.hand_label = hand_label
        self._lock = threading.RLock()
        self._clock = FrameClock()
        self.started = False
        self.finished = False
        self.frames = 0

    @property
    def runner(self) -> _Sink | None:
        return self if self.started else None

    def finish(self) -> None:
        with self._lock:
            self.finished = True

    def _hand(self, frame: BrowserFrame) -> DetectedHand | None:
        hand = frame.hand
        if hand is not None and self.hand_label and hand.label != self.hand_label:
            return None
        return hand

    def process_batch(self, frames: list[BrowserFrame]) -> None:
        with self._lock:
            if frames:
                self._clock.anchor(max(frame.at_ms for frame in frames))
            for frame in frames:
                self.started = True
                self.frames += 1
                self._consume(self._hand(frame), self._clock(frame.at_ms))

    def _consume(self, hand: DetectedHand | None, at: float) -> None:  # pragma: no cover
        raise NotImplementedError


class TutorialSink(_Sink):
    """Feeds the gesture learning path."""

    def __init__(
        self,
        hand_label: str | None = None,
        calibration: CalibrationProfile | None = None,
        finger_colors: dict[str, str] | None = None,
    ) -> None:
        super().__init__(hand_label)
        kwargs = {"finger_colors": dict(finger_colors)} if finger_colors else {}
        self.engine = TutorialEngine(
            calibration=calibration,
            pinch=calibration.pinch if calibration else None,
            **kwargs,
        )

    def _consume(self, hand: DetectedHand | None, at: float) -> None:
        self.engine.feed(hand.landmarks if hand else None, at)

    def skip(self) -> None:
        with self._lock:
            self.engine.skip()

    def status(self) -> TutorialStatus:
        with self._lock:
            return self.engine.status()


class GestureCollector(_Sink):
    """Records labelled examples for the classifier, or predicts with a trained one."""

    def __init__(self, model: GestureClassifier | None = None) -> None:
        super().__init__(None)
        self.model = model or GestureClassifier()
        self.label: str | None = None
        self.record_from = 0.0
        self.record_until = 0.0
        self.predicting = False
        self.prediction: Prediction | None = None
        self.hand_visible = False
        self._last_sample = 0.0

    def record(self, label: str, seconds: float = 3.0, countdown: float = 1.5) -> None:
        """Collect examples of ``label`` after a short countdown, for ``seconds``."""
        with self._lock:
            now = time.time()
            self.label = label.strip()
            self.record_from = now + countdown
            self.record_until = self.record_from + seconds
            self.predicting = False

    def recording(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return self.label is not None and current < self.record_until

    def start_predicting(self) -> None:
        with self._lock:
            self.label = None
            self.predicting = True
            self.prediction = None

    def _consume(self, hand: DetectedHand | None, at: float) -> None:
        self.hand_visible = hand is not None
        if hand is None:
            if self.predicting:
                self.prediction = None
            return
        features = pose_features(hand.landmarks, hand.label)
        if self.predicting and self.model.trained:
            self.prediction = self.model.predict(features)
            return
        now = time.time()
        if (
            self.label
            and self.record_from <= now < self.record_until
            and now - self._last_sample >= SAMPLE_INTERVAL_SECONDS
        ):
            self.model.add(self.label, features)
            self._last_sample = now


class PinchCalibrationSink(_Sink):
    """Collects 'fingers apart' and 'pinching' frames to learn the click thresholds."""

    def __init__(self, hand_label: str | None = None) -> None:
        super().__init__(hand_label)
        self.samples: dict[str, list[HandLandmarks]] = {"apart": [], "pinch": []}
        self.phase: str | None = None
        self.until = 0.0
        self.ratio: float | None = None

    def capture(self, phase: str, seconds: float = 2.0) -> None:
        if phase not in self.samples:
            raise ValueError("phase must be 'apart' or 'pinch'")
        with self._lock:
            self.samples[phase] = []
            self.phase = phase
            self.until = time.time() + seconds

    def _consume(self, hand: DetectedHand | None, at: float) -> None:
        self.ratio = pinch_ratio(hand.landmarks) if hand else None
        if hand is not None and self.phase and time.time() < self.until:
            self.samples[self.phase].append(hand.landmarks)

    def thresholds(self) -> PinchThresholds:
        with self._lock:
            return calibrate_pinch(self.samples["apart"], self.samples["pinch"])
