"""Runs a list of exercises over a stream of timestamped landmarks.

This is the browser counterpart of the desktop physiotherapy mode, without OpenCV: the
same finger-state estimator and calibration, the same metric recorder and the same run
windows for the reports, driven by the timestamps of the analysed frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectra.detection.landmarks import HandLandmarks
from spectra.exercises.catalog import ExerciseSpec
from spectra.exercises.detectors import Cue, Detector, build_detector
from spectra.gestures.calibration import CalibrationProfile, FingerStateEstimator
from spectra.gestures.features import FingerStates
from spectra.metrics.analysis import RunWindow
from spectra.metrics.recorder import SessionRecorder

REP_COOLDOWN_SECONDS = 0.3
SWITCH_COOLDOWN_SECONDS = 1.0


@dataclass
class RunProgress:
    """Repetitions and time window of one exercise run."""

    spec: ExerciseSpec
    started_at: float
    reps: int = 0
    ended_at: float | None = None
    rep_times: list[float] = field(default_factory=list)

    @property
    def target(self) -> int:
        return self.spec.target_reps

    @property
    def completed(self) -> bool:
        return self.reps >= self.target

    def window(self) -> RunWindow:
        return RunWindow(
            name=self.spec.key,
            kind="physio",
            started_at=self.started_at,
            ended_at=self.ended_at if self.ended_at is not None else self.started_at,
            reps=self.reps,
            target=self.target,
            rep_times=tuple(self.rep_times),
            title=self.spec.title,
            details={"movement": self.spec.movement.value, **self.spec.params()},
        )


class ExerciseRunner:
    """Counts repetitions for a sequence of exercises and records the session."""

    def __init__(
        self,
        specs: tuple[ExerciseSpec, ...],
        calibration: CalibrationProfile | None = None,
        recorder: SessionRecorder | None = None,
        started_at: float = 0.0,
    ) -> None:
        if not specs:
            raise ValueError("at least one exercise is required")
        self.specs = specs
        self.recorder = recorder if recorder is not None else SessionRecorder()
        self.estimator = FingerStateEstimator(calibration)
        self.index = 0
        self.history: list[RunProgress] = []
        self.progress = RunProgress(specs[0], started_at)
        self.detector: Detector = build_detector(specs[0])
        self.states: FingerStates | None = None
        self.hand_visible = False
        self.last_at = started_at
        self._cooldown_until = started_at

    @property
    def current(self) -> ExerciseSpec:
        return self.specs[self.index]

    @property
    def total_reps(self) -> int:
        return self.progress.reps + sum(run.reps for run in self.history)

    def cue(self) -> Cue:
        if not self.hand_visible:
            return Cue("exercise.cue.show_hand")
        if self.progress.completed:
            return Cue("exercise.cue.done")
        return self.detector.cue()

    def hold_fraction(self) -> float:
        return self.detector.hold_fraction(self.last_at)

    def feed(self, landmarks: HandLandmarks | None, confidence: float, at: float) -> bool:
        """Account for one analysed frame; returns ``True`` when a repetition counted."""
        self.last_at = at
        self.recorder.observe(landmarks, confidence, now=at)
        if landmarks is None:
            self.hand_visible = False
            self.states = None
            self.estimator.reset()
            self.detector.hand_lost()
            return False
        self.hand_visible = True
        self.states = self.estimator.update(landmarks)
        if at < self._cooldown_until:
            return False
        if self.detector.update(landmarks, self.states, at):
            self.progress.reps += 1
            self.progress.rep_times.append(at)
            self._cooldown_until = at + REP_COOLDOWN_SECONDS
            return True
        return False

    def next_exercise(self, at: float | None = None) -> None:
        moment = self.last_at if at is None else at
        self.progress.ended_at = moment
        self.history.append(self.progress)
        self.index = (self.index + 1) % len(self.specs)
        self.progress = RunProgress(self.current, moment)
        self.detector = build_detector(self.current)
        self._cooldown_until = moment + SWITCH_COOLDOWN_SECONDS

    def close(self, at: float | None = None) -> None:
        if self.progress.ended_at is None:
            self.progress.ended_at = self.last_at if at is None else at

    def windows(self) -> list[RunWindow]:
        """Runs with at least one repetition, for metrics and reports."""
        return [run.window() for run in [*self.history, self.progress] if run.reps > 0]
