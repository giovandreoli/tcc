"""Guided physiotherapy exercises with repetition counting."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.detection.landmarks import INDEX_TIP, THUMB_TIP, HandLandmarks
from spectra.gestures.features import FingerStates
from spectra.i18n import t
from spectra.modes.base import ACTION_NEXT, BaseMode, ModeContext
from spectra.ui.text import draw_text, draw_text_centered
from spectra.ui.widgets import dim_frame, draw_finger_hud, draw_progress_bar

ACTION_SAVE_SESSION = "save_session"

REPS_TARGET = 8
PACE_WINDOW_SECONDS = 10.0
PINCH_THRESHOLD_PX = 40.0
OPEN_HAND_FINGERS = 4


class PhysioExercise(Enum):
    """Exercise identifiers; values double as i18n keys under ``physio.exercise.``."""

    OPEN_CLOSE = "open_close"
    FINGER_TOUCH = "finger_touch"
    FINGER_WAVE = "finger_wave"

    @property
    def title(self) -> str:
        return t(f"physio.exercise.{self.value}")

    @property
    def instructions(self) -> list[str]:
        return [t(f"physio.instructions.{self.value}_{i}") for i in (1, 2, 3)]


# --------------------------------------------------------------------- detectors
class OpenCloseDetector:
    """Counts one repetition per open-hand → closed-fist cycle."""

    def __init__(self) -> None:
        self.phase = "open"

    def update(self, states: FingerStates) -> bool:
        count = states.extended_count
        if self.phase == "open" and count >= OPEN_HAND_FINGERS:
            self.phase = "close"
        elif self.phase == "close" and count == 0:
            self.phase = "open"
            return True
        return False

    def reset(self) -> None:
        self.phase = "open"


class PinchDetector:
    """Counts one repetition per thumb-to-index-tip contact (rising edge)."""

    def __init__(self, threshold_px: float = PINCH_THRESHOLD_PX) -> None:
        self.threshold_px = threshold_px
        self.touching = False

    def update(self, landmarks: HandLandmarks, frame_shape: tuple[int, ...]) -> bool:
        height, width = frame_shape[:2]
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        distance = math.hypot(
            (thumb.x - index.x) * width,
            (thumb.y - index.y) * height,
        )
        touching = distance < self.threshold_px
        rep = touching and not self.touching
        self.touching = touching
        return rep

    def reset(self) -> None:
        self.touching = False


class FingerWaveDetector:
    """Counts one repetition per index → middle → ring → pinky lift sequence."""

    SEQUENCE = (1, 2, 3, 4)

    def __init__(self) -> None:
        self.step = 0
        self.lifted = False

    def update(self, states: FingerStates) -> bool:
        expected = self.SEQUENCE[self.step]
        if states[expected] and not self.lifted:
            self.lifted = True
        elif not states[expected] and self.lifted:
            self.lifted = False
            self.step += 1
            if self.step >= len(self.SEQUENCE):
                self.step = 0
                return True
        return False

    def reset(self) -> None:
        self.step = 0
        self.lifted = False


@dataclass
class ExerciseProgress:
    """Repetition tally and the time window of one exercise run.

    The window is what lets the metrics engine slice the session recording per exercise
    without the mode having to know anything about ROM or tremor.
    """

    exercise: PhysioExercise
    target: int = REPS_TARGET
    reps: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    rep_times: list[float] = field(default_factory=list)

    def add_rep(self, now: float | None = None) -> None:
        self.reps += 1
        self.rep_times.append(time.time() if now is None else now)

    def close(self, now: float | None = None) -> None:
        self.ended_at = time.time() if now is None else now

    @property
    def duration(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def window(self) -> tuple[float, float]:
        return (self.started_at, self.ended_at if self.ended_at is not None else time.time())

    @property
    def completed(self) -> bool:
        return self.reps >= self.target

    def pace_rpm(
        self, now: float | None = None, window: float = PACE_WINDOW_SECONDS
    ) -> float | None:
        """Repetitions per minute over the trailing ``window`` seconds."""
        current = time.time() if now is None else now
        recent = [stamp for stamp in self.rep_times if current - stamp <= window]
        if len(recent) < 2:
            return None
        span = recent[-1] - recent[0]
        if span <= 0:
            return None
        return len(recent) / span * 60.0


# -------------------------------------------------------------------------- mode
class PhysioMode(BaseMode):
    records_metrics = True

    EXERCISES = (
        PhysioExercise.OPEN_CLOSE,
        PhysioExercise.FINGER_TOUCH,
        PhysioExercise.FINGER_WAVE,
    )

    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        self.exercise_index = 0
        self.session_start = time.time()
        self.progress = ExerciseProgress(self.current_exercise)
        self.history: list[ExerciseProgress] = []
        self._detectors = {
            PhysioExercise.OPEN_CLOSE: OpenCloseDetector(),
            PhysioExercise.FINGER_TOUCH: PinchDetector(),
            PhysioExercise.FINGER_WAVE: FingerWaveDetector(),
        }
        self._cooldown_until = 0.0
        self._message = ""
        self._message_at = 0.0

        self.buttons = self.navigation_buttons(include_next=True)
        self.buttons.insert(
            0,
            self.make_button(
                10,
                self.height - 44 - 70,
                138,
                44,
                t("physio.save_session"),
                (20, 80, 20),
                value=ACTION_SAVE_SESSION,
            ),
        )

    @property
    def current_exercise(self) -> PhysioExercise:
        return self.EXERCISES[self.exercise_index]

    def _message_now(self, text: str) -> None:
        self._message = text
        self._message_at = time.time()

    def _next_exercise(self) -> None:
        self.progress.close()
        self.history.append(self.progress)
        self.exercise_index = (self.exercise_index + 1) % len(self.EXERCISES)
        self.progress = ExerciseProgress(self.current_exercise)
        self._detectors[self.current_exercise].reset()
        self._cooldown_until = time.time() + 1.0

    def _register_rep(self) -> None:
        self.progress.add_rep()
        self._message_now(t("physio.rep_done", count=self.progress.reps))
        self._cooldown_until = time.time() + 0.4

    # -------------------------------------------------------------- process
    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        height, width = frame.shape[:2]
        dim_frame(frame, (22, 10, 10), 0.65)

        exercise = self.current_exercise
        draw_text(frame, t("physio.title"), (10, 32), 0.8, (0, 220, 255), 2)
        draw_text(frame, exercise.title, (10, 62), 0.85, (255, 255, 255), 2)

        ratio = min(1.0, self.progress.reps / self.progress.target)
        draw_progress_bar(frame, (10, 76), (width - 20, 20), ratio, (0, 220, 80))
        draw_text(
            frame,
            t("physio.reps", done=self.progress.reps, target=self.progress.target),
            (10, 114),
            0.6,
            (200, 255, 200),
        )
        draw_text(
            frame,
            t("physio.exercise_index", current=self.exercise_index + 1, total=len(self.EXERCISES)),
            (width - 115, 32),
            0.55,
            (155, 155, 155),
        )
        minutes, seconds = divmod(int(time.time() - self.session_start), 60)
        draw_text(
            frame,
            t("physio.session_timer", mm=f"{minutes:02d}", ss=f"{seconds:02d}"),
            (10, height - 178),
            0.58,
            (170, 170, 170),
        )
        for i, line in enumerate(exercise.instructions):
            draw_text(frame, line, (10, 140 + i * 28), 0.58, (200, 200, 200))

        frame_context = self.observe(frame.shape, detection)
        pointer = frame_context.pointer
        landmarks = frame_context.landmarks
        states = frame_context.states

        if landmarks is not None and states is not None and time.time() > self._cooldown_until:
            if pointer is not None:
                cv2.circle(frame, pointer, 10, (0, 220, 255), -1)
            if self._update_detector(exercise, landmarks, states, frame.shape):
                self._register_rep()
            self._draw_finger_indicators(frame, states)

        pace = self.progress.pace_rpm()
        if pace is not None:
            draw_text(
                frame, t("physio.pace", rpm=f"{pace:.0f}"), (width - 230, 62), 0.58, (255, 220, 0)
            )

        if self.progress.completed:
            dim_frame(frame, (0, 55, 0), 0.28)
            draw_text_centered(
                frame, t("physio.completed"), width // 2, height // 2, 1.3, (0, 255, 80), 3
            )
            draw_text_centered(
                frame, t("physio.advance"), width // 2, height // 2 + 50, 0.75, (180, 255, 180), 2
            )

        if self._message and (time.time() - self._message_at) < 1.6:
            draw_text_centered(
                frame, self._message, width // 2, height - 130, 1.1, (0, 255, 150), 3
            )

        action = self.draw_buttons(frame, pointer)
        if action == ACTION_NEXT:
            self._next_exercise()
            action = None
        elif action == ACTION_SAVE_SESSION:
            self._save_session()
            action = None

        if states is not None:
            draw_finger_hud(frame, states, width - 155, height - 68)
        return frame, action

    def _update_detector(
        self,
        exercise: PhysioExercise,
        landmarks: HandLandmarks,
        states: FingerStates,
        frame_shape: tuple[int, ...],
    ) -> bool:
        detector = self._detectors[exercise]
        if isinstance(detector, PinchDetector):
            return detector.update(landmarks, frame_shape)
        return detector.update(states)

    def _draw_finger_indicators(self, frame: np.ndarray, states: FingerStates) -> None:
        height = frame.shape[0]
        baseline = height - 163
        keys = ("thumb", "index", "middle", "ring", "pinky")
        for i, (key, extended) in enumerate(zip(keys, states, strict=True)):
            x = 18 + i * 52
            color = (0, 220, 100) if extended else (58, 58, 58)
            cv2.rectangle(frame, (x, baseline - 32), (x + 44, baseline), color, -1)
            cv2.rectangle(frame, (x, baseline - 32), (x + 44, baseline), (145, 145, 145), 1)
            draw_text(frame, t(f"finger.short.{key}"), (x + 12, baseline - 10), 0.58)
        detector = self._detectors[self.current_exercise]
        if isinstance(detector, OpenCloseDetector):
            phase = t(f"physio.phase_{detector.phase}")
            draw_text(
                frame, t("physio.phase", phase=phase), (310, height - 148), 0.58, (0, 220, 255)
            )

    # --------------------------------------------------------------- export
    def _save_session(self) -> None:
        try:
            path = self.export_csv(self.context.config.exports_dir)
        except OSError:
            self._message_now(t("physio.session_save_failed"))
            self.context.sound.error()
            return
        self._message_now(t("physio.session_saved", filename=path.name))
        self.context.sound.success()

    def export_csv(self, directory: Path) -> Path:
        """Write one row per exercise run to a timestamped CSV and return its path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"physio_session_{int(self.session_start)}.csv"
        runs = [*self.history, self.progress]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["session_start", "exercise", "reps", "target", "duration_s", "rep_times"]
            )
            for run in runs:
                writer.writerow(
                    [
                        int(self.session_start),
                        run.exercise.value,
                        run.reps,
                        run.target,
                        f"{run.duration:.2f}",
                        ";".join(f"{stamp:.3f}" for stamp in run.rep_times),
                    ]
                )
        return path
