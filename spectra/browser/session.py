"""Thread-safe adapter from browser video frames to the existing exercise engine."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

import cv2
import numpy as np

from spectra.app import SpectraApp
from spectra.browser import live
from spectra.browser.rooms import RoomAdmission, Rooms
from spectra.config import AppConfig
from spectra.detection.hand_detector import HandDetector
from spectra.i18n import t
from spectra.modes.base import ACTION_MENU, ACTION_QUIT, AppMode
from spectra.modes.physio import ExerciseProgress, PhysioExercise, PhysioMode
from spectra.service import SpectraService
from spectra.ui.text import draw_text_centered

logger = logging.getLogger(__name__)


class BrowserSession:
    """A single admitted browser, with no OpenCV windows or hardware capture."""

    def __init__(self, config: AppConfig, admission: RoomAdmission) -> None:
        self.config = replace(
            config, sound_enabled=False, session_limit_minutes=admission.duration_minutes
        )
        self.admission = admission
        self._lock = threading.RLock()
        self._app: SpectraApp | None = None
        self._detector: HandDetector | None = None
        self._last_heartbeat = 0.0
        self.finished = False
        self.saved = False
        self.error_key: str | None = None
        self.pain_before: int | None = None
        self._reps = 0
        self._target = 0
        self._total_reps = 0
        self._exercise_index = 0
        self._hand_visible = False
        self._exercise = admission.exercises[0]
        live.register(admission.session.id, self)

    def _start(self, frame: np.ndarray) -> None:
        self._detector = HandDetector(
            self.config.model_path(),
            detection_size=(self.config.detection_width, self.config.detection_height),
        )
        with SpectraService.open(self.config) as service, service.db.transaction() as connection:
            connection.execute(
                "UPDATE session SET started_at = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), self.admission.session.id),
            )
        app = SpectraApp(self.config, session_id=self.admission.session.id)
        self._app = app
        app.attach_session()
        height, width = frame.shape[:2]
        app.build_modes(width, height, require_pin=False)
        mode = app.modes[AppMode.PHYSIO]
        if not isinstance(mode, PhysioMode):
            raise RuntimeError("physiotherapy mode is unavailable")
        mode.EXERCISES = tuple(PhysioExercise(name) for name in self.admission.exercises)
        mode.progress = ExerciseProgress(mode.current_exercise)
        app.switch_to(AppMode.PHYSIO)
        app.guard.start()

    def process(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            if self.finished:
                output = np.zeros_like(frame)
                draw_text_centered(
                    output, t("browser.ended"), output.shape[1] // 2, output.shape[0] // 2
                )
                return output
            try:
                now = time.monotonic()
                if now - self._last_heartbeat >= 5:
                    with SpectraService.open(self.config) as service:
                        Rooms(service).heartbeat(self.admission)
                    self._last_heartbeat = now
                if self._app is None:
                    self._start(frame)
                app, detector = self._app, self._detector
                if app is None or detector is None:
                    raise RuntimeError("browser session initialization failed")
                mirrored = cv2.flip(frame, 1)
                detection = detector.detect(mirrored)
                self._hand_visible = bool(detection.hands)
                output, action = app.render_frame(mirrored, detection)
                mode = app.modes[AppMode.PHYSIO]
                if isinstance(mode, PhysioMode):
                    self._reps = mode.progress.reps
                    self._target = mode.progress.target
                    self._exercise_index = mode.exercise_index
                    self._total_reps = mode.progress.reps + sum(r.reps for r in mode.history)
                    self._exercise = mode.current_exercise.value
                if action in (ACTION_MENU, ACTION_QUIT) or app.guard.expired:
                    self.finish()
                return output
            except Exception:
                self.error_key = "browser.processing_error"
                logger.exception("browser exercise processing failed")
                raise
            finally:
                if self._app is not None:
                    self._app.close_storage()

    def finish(self) -> None:
        with self._lock:
            if self.finished:
                return
            try:
                if self._app is not None and self._detector is not None:
                    with SpectraService.open(self.config) as service:
                        rooms = Rooms(service)
                        rooms.heartbeat(self.admission)
                        mode = self._app.modes[AppMode.PHYSIO]
                        if isinstance(mode, PhysioMode):
                            mode.progress.close()
                        self._app.finish_session(pain_before=self.pain_before)
                        rooms.complete(self.admission)
                    self.saved = True
                else:
                    with SpectraService.open(self.config) as service:
                        Rooms(service).release(self.admission)
                self.finished = True
            except Exception:
                self.error_key = "browser.save_error"
                logger.exception("browser session finalization failed")
                raise
            finally:
                self.finished = True
                if self._detector is not None:
                    self._detector.close()
                    self._detector = None
                if self._app is not None:
                    self._app.close_storage()

    def snapshot(self) -> tuple[bool, bool, str | None, str, int, int]:
        with self._lock:
            frames = self._app.recorder.frames_total if self._app is not None else 0
            return self.finished, self.saved, self.error_key, self._exercise, self._reps, frames

    def status(self) -> live.LiveStatus:
        with self._lock:
            app = self._app
            return live.LiveStatus(
                finished=self.finished,
                saved=self.saved,
                error_key=self.error_key,
                exercise=self._exercise,
                reps=self._reps,
                target=self._target,
                total_reps=self._total_reps,
                frames=app.recorder.frames_total if app is not None else 0,
                hand_visible=self._hand_visible,
                elapsed_seconds=app.guard.elapsed() if app is not None else 0.0,
                limit_seconds=self.admission.duration_minutes * 60.0,
                exercise_index=self._exercise_index,
                exercise_total=len(self.admission.exercises),
            )
