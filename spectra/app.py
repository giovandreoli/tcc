"""Patient application shell: camera loop, mode routing and session lifecycle.

Two ways to run:

* **standalone** (``python -m spectra``) — free exploration, nothing is persisted;
* **attached to a session** (``python -m spectra --session <uuid>``) — the therapist opened
  the session from the panel; the patient confirms with their PIN, the patient's own gesture
  profile and calibration are applied, and on exit the metrics are computed and stored.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from spectra.config import AppConfig, load_config
from spectra.core.fps import FpsCounter
from spectra.core.session_guard import SafetyPrompt, SessionGuard
from spectra.detection.camera import Camera
from spectra.detection.hand_detector import DetectionResult, HandDetector
from spectra.gestures.profiles import get_preset
from spectra.i18n import set_locale, t
from spectra.metrics.analysis import RunWindow
from spectra.metrics.recorder import SessionRecorder
from spectra.modes.base import ACTION_MENU, ACTION_QUIT, AppMode, BaseMode, ModeContext
from spectra.modes.edu_colors import EduColorsMode
from spectra.modes.edu_count import EduCountMode
from spectra.modes.free_draw import FreeDrawMode
from spectra.modes.guided_draw import GuidedDrawMode
from spectra.modes.menu import MenuMode
from spectra.modes.physio import PhysioMode
from spectra.modes.pin_entry import ACTION_AUTHENTICATED, PinEntryMode
from spectra.ui.sound import SoundPlayer
from spectra.ui.text import draw_text, draw_text_centered
from spectra.ui.widgets import dim_frame, draw_hand_landmarks, draw_status_bar

logger = logging.getLogger(__name__)

WINDOW_NAME = "SPECTRA"
ESC_KEY = 27
PROMPT_SECONDS = 4.0

MODE_FACTORIES: dict[AppMode, type[BaseMode]] = {
    AppMode.MENU: MenuMode,
    AppMode.FREE_DRAW: FreeDrawMode,
    AppMode.GUIDED_DRAW: GuidedDrawMode,
    AppMode.EDU_COLORS: EduColorsMode,
    AppMode.EDU_COUNT: EduCountMode,
    AppMode.PHYSIO: PhysioMode,
}


class SpectraApp:
    """Owns the capture loop and dispatches frames to the active mode."""

    def __init__(
        self,
        config: AppConfig | None = None,
        hand_label: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.config.ensure_directories()
        set_locale(self.config.locale)
        self.hand_label = hand_label
        self.session_id = session_id
        self.mode = AppMode.MENU
        self.modes: dict[AppMode, BaseMode] = {}
        self.fps = FpsCounter()
        self.sound = SoundPlayer(self.config.sound_enabled)
        self.recorder = SessionRecorder()
        self.guard = SessionGuard(
            limit_minutes=self.config.session_limit_minutes,
            rest_every_minutes=self.config.rest_prompt_minutes,
        )
        self._prompt_key: str | None = None
        self._prompt_at = 0.0
        self._context: ModeContext | None = None
        self._pin_mode: PinEntryMode | None = None
        self._service = None
        self._session = None

    # ------------------------------------------------------------ session
    def attach_session(self) -> None:
        """Load the panel-created session, its gesture profile and its calibration."""
        if self.session_id is None:
            return
        from spectra.service import SpectraService

        self._service = SpectraService.open(self.config)
        self._session = self._service.db.get_session(self.session_id)
        if self._session is None:
            self._service.close()
            self._service = None
            raise RuntimeError(t("app.unknown_session"))
        self.hand_label = self._session.hand.value

    @property
    def attached(self) -> bool:
        return self._session is not None

    def verify_pin(self, pin: str) -> bool:
        from spectra.storage.db import AccountLockedError

        if self._service is None or self._session is None:
            return True
        try:
            return self._service.db.authenticate_patient(
                self._session.patient_id, pin
            ).authenticated
        except AccountLockedError:
            logger.warning("patient account is locked")
            return False

    def collect_runs(self) -> list[RunWindow]:
        """Exercise windows gathered from the physiotherapy mode."""
        physio = self.modes.get(AppMode.PHYSIO)
        if not isinstance(physio, PhysioMode):
            return []
        runs = [*physio.history, physio.progress]
        return [
            RunWindow(
                name=run.exercise.value,
                kind="physio",
                started_at=run.started_at,
                ended_at=run.ended_at or run.started_at,
                reps=run.reps,
                target=run.target,
                rep_times=tuple(run.rep_times),
            )
            for run in runs
            if run.reps > 0
        ]

    def collect_traces(self) -> list:
        guided = self.modes.get(AppMode.GUIDED_DRAW)
        return list(guided.results) if isinstance(guided, GuidedDrawMode) else []

    def finish_session(self) -> None:
        """Compute and persist the metrics, then close the service."""
        if self._service is None or self._session is None:
            return
        try:
            self._service.finish_session(
                self._session,
                self.recorder,
                runs=self.collect_runs(),
                traces=self.collect_traces(),
            )
            logger.info("%s", t("app.session_saved"))
        finally:
            self._service.close()
            self._service = None

    # ------------------------------------------------------------------ setup
    def build_modes(self, width: int, height: int) -> None:
        profile_name = self._session.gesture_profile_name if self._session else "standard"
        calibration = (
            self._service.calibration_for(self._session.patient_id)
            if self._service and self._session
            else None
        )
        context = ModeContext(
            config=self.config,
            width=width,
            height=height,
            sound=self.sound,
            hand_label=self.hand_label,
            gesture_profile=get_preset(profile_name),
            recorder=self.recorder,
            guard=self.guard,
            **({"calibration": calibration} if calibration else {}),
        )
        self.modes = {mode: factory(context) for mode, factory in MODE_FACTORIES.items()}
        self._context = context
        if self.attached:
            self._pin_mode = PinEntryMode(context, verify=self.verify_pin)

    def switch_to(self, mode: AppMode) -> None:
        self.mode = mode
        self.modes[mode].on_enter()

    def reset_mode(self, mode: AppMode) -> None:
        """Rebuild a mode from scratch, discarding its state."""
        self.modes[mode] = MODE_FACTORIES[mode](self._context)

    # -------------------------------------------------------------- per frame
    def render_frame(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        if self._pin_mode is not None:
            output, action = self._pin_mode.process(frame, detection)
            if action == ACTION_AUTHENTICATED:
                self._pin_mode = None
                self.guard.start()
                return output, None
            return output, action

        handler = self.modes[self.mode]
        output, action = handler.process(frame, detection)
        for hand in detection.hands:
            draw_hand_landmarks(output, hand.landmarks)
        draw_status_bar(output, self.mode.title, t("app.interact_hint"))
        draw_text(
            output,
            t("common.fps", fps=f"{self.fps.tick():.0f}"),
            (output.shape[1] - 78, 18),
            0.46,
            (110, 110, 110),
        )
        self._draw_safety(output)
        return output, action

    def _draw_safety(self, frame: np.ndarray) -> None:
        """Show rest prompts and stop the session when the time limit is reached."""
        import time

        prompt = self.guard.check()
        if prompt is not SafetyPrompt.NONE:
            self._prompt_key = f"safety.{prompt.value}"
            self._prompt_at = time.time()
            self.sound.success()
        if self._prompt_key and (time.time() - self._prompt_at) < PROMPT_SECONDS:
            dim_frame(frame, (0, 20, 40), 0.35)
            draw_text_centered(
                frame, t(self._prompt_key), frame.shape[1] // 2, 120, 0.9, (0, 220, 255), 2
            )

    def handle_action(self, action: Any) -> bool:
        """Apply a mode action; returns ``False`` when the app should stop."""
        if action == ACTION_QUIT:
            return False
        if action == ACTION_MENU:
            self.reset_mode(AppMode.MENU)
            self.switch_to(AppMode.MENU)
        elif isinstance(action, AppMode):
            self.switch_to(action)
        return True

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        self.attach_session()
        try:
            camera = Camera(
                self.config.camera_index,
                self.config.capture_width,
                self.config.capture_height,
                self.config.capture_fps,
            )
        except RuntimeError as exc:
            raise RuntimeError(t("app.camera_error")) from exc

        with camera:
            first = camera.read()
            if first is None:
                raise RuntimeError(t("app.frame_error"))
            height, width = first.shape[:2]
            self.build_modes(width, height)
            if not self.attached:
                self.guard.start()

            detector = HandDetector(
                self.config.model_path(),
                detection_size=(self.config.detection_width, self.config.detection_height),
            )
            logger.info("%s", t("app.started"))
            logger.info("%s", t("app.hint"))
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, width, height)
            try:
                while True:
                    frame = camera.read()
                    if frame is None:
                        break
                    output, action = self.render_frame(frame, detector.detect(frame))
                    cv2.imshow(WINDOW_NAME, output)
                    if cv2.waitKey(1) & 0xFF == ESC_KEY:  # emergency exit for the therapist
                        break
                    if not self.handle_action(action):
                        break
                    if self.guard.expired:
                        break
            except KeyboardInterrupt:
                pass
            finally:
                detector.close()
                cv2.destroyAllWindows()
                self.finish_session()
                logger.info("%s", t("app.stopped"))
