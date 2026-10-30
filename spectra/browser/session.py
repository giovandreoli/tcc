"""Browser sessions driven by landmarks analysed on the patient's own computer.

The browser runs MediaPipe locally and sends batches of hand landmarks, so the patient's
video never leaves their computer. This module turns those batches into repetitions,
metrics and safety prompts with the same exercise engine, recorder and report pipeline
as the desktop application.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

from spectra.browser import live
from spectra.browser.frames import BrowserFrame
from spectra.browser.rooms import RoomAdmission, Rooms
from spectra.config import AppConfig
from spectra.core.session_guard import SafetyPrompt, SessionGuard
from spectra.exercises.catalog import ExerciseSpec, finger_word
from spectra.exercises.runner import ExerciseRunner
from spectra.gestures.calibration import CalibrationProfile
from spectra.i18n import t
from spectra.service import SpectraService

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 5.0
SAFETY_PROMPT_SECONDS = 8.0
#: Frames this far in the future (clock skew) are pulled back to "now".
MAX_FUTURE_SECONDS = 0.0


class LandmarkSession:
    """Exercise counting over browser landmark batches, without any persistence.

    Used directly for the therapist's "try this exercise" preview, and as the base of
    :class:`BrowserSession`, which adds the room, the database and the safety limits.
    """

    def __init__(
        self,
        specs: tuple[ExerciseSpec, ...],
        hand_label: str | None = None,
        calibration: CalibrationProfile | None = None,
    ) -> None:
        if not specs:
            raise ValueError("at least one exercise is required")
        self.specs = specs
        self.hand_label = hand_label
        self.calibration = calibration
        self._lock = threading.RLock()
        self._runner: ExerciseRunner | None = None
        self._offset: float | None = None
        self._last_at = 0.0
        self.finished = False
        self.saved = False
        self.error_key: str | None = None
        self.started_wall: float | None = None

    # ------------------------------------------------------------- timing
    def _timestamp(self, at_ms: float) -> float:
        """Map the browser clock onto wall time, monotonic and never in the future."""
        now = time.time()
        if self._offset is None:
            self._offset = now - at_ms / 1000.0
        moment = min(self._offset + at_ms / 1000.0, now + MAX_FUTURE_SECONDS)
        moment = max(moment, self._last_at + 0.001)
        self._last_at = moment
        return moment

    # --------------------------------------------------------- processing
    def _on_start(self, at: float) -> None:
        """Hook for subclasses: the first frame arrived."""

    def _after_batch(self) -> None:
        """Hook for subclasses: called after every processed batch."""

    def process_batch(self, frames: list[BrowserFrame]) -> None:
        with self._lock:
            if self.finished or not frames:
                return
            try:
                self._before_batch()
                if self._offset is None:
                    # Anchor on the newest frame so the first batch keeps its real spacing.
                    self._offset = time.time() - max(f.at_ms for f in frames) / 1000.0
                for frame in frames:
                    at = self._timestamp(frame.at_ms)
                    if self._runner is None:
                        self.started_wall = time.time()
                        self._runner = ExerciseRunner(self.specs, self.calibration, started_at=at)
                        self._on_start(at)
                    hand = frame.hand
                    if hand is not None and self.hand_label and hand.label != self.hand_label:
                        hand = None
                    self._runner.feed(
                        hand.landmarks if hand else None, hand.score if hand else 0.0, at
                    )
                self._after_batch()
            except Exception:
                self.error_key = "browser.processing_error"
                logger.exception("browser exercise processing failed")
                raise

    def _before_batch(self) -> None:
        """Hook for subclasses: called before every batch is processed."""

    def next_exercise(self) -> None:
        with self._lock:
            if self.finished or self.error_key or self._runner is None:
                raise RuntimeError("cannot advance an inactive browser session")
            self._runner.next_exercise()

    def finish(self) -> None:
        with self._lock:
            self.finished = True

    # -------------------------------------------------------------- status
    @property
    def runner(self) -> ExerciseRunner | None:
        return self._runner

    def _elapsed(self) -> float:
        return time.time() - self.started_wall if self.started_wall else 0.0

    def _limit_seconds(self) -> float:
        return 0.0

    def _safety_key(self) -> str | None:
        return None

    def status(self) -> live.LiveStatus:
        with self._lock:
            runner = self._runner
            spec = runner.current if runner else self.specs[0]
            cue = runner.cue() if runner else None
            return live.LiveStatus(
                finished=self.finished,
                saved=self.saved,
                error_key=self.error_key,
                exercise=spec.key,
                reps=runner.progress.reps if runner else 0,
                target=spec.target_reps,
                total_reps=runner.total_reps if runner else 0,
                frames=runner.recorder.frames_total if runner else 0,
                hand_visible=runner.hand_visible if runner else False,
                elapsed_seconds=self._elapsed(),
                limit_seconds=self._limit_seconds(),
                exercise_index=runner.index if runner else 0,
                exercise_total=len(self.specs),
                safety_key=self._safety_key(),
                exercise_title=spec.display_title(),
                cue=(
                    t(cue.key, finger=finger_word(cue.finger) if cue.finger else "") if cue else ""
                ),
                hold_fraction=runner.hold_fraction() if runner else 0.0,
                instructions=tuple(spec.display_instructions()),
            )


class BrowserSession(LandmarkSession):
    """A single admitted browser: room lease, persistence and safety limits."""

    def __init__(
        self,
        config: AppConfig,
        admission: RoomAdmission,
        calibration: CalibrationProfile | None = None,
    ) -> None:
        self.config = replace(
            config, sound_enabled=False, session_limit_minutes=admission.duration_minutes
        )
        self.admission = admission
        super().__init__(
            admission.exercises,
            hand_label=admission.session.hand.value,
            calibration=calibration,
        )
        self.guard = SessionGuard(
            limit_minutes=admission.duration_minutes,
            rest_every_minutes=self.config.rest_prompt_minutes,
        )
        self._last_heartbeat = 0.0
        self.pain_before: int | None = None
        self._safety: str | None = None
        self._safety_until = 0.0
        live.register(admission.session.id, self)

    def _before_batch(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_SECONDS:
            with SpectraService.open(self.config) as service:
                Rooms(service).heartbeat(self.admission)
            self._last_heartbeat = now

    def _on_start(self, at: float) -> None:
        with SpectraService.open(self.config) as service, service.db.transaction() as connection:
            connection.execute(
                "UPDATE session SET started_at = ? WHERE id = ? AND ended_at IS NULL",
                (at, self.admission.session.id),
            )
        self.guard.start()

    def _after_batch(self) -> None:
        prompt = self.guard.check()
        if prompt is not SafetyPrompt.NONE:
            self._safety = f"safety.{prompt.value}"
            self._safety_until = time.monotonic() + SAFETY_PROMPT_SECONDS
        if self.guard.expired:
            self.finish()

    def _elapsed(self) -> float:
        return self.guard.elapsed()

    def _limit_seconds(self) -> float:
        return self.admission.duration_minutes * 60.0

    def _safety_key(self) -> str | None:
        return self._safety if time.monotonic() < self._safety_until else None

    def finish(self) -> None:
        with self._lock:
            if self.finished:
                return
            try:
                with SpectraService.open(self.config) as service:
                    rooms = Rooms(service)
                    runner = self._runner
                    if runner is not None:
                        rooms.heartbeat(self.admission)
                        runner.close()
                        session = service.db.get_session(self.admission.session.id)
                        if session is None:
                            raise RuntimeError("the session disappeared before saving")
                        service.finish_session(
                            session,
                            runner.recorder,
                            runs=runner.windows(),
                            pain_before=self.pain_before,
                        )
                        rooms.complete(self.admission)
                        self.saved = True
                    else:
                        rooms.release(self.admission)
            except Exception:
                self.error_key = "browser.save_error"
                logger.exception("browser session finalization failed")
                raise
            finally:
                self.finished = True
