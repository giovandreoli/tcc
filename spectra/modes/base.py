"""Shared scaffolding for the interactive modes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from spectra.config import AppConfig
from spectra.core.kalman import make_point_filter
from spectra.core.session_guard import SessionGuard
from spectra.detection.hand_detector import DetectedHand, DetectionResult
from spectra.detection.landmarks import HandLandmarks
from spectra.gestures.calibration import CalibrationProfile
from spectra.gestures.click import PinchClick
from spectra.gestures.depth import DepthEstimator, DepthMethod, DepthReading
from spectra.gestures.features import FingerStates, pointer_position
from spectra.gestures.profiles import STANDARD_PROFILE, GestureProfile
from spectra.gestures.state_machine import GestureEngine, GestureState
from spectra.i18n import t
from spectra.metrics.recorder import SessionRecorder
from spectra.ui.button import Button
from spectra.ui.sound import SoundPlayer

ACTION_QUIT = "quit"
ACTION_MENU = "menu"
ACTION_NEXT = "next"
#: A click lands on the last pointer position seen within this window (the index finger
#: bends slightly while pinching, which can hide the pointer for a frame or two).
CLICK_POINTER_SECONDS = 0.4

Point = tuple[int, int]


class AppMode(Enum):
    """Top-level screens; values double as i18n keys under ``mode.``."""

    MENU = "menu"
    FREE_DRAW = "free_draw"
    GUIDED_DRAW = "guided_draw"
    EDU_COLORS = "edu_colors"
    EDU_COUNT = "edu_count"
    PHYSIO = "physio"
    STERILE = "sterile"
    USABILITY = "usability"

    @property
    def title(self) -> str:
        return t(f"mode.{self.value}")


@dataclass
class ModeContext:
    """Everything a mode needs from the application shell."""

    config: AppConfig
    width: int
    height: int
    sound: SoundPlayer = field(default_factory=SoundPlayer)
    #: Handedness label of the hand being treated; ``None`` means "first hand seen".
    hand_label: str | None = None
    gesture_profile: GestureProfile = STANDARD_PROFILE
    calibration: CalibrationProfile = field(default_factory=CalibrationProfile)
    #: Shared across modes so one session produces one continuous recording.
    recorder: SessionRecorder = field(default_factory=SessionRecorder)
    guard: SessionGuard = field(default_factory=SessionGuard)
    camera_overlay: bool = True
    #: Gestures learned from this user (classifier) and the commands bound to them.
    gesture_model: Any = None
    learned_bindings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FrameContext:
    """Everything a mode learns about the current frame, computed once."""

    hand: DetectedHand | None
    landmarks: HandLandmarks | None
    states: FingerStates | None
    pointer: Point | None
    gesture: GestureState
    normalized_pointer: tuple[float, float] | None
    #: True on the frame a pinch "click" happens.
    clicked: bool = False
    depth: DepthReading | None = None
    at: float = 0.0

    @property
    def has_hand(self) -> bool:
        return self.landmarks is not None

    @property
    def paused(self) -> bool:
        return self.gesture is GestureState.PAUSED


class BaseMode:
    """Base class handling the gesture engine, pointer smoothing and dwell buttons."""

    #: Observation-only screens (menu, quizzes) do not feed the metric recorder.
    records_metrics = False

    def __init__(self, context: ModeContext) -> None:
        self.context = context
        self.width = context.width
        self.height = context.height
        self.buttons: list[Button] = []
        self.engine = GestureEngine(context.gesture_profile, context.calibration)
        config = context.config
        self._pointer = make_point_filter(
            config.pointer_filter,
            alpha=config.pointer_smooth_alpha,
            measurement_std=config.kalman_measurement_std,
            acceleration_std=config.kalman_acceleration_std,
        )
        self.click = PinchClick(context.calibration.pinch)
        self.depth = DepthEstimator(
            DepthMethod(config.depth_method), image_aspect=context.width / max(1, context.height)
        )
        self.clicked = False
        self._last_pointer: Point | None = None
        self._last_pointer_at = 0.0

    @property
    def pinch_clicks_enabled(self) -> bool:
        """Off when the gesture profile already uses the pinch (e.g. to paint)."""
        if not self.context.config.pinch_click:
            return False
        return not any(trigger.pinch for trigger in self.engine.profile.bindings.values())

    # -------------------------------------------------------------------- input
    def active_hand(self, detection: DetectionResult) -> DetectedHand | None:
        return detection.for_label(self.context.hand_label)

    def observe(self, frame_shape: tuple[int, ...], detection: DetectionResult) -> FrameContext:
        """Run the gesture engine, the click, the depth and the recorder once per frame."""
        now = time.time()
        hand = self.active_hand(detection)
        landmarks = hand.landmarks if hand else None
        confidence = hand.score if hand else 0.0
        gesture = self.engine.update(landmarks, confidence)
        if self.records_metrics:
            self.context.recorder.observe(landmarks, confidence)

        pointer: Point | None = None
        normalized: tuple[float, float] | None = None
        if landmarks is None:
            self._pointer.reset()
        else:
            pointer = self._pointer.update(pointer_position(landmarks, frame_shape), now)
            if pointer is not None:
                height, width = frame_shape[:2]
                normalized = (pointer[0] / width, pointer[1] / height)
                self._last_pointer, self._last_pointer_at = pointer, now

        self.clicked = self.pinch_clicks_enabled and self.click.update(landmarks, now)
        depth = self.depth.update(landmarks)

        return FrameContext(
            hand=hand,
            landmarks=landmarks,
            states=self.engine.signal.fingers if landmarks is not None else None,
            pointer=pointer,
            gesture=gesture,
            normalized_pointer=normalized,
            clicked=self.clicked,
            depth=depth,
            at=now,
        )

    def click_point(self, pointer: Point | None) -> Point | None:
        """Where a click lands: the pointer, or the last one seen a moment ago."""
        if pointer is not None:
            return pointer
        if (
            self._last_pointer is not None
            and time.time() - self._last_pointer_at <= CLICK_POINTER_SECONDS
        ):
            return self._last_pointer
        return None

    # ----------------------------------------------------------------------- ui
    def make_button(self, *args: Any, **kwargs: Any) -> Button:
        kwargs.setdefault("hover_seconds", self.context.config.hover_select_secs)
        return Button(*args, **kwargs)

    def draw_buttons(self, frame: np.ndarray, pointer: Point | None) -> Any:
        """Dwell on a button, or pinch-click it, to select it."""
        action: Any = None
        target = self.click_point(pointer) if self.clicked else None
        for button in self.buttons:
            if button.update_hover(button.contains(pointer)):
                action = button.value
            elif target is not None and button.contains(target):
                button.reset()
                action = button.value
            button.draw(frame)
        return action

    def navigation_buttons(self, include_next: bool = False) -> list[Button]:
        """Bottom-right "next / back / quit" cluster shared by every mode."""
        width, height = 138, 44
        y = self.height - height - 10
        buttons: list[Button] = []
        if include_next:
            buttons.append(
                self.make_button(
                    self.width - 3 * width - 30,
                    y,
                    width,
                    height,
                    t("common.next"),
                    (60, 60, 15),
                    value=ACTION_NEXT,
                )
            )
        buttons.append(
            self.make_button(
                self.width - 2 * width - 20,
                y,
                width,
                height,
                t("common.back"),
                (20, 20, 80),
                value=ACTION_MENU,
            )
        )
        buttons.append(
            self.make_button(
                self.width - width - 10,
                y,
                width,
                height,
                t("common.quit"),
                (120, 15, 15),
                value=ACTION_QUIT,
            )
        )
        return buttons

    # ------------------------------------------------------------------ contract
    def process(
        self, frame: np.ndarray, detection: DetectionResult
    ) -> tuple[np.ndarray, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_enter(self) -> None:
        """Called every time the mode becomes active."""
        self._pointer.reset()
        self.engine.reset()
        self.click.reset()
        self.depth.recalibrate()
        for button in self.buttons:
            button.reset()
