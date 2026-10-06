"""Gesture state machine with hold time, hysteresis and a safe pause.

Interaction is modelled as explicit states instead of "one combination equals one
colour". A gesture only takes effect after being held for ``hold_seconds``, which stops
transitional postures (a hand on its way from open to closed) from firing actions.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from spectra.detection.landmarks import HandLandmarks
from spectra.gestures.calibration import CalibrationProfile, FingerStateEstimator
from spectra.gestures.features import pinch_ratio
from spectra.gestures.profiles import (
    STANDARD_PROFILE,
    GestureAction,
    GestureProfile,
    HandSignal,
)

PINCH_ON_RATIO = 0.35
PINCH_OFF_RATIO = 0.45

#: Frame timestamps are floats; without a tolerance an exact-duration hold can miss.
HOLD_EPSILON = 1e-6


class GestureState(Enum):
    """Values double as i18n keys under ``gesture.state.``."""

    IDLE = "idle"
    POINTING = "pointing"
    PAINTING = "painting"
    ERASING = "erasing"
    COLOR_MENU = "color_menu"
    PAUSED = "paused"


ACTION_STATES: dict[GestureAction, GestureState] = {
    GestureAction.POINT: GestureState.POINTING,
    GestureAction.PAINT: GestureState.PAINTING,
    GestureAction.ERASE: GestureState.ERASING,
    GestureAction.COLOR_MENU: GestureState.COLOR_MENU,
    GestureAction.PAUSE: GestureState.PAUSED,
}


@dataclass(frozen=True)
class GestureEvent:
    """A state transition, for logging, metrics and tests."""

    previous: GestureState
    current: GestureState
    action: GestureAction | None
    at: float


class GestureStateMachine:
    """Drives :class:`GestureState` from a stream of :class:`HandSignal`."""

    def __init__(self, profile: GestureProfile | None = None, history_size: int = 32) -> None:
        self.profile = profile or STANDARD_PROFILE
        self.state = GestureState.IDLE
        self.events: deque[GestureEvent] = deque(maxlen=history_size)
        self._candidate: GestureAction | None = None
        self._candidate_since = 0.0
        self._absent_since: float | None = None
        #: ``False`` right after a pause toggle: the gesture must be released first.
        self._pause_armed = True

    # ------------------------------------------------------------------ helpers
    @property
    def paused(self) -> bool:
        return self.state is GestureState.PAUSED

    @property
    def hold_progress(self) -> float:
        """0..1 progress of the gesture currently being held, for on-screen feedback."""
        if self._candidate is None or self.profile.hold_seconds <= 0:
            return 0.0
        elapsed = time.time() - self._candidate_since
        return min(1.0, elapsed / self.profile.hold_seconds)

    def reset(self) -> None:
        self.state = GestureState.IDLE
        self._candidate = None
        self._absent_since = None
        self._pause_armed = True

    def _transition(self, target: GestureState, action: GestureAction | None, now: float) -> None:
        if target is self.state:
            return
        self.events.append(GestureEvent(self.state, target, action, now))
        self.state = target

    # -------------------------------------------------------------------- update
    def update(self, signal: HandSignal, now: float | None = None) -> GestureState:
        current = time.time() if now is None else now
        action = self.profile.action_for(signal) if signal.present else None

        if action is not self._candidate:
            self._candidate = action
            self._candidate_since = current
        held = current - self._candidate_since

        if not signal.present:
            return self._update_absent(current)

        self._absent_since = None
        if self.state is GestureState.PAUSED:
            return self._update_paused(action, held, current)
        if action is None:
            if held + HOLD_EPSILON >= self.profile.release_seconds:
                self._transition(GestureState.IDLE, None, current)
            return self.state
        if held + HOLD_EPSILON >= self.profile.hold_seconds:
            if action is GestureAction.PAUSE:
                self._pause_armed = False
            self._transition(ACTION_STATES[action], action, current)
        return self.state

    def _update_absent(self, now: float) -> GestureState:
        """A lost hand must never silently resume a paused session."""
        if self._absent_since is None:
            self._absent_since = now
        if self.state is GestureState.PAUSED:
            return self.state
        if now - self._absent_since + HOLD_EPSILON >= self.profile.release_seconds:
            self._transition(GestureState.IDLE, None, now)
        return self.state

    def _update_paused(self, action: GestureAction | None, held: float, now: float) -> GestureState:
        if action is not GestureAction.PAUSE:
            self._pause_armed = True
            return self.state
        if self._pause_armed and held + HOLD_EPSILON >= self.profile.hold_seconds:
            self._pause_armed = False
            self._transition(GestureState.IDLE, GestureAction.PAUSE, now)
        return self.state


class PinchTracker:
    """Hysteretic thumb-to-index contact detector (lower ratio means closer)."""

    def __init__(self, on_ratio: float = PINCH_ON_RATIO, off_ratio: float = PINCH_OFF_RATIO):
        if on_ratio >= off_ratio:
            raise ValueError("pinch needs on_ratio < off_ratio")
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.pinching = False

    def update(self, landmarks: HandLandmarks) -> bool:
        ratio = pinch_ratio(landmarks)
        self.pinching = ratio < self.off_ratio if self.pinching else ratio <= self.on_ratio
        return self.pinching

    def reset(self) -> None:
        self.pinching = False


class GestureEngine:
    """Landmarks in, :class:`GestureState` out: estimator, pinch and state machine."""

    def __init__(
        self,
        profile: GestureProfile | None = None,
        calibration: CalibrationProfile | None = None,
        smoothing_frames: int = 3,
    ) -> None:
        self.estimator = FingerStateEstimator(calibration, smoothing_frames)
        self.pinch = PinchTracker()
        self.machine = GestureStateMachine(profile)
        self.signal = HandSignal.absent()

    @property
    def state(self) -> GestureState:
        return self.machine.state

    @property
    def profile(self) -> GestureProfile:
        return self.machine.profile

    def reset(self) -> None:
        self.estimator.reset()
        self.pinch.reset()
        self.machine.reset()
        self.signal = HandSignal.absent()

    def update(
        self,
        landmarks: HandLandmarks | None,
        confidence: float = 0.0,
        now: float | None = None,
    ) -> GestureState:
        if landmarks is None:
            self.estimator.reset()
            self.pinch.reset()
            self.signal = HandSignal.absent()
        else:
            self.signal = HandSignal(
                present=True,
                fingers=self.estimator.update(landmarks),
                pinching=self.pinch.update(landmarks),
                confidence=confidence,
            )
        return self.machine.update(self.signal, now)
