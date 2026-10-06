from __future__ import annotations

import pytest

from spectra.core.session_guard import SafetyPrompt, SessionGuard
from spectra.gestures.features import FingerStates
from spectra.gestures.profiles import STANDARD_PROFILE, GestureAction, HandSignal
from spectra.gestures.state_machine import (
    GestureEngine,
    GestureState,
    GestureStateMachine,
    PinchTracker,
)
from tests.conftest import make_hand, pinching_hand

HOLD = STANDARD_PROFILE.hold_seconds

POINT = (False, True, False, False, False)
PAINT = (False, True, True, False, False)
ERASE = (False, True, True, True, False)
OPEN = (True, True, True, True, True)
FIST = (False, False, False, False, False)


def signal(flags) -> HandSignal:
    return HandSignal(present=True, fingers=FingerStates(*flags), confidence=0.9)


def hold(machine: GestureStateMachine, flags, start: float, duration: float = HOLD) -> float:
    """Feed the same gesture for ``duration`` seconds; returns the end timestamp."""
    machine.update(signal(flags), now=start)
    end = start + duration
    machine.update(signal(flags), now=end)
    return end


class TestTransitions:
    def test_starts_idle(self):
        assert GestureStateMachine().state is GestureState.IDLE

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            (POINT, GestureState.POINTING),
            (PAINT, GestureState.PAINTING),
            (ERASE, GestureState.ERASING),
            (OPEN, GestureState.COLOR_MENU),
            (FIST, GestureState.PAUSED),
        ],
    )
    def test_each_gesture_reaches_its_state(self, flags, expected):
        machine = GestureStateMachine()
        hold(machine, flags, start=0.0)
        assert machine.state is expected

    def test_a_brief_gesture_is_ignored(self):
        machine = GestureStateMachine()
        hold(machine, PAINT, start=0.0, duration=HOLD / 4)
        assert machine.state is GestureState.IDLE

    def test_a_transitional_posture_does_not_fire(self):
        """Opening the hand passes through paint and erase; neither should trigger."""
        machine = GestureStateMachine()
        now = 0.0
        for flags in (POINT, PAINT, ERASE, OPEN):
            now += HOLD / 5
            machine.update(signal(flags), now=now)
        assert machine.state is GestureState.IDLE

    def test_switching_gestures_restarts_the_hold_timer(self):
        machine = GestureStateMachine()
        machine.update(signal(PAINT), now=0.0)
        machine.update(signal(ERASE), now=HOLD * 0.9)
        assert machine.state is GestureState.IDLE
        machine.update(signal(ERASE), now=HOLD * 0.9 + HOLD)
        assert machine.state is GestureState.ERASING

    def test_an_unbound_pattern_returns_to_idle(self):
        machine = GestureStateMachine()
        now = hold(machine, PAINT, start=0.0)
        machine.update(signal((False, False, False, True, False)), now=now)
        machine.update(signal((False, False, False, True, False)), now=now + 1.0)
        assert machine.state is GestureState.IDLE

    def test_events_record_the_transition(self):
        machine = GestureStateMachine()
        hold(machine, PAINT, start=0.0)
        event = machine.events[-1]
        assert (event.previous, event.current) == (GestureState.IDLE, GestureState.PAINTING)
        assert event.action is GestureAction.PAINT

    def test_holding_the_same_gesture_emits_a_single_event(self):
        machine = GestureStateMachine()
        for i in range(20):
            machine.update(signal(PAINT), now=i * HOLD)
        assert len(machine.events) == 1


class TestHandLoss:
    def test_losing_the_hand_returns_to_idle(self):
        machine = GestureStateMachine()
        now = hold(machine, PAINT, start=0.0)
        machine.update(HandSignal.absent(), now=now)
        machine.update(HandSignal.absent(), now=now + 1.0)
        assert machine.state is GestureState.IDLE

    def test_a_single_dropped_frame_does_not_interrupt_painting(self):
        machine = GestureStateMachine()
        now = hold(machine, PAINT, start=0.0)
        machine.update(HandSignal.absent(), now=now + 0.01)
        assert machine.state is GestureState.PAINTING

    def test_losing_the_hand_never_resumes_a_paused_session(self):
        machine = GestureStateMachine()
        now = hold(machine, FIST, start=0.0)
        machine.update(HandSignal.absent(), now=now + 10.0)
        assert machine.state is GestureState.PAUSED


class TestPause:
    def test_pause_is_a_toggle(self):
        machine = GestureStateMachine()
        now = hold(machine, FIST, start=0.0)
        assert machine.paused
        machine.update(signal(POINT), now=now + 0.1)  # release the fist
        now = hold(machine, FIST, start=now + 0.2)
        assert machine.state is GestureState.IDLE

    def test_holding_the_fist_does_not_bounce_out_of_pause(self):
        machine = GestureStateMachine()
        now = 0.0
        machine.update(signal(FIST), now=now)
        for i in range(1, 20):
            machine.update(signal(FIST), now=i * HOLD)
        assert machine.paused

    def test_other_gestures_are_ignored_while_paused(self):
        machine = GestureStateMachine()
        now = hold(machine, FIST, start=0.0)
        hold(machine, PAINT, start=now + 0.1, duration=HOLD * 3)
        assert machine.paused

    def test_reset_clears_the_pause(self):
        machine = GestureStateMachine()
        hold(machine, FIST, start=0.0)
        machine.reset()
        assert machine.state is GestureState.IDLE


class TestPinchTracker:
    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError):
            PinchTracker(on_ratio=0.5, off_ratio=0.2)

    def test_detects_contact_and_release(self):
        tracker = PinchTracker()
        assert tracker.update(pinching_hand(distance=0.005)) is True
        assert tracker.update(make_hand(thumb=True, index=True)) is False

    def test_hysteresis_keeps_the_pinch_through_jitter(self):
        tracker = PinchTracker(on_ratio=0.35, off_ratio=0.45)
        tracker.update(pinching_hand(distance=0.005))
        assert tracker.update(pinching_hand(distance=0.10)) is True


class TestGestureEngine:
    def test_landmarks_drive_the_state_machine(self):
        engine = GestureEngine()
        for i in range(6):
            engine.update(make_hand(*PAINT), now=i * HOLD)
        assert engine.state is GestureState.PAINTING

    def test_a_missing_hand_resets_the_signal(self):
        engine = GestureEngine()
        engine.update(make_hand(*PAINT), now=0.0)
        engine.update(None, now=1.0)
        assert engine.signal.present is False

    def test_reset_returns_everything_to_idle(self):
        engine = GestureEngine()
        for i in range(6):
            engine.update(make_hand(*FIST), now=i * HOLD)
        engine.reset()
        assert engine.state is GestureState.IDLE
        assert engine.signal.present is False


class TestSessionGuard:
    def test_no_prompt_before_the_session_starts(self):
        assert SessionGuard().check() is SafetyPrompt.NONE

    def test_rest_prompt_fires_once_per_interval(self):
        guard = SessionGuard(limit_minutes=30, rest_every_minutes=5)
        guard.start(now=0.0)
        assert guard.check(now=299.0) is SafetyPrompt.NONE
        assert guard.check(now=301.0) is SafetyPrompt.REST
        assert guard.check(now=310.0) is SafetyPrompt.NONE
        assert guard.check(now=601.0) is SafetyPrompt.REST

    def test_time_limit_fires_once(self):
        guard = SessionGuard(limit_minutes=1)
        guard.start(now=0.0)
        assert guard.check(now=61.0) is SafetyPrompt.TIME_LIMIT
        assert guard.check(now=62.0) is SafetyPrompt.NONE
        assert guard.expired is True

    def test_remaining_time_never_goes_negative(self):
        guard = SessionGuard(limit_minutes=1)
        guard.start(now=0.0)
        assert guard.remaining(now=500.0) == 0.0

    def test_rest_prompts_can_be_disabled(self):
        guard = SessionGuard(limit_minutes=30, rest_every_minutes=0)
        guard.start(now=0.0)
        assert guard.check(now=1000.0) is SafetyPrompt.NONE
