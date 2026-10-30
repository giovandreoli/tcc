"""Custom exercises: validated movements, parameter limits and repetition detectors."""

from __future__ import annotations

import pytest

from spectra.detection.landmarks import Landmark
from spectra.exercises import ExerciseRunner, ExerciseSpec, Movement
from spectra.exercises.catalog import (
    BUILTIN_SPECS,
    MOVEMENT_PARAMS,
    specs_from_json,
)
from spectra.exercises.detectors import (
    CONTACT_THRESHOLDS,
    ContactCounter,
    FingerLiftCounter,
    FingerWaveCounter,
    OpenCloseCounter,
    ThumbOppositionCounter,
    build_detector,
    contact_ratio,
)
from spectra.gestures.features import FingerStates
from tests.conftest import make_hand, pinching_hand, rotate_hand

OPEN = FingerStates(True, True, True, True, True)
FIST = FingerStates(False, False, False, False, False)


def states(*fingers: bool) -> FingerStates:
    return FingerStates(*fingers)


def touching(finger: str, gap: float = 0.0, along: bool = False) -> list[Landmark]:
    """A hand with the thumb tip on a fingertip, or ``gap`` away from it.

    By default the gap is sideways (towards the next finger); ``along`` moves the thumb
    beyond the fingertip instead, so the target stays the nearest fingertip.
    """
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    hand = make_hand(index=True, middle=True, ring=True, pinky=True)
    tip = hand[tips[finger]]
    hand[4] = Landmark(tip.x, tip.y - gap, 0.0) if along else Landmark(tip.x + gap, tip.y, 0.0)
    return hand


def apart() -> list[Landmark]:
    return make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)


# ------------------------------------------------------------------ catalog
def test_builtin_exercises_keep_their_historical_keys_and_titles():
    assert list(BUILTIN_SPECS) == ["open_close", "finger_touch", "finger_wave"]
    assert BUILTIN_SPECS["finger_touch"].display_title() == "Oposição do Polegar (Pinça)"
    assert len(BUILTIN_SPECS["open_close"].display_instructions()) == 3


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"title": "ab"}, "exercise.error.title"),
        ({"title": "x" * 61}, "exercise.error.title"),
        ({"target_reps": 0}, "exercise.error.reps"),
        ({"target_reps": 31}, "exercise.error.reps"),
        ({"hold_seconds": 5.5}, "exercise.error.hold"),
        ({"hold_seconds": -1}, "exercise.error.hold"),
        ({"finger": "thumb"}, "exercise.error.finger"),
        ({"tolerance": "loose"}, "exercise.error.tolerance"),
        ({"instructions": ("a", "b", "c", "d")}, "exercise.error.instructions"),
        ({"instructions": ("x" * 141,)}, "exercise.error.instructions"),
    ],
)
def test_out_of_range_parameters_are_rejected(changes, problem):
    base = {"key": "", "movement": Movement.PINCH, "title": "Pinça leve"}
    assert problem in ExerciseSpec(**{**base, **changes}).validate()


def test_hold_time_is_rejected_for_movements_without_a_hold():
    spec = ExerciseSpec("", Movement.FINGER_WAVE, title="Onda", hold_seconds=1.0)
    assert "exercise.error.hold" in spec.validate()


@pytest.mark.parametrize("movement", list(Movement))
def test_every_movement_round_trips_and_has_a_detector(movement):
    spec = ExerciseSpec("custom:x", movement, title="Exercício", target_reps=5)
    assert spec.validate() == []
    restored = ExerciseSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert set(spec.params()) == MOVEMENT_PARAMS[movement]
    assert build_detector(spec) is not None
    assert len(spec.display_instructions()) == 3
    assert spec.summary().startswith("5 repetições")


def test_unused_parameters_are_dropped_on_save():
    data = ExerciseSpec("k", Movement.OPEN_CLOSE, title="Punho", finger="ring").to_dict()
    assert "finger" not in data
    assert ExerciseSpec.from_dict(data).finger == "index"


def test_room_json_accepts_legacy_keys_and_full_specs():
    custom = ExerciseSpec("custom:1", Movement.FINGER_LIFT, title="Médio", finger="middle")
    specs = specs_from_json(["open_close", custom.to_dict()])
    assert specs == (BUILTIN_SPECS["open_close"], custom)


def test_generated_instructions_name_the_finger_and_the_hold():
    spec = ExerciseSpec("", Movement.PINCH, title="Pinça", finger="ring", hold_seconds=2)
    lines = spec.display_instructions()
    assert "anelar" in lines[0]
    assert "2 s" in lines[2]


# ---------------------------------------------------------------- detectors
def test_contact_ratio_is_independent_of_distance_to_the_camera():
    near = touching("index", gap=0.05)
    far = [Landmark(0.5 + (p.x - 0.5) * 0.5, 0.5 + (p.y - 0.5) * 0.5, 0) for p in near]
    assert contact_ratio(near, "index") == pytest.approx(contact_ratio(far, "index"))


def test_contact_ratio_is_independent_of_hand_rotation():
    hand = touching("middle", gap=0.04)
    rotated = rotate_hand(hand, 35)
    assert contact_ratio(hand, "middle") == pytest.approx(contact_ratio(rotated, "middle"))


def test_open_close_counts_one_rep_per_cycle_and_ignores_jitter():
    counter = OpenCloseCounter()
    hand = apart()
    sequence = [OPEN, OPEN, FIST, FIST, OPEN, states(False, True, False, False, False), FIST]
    reps = [counter.update(hand, item, i * 0.1) for i, item in enumerate(sequence)]
    assert reps.count(True) == 2


def test_open_close_hold_requires_the_fist_to_be_kept():
    counter = OpenCloseCounter(hold_seconds=1.0)
    hand = apart()
    counter.update(hand, OPEN, 0.0)
    assert not counter.update(hand, FIST, 0.1)
    assert counter.cue().key == "exercise.cue.hold"
    assert not counter.update(hand, OPEN, 0.5)  # Opened too early: no rep.
    assert not counter.update(hand, FIST, 0.6)
    assert counter.hold_fraction(1.1) == pytest.approx(0.5)
    assert counter.update(hand, FIST, 1.6)


def test_open_close_hold_restarts_when_the_hand_is_lost():
    counter = OpenCloseCounter(hold_seconds=1.0)
    hand = apart()
    counter.update(hand, OPEN, 0.0)
    counter.update(hand, FIST, 0.1)
    counter.hand_lost()
    assert not counter.update(hand, FIST, 1.0)
    assert counter.update(hand, FIST, 2.0)


@pytest.mark.parametrize("finger", ["index", "middle", "ring", "pinky"])
def test_pinch_counts_contacts_with_the_chosen_finger_only(finger):
    counter = ContactCounter(finger)
    others = [name for name in ("index", "middle", "ring", "pinky") if name != finger]
    assert not counter.update(touching(others[0]), OPEN, 0.0)
    assert counter.update(touching(finger), OPEN, 0.1)
    assert not counter.update(touching(finger), OPEN, 0.2)  # Still touching: one rep.
    assert counter.cue().key == "exercise.cue.release"
    counter.update(apart(), OPEN, 0.3)
    assert counter.update(touching(finger), OPEN, 0.4)


def test_pinch_hysteresis_ignores_jitter_around_the_threshold():
    counter = ContactCounter("index", "normal")
    on, off = CONTACT_THRESHOLDS["normal"]
    near_on = pinching_hand(0.0)
    assert counter.update(near_on, OPEN, 0.0)
    # Oscillating between "on" and "off" never releases, so no extra repetitions.
    for i in range(10):
        gap = 0.0 if i % 2 else (on + off) / 2 * 0.25
        assert not counter.update(touching("index", gap), OPEN, 0.1 * (i + 1))


@pytest.mark.parametrize(("tolerance", "counts"), [("wide", True), ("strict", False)])
def test_pinch_tolerance_changes_how_close_the_contact_must_be(tolerance, counts):
    scale = contact_ratio(touching("index", 0.1, along=True), "index") / 0.1
    gap = 0.40 / scale  # Between the strict and the wide thresholds.
    near = touching("index", gap, along=True)
    assert ContactCounter("index", tolerance).update(near, OPEN, 0.0) is counts
    assert ContactCounter("index", tolerance).update(touching("index"), OPEN, 0.0)


def test_touching_between_two_fingers_counts_only_for_the_nearest():
    hand = touching("index", gap=0.02)  # Slightly towards the middle finger.
    assert ContactCounter("index").update(hand, OPEN, 0.0)
    assert not ContactCounter("middle").update(hand, OPEN, 0.0)


def test_finger_wave_backward_starts_at_the_pinky():
    counter = FingerWaveCounter("backward")
    hand = apart()
    assert counter.cue().finger == "pinky"
    reps = 0
    for index in (4, 3, 2, 1):
        lifted = [False] * 5
        lifted[index] = True
        counter.update(hand, states(*lifted), 0.0)
        reps += counter.update(hand, FIST, 0.1)
    assert reps == 1


def test_finger_wave_out_of_order_does_not_count():
    counter = FingerWaveCounter()
    hand = apart()
    for index in (2, 1, 3, 4):
        lifted = [False] * 5
        lifted[index] = True
        counter.update(hand, states(*lifted), 0.0)
        assert not counter.update(hand, FIST, 0.1)


def test_isolated_finger_lift_requires_the_other_fingers_down():
    counter = FingerLiftCounter("ring", isolated=True)
    hand = apart()
    assert not counter.update(hand, states(False, False, True, True, False), 0.0)
    assert counter.cue().key == "exercise.cue.lift_only"
    assert counter.update(hand, states(False, False, False, True, False), 0.1)
    assert not counter.update(hand, states(False, False, False, True, False), 0.2)
    counter.update(hand, FIST, 0.3)
    assert counter.update(hand, states(True, False, False, True, False), 0.4)  # Thumb is free.


def test_non_isolated_finger_lift_accepts_neighbours():
    counter = FingerLiftCounter("ring")
    assert counter.update(apart(), states(False, False, True, True, False), 0.0)


def test_thumb_opposition_needs_every_finger_in_order():
    counter = ThumbOppositionCounter()
    reps = []
    for finger in ("index", "middle", "ring", "pinky"):
        reps.append(counter.update(touching(finger), OPEN, 0.0))
        assert counter.cue().key == "exercise.cue.release"
        counter.update(apart(), OPEN, 0.1)
    assert reps == [False, False, False, True]
    assert counter.cue().finger == "index"


def test_thumb_opposition_ignores_the_wrong_finger():
    counter = ThumbOppositionCounter()
    assert not counter.update(touching("ring"), OPEN, 0.0)
    assert counter.cue().finger == "index"


# ------------------------------------------------------------------- runner
def test_runner_records_windows_with_titles_and_parameters():
    spec = ExerciseSpec("custom:a", Movement.PINCH, title="Pinça", finger="middle", target_reps=2)
    runner = ExerciseRunner((spec, BUILTIN_SPECS["open_close"]), started_at=100.0)
    moment = 100.0
    for _ in range(2):
        for hand in (touching("middle"), apart()):
            moment += 0.5
            runner.feed(hand, 0.9, moment)
    assert runner.progress.completed
    assert runner.cue().key == "exercise.cue.done"
    runner.next_exercise()
    runner.feed(None, 0.0, moment + 1)
    assert runner.cue().key == "exercise.cue.show_hand"
    runner.close()
    [window] = runner.windows()
    assert window.name == "custom:a" and window.reps == 2 and window.target == 2
    assert window.title == "Pinça"
    assert window.details == {
        "movement": "pinch",
        "finger": "middle",
        "tolerance": "normal",
        "hold_seconds": 0.0,
    }
    assert runner.recorder.frames_total == 5


def test_runner_cooldown_prevents_a_double_count_on_the_same_contact():
    runner = ExerciseRunner((BUILTIN_SPECS["finger_touch"],), started_at=0.0)
    assert runner.feed(pinching_hand(0.0), 0.9, 0.0)
    assert not runner.feed(make_hand(thumb=True, index=True), 0.9, 0.05)
    assert not runner.feed(pinching_hand(0.0), 0.9, 0.1)
    assert runner.progress.reps == 1


def test_runner_requires_at_least_one_exercise():
    with pytest.raises(ValueError):
        ExerciseRunner(())
