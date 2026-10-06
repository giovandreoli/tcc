from __future__ import annotations

from spectra.gestures.features import FingerStates
from spectra.modes.physio import (
    ExerciseProgress,
    FingerWaveDetector,
    OpenCloseDetector,
    PhysioExercise,
    PinchDetector,
)
from tests.conftest import make_hand, pinching_hand

OPEN = FingerStates(True, True, True, True, True)
FIST = FingerStates(False, False, False, False, False)
FRAME = (720, 1280)


class TestOpenCloseDetector:
    def test_counts_one_rep_per_open_close_cycle(self):
        detector = OpenCloseDetector()
        assert detector.update(OPEN) is False
        assert detector.update(FIST) is True
        assert detector.update(OPEN) is False
        assert detector.update(FIST) is True

    def test_holding_the_hand_open_does_not_count(self):
        detector = OpenCloseDetector()
        assert [detector.update(OPEN) for _ in range(5)] == [False] * 5

    def test_reset_returns_to_the_open_phase(self):
        detector = OpenCloseDetector()
        detector.update(OPEN)
        detector.reset()
        assert detector.phase == "open"
        assert detector.update(FIST) is False


class TestPinchDetector:
    def test_counts_on_the_rising_edge_only(self):
        detector = PinchDetector()
        closed = pinching_hand(distance=0.001)
        assert detector.update(closed, FRAME) is True
        assert detector.update(closed, FRAME) is False

    def test_releasing_allows_the_next_rep(self):
        detector = PinchDetector()
        closed = pinching_hand(distance=0.001)
        far = make_hand(thumb=True, index=True)
        detector.update(closed, FRAME)
        detector.update(far, FRAME)
        assert detector.update(closed, FRAME) is True

    def test_open_hand_is_not_a_pinch(self):
        assert PinchDetector().update(make_hand(thumb=True, index=True), FRAME) is False


class TestFingerWaveDetector:
    def test_full_sequence_counts_one_rep(self):
        detector = FingerWaveDetector()
        results = []
        for finger in (1, 2, 3, 4):
            flags = [False] * 5
            flags[finger] = True
            results.append(detector.update(FingerStates(*flags)))
            results.append(detector.update(FIST))
        assert results[-1] is True
        assert sum(results) == 1

    def test_out_of_order_fingers_do_not_advance(self):
        detector = FingerWaveDetector()
        detector.update(FingerStates(False, False, False, False, True))
        assert detector.step == 0


class TestExerciseProgress:
    def test_completion_follows_the_target(self):
        progress = ExerciseProgress(PhysioExercise.OPEN_CLOSE, target=2)
        progress.add_rep()
        assert progress.completed is False
        progress.add_rep()
        assert progress.completed is True

    def test_pace_needs_at_least_two_reps(self):
        progress = ExerciseProgress(PhysioExercise.OPEN_CLOSE)
        progress.add_rep(now=0.0)
        assert progress.pace_rpm(now=1.0) is None

    def test_pace_is_reps_per_minute(self):
        progress = ExerciseProgress(PhysioExercise.OPEN_CLOSE)
        for i in range(5):
            progress.add_rep(now=float(i))
        assert progress.pace_rpm(now=4.0) == 75.0

    def test_pace_ignores_reps_outside_the_window(self):
        progress = ExerciseProgress(PhysioExercise.OPEN_CLOSE)
        progress.add_rep(now=0.0)
        progress.add_rep(now=1.0)
        assert progress.pace_rpm(now=100.0) is None


def test_exercise_titles_and_instructions_are_translated():
    exercise = PhysioExercise.OPEN_CLOSE
    assert exercise.title == "Abrir e Fechar a Mão"
    assert len(exercise.instructions) == 3
    assert all(not line.startswith("physio.") for line in exercise.instructions)
