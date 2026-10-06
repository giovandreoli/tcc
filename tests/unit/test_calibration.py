from __future__ import annotations

import pytest

from spectra.gestures.calibration import (
    DEFAULT_FINGER,
    DEFAULT_THUMB,
    CalibrationProfile,
    FingerStateEstimator,
    Hysteresis,
    build_profile,
    finger_signals,
    finger_states,
)
from tests.conftest import make_hand, rotate_hand

PATTERNS = [
    (False, False, False, False, False),
    (False, True, False, False, False),
    (False, True, True, False, False),
    (False, False, False, True, True),
    (True, False, False, False, False),
    (True, True, True, True, True),
]

OPEN_HAND = (True, True, True, True, True)
FIST = (False, False, False, False, False)


class TestHysteresis:
    def test_on_threshold_must_exceed_off(self):
        with pytest.raises(ValueError):
            Hysteresis(on=10.0, off=20.0)

    def test_rising_edge_needs_the_on_threshold(self):
        band = Hysteresis(on=150.0, off=120.0)
        assert band.apply(130.0, previously_on=False) is False
        assert band.apply(150.0, previously_on=False) is True

    def test_falling_edge_uses_the_off_threshold(self):
        band = Hysteresis(on=150.0, off=120.0)
        assert band.apply(130.0, previously_on=True) is True
        assert band.apply(119.0, previously_on=True) is False

    def test_the_dead_band_prevents_flicker(self):
        band = Hysteresis(on=150.0, off=120.0)
        borderline = 135.0
        assert band.apply(borderline, previously_on=False) is False
        assert band.apply(borderline, previously_on=True) is True


class TestStatelessReading:
    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("label", ["Right", "Left"])
    def test_recovers_the_synthetic_pattern(self, pattern, label):
        assert tuple(finger_states(make_hand(*pattern, label=label), label)) == pattern

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_survives_a_tilted_hand(self, pattern):
        tilted = rotate_hand(make_hand(*pattern), 45)
        assert tuple(finger_states(tilted)) == pattern


class TestCalibration:
    def test_defaults_are_used_without_samples(self):
        profile = build_profile([], [])
        assert profile.thresholds[0] == DEFAULT_THUMB
        assert profile.is_calibrated is False

    def test_thresholds_sit_between_the_two_postures(self):
        profile = build_profile([make_hand(*OPEN_HAND)] * 5, [make_hand(*FIST)] * 5)
        open_signals = finger_signals(make_hand(*OPEN_HAND))
        closed_signals = finger_signals(make_hand(*FIST))
        for band, low, high in zip(profile.thresholds, closed_signals, open_signals, strict=True):
            assert low < band.off < band.on < high

    def test_calibrated_thresholds_still_classify_both_postures(self):
        profile = build_profile([make_hand(*OPEN_HAND)] * 3, [make_hand(*FIST)] * 3)
        assert tuple(finger_states(make_hand(*OPEN_HAND), profile=profile)) == OPEN_HAND
        assert tuple(finger_states(make_hand(*FIST), profile=profile)) == FIST

    def test_identical_postures_fall_back_to_defaults(self):
        profile = build_profile([make_hand(*FIST)] * 3, [make_hand(*FIST)] * 3)
        assert set(profile.uncalibrated) == {"thumb", "index", "middle", "ring", "pinky"}
        assert profile.thresholds[1] == DEFAULT_FINGER

    def test_round_trip_through_dict(self):
        profile = build_profile([make_hand(*OPEN_HAND)], [make_hand(*FIST)])
        assert CalibrationProfile.from_dict(profile.to_dict()) == profile


class TestFingerStateEstimator:
    def test_majority_vote_absorbs_a_single_bad_frame(self):
        estimator = FingerStateEstimator(smoothing_frames=3)
        pointing = make_hand(index=True)
        for _ in range(3):
            estimator.update(pointing)
        assert estimator.states.index is True
        estimator.update(make_hand())  # one dropped frame
        assert estimator.states.index is True

    def test_a_sustained_change_is_accepted(self):
        estimator = FingerStateEstimator(smoothing_frames=3)
        for _ in range(3):
            estimator.update(make_hand(index=True))
        for _ in range(3):
            estimator.update(make_hand())
        assert estimator.states.index is False

    def test_reset_clears_the_history(self):
        estimator = FingerStateEstimator()
        for _ in range(3):
            estimator.update(make_hand(index=True))
        estimator.reset()
        assert estimator.states.extended_count == 0

    def test_estimator_tracks_the_full_open_close_cycle(self):
        estimator = FingerStateEstimator(smoothing_frames=1)
        assert tuple(estimator.update(make_hand(*OPEN_HAND))) == OPEN_HAND
        assert tuple(estimator.update(make_hand(*FIST))) == FIST
