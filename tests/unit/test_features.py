from __future__ import annotations

import math

import pytest

from spectra.detection.landmarks import Landmark
from spectra.gestures.features import (
    finger_extension_angles,
    finger_flexion_angles,
    finger_states_from_tips,
    joint_angle,
    opposition_ratios,
    palm_size,
    pinch_ratio,
    pointer_position,
    thumb_abduction_ratio,
)
from tests.conftest import make_hand, pinching_hand, rotate_hand

PATTERNS = [
    (False, False, False, False, False),
    (False, True, False, False, False),
    (False, True, True, False, False),
    (False, False, False, True, True),
    (True, False, False, False, False),
    (True, True, True, True, True),
]


class TestJointAngle:
    def test_collinear_points_measure_180_degrees(self):
        a, b, c = Landmark(0, 0), Landmark(1, 0), Landmark(2, 0)
        assert joint_angle(a, b, c) == pytest.approx(180.0)

    def test_right_angle(self):
        a, b, c = Landmark(0, 0), Landmark(1, 0), Landmark(1, 1)
        assert joint_angle(a, b, c) == pytest.approx(90.0)

    def test_folded_back_measures_zero(self):
        a, b, c = Landmark(0, 0), Landmark(1, 0), Landmark(0, 0)
        assert joint_angle(a, b, c) == pytest.approx(0.0)

    def test_degenerate_points_default_to_straight(self):
        a = b = c = Landmark(0.5, 0.5)
        assert joint_angle(a, b, c) == 180.0


class TestExtensionAngles:
    def test_extended_fingers_are_near_straight(self):
        angles = finger_extension_angles(make_hand(index=True, middle=True))
        assert angles[1] > 170
        assert angles[2] > 170

    def test_curled_fingers_are_clearly_flexed(self):
        angles = finger_extension_angles(make_hand())
        assert all(angle < 140 for angle in angles[1:])

    def test_flexion_is_the_complement_of_extension(self):
        landmarks = make_hand(index=True)
        extension = finger_extension_angles(landmarks)
        flexion = finger_flexion_angles(landmarks)
        assert all(e + f == pytest.approx(180.0) for e, f in zip(extension, flexion, strict=True))

    def test_angles_are_invariant_to_rotation(self):
        upright = finger_extension_angles(make_hand(index=True, middle=True))
        tilted = finger_extension_angles(rotate_hand(make_hand(index=True, middle=True), 40))
        assert all(abs(a - b) < 1.0 for a, b in zip(upright, tilted, strict=True))

    def test_angles_are_invariant_to_scale(self):
        hand = make_hand(index=True)
        small = [Landmark(0.5 + (p.x - 0.5) * 0.4, 0.5 + (p.y - 0.5) * 0.4) for p in hand]
        assert finger_extension_angles(hand)[1] == pytest.approx(
            finger_extension_angles(small)[1], abs=0.5
        )


class TestThumbAndPinch:
    def test_abduction_separates_open_from_closed_thumb(self):
        assert thumb_abduction_ratio(make_hand(thumb=True)) > 0.6
        assert thumb_abduction_ratio(make_hand(thumb=False)) < 0.4

    def test_pinch_ratio_is_small_when_tips_touch(self):
        assert pinch_ratio(pinching_hand(distance=0.005)) < 0.1

    def test_pinch_ratio_is_large_when_the_hand_is_open(self):
        assert pinch_ratio(make_hand(thumb=True, index=True)) > 0.8

    def test_opposition_returns_one_ratio_per_finger(self):
        assert len(opposition_ratios(make_hand(thumb=True))) == 4


class TestPalmSize:
    def test_palm_size_is_positive(self):
        assert palm_size(make_hand()) > 0

    def test_palm_size_is_never_zero(self):
        flat = [Landmark(0.5, 0.5) for _ in range(21)]
        assert palm_size(flat) > 0


class TestPointer:
    def test_pointer_is_none_when_the_index_is_curled(self):
        assert pointer_position(make_hand(middle=True), (720, 1280)) is None

    def test_pointer_is_scaled_to_pixels(self):
        landmarks = make_hand(index=True)
        assert pointer_position(landmarks, (720, 1280)) == (
            int(landmarks[8].x * 1280),
            int(landmarks[8].y * 720),
        )

    def test_pointer_survives_a_tilted_hand(self):
        assert pointer_position(rotate_hand(make_hand(index=True), 35), (720, 1280)) is not None


class TestLegacyBaseline:
    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_baseline_still_reads_upright_hands(self, pattern):
        assert tuple(finger_states_from_tips(make_hand(*pattern))) == pattern

    def test_baseline_breaks_on_a_rotated_hand(self):
        """Documents exactly why the estimator moved to joint angles."""
        upright = make_hand(index=True, middle=True)
        readings = {
            tuple(finger_states_from_tips(rotate_hand(upright, angle)))
            for angle in (0, 45, 90, 135)
        }
        assert len(readings) > 1


def test_rotation_helper_preserves_distances():
    hand = make_hand(index=True)
    rotated = rotate_hand(hand, 90)
    before = math.hypot(hand[0].x - hand[8].x, hand[0].y - hand[8].y)
    after = math.hypot(rotated[0].x - rotated[8].x, rotated[0].y - rotated[8].y)
    assert before == pytest.approx(after)
