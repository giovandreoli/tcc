from __future__ import annotations

import pytest

from spectra.gestures.features import FingerStates
from spectra.gestures.profiles import (
    PRESETS,
    SIMPLIFIED_PROFILE,
    STANDARD_PROFILE,
    GestureAction,
    GestureProfile,
    HandSignal,
    Severity,
    Trigger,
    all_patterns,
    build_custom_profile,
    get_preset,
    pattern,
    validate_profile,
)


def signal(*flags: bool, pinching: bool = False) -> HandSignal:
    return HandSignal(present=True, fingers=FingerStates(*flags), pinching=pinching)


class TestTrigger:
    def test_a_trigger_must_constrain_something(self):
        with pytest.raises(ValueError):
            Trigger()

    def test_finger_trigger_matches_the_exact_pattern(self):
        trigger = Trigger(fingers=pattern(index=True))
        assert trigger.matches(signal(False, True, False, False, False))
        assert not trigger.matches(signal(False, True, True, False, False))

    def test_absent_hands_never_match(self):
        assert not Trigger(fingers=pattern()).matches(HandSignal.absent())

    def test_pinch_only_trigger_ignores_the_finger_pattern(self):
        trigger = Trigger(pinch=True)
        assert trigger.matches(signal(True, True, False, False, False, pinching=True))
        assert trigger.matches(signal(True, True, True, True, True, pinching=True))
        assert not trigger.matches(signal(True, True, False, False, False))

    def test_combined_triggers_are_more_specific(self):
        assert (
            Trigger(fingers=pattern(index=True), pinch=False).specificity
            > Trigger(fingers=pattern(index=True)).specificity
        )

    def test_describe_is_human_readable(self):
        assert Trigger(fingers=pattern(index=True, middle=True)).describe() == "index+middle"
        assert Trigger(fingers=pattern()).describe() == "fist"
        assert Trigger(pinch=True).describe() == "pinch"

    def test_round_trip_through_dict(self):
        trigger = Trigger(fingers=pattern(index=True), pinch=False)
        assert Trigger.from_dict(trigger.to_dict()) == trigger


class TestStandardProfile:
    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ((False, True, False, False, False), GestureAction.POINT),
            ((False, True, True, False, False), GestureAction.PAINT),
            ((False, True, True, True, False), GestureAction.ERASE),
            ((True, True, True, True, True), GestureAction.COLOR_MENU),
            ((False, False, False, False, False), GestureAction.PAUSE),
        ],
    )
    def test_default_bindings(self, flags, expected):
        assert STANDARD_PROFILE.action_for(signal(*flags)) is expected

    def test_unbound_pattern_yields_no_action(self):
        assert STANDARD_PROFILE.action_for(signal(False, False, False, True, False)) is None

    def test_the_standard_profile_is_clean(self):
        assert validate_profile(STANDARD_PROFILE) == []


class TestSimplifiedProfile:
    def test_pinch_paints_regardless_of_the_finger_pattern(self):
        assert (
            SIMPLIFIED_PROFILE.action_for(signal(True, True, True, False, False, pinching=True))
            is GestureAction.PAINT
        )

    def test_pointing_requires_no_pinch(self):
        assert (
            SIMPLIFIED_PROFILE.action_for(signal(False, True, False, False, False))
            is GestureAction.POINT
        )

    def test_the_more_specific_trigger_wins_when_both_match(self):
        # index-only while pinching matches both POINT (fingers) and PAINT (pinch)
        both = signal(False, True, False, False, False, pinching=True)
        assert SIMPLIFIED_PROFILE.action_for(both) is GestureAction.PAINT

    def test_it_has_a_longer_hold_time_for_low_dexterity(self):
        assert SIMPLIFIED_PROFILE.hold_seconds > STANDARD_PROFILE.hold_seconds

    def test_the_simplified_profile_has_no_errors(self):
        assert not any(issue.is_error for issue in validate_profile(SIMPLIFIED_PROFILE))


class TestValidation:
    def test_missing_action_is_an_error(self):
        profile = GestureProfile(
            "partial", {GestureAction.POINT: Trigger(fingers=pattern(index=True))}
        )
        codes = {issue.code for issue in validate_profile(profile) if issue.is_error}
        assert codes == {"missing_action"}

    def test_duplicate_triggers_are_an_error(self):
        same = Trigger(fingers=pattern(index=True))
        profile = GestureProfile(
            "clash",
            {
                GestureAction.POINT: same,
                GestureAction.PAINT: same,
                GestureAction.ERASE: Trigger(fingers=pattern(index=True, middle=True, ring=True)),
                GestureAction.COLOR_MENU: Trigger(fingers=pattern(True, True, True, True, True)),
                GestureAction.PAUSE: Trigger(fingers=pattern()),
            },
        )
        issues = [i for i in validate_profile(profile) if i.code == "duplicate_trigger"]
        assert len(issues) == 1
        assert set(issues[0].actions) == {GestureAction.POINT, GestureAction.PAINT}

    def test_patterns_one_finger_apart_raise_a_warning(self):
        _, issues = build_custom_profile(
            {
                GestureAction.POINT: Trigger(fingers=pattern(middle=True, ring=True)),
                GestureAction.PAINT: Trigger(fingers=pattern(middle=True, ring=True, pinky=True)),
            }
        )
        warnings = [i for i in issues if i.code == "ambiguous_patterns"]
        assert warnings
        assert all(i.severity is Severity.WARNING for i in warnings)

    def test_the_deliberate_finger_ladder_is_not_flagged(self):
        """1, 2 and 3 fingers differ by one finger on purpose; that is the design."""
        issues = validate_profile(STANDARD_PROFILE)
        assert not [i for i in issues if i.code == "ambiguous_patterns"]

    @pytest.mark.parametrize(
        ("fingers", "code"),
        [
            (pattern(ring=True), "ring_without_middle"),
            (pattern(ring=True, pinky=True), "ring_without_middle"),
            (pattern(thumb=True), "thumb_dependent"),
            (pattern(thumb=True, pinky=True), "thumb_dependent"),
        ],
    )
    def test_anatomically_hard_patterns_are_flagged(self, fingers, code):
        _, issues = build_custom_profile({GestureAction.PAINT: Trigger(fingers=fingers)})
        assert code in {issue.code for issue in issues}

    def test_middle_and_ring_together_is_acceptable(self):
        _, issues = build_custom_profile(
            {GestureAction.PAINT: Trigger(fingers=pattern(middle=True, ring=True))}
        )
        assert not any(issue.code.startswith("ring") for issue in issues)

    def test_every_issue_exposes_an_i18n_key(self):
        _, issues = build_custom_profile({GestureAction.PAINT: Trigger(fingers=pattern(ring=True))})
        assert all(issue.message_key.startswith("gesture.issue.") for issue in issues)


class TestPresetsAndSerialisation:
    def test_known_presets(self):
        assert set(PRESETS) == {"standard", "simplified"}

    def test_unknown_preset_falls_back_to_standard(self):
        assert get_preset("nope") is STANDARD_PROFILE

    def test_profile_round_trip_through_dict(self):
        restored = GestureProfile.from_dict(STANDARD_PROFILE.to_dict())
        assert restored.bindings == STANDARD_PROFILE.bindings
        assert restored.hold_seconds == STANDARD_PROFILE.hold_seconds

    def test_all_patterns_enumerates_the_32_combinations(self):
        patterns = list(all_patterns())
        assert len(patterns) == 32
        assert len(set(patterns)) == 32
