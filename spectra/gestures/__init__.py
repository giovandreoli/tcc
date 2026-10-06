"""Gesture engine: features, calibration, profiles and the state machine."""

from spectra.gestures.calibration import (
    CalibrationProfile,
    FingerStateEstimator,
    Hysteresis,
    build_profile,
    finger_states,
)
from spectra.gestures.features import (
    FINGER_NAMES,
    FingerStates,
    extended_count,
    finger_extension_angles,
    finger_flexion_angles,
    pointer_position,
)
from spectra.gestures.profiles import (
    PRESETS,
    SIMPLIFIED_PROFILE,
    STANDARD_PROFILE,
    GestureAction,
    GestureProfile,
    HandSignal,
    ProfileIssue,
    Severity,
    Trigger,
    get_preset,
    validate_profile,
)
from spectra.gestures.state_machine import GestureEngine, GestureState, GestureStateMachine

__all__ = [
    "FINGER_NAMES",
    "PRESETS",
    "SIMPLIFIED_PROFILE",
    "STANDARD_PROFILE",
    "CalibrationProfile",
    "FingerStateEstimator",
    "FingerStates",
    "GestureAction",
    "GestureEngine",
    "GestureProfile",
    "GestureState",
    "GestureStateMachine",
    "HandSignal",
    "Hysteresis",
    "ProfileIssue",
    "Severity",
    "Trigger",
    "build_profile",
    "extended_count",
    "finger_extension_angles",
    "finger_flexion_angles",
    "finger_states",
    "get_preset",
    "pointer_position",
    "validate_profile",
]
