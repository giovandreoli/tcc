"""Data-driven mapping from finger patterns to gesture actions.

The original SPECTRA hard-coded "finger combination equals colour". Here the mapping is
a *profile*: a table of ``action -> trigger`` that a therapist can swap or customise per
patient. Any of the 32 finger combinations may be used, optionally combined with a pinch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from spectra.gestures.features import FINGER_NAMES, FingerStates

FingerPattern = tuple[bool, bool, bool, bool, bool]

NO_FINGERS = FingerStates(False, False, False, False, False)


class GestureAction(Enum):
    """Actions a patient can trigger; values double as i18n keys under ``gesture.action.``."""

    POINT = "point"
    PAINT = "paint"
    ERASE = "erase"
    COLOR_MENU = "color_menu"
    PAUSE = "pause"


@dataclass(frozen=True)
class HandSignal:
    """Everything the gesture engine needs to know about the current frame."""

    present: bool = False
    fingers: FingerStates = NO_FINGERS
    pinching: bool = False
    confidence: float = 0.0

    @classmethod
    def absent(cls) -> HandSignal:
        return cls(present=False)


@dataclass(frozen=True)
class Trigger:
    """A gesture condition: a finger pattern, a pinch, or both."""

    fingers: FingerPattern | None = None
    pinch: bool | None = None

    def __post_init__(self) -> None:
        if self.fingers is None and self.pinch is None:
            raise ValueError("a trigger must constrain the fingers, the pinch, or both")

    @property
    def specificity(self) -> int:
        """Higher means more constrained; used to resolve overlapping triggers."""
        return (0 if self.fingers is None else 5) + (0 if self.pinch is None else 3)

    def matches(self, signal: HandSignal) -> bool:
        if not signal.present:
            return False
        if self.pinch is not None and signal.pinching != self.pinch:
            return False
        return self.fingers is None or tuple(signal.fingers) == self.fingers

    def describe(self) -> str:
        """Short machine-readable description, e.g. ``index+middle`` or ``pinch``."""
        parts = []
        if self.pinch:
            parts.append("pinch")
        if self.fingers is not None:
            names = [name for name, up in zip(FINGER_NAMES, self.fingers, strict=True) if up]
            parts.append("+".join(names) if names else "fist")
        return " & ".join(parts)

    def to_dict(self) -> dict:
        return {"fingers": list(self.fingers) if self.fingers else None, "pinch": self.pinch}

    @classmethod
    def from_dict(cls, data: Mapping) -> Trigger:
        fingers = data.get("fingers")
        return cls(
            fingers=tuple(bool(flag) for flag in fingers) if fingers else None,
            pinch=data.get("pinch"),
        )


def pattern(thumb=False, index=False, middle=False, ring=False, pinky=False) -> FingerPattern:
    return (thumb, index, middle, ring, pinky)


@dataclass(frozen=True)
class GestureProfile:
    """A complete action-to-trigger mapping plus its timing parameters."""

    name: str
    bindings: Mapping[GestureAction, Trigger]
    #: How long a gesture must be held before the state machine accepts it.
    hold_seconds: float = 0.4
    #: Grace period before a hand that disappeared is treated as gone.
    release_seconds: float = 0.15

    def trigger_for(self, action: GestureAction) -> Trigger | None:
        return self.bindings.get(action)

    def action_for(self, signal: HandSignal) -> GestureAction | None:
        """Most specific action matching ``signal``, or ``None``."""
        matches = [
            (trigger.specificity, action)
            for action, trigger in self.bindings.items()
            if trigger.matches(signal)
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: item[0])[1]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hold_seconds": self.hold_seconds,
            "release_seconds": self.release_seconds,
            "bindings": {
                action.value: trigger.to_dict() for action, trigger in self.bindings.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> GestureProfile:
        bindings = {
            GestureAction(key): Trigger.from_dict(value)
            for key, value in data.get("bindings", {}).items()
        }
        return cls(
            name=data.get("name", "custom"),
            bindings=bindings,
            hold_seconds=float(data.get("hold_seconds", 0.4)),
            release_seconds=float(data.get("release_seconds", 0.15)),
        )


STANDARD_PROFILE = GestureProfile(
    name="standard",
    bindings={
        GestureAction.POINT: Trigger(fingers=pattern(index=True)),
        GestureAction.PAINT: Trigger(fingers=pattern(index=True, middle=True)),
        GestureAction.ERASE: Trigger(fingers=pattern(index=True, middle=True, ring=True)),
        GestureAction.COLOR_MENU: Trigger(
            fingers=pattern(thumb=True, index=True, middle=True, ring=True, pinky=True)
        ),
        GestureAction.PAUSE: Trigger(fingers=pattern()),
    },
)

#: For low dexterity: painting is a pinch, which needs no independent finger control.
SIMPLIFIED_PROFILE = GestureProfile(
    name="simplified",
    bindings={
        GestureAction.POINT: Trigger(fingers=pattern(index=True), pinch=False),
        GestureAction.PAINT: Trigger(pinch=True),
        GestureAction.ERASE: Trigger(fingers=pattern(index=True, middle=True), pinch=False),
        GestureAction.COLOR_MENU: Trigger(
            fingers=pattern(thumb=True, index=True, middle=True, ring=True, pinky=True),
            pinch=False,
        ),
        GestureAction.PAUSE: Trigger(fingers=pattern(), pinch=False),
    },
    hold_seconds=0.6,
)

PRESETS: dict[str, GestureProfile] = {
    STANDARD_PROFILE.name: STANDARD_PROFILE,
    SIMPLIFIED_PROFILE.name: SIMPLIFIED_PROFILE,
}


def get_preset(name: str) -> GestureProfile:
    """Return a preset by name, defaulting to ``standard``."""
    return PRESETS.get(name, STANDARD_PROFILE)


# ------------------------------------------------------------------- validation
class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ProfileIssue:
    """A problem found in a gesture profile, ready to be shown in the panel."""

    severity: Severity
    code: str
    actions: tuple[GestureAction, ...] = ()
    detail: str = ""

    @property
    def message_key(self) -> str:
        return f"gesture.issue.{self.code}"

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR


def _hard_pattern_code(fingers: FingerPattern) -> str | None:
    """Flag combinations that are anatomically hard or easily confused."""
    thumb, _index, middle, ring, pinky = fingers
    extended = sum(fingers)
    if ring and not middle:
        # The ring finger shares tendons with the middle finger; isolating it is hard.
        return "ring_without_middle"
    if ring and pinky and extended == 2:
        return "ring_and_pinky"
    if thumb and extended <= 2:
        # Thumb extension is the least reliable signal from a 2D webcam.
        return "thumb_dependent"
    return None


def _differ_by_one(a: FingerPattern, b: FingerPattern) -> bool:
    return sum(1 for x, y in zip(a, b, strict=True) if x != y) == 1


#: Deliberate "count the fingers" ladder: fist, index, +middle, +ring, +pinky, open hand.
#: Neighbours in this ladder differ by one finger on purpose, so they are not flagged.
PROGRESSIVE_PATTERNS: frozenset[FingerPattern] = frozenset(
    {
        pattern(),
        pattern(index=True),
        pattern(index=True, middle=True),
        pattern(index=True, middle=True, ring=True),
        pattern(index=True, middle=True, ring=True, pinky=True),
        pattern(thumb=True, index=True, middle=True, ring=True, pinky=True),
    }
)


def _is_deliberate_ladder(a: FingerPattern, b: FingerPattern) -> bool:
    return a in PROGRESSIVE_PATTERNS and b in PROGRESSIVE_PATTERNS


def validate_profile(profile: GestureProfile) -> list[ProfileIssue]:
    """Check a (possibly custom) profile for conflicts and risky patterns."""
    issues: list[ProfileIssue] = []

    missing = [action for action in GestureAction if action not in profile.bindings]
    issues.extend(ProfileIssue(Severity.ERROR, "missing_action", (action,)) for action in missing)

    bound = list(profile.bindings.items())
    for i, (action_a, trigger_a) in enumerate(bound):
        for action_b, trigger_b in bound[i + 1 :]:
            if trigger_a == trigger_b:
                issues.append(
                    ProfileIssue(
                        Severity.ERROR,
                        "duplicate_trigger",
                        (action_a, action_b),
                        trigger_a.describe(),
                    )
                )
            elif (
                trigger_a.fingers is not None
                and trigger_b.fingers is not None
                and trigger_a.pinch == trigger_b.pinch
                and _differ_by_one(trigger_a.fingers, trigger_b.fingers)
                and not _is_deliberate_ladder(trigger_a.fingers, trigger_b.fingers)
            ):
                issues.append(
                    ProfileIssue(
                        Severity.WARNING,
                        "ambiguous_patterns",
                        (action_a, action_b),
                        f"{trigger_a.describe()} / {trigger_b.describe()}",
                    )
                )

    for action, trigger in bound:
        if trigger.fingers is None:
            continue
        code = _hard_pattern_code(trigger.fingers)
        if code:
            issues.append(ProfileIssue(Severity.WARNING, code, (action,), trigger.describe()))

    return issues


def build_custom_profile(
    bindings: Mapping[GestureAction, Trigger],
    name: str = "custom",
    hold_seconds: float = 0.4,
) -> tuple[GestureProfile, list[ProfileIssue]]:
    """Create a custom profile and return it together with its validation issues."""
    profile = GestureProfile(name=name, bindings=dict(bindings), hold_seconds=hold_seconds)
    return profile, validate_profile(profile)


def all_patterns() -> Iterable[FingerPattern]:
    """The 32 finger combinations, for the profile editor in the therapist panel."""
    for mask in range(32):
        yield tuple(bool(mask & (1 << i)) for i in range(5))  # type: ignore[misc]


__all__ = [
    "PRESETS",
    "SIMPLIFIED_PROFILE",
    "STANDARD_PROFILE",
    "FingerPattern",
    "GestureAction",
    "GestureProfile",
    "HandSignal",
    "ProfileIssue",
    "Severity",
    "Trigger",
    "all_patterns",
    "build_custom_profile",
    "get_preset",
    "pattern",
    "validate_profile",
]
