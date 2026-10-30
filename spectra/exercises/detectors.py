"""Repetition detectors for the validated movements.

All detectors are pure and receive explicit timestamps, so they behave identically on
frames analysed by the browser and in the unit tests. Contacts use thumb-to-fingertip
distances **normalised by the palm size**, which makes them independent of how far the
patient sits from the camera and of the video resolution. Every contact has hysteresis:
it must close below ``on`` and open above ``off`` before the next repetition can count,
so landmark jitter around a single threshold never produces double counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from spectra.detection.landmarks import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    HandLandmarks,
)
from spectra.exercises.catalog import ExerciseSpec, Movement
from spectra.gestures.features import FingerStates, distance, palm_size

OPEN_HAND_FINGERS = 4
CLOSED_HAND_MAX_EXTENDED = 1

#: (on, off) palm-normalised thumb-to-fingertip distances per tolerance.
CONTACT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "wide": (0.45, 0.65),
    "normal": (0.35, 0.55),
    "strict": (0.25, 0.45),
}

FINGER_INDEX = {"index": 1, "middle": 2, "ring": 3, "pinky": 4}
FINGER_TIP = {"index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP, "pinky": PINKY_TIP}
FINGER_ORDER = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class Cue:
    """What the patient should do next, as an i18n key plus an optional finger."""

    key: str
    finger: str | None = None


def contact_ratio(landmarks: HandLandmarks, finger: str) -> float:
    return distance(landmarks[THUMB_TIP], landmarks[FINGER_TIP[finger]]) / palm_size(landmarks)


def in_contact(landmarks: HandLandmarks, finger: str, threshold: float) -> bool:
    """Thumb within ``threshold`` of ``finger`` **and** closer to it than to any other tip.

    Adjacent fingertips sit about a quarter of a palm apart, inside any useful threshold, so
    distance alone would count a touch on the neighbouring finger. Requiring the target to be
    the nearest fingertip makes finger-specific exercises discriminative.
    """
    ratios = {name: contact_ratio(landmarks, name) for name in FINGER_ORDER}
    target = ratios[finger]
    return target <= threshold and all(target <= value for value in ratios.values())


class HoldGate:
    """True once a condition has stayed true for ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds = max(0.0, seconds)
        self.since: float | None = None

    def update(self, active: bool, at: float) -> bool:
        if not active:
            self.since = None
            return False
        if self.since is None:
            self.since = at
        return at - self.since + 1e-9 >= self.seconds

    def fraction(self, at: float) -> float:
        if self.since is None:
            return 0.0
        if self.seconds <= 0:
            return 1.0
        return min(1.0, (at - self.since) / self.seconds)

    def reset(self) -> None:
        self.since = None


class Detector(Protocol):
    def update(self, landmarks: HandLandmarks, states: FingerStates, at: float) -> bool: ...

    def cue(self) -> Cue: ...

    def hold_fraction(self, at: float) -> float: ...

    def hand_lost(self) -> None: ...

    def reset(self) -> None: ...


class OpenCloseCounter:
    """One repetition per open hand followed by a (optionally held) closed fist."""

    def __init__(self, hold_seconds: float = 0.0) -> None:
        self.gate = HoldGate(hold_seconds)
        self.phase = "open"

    def update(self, landmarks: HandLandmarks, states: FingerStates, at: float) -> bool:
        count = states.extended_count
        if self.phase == "open":
            if count >= OPEN_HAND_FINGERS:
                self.phase = "close"
            return False
        if self.gate.update(count <= CLOSED_HAND_MAX_EXTENDED, at):
            self.gate.reset()
            self.phase = "open"
            return True
        return False

    def cue(self) -> Cue:
        if self.phase == "open":
            return Cue("exercise.cue.open_hand")
        return Cue(
            "exercise.cue.hold" if self.gate.since is not None else "exercise.cue.close_hand"
        )

    def hold_fraction(self, at: float) -> float:
        return self.gate.fraction(at) if self.gate.seconds else 0.0

    def hand_lost(self) -> None:
        self.gate.reset()

    def reset(self) -> None:
        self.gate.reset()
        self.phase = "open"


class ContactCounter:
    """One repetition per thumb contact with ``finger``, released before the next."""

    def __init__(self, finger: str, tolerance: str = "normal", hold_seconds: float = 0.0):
        self.finger = finger
        self.on, self.off = CONTACT_THRESHOLDS[tolerance]
        self.gate = HoldGate(hold_seconds)
        self.phase = "reach"

    def update(self, landmarks: HandLandmarks, states: FingerStates, at: float) -> bool:
        if self.phase == "release":
            if contact_ratio(landmarks, self.finger) > self.off:
                self.phase = "reach"
            return False
        if self.gate.update(in_contact(landmarks, self.finger, self.on), at):
            self.gate.reset()
            self.phase = "release"
            return True
        return False

    def cue(self) -> Cue:
        if self.phase == "release":
            return Cue("exercise.cue.release")
        if self.gate.since is not None:
            return Cue("exercise.cue.hold")
        return Cue("exercise.cue.touch", self.finger)

    def hold_fraction(self, at: float) -> float:
        return self.gate.fraction(at) if self.gate.seconds else 0.0

    def hand_lost(self) -> None:
        self.gate.reset()

    def reset(self) -> None:
        self.gate.reset()
        self.phase = "reach"


class FingerWaveCounter:
    """One repetition per lift-and-lower of each finger, in order."""

    def __init__(self, direction: str = "forward") -> None:
        self.sequence = FINGER_ORDER if direction == "forward" else FINGER_ORDER[::-1]
        self.step = 0
        self.lifted = False

    def update(self, landmarks: HandLandmarks, states: FingerStates, at: float) -> bool:
        extended = states[FINGER_INDEX[self.sequence[self.step]]]
        if extended and not self.lifted:
            self.lifted = True
        elif not extended and self.lifted:
            self.lifted = False
            self.step += 1
            if self.step >= len(self.sequence):
                self.step = 0
                return True
        return False

    def cue(self) -> Cue:
        finger = self.sequence[self.step]
        return Cue("exercise.cue.lower" if self.lifted else "exercise.cue.lift", finger)

    def hold_fraction(self, at: float) -> float:
        return 0.0

    def hand_lost(self) -> None:
        return None

    def reset(self) -> None:
        self.step = 0
        self.lifted = False


class FingerLiftCounter:
    """One repetition per lift of a single finger, optionally with the others down."""

    def __init__(self, finger: str, isolated: bool = False, hold_seconds: float = 0.0):
        self.finger = finger
        self.index = FINGER_INDEX[finger]
        self.isolated = isolated
        self.gate = HoldGate(hold_seconds)
        self.phase = "lift"

    def _target(self, states: FingerStates) -> bool:
        if not states[self.index]:
            return False
        if not self.isolated:
            return True
        return not any(states[i] for i in FINGER_INDEX.values() if i != self.index)

    def update(self, landmarks: HandLandmarks, states: FingerStates, at: float) -> bool:
        if self.phase == "lower":
            if not states[self.index]:
                self.phase = "lift"
            return False
        if self.gate.update(self._target(states), at):
            self.gate.reset()
            self.phase = "lower"
            return True
        return False

    def cue(self) -> Cue:
        if self.phase == "lower":
            return Cue("exercise.cue.lower", self.finger)
        if self.gate.since is not None:
            return Cue("exercise.cue.hold")
        return Cue("exercise.cue.lift_only" if self.isolated else "exercise.cue.lift", self.finger)

    def hold_fraction(self, at: float) -> float:
        return self.gate.fraction(at) if self.gate.seconds else 0.0

    def hand_lost(self) -> None:
        self.gate.reset()

    def reset(self) -> None:
        self.gate.reset()
        self.phase = "lift"


class ThumbOppositionCounter:
    """One repetition per thumb contact with each fingertip, index to pinky."""

    def __init__(self, tolerance: str = "normal") -> None:
        self.on, self.off = CONTACT_THRESHOLDS[tolerance]
        self.step = 0
        self.touching = False

    def update(self, landmarks: HandLandmarks, states: FingerStates, at: float) -> bool:
        finger = FINGER_ORDER[self.step]
        if self.touching:
            if contact_ratio(landmarks, finger) > self.off:
                self.touching = False
                self.step = (self.step + 1) % len(FINGER_ORDER)
            return False
        if in_contact(landmarks, finger, self.on):
            self.touching = True
            return self.step == len(FINGER_ORDER) - 1
        return False

    def cue(self) -> Cue:
        if self.touching:
            return Cue("exercise.cue.release")
        return Cue("exercise.cue.touch", FINGER_ORDER[self.step])

    def hold_fraction(self, at: float) -> float:
        return 0.0

    def hand_lost(self) -> None:
        return None

    def reset(self) -> None:
        self.step = 0
        self.touching = False


def build_detector(spec: ExerciseSpec) -> Detector:
    """The detector for a validated spec."""
    if spec.movement is Movement.OPEN_CLOSE:
        return OpenCloseCounter(spec.hold_seconds)
    if spec.movement is Movement.PINCH:
        return ContactCounter(spec.finger, spec.tolerance, spec.hold_seconds)
    if spec.movement is Movement.FINGER_WAVE:
        return FingerWaveCounter(spec.direction)
    if spec.movement is Movement.FINGER_LIFT:
        return FingerLiftCounter(spec.finger, spec.isolated, spec.hold_seconds)
    return ThumbOppositionCounter(spec.tolerance)
