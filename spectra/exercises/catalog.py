"""Exercise specifications built from validated movements.

The therapist does not write detection code. A custom exercise is a :class:`Movement`
that already has a tested detector, plus parameters with enforced limits: repetitions,
hold time, the finger involved and how close a contact must be. That is what keeps a
patient-specific exercise reliable: every combination maps to a detector covered by the
unit tests, and out-of-range values are rejected before a session can use them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from spectra.i18n import t

CUSTOM_PREFIX = "custom:"
FINGERS = ("index", "middle", "ring", "pinky")
TOLERANCES = ("wide", "normal", "strict")
DIRECTIONS = ("forward", "backward")

MIN_REPS, MAX_REPS = 1, 30
MAX_HOLD_SECONDS = 5.0
TITLE_MIN, TITLE_MAX = 3, 60
MAX_INSTRUCTIONS = 3
INSTRUCTION_MAX = 140


class Movement(Enum):
    """Validated movements; values double as i18n keys under ``exercise.movement.``."""

    OPEN_CLOSE = "open_close"
    PINCH = "pinch"
    FINGER_WAVE = "finger_wave"
    FINGER_LIFT = "finger_lift"
    THUMB_OPPOSITION = "thumb_opposition"

    @property
    def title(self) -> str:
        return t(f"exercise.movement.{self.value}")

    @property
    def description(self) -> str:
        return t(f"exercise.movement_help.{self.value}")


#: Which parameters each movement accepts. Anything else is ignored on save.
MOVEMENT_PARAMS: dict[Movement, frozenset[str]] = {
    Movement.OPEN_CLOSE: frozenset({"hold_seconds"}),
    Movement.PINCH: frozenset({"finger", "tolerance", "hold_seconds"}),
    Movement.FINGER_WAVE: frozenset({"direction"}),
    Movement.FINGER_LIFT: frozenset({"finger", "isolated", "hold_seconds"}),
    Movement.THUMB_OPPOSITION: frozenset({"tolerance"}),
}


@dataclass(frozen=True)
class ExerciseSpec:
    """Everything a session needs to run, count and report one exercise."""

    key: str
    movement: Movement
    title: str = ""
    instructions: tuple[str, ...] = ()
    target_reps: int = 8
    hold_seconds: float = 0.0
    finger: str = "index"
    tolerance: str = "normal"
    direction: str = "forward"
    isolated: bool = False
    builtin: bool = False

    # ---------------------------------------------------------------- display
    @property
    def is_custom(self) -> bool:
        return self.key.startswith(CUSTOM_PREFIX)

    def display_title(self) -> str:
        if self.title:
            return self.title
        return t(f"physio.exercise.{self.key}")

    def display_instructions(self) -> list[str]:
        if self.instructions:
            return list(self.instructions)
        if self.builtin:
            return [t(f"physio.instructions.{self.key}_{i}") for i in (1, 2, 3)]
        return default_instructions(self)

    def summary(self) -> str:
        """One line describing the parameters, for the therapist."""
        parts = [t("exercise.summary.reps", reps=self.target_reps)]
        accepted = MOVEMENT_PARAMS[self.movement]
        if "finger" in accepted:
            parts.append(t(f"finger.{self.finger}"))
        if "hold_seconds" in accepted and self.hold_seconds:
            parts.append(t("exercise.summary.hold", seconds=f"{self.hold_seconds:g}"))
        if "tolerance" in accepted:
            parts.append(t(f"exercise.tolerance.{self.tolerance}"))
        if "direction" in accepted:
            parts.append(t(f"exercise.direction.{self.direction}"))
        if "isolated" in accepted and self.isolated:
            parts.append(t("exercise.summary.isolated"))
        return " · ".join(parts)

    # ------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return i18n keys for every problem; empty means the spec is safe to run."""
        problems: list[str] = []
        if not self.builtin and not TITLE_MIN <= len(self.title.strip()) <= TITLE_MAX:
            problems.append("exercise.error.title")
        if len(self.instructions) > MAX_INSTRUCTIONS or any(
            len(line) > INSTRUCTION_MAX for line in self.instructions
        ):
            problems.append("exercise.error.instructions")
        if not MIN_REPS <= self.target_reps <= MAX_REPS:
            problems.append("exercise.error.reps")
        accepted = MOVEMENT_PARAMS[self.movement]
        if not 0 <= self.hold_seconds <= MAX_HOLD_SECONDS or (
            self.hold_seconds and "hold_seconds" not in accepted
        ):
            problems.append("exercise.error.hold")
        if self.finger not in FINGERS:
            problems.append("exercise.error.finger")
        if self.tolerance not in TOLERANCES:
            problems.append("exercise.error.tolerance")
        if self.direction not in DIRECTIONS:
            problems.append("exercise.error.direction")
        return problems

    # ---------------------------------------------------------- serialisation
    def params(self) -> dict[str, Any]:
        """Only the parameters that the movement actually uses."""
        values = {
            "hold_seconds": self.hold_seconds,
            "finger": self.finger,
            "tolerance": self.tolerance,
            "direction": self.direction,
            "isolated": self.isolated,
        }
        return {key: values[key] for key in sorted(MOVEMENT_PARAMS[self.movement])}

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "movement": self.movement.value,
            "title": self.title,
            "instructions": list(self.instructions),
            "target_reps": self.target_reps,
            "builtin": self.builtin,
            **self.params(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExerciseSpec:
        movement = Movement(data["movement"])
        accepted = MOVEMENT_PARAMS[movement]
        kwargs: dict[str, Any] = {
            "key": str(data["key"]),
            "movement": movement,
            "title": str(data.get("title") or ""),
            "instructions": tuple(str(line) for line in data.get("instructions") or ()),
            "target_reps": int(data.get("target_reps", 8)),
            "builtin": bool(data.get("builtin", False)),
        }
        if "hold_seconds" in accepted:
            kwargs["hold_seconds"] = float(data.get("hold_seconds", 0.0))
        for name in ("finger", "tolerance", "direction"):
            if name in accepted and data.get(name) is not None:
                kwargs[name] = str(data[name])
        if "isolated" in accepted:
            kwargs["isolated"] = bool(data.get("isolated", False))
        return cls(**kwargs)

    def with_key(self, key: str) -> ExerciseSpec:
        return replace(self, key=key)


def finger_word(finger: str) -> str:
    """Finger name for use inside a sentence ("Levante o médio")."""
    return t(f"finger.{finger}").lower()


def default_instructions(spec: ExerciseSpec) -> list[str]:
    """Plain-language steps generated from the movement and its parameters."""
    finger = finger_word(spec.finger)
    lines = [t(f"exercise.steps.{spec.movement.value}_{i}", finger=finger) for i in (1, 2)]
    if spec.hold_seconds and "hold_seconds" in MOVEMENT_PARAMS[spec.movement]:
        lines.append(t("exercise.steps.hold", seconds=f"{spec.hold_seconds:g}"))
    else:
        lines.append(t("exercise.steps.gentle"))
    return lines


BUILTIN_SPECS: dict[str, ExerciseSpec] = {
    spec.key: spec
    for spec in (
        ExerciseSpec("open_close", Movement.OPEN_CLOSE, builtin=True),
        ExerciseSpec("finger_touch", Movement.PINCH, finger="index", builtin=True),
        ExerciseSpec("finger_wave", Movement.FINGER_WAVE, builtin=True),
    )
}
BUILTIN_KEYS: tuple[str, ...] = tuple(BUILTIN_SPECS)


def builtin_spec(key: str) -> ExerciseSpec:
    return BUILTIN_SPECS[key]


def specs_from_json(entries: list[Any]) -> tuple[ExerciseSpec, ...]:
    """Load a room's exercise list; plain strings are legacy built-in keys."""
    return tuple(
        builtin_spec(entry) if isinstance(entry, str) else ExerciseSpec.from_dict(entry)
        for entry in entries
    )
