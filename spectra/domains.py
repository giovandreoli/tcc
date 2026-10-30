"""Application domains: the same gestures mapped to different commands.

The interface is built in two layers so it can be re-targeted without touching detection:

1. :class:`GestureTokenizer` turns each frame into discrete **gesture tokens**: a posture
   held long enough (``open_hand``, ``fist``, ``point``, ``two``, ``three``), a pinch ``click``,
   a ``push`` towards or ``pull`` away from the camera (depth, ignoring rotations), and the
   onset of a gesture learned by the classifier (``ml:<label>``).
2. A :class:`DomainProfile` maps tokens to **commands**. Rehabilitation, a sterile
   environment (operating room) and drawing are presets; any mapping can be loaded from JSON.

The sterile preset includes a **lock**: a closed fist toggles it, and while locked every other
command is ignored, so an accidental movement near the screen cannot change the image during
a procedure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from spectra.gestures.depth import DepthReading, Motion
from spectra.gestures.features import FingerStates

POSTURE_HOLD_SECONDS = 0.5
COMMAND_COOLDOWN_SECONDS = 0.8
PUSH_ON, PUSH_OFF = 1.20, 1.10
PULL_ON, PULL_OFF = 0.83, 0.90


class Command(Enum):
    """Values double as i18n keys under ``domain.command.``."""

    SELECT = "select"
    NEXT = "next"
    PREVIOUS = "previous"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    RESET_VIEW = "reset_view"
    LOCK = "lock"
    MENU = "menu"
    PAUSE = "pause"


POSTURES: dict[str, tuple[bool, bool, bool, bool]] = {
    "open_hand": (True, True, True, True),
    "fist": (False, False, False, False),
    "point": (True, False, False, False),
    "two": (True, True, False, False),
    "three": (True, True, True, False),
}
BASE_TOKENS = (*POSTURES, "click", "push", "pull")
ML_PREFIX = "ml:"


def posture_of(states: FingerStates | None) -> str | None:
    """Posture name from the four fingers (the thumb is ignored: it varies between people)."""
    if states is None:
        return None
    fingers = (states.index, states.middle, states.ring, states.pinky)
    return next((name for name, pattern in POSTURES.items() if pattern == fingers), None)


class GestureTokenizer:
    """Discrete gesture tokens from per-frame observations, each emitted once per onset."""

    def __init__(self, hold_seconds: float = POSTURE_HOLD_SECONDS) -> None:
        self.hold_seconds = hold_seconds
        self._posture: str | None = None
        self._since: float | None = None
        self._emitted: str | None = None
        self._pushed = False
        self._pulled = False

    def reset(self) -> None:
        self._posture = self._since = self._emitted = None
        self._pushed = self._pulled = False

    def update(
        self,
        states: FingerStates | None,
        at: float,
        *,
        clicked: bool = False,
        depth: DepthReading | None = None,
        learned: str | None = None,
    ) -> list[str]:
        tokens: list[str] = []
        posture = posture_of(states)
        if posture != self._posture:
            self._posture, self._since = posture, at
            if posture != self._emitted:
                self._emitted = None
        if (
            posture is not None
            and self._since is not None
            and at - self._since + 1e-9 >= self.hold_seconds
            and self._emitted != posture
        ):
            self._emitted = posture
            tokens.append(posture)
        if clicked:
            tokens.append("click")
        if depth is not None and depth.motion is not Motion.ROTATION:
            if not self._pushed and depth.scale >= PUSH_ON:
                self._pushed = True
                tokens.append("push")
            elif self._pushed and depth.scale <= PUSH_OFF:
                self._pushed = False
            if not self._pulled and depth.scale <= PULL_ON:
                self._pulled = True
                tokens.append("pull")
            elif self._pulled and depth.scale >= PULL_OFF:
                self._pulled = False
        if learned:
            tokens.append(ML_PREFIX + learned)
        return tokens


@dataclass(frozen=True)
class DomainProfile:
    """Token-to-command bindings for one application domain."""

    name: str
    bindings: Mapping[str, Command]
    cooldown_seconds: float = COMMAND_COOLDOWN_SECONDS
    lockable: bool = False

    def __post_init__(self) -> None:
        for token in self.bindings:
            if token not in BASE_TOKENS and not token.startswith(ML_PREFIX):
                raise ValueError(f"unknown gesture token: {token}")
        if list(self.bindings.values()).count(Command.LOCK) > 1:
            raise ValueError("the lock command must have a single gesture")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bindings": {token: command.value for token, command in self.bindings.items()},
            "cooldown_seconds": self.cooldown_seconds,
            "lockable": self.lockable,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> DomainProfile:
        return cls(
            name=str(data.get("name", "custom")),
            bindings={str(k): Command(v) for k, v in dict(data["bindings"]).items()},
            cooldown_seconds=float(data.get("cooldown_seconds", COMMAND_COOLDOWN_SECONDS)),
            lockable=bool(data.get("lockable", False)),
        )

    def with_learned(self, learned: Mapping[str, Command]) -> DomainProfile:
        """Add bindings for gestures learned by the classifier (label → command)."""
        extra = {ML_PREFIX + label: command for label, command in learned.items()}
        return DomainProfile(
            self.name, {**self.bindings, **extra}, self.cooldown_seconds, self.lockable
        )


DOMAINS: dict[str, DomainProfile] = {
    "rehab": DomainProfile(
        "rehab",
        {"click": Command.SELECT, "open_hand": Command.MENU, "fist": Command.PAUSE},
    ),
    "sterile": DomainProfile(
        "sterile",
        {
            "click": Command.SELECT,
            "two": Command.NEXT,
            "three": Command.PREVIOUS,
            "push": Command.ZOOM_IN,
            "pull": Command.ZOOM_OUT,
            "open_hand": Command.RESET_VIEW,
            "fist": Command.LOCK,
        },
        lockable=True,
    ),
    "drawing": DomainProfile(
        "drawing",
        {"click": Command.SELECT, "fist": Command.PAUSE, "open_hand": Command.MENU},
    ),
}


@dataclass
class CommandMapper:
    """Applies a domain profile with a per-command cooldown and the optional lock."""

    profile: DomainProfile
    locked: bool = False
    _last: dict[Command, float] = field(default_factory=dict)

    def update(self, tokens: Iterable[str], at: float) -> list[Command]:
        commands: list[Command] = []
        for token in tokens:
            command = self.profile.bindings.get(token)
            if command is None:
                continue
            last = self._last.get(command)
            if last is not None and at - last < self.profile.cooldown_seconds:
                continue
            if self.profile.lockable and command is Command.LOCK:
                self.locked = not self.locked
            elif self.locked:
                continue
            self._last[command] = at
            commands.append(command)
        return commands


@dataclass
class ImageViewer:
    """State of the touchless image viewer used in the sterile-environment demonstration."""

    count: int
    index: int = 0
    zoom: float = 1.0
    min_zoom: float = 1.0
    max_zoom: float = 4.0
    zoom_step: float = 1.25

    def apply(self, command: Command) -> bool:
        """Apply a command; returns ``True`` if the view changed."""
        before = (self.index, self.zoom)
        if command is Command.NEXT:
            self.index = (self.index + 1) % self.count
            self.zoom = self.min_zoom
        elif command is Command.PREVIOUS:
            self.index = (self.index - 1) % self.count
            self.zoom = self.min_zoom
        elif command is Command.ZOOM_IN:
            self.zoom = min(self.max_zoom, self.zoom * self.zoom_step)
        elif command is Command.ZOOM_OUT:
            self.zoom = max(self.min_zoom, self.zoom / self.zoom_step)
        elif command is Command.RESET_VIEW:
            self.zoom = self.min_zoom
        return (self.index, self.zoom) != before
