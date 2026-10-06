"""Finger-combination palette used by the free-draw mode.

This static mapping is the original SPECTRA interaction. It is kept for the painting
colours only; navigation and tool selection move to the gesture state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from spectra.gestures.features import FingerStates
from spectra.i18n import t

ColorBGR = tuple[int, int, int]

COMMAND_UNDO = "undo"
COMMAND_CLEAR = "clear"
COMMAND_ERASER = "eraser"


@dataclass(frozen=True)
class PaletteEntry:
    """A finger combination bound either to a colour or to a canvas command."""

    pattern: FingerStates
    label_key: str
    color: ColorBGR | None = None
    command: str | None = None

    @property
    def label(self) -> str:
        return t(self.label_key)

    @property
    def is_command(self) -> bool:
        return self.command is not None


def _states(thumb: bool, index: bool, middle: bool, ring: bool, pinky: bool) -> FingerStates:
    return FingerStates(thumb, index, middle, ring, pinky)


PALETTE: tuple[PaletteEntry, ...] = (
    PaletteEntry(_states(False, True, False, False, False), "color.red", (0, 0, 255)),
    PaletteEntry(_states(False, False, True, False, False), "color.green", (0, 255, 0)),
    PaletteEntry(_states(False, False, False, True, False), "color.blue", (255, 0, 0)),
    PaletteEntry(_states(False, False, False, False, True), "color.yellow", (0, 255, 255)),
    PaletteEntry(_states(False, True, True, False, False), "color.orange", (0, 165, 255)),
    PaletteEntry(_states(False, True, False, True, False), "color.purple", (128, 0, 128)),
    PaletteEntry(_states(False, True, False, False, True), "color.pink", (147, 20, 255)),
    PaletteEntry(_states(False, False, True, True, False), "color.cyan", (255, 255, 0)),
    PaletteEntry(_states(False, False, True, False, True), "color.brown", (42, 42, 165)),
    PaletteEntry(_states(False, True, True, True, False), "color.white", (255, 255, 255)),
    PaletteEntry(
        _states(False, False, False, True, True), "paint.command_undo", command=COMMAND_UNDO
    ),
    PaletteEntry(_states(True, False, False, False, False), "color.eraser", command=COMMAND_ERASER),
    PaletteEntry(
        _states(True, True, True, True, True), "paint.command_clear", command=COMMAND_CLEAR
    ),
)

#: Colour entries only, used by the educational colour quiz.
COLOR_ENTRIES: tuple[PaletteEntry, ...] = tuple(e for e in PALETTE if not e.is_command)


def resolve(states: FingerStates | None, fuzzy_tolerance: int = 1) -> PaletteEntry | None:
    """Match ``states`` to a palette entry, allowing one wrong finger for colours."""
    if states is None:
        return None
    pattern = tuple(bool(flag) for flag in states)
    for entry in PALETTE:
        if pattern == tuple(entry.pattern):
            return entry
    best: PaletteEntry | None = None
    best_difference = fuzzy_tolerance + 1
    for entry in COLOR_ENTRIES:
        difference = sum(1 for a, b in zip(pattern, entry.pattern, strict=True) if a != b)
        if difference < best_difference:
            best_difference = difference
            best = entry
    return best


def combination_hint(states: FingerStates) -> str:
    """Human-readable finger list, e.g. ``"Indicador + Médio"``."""
    names = [
        t(f"finger.{name}")
        for name, extended in zip(
            ("thumb", "index", "middle", "ring", "pinky"), states, strict=True
        )
        if extended
    ]
    return " + ".join(names) if names else t("common.none")
