from __future__ import annotations

from spectra.gestures.features import FingerStates
from spectra.modes.palette import (
    COLOR_ENTRIES,
    COMMAND_CLEAR,
    COMMAND_ERASER,
    COMMAND_UNDO,
    PALETTE,
    combination_hint,
    resolve,
)


def states(*flags: bool) -> FingerStates:
    return FingerStates(*flags)


def test_every_palette_pattern_is_unique():
    patterns = [tuple(entry.pattern) for entry in PALETTE]
    assert len(patterns) == len(set(patterns))


def test_exact_match_returns_the_colour():
    entry = resolve(states(False, True, False, False, False))
    assert entry is not None
    assert entry.color == (0, 0, 255)


def test_commands_are_matched_exactly():
    assert resolve(states(True, True, True, True, True)).command == COMMAND_CLEAR
    assert resolve(states(False, False, False, True, True)).command == COMMAND_UNDO
    assert resolve(states(True, False, False, False, False)).command == COMMAND_ERASER


def test_fuzzy_match_tolerates_one_wrong_finger():
    # index + middle + pinky is one finger away from orange (index + middle)
    entry = resolve(states(False, True, True, False, True))
    assert entry is not None
    assert entry.label_key == "color.orange"


def test_fuzzy_match_never_returns_a_command():
    entry = resolve(states(False, True, True, True, True))
    assert entry is not None
    assert not entry.is_command


def test_fuzzy_match_can_be_disabled():
    assert resolve(states(False, True, True, False, True), fuzzy_tolerance=0) is None


def test_resolve_none_returns_none():
    assert resolve(None) is None


def test_colour_entries_exclude_commands():
    assert all(not entry.is_command for entry in COLOR_ENTRIES)


def test_combination_hint_lists_extended_fingers():
    assert combination_hint(states(False, True, True, False, False)) == "Indicador + Médio"


def test_combination_hint_for_closed_fist():
    assert combination_hint(states(False, False, False, False, False)) == "Nenhum"
