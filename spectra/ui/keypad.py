"""On-screen numeric keypad driven by dwell selection.

Typing eleven CPF digits by dwelling would take over a minute, so the gesture keypad is
used only for the 4-digit PIN that confirms a session the therapist already opened. The
full CPF + PIN login is available by keyboard at application start.

The layout and the entry logic are pure, so the whole keypad is unit tested without a
camera: only :meth:`GesturePinKeypad.draw` touches OpenCV.
"""

from __future__ import annotations

import numpy as np

from spectra.i18n import t
from spectra.ui.button import Button
from spectra.ui.text import draw_text_centered

PIN_LENGTH = 4

ACTION_CLEAR = "clear"
ACTION_BACKSPACE = "backspace"

#: Phone layout, with clear and backspace flanking the zero.
LAYOUT: tuple[tuple[str, ...], ...] = (
    ("1", "2", "3"),
    ("4", "5", "6"),
    ("7", "8", "9"),
    (ACTION_CLEAR, "0", ACTION_BACKSPACE),
)


class GesturePinKeypad:
    """A dwell-activated PIN pad. Call :meth:`update` once per frame."""

    def __init__(
        self,
        width: int,
        height: int,
        hover_seconds: float = 1.2,
        pin_length: int = PIN_LENGTH,
    ) -> None:
        self.width = width
        self.height = height
        self.pin_length = pin_length
        self.entry = ""
        self.submitted: str | None = None
        self.keys = self._build_keys(hover_seconds)

    def _build_keys(self, hover_seconds: float) -> list[Button]:
        key_size, gap = 96, 14
        columns, rows = 3, len(LAYOUT)
        total_width = columns * key_size + (columns - 1) * gap
        total_height = rows * key_size + (rows - 1) * gap
        x0 = (self.width - total_width) // 2
        y0 = (self.height - total_height) // 2 + 30

        keys: list[Button] = []
        for row_index, row in enumerate(LAYOUT):
            for column_index, value in enumerate(row):
                label = {
                    ACTION_CLEAR: t("keypad.clear"),
                    ACTION_BACKSPACE: t("keypad.backspace"),
                }.get(value, value)
                background = (
                    (60, 20, 20) if value in (ACTION_CLEAR, ACTION_BACKSPACE) else (40, 40, 55)
                )
                keys.append(
                    Button(
                        x0 + column_index * (key_size + gap),
                        y0 + row_index * (key_size + gap),
                        key_size,
                        key_size,
                        label,
                        background,
                        value=value,
                        hover_seconds=hover_seconds,
                    )
                )
        return keys

    # ------------------------------------------------------------------ logic
    @property
    def complete(self) -> bool:
        return len(self.entry) >= self.pin_length

    @property
    def masked(self) -> str:
        return "•" * len(self.entry) + "_" * (self.pin_length - len(self.entry))

    def reset(self) -> None:
        self.entry = ""
        self.submitted = None
        for key in self.keys:
            key.reset()

    def press(self, value: str) -> str | None:
        """Apply one key press; returns the PIN once it is complete."""
        if value == ACTION_CLEAR:
            self.entry = ""
        elif value == ACTION_BACKSPACE:
            self.entry = self.entry[:-1]
        elif value.isdigit() and not self.complete:
            self.entry += value
        if self.complete:
            self.submitted = self.entry
            return self.submitted
        return None

    def update(self, pointer: tuple[int, int] | None, now: float | None = None) -> str | None:
        """Advance every dwell timer; returns the PIN on the frame it completes."""
        result = None
        for key in self.keys:
            if key.update_hover(key.contains(pointer), now=now):
                result = self.press(key.value)
        return result

    # ----------------------------------------------------------------- render
    def draw(self, frame: np.ndarray) -> None:
        draw_text_centered(frame, t("keypad.prompt"), self.width // 2, 90, 0.7, (230, 230, 230), 2)
        draw_text_centered(frame, self.masked, self.width // 2, 150, 1.6, (0, 220, 255), 3)
        for key in self.keys:
            key.draw(frame)
