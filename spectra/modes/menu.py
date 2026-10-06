"""Main menu: dwell on an entry with the index finger to open a mode."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.i18n import t
from spectra.modes.base import ACTION_QUIT, AppMode, BaseMode, ModeContext
from spectra.ui.text import draw_text_centered
from spectra.ui.widgets import dim_frame, draw_finger_hud, draw_status_bar

MENU_ENTRIES: tuple[tuple[AppMode | str, tuple[int, int, int]], ...] = (
    (AppMode.FREE_DRAW, (20, 90, 20)),
    (AppMode.GUIDED_DRAW, (90, 60, 20)),
    (AppMode.EDU_COLORS, (20, 20, 120)),
    (AppMode.EDU_COUNT, (80, 20, 120)),
    (AppMode.PHYSIO, (120, 55, 15)),
    (ACTION_QUIT, (120, 15, 15)),
)


class MenuMode(BaseMode):
    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        width, height, gap = 320, 64, 18
        x = (self.width - width) // 2
        count = len(MENU_ENTRIES)
        top = (self.height - (count * height + (count - 1) * gap)) // 2
        self.buttons = [
            self.make_button(
                x,
                top + i * (height + gap),
                width,
                height,
                t("common.quit") if value == ACTION_QUIT else value.title,
                background,
                value=value,
            )
            for i, (value, background) in enumerate(MENU_ENTRIES)
        ]

    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        dim_frame(frame, (12, 12, 12), 0.45)
        draw_status_bar(frame, t("app.title"), AppMode.MENU.title)
        draw_text_centered(frame, t("menu.prompt"), self.width // 2, 90, 0.58, (220, 220, 220))

        pointer = self.observe(frame.shape, detection).pointer
        states = self.engine.signal.fingers if self.engine.signal.present else None
        if pointer is not None:
            cv2.circle(frame, pointer, 14, (0, 220, 255), -1)
            cv2.circle(frame, pointer, 14, (255, 255, 255), 2)

        selected = self.draw_buttons(frame, pointer)
        if states is not None:
            draw_finger_hud(frame, states, self.width - 155, self.height - 22)
        return frame, selected
