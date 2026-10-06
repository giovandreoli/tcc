"""Educational quiz: reproduce the finger combination bound to a shown colour."""

from __future__ import annotations

import math
import random
import time
from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.i18n import t
from spectra.modes.base import ACTION_NEXT, BaseMode, ModeContext
from spectra.modes.palette import COLOR_ENTRIES, PaletteEntry, combination_hint
from spectra.modes.quiz import ANSWER_COOLDOWN_SECONDS, QuizScore, ResultBanner, draw_scoreboard
from spectra.ui.text import draw_text_centered
from spectra.ui.widgets import dim_frame, draw_finger_hud

HOLD_SECONDS = 0.8
QUIZ_ENTRIES: tuple[PaletteEntry, ...] = tuple(
    entry
    for entry in COLOR_ENTRIES
    if entry.label_key
    in {
        "color.red",
        "color.green",
        "color.blue",
        "color.yellow",
        "color.orange",
        "color.purple",
        "color.cyan",
    }
)


class EduColorsMode(BaseMode):
    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        self.score = QuizScore()
        self.banner = ResultBanner()
        self.current: PaletteEntry = random.choice(QUIZ_ENTRIES)
        self._cooldown_until = 0.0
        self._hold_start: float | None = None
        self._last_states = None
        self.buttons = self.navigation_buttons(include_next=True)
        self._next_question()

    def _next_question(self) -> None:
        self.current = random.choice(QUIZ_ENTRIES)
        self._cooldown_until = time.time() + ANSWER_COOLDOWN_SECONDS
        self._hold_start = None
        self._last_states = None

    def _answer(self, success: bool, hint: str) -> None:
        self.score.register(success)
        if success:
            self.banner.show(t("edu_colors.correct", streak=self.score.streak), True)
            self.context.sound.success()
        else:
            self.banner.show(t("edu_colors.wrong", hint=hint), False)
            self.context.sound.error()
        self._next_question()

    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        height, width = frame.shape[:2]
        entry = self.current
        hint = combination_hint(entry.pattern)

        dim_frame(frame, (10, 10, 22), 0.72)

        cx, cy, radius = width // 2, height // 2 - 52, 100
        pulse = int(5 * abs(math.sin(time.time() * 2.8)))
        cv2.circle(frame, (cx, cy), radius + pulse, entry.color, 7)
        cv2.circle(frame, (cx, cy), radius, entry.color, -1)
        cv2.circle(frame, (cx, cy), radius + 3, (255, 255, 255), 3)

        draw_text_centered(
            frame, t("edu_colors.prompt"), cx, cy - radius - 32, 0.65, (210, 210, 210)
        )
        draw_text_centered(frame, entry.label, cx, cy + radius + 46, 1.3, entry.color, 3)
        draw_text_centered(
            frame, t("edu_colors.hint", hint=hint), cx, cy + radius + 88, 0.6, (150, 150, 150)
        )
        draw_scoreboard(frame, self.score)

        frame_context = self.observe(frame.shape, detection)
        pointer = frame_context.pointer
        states = frame_context.states
        if pointer is not None:
            cv2.circle(frame, pointer, 10, (0, 220, 255), -1)

        if states is not None and time.time() > self._cooldown_until:
            self._evaluate(states, hint)

        if self.banner.visible:
            color = (0, 255, 80) if self.banner.success else (0, 50, 255)
            draw_text_centered(frame, self.banner.text, width // 2, 98, 0.9, color, 2)

        action = self.draw_buttons(frame, pointer)
        if action == ACTION_NEXT:
            self._next_question()
            action = None

        if states is not None:
            draw_finger_hud(frame, states, width - 155, height - 68)
        return frame, action

    def _evaluate(self, states, hint: str) -> None:
        """Confirm an answer only after the same combination is held steady."""
        correct = tuple(states) == tuple(self.current.pattern)
        if not correct and states.extended_count == 0:
            self._hold_start = None
            self._last_states = states
            return
        if self._last_states != states:
            self._hold_start = time.time()
            self._last_states = states
            return
        if self._hold_start is None:
            self._hold_start = time.time()
            return
        if time.time() - self._hold_start >= HOLD_SECONDS:
            self._answer(correct, hint)
