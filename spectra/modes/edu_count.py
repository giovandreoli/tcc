"""Educational quiz: raise exactly N fingers."""

from __future__ import annotations

import random
import time
from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.i18n import t
from spectra.modes.base import BaseMode, ModeContext
from spectra.modes.quiz import ANSWER_COOLDOWN_SECONDS, QuizScore, ResultBanner, draw_scoreboard
from spectra.ui.text import draw_text, draw_text_centered
from spectra.ui.widgets import dim_frame, draw_finger_hud, draw_progress_bar

HOLD_SECONDS = 0.8


class EduCountMode(BaseMode):
    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        self.score = QuizScore()
        self.banner = ResultBanner()
        self.target = 0
        self._cooldown_until = 0.0
        self._hold_start: float | None = None
        self._last_count = -1
        self.buttons = self.navigation_buttons()
        self._next_question()

    def _next_question(self) -> None:
        self.target = random.randint(0, 5)
        self._cooldown_until = time.time() + ANSWER_COOLDOWN_SECONDS
        self._hold_start = None
        self._last_count = -1

    def _answer(self, count: int) -> None:
        success = count == self.target
        self.score.register(success)
        if success:
            self.banner.show(t("edu_count.correct", count=count, streak=self.score.streak), True)
            self.context.sound.success()
        else:
            self.banner.show(t("edu_count.wrong", target=self.target), False)
            self.context.sound.error()
        self._next_question()

    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        height, width = frame.shape[:2]
        dim_frame(frame, (10, 22, 10), 0.72)

        number = str(self.target)
        draw_text_centered(frame, number, width // 2, height // 2 - 8, 6.5, (40, 40, 40), 18)
        draw_text_centered(frame, number, width // 2, height // 2 - 8, 6.5, (255, 255, 255), 10)
        draw_text_centered(
            frame, t("edu_count.prompt"), width // 2, height // 2 + 112, 0.75, (200, 200, 200), 2
        )
        draw_text_centered(
            frame,
            t("edu_count.hold_hint", secs=HOLD_SECONDS),
            width // 2,
            height // 2 + 140,
            0.5,
            (130, 130, 130),
        )
        draw_scoreboard(frame, self.score)

        frame_context = self.observe(frame.shape, detection)
        pointer = frame_context.pointer
        states = frame_context.states

        if states is None:
            self._hold_start = None
            self._last_count = -1
        else:
            count = states.extended_count
            draw_text(
                frame, t("edu_count.your_fingers", count=count), (10, 98), 0.8, (255, 220, 0), 2
            )
            if pointer is not None:
                cv2.circle(frame, pointer, 10, (0, 220, 255), -1)
            if time.time() > self._cooldown_until:
                self._evaluate(frame, count)

        if self.banner.visible:
            color = (0, 255, 80) if self.banner.success else (0, 50, 255)
            draw_text_centered(frame, self.banner.text, width // 2, 98, 0.9, color, 2)

        action = self.draw_buttons(frame, pointer)
        if states is not None:
            draw_finger_hud(frame, states, width - 155, height - 68)
        return frame, action

    def _evaluate(self, frame: np.ndarray, count: int) -> None:
        height, width = frame.shape[:2]
        if count != self._last_count:
            self._hold_start = time.time()
            self._last_count = count
            return
        if self._hold_start is None:
            self._hold_start = time.time()
            return
        held = time.time() - self._hold_start
        draw_progress_bar(
            frame, (width // 2 - 100, height // 2 + 154), (200, 13), held / HOLD_SECONDS
        )
        if held >= HOLD_SECONDS:
            self._answer(count)
