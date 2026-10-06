"""Scoreboard shared by the educational quiz modes."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from spectra.i18n import t
from spectra.ui.text import draw_text

RESULT_SECONDS = 2.6
ANSWER_COOLDOWN_SECONDS = 2.5


@dataclass
class QuizScore:
    """Right/wrong tally plus the current and best streak."""

    correct: int = 0
    total: int = 0
    streak: int = 0
    best_streak: int = 0

    def register(self, success: bool) -> None:
        self.total += 1
        if success:
            self.correct += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0


class ResultBanner:
    """Transient "correct"/"wrong" message shown after each answer."""

    def __init__(self, duration: float = RESULT_SECONDS) -> None:
        self.duration = duration
        self.text = ""
        self.success = True
        self._shown_at = 0.0

    def show(self, text: str, success: bool) -> None:
        self.text = text
        self.success = success
        self._shown_at = time.time()

    @property
    def visible(self) -> bool:
        return bool(self.text) and (time.time() - self._shown_at) < self.duration


def draw_scoreboard(frame: np.ndarray, score: QuizScore) -> None:
    draw_text(
        frame,
        t("common.score", score=score.correct, total=score.total),
        (10, 38),
        0.75,
        (0, 220, 255),
        2,
    )
    color = (0, 255, 100) if score.streak >= 3 else (190, 190, 190)
    draw_text(
        frame,
        t("common.streak", streak=score.streak, record=score.best_streak),
        (10, 64),
        0.52,
        color,
    )
