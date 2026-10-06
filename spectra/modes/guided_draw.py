"""Guided drawing: trace a target shape inside a tolerance corridor.

This is the assessment mode. Unlike free drawing, every sample is scored against a
geometric target, which gives the therapist a number that can be compared across sessions.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.gestures.state_machine import GestureState
from spectra.guided.scoring import TraceResult, TraceScorer
from spectra.guided.shapes import SHAPE_ORDER, Difficulty, TargetPath, build_path
from spectra.i18n import t
from spectra.modes.base import ACTION_NEXT, BaseMode, FrameContext, ModeContext
from spectra.ui.text import draw_text, draw_text_centered
from spectra.ui.widgets import dim_frame, draw_progress_bar

ACTION_FINISH = "finish"
ACTION_RESTART = "restart"

CORRIDOR_COLOR = (70, 70, 70)
TARGET_COLOR = (0, 200, 255)
INSIDE_COLOR = (0, 255, 120)
OUTSIDE_COLOR = (0, 80, 255)

#: Gestures during which the dwell buttons stay live.
INTERACTIVE_STATES = (GestureState.IDLE, GestureState.POINTING)


class GuidedDrawMode(BaseMode):
    records_metrics = True

    def __init__(
        self,
        context: ModeContext,
        shape_key: str = SHAPE_ORDER[0],
        difficulty: Difficulty = Difficulty.EASY,
    ) -> None:
        super().__init__(context)
        self.difficulty = difficulty
        self.shape_index = SHAPE_ORDER.index(shape_key) if shape_key in SHAPE_ORDER else 0
        self.attempt = 1
        self.results: list[TraceResult] = []
        self.last_result: TraceResult | None = None
        self.path: TargetPath = build_path(self.shape_key, difficulty)
        self.scorer = TraceScorer(self.path)
        self._stroke: list[tuple[int, int]] = []

        button_width, button_height = 112, 42
        y = self.height - button_height - 10
        self.buttons = [
            self.make_button(
                10,
                y,
                button_width,
                button_height,
                t("guided.finish"),
                (20, 70, 20),
                value=ACTION_FINISH,
            ),
            self.make_button(
                130,
                y,
                button_width,
                button_height,
                t("guided.restart"),
                (80, 60, 10),
                value=ACTION_RESTART,
            ),
        ]
        self.buttons += self.navigation_buttons(include_next=True)

    # ------------------------------------------------------------------ challenge
    @property
    def shape_key(self) -> str:
        return SHAPE_ORDER[self.shape_index]

    def load_challenge(
        self, shape_key: str | None = None, difficulty: Difficulty | None = None
    ) -> None:
        if shape_key is not None:
            self.shape_index = SHAPE_ORDER.index(shape_key)
        if difficulty is not None:
            self.difficulty = difficulty
        self.path = build_path(self.shape_key, self.difficulty)
        self.scorer = TraceScorer(self.path)
        self._stroke.clear()

    def next_shape(self) -> None:
        self.shape_index = (self.shape_index + 1) % len(SHAPE_ORDER)
        self.attempt = 1
        self.load_challenge()

    def restart(self) -> None:
        self.attempt += 1
        self.load_challenge()

    def finish(self) -> TraceResult:
        """Close the attempt, keep its result and reset the scorer."""
        result = self.scorer.result()
        self.last_result = result
        self.results.append(result)
        self.context.sound.success()
        self.restart()
        return result

    # -------------------------------------------------------------------- render
    def _draw_target(self, frame: np.ndarray) -> None:
        points = self.path.to_pixels(self.width, self.height)
        corridor = max(2, int(self.path.tolerance * self.height))
        closed = self.path.closed
        cv2.polylines(
            frame, [np.array(points, dtype=np.int32)], closed, CORRIDOR_COLOR, corridor * 2
        )
        cv2.polylines(
            frame, [np.array(points, dtype=np.int32)], closed, TARGET_COLOR, 2, cv2.LINE_AA
        )
        cv2.circle(frame, points[0], 10, INSIDE_COLOR, -1)

    def _draw_stroke(self, frame: np.ndarray) -> None:
        if len(self._stroke) > 1:
            cv2.polylines(
                frame,
                [np.array(self._stroke, dtype=np.int32)],
                False,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

    def _draw_hud(self, frame: np.ndarray, context: FrameContext) -> None:
        cv2.rectangle(frame, (0, 0), (self.width, 54), (15, 15, 15), -1)
        draw_text(frame, t("guided.title"), (10, 24), 0.7, (0, 220, 255), 2)
        draw_text(
            frame, t("guided.shape_label", shape=t(f"guided.shape.{self.shape_key}")), (10, 46), 0.5
        )
        draw_text(
            frame,
            t("guided.difficulty_label", level=t(f"guided.difficulty.{self.difficulty.value}")),
            (240, 46),
            0.5,
        )
        draw_text(frame, t("guided.attempt", number=self.attempt), (440, 46), 0.5, (180, 180, 180))

        completion = self.scorer.completion
        draw_text(
            frame,
            t("guided.completion", percent=f"{completion * 100:.0f}"),
            (self.width - 300, 24),
            0.55,
        )
        draw_progress_bar(frame, (self.width - 300, 32), (280, 10), completion, INSIDE_COLOR)
        draw_text(
            frame,
            t("guided.time", seconds=f"{self.scorer.duration:.0f}"),
            (self.width - 300, 52),
            0.45,
            (180, 180, 180),
        )

        if context.gesture is GestureState.IDLE and not self._stroke:
            draw_text_centered(
                frame,
                t("guided.start_hint"),
                self.width // 2,
                self.height - 120,
                0.6,
                (200, 200, 200),
            )
        else:
            draw_text_centered(
                frame, t("guided.prompt"), self.width // 2, self.height - 120, 0.6, (200, 200, 200)
            )

    # ------------------------------------------------------------------- process
    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        context = self.observe(frame.shape, detection)
        dim_frame(frame, (12, 12, 18), 0.6)
        self._draw_target(frame)

        tracing = (
            context.gesture is GestureState.PAINTING and context.normalized_pointer is not None
        )
        if tracing:
            deviation = self.scorer.add(context.normalized_pointer, time.time())
            self._stroke.append(context.pointer)
            inside = deviation <= self.path.tolerance
            cv2.circle(frame, context.pointer, 9, INSIDE_COLOR if inside else OUTSIDE_COLOR, -1)
        elif context.pointer is not None:
            cv2.circle(frame, context.pointer, 9, (220, 220, 220), 2)

        self._draw_stroke(frame)
        self._draw_hud(frame, context)

        if context.paused:
            dim_frame(frame, (0, 0, 40), 0.6)
            draw_text_centered(
                frame, t("safety.paused"), self.width // 2, self.height // 2, 0.9, (0, 220, 255), 2
            )
        elif self.last_result is not None and not self._stroke:
            draw_text_centered(
                frame,
                t(
                    "guided.result",
                    score=f"{self.last_result.score:.0f}",
                    percent=f"{self.last_result.completion * 100:.0f}",
                ),
                self.width // 2,
                self.height // 2,
                0.9,
                (0, 255, 120),
                2,
            )

        interactive = context.gesture in INTERACTIVE_STATES
        if not interactive:
            for button in self.buttons:
                button.reset()
        action = self.draw_buttons(frame, context.pointer if interactive else None)

        if action == ACTION_FINISH:
            self.finish()
            return frame, None
        if action == ACTION_RESTART:
            self.restart()
            return frame, None
        if action == ACTION_NEXT:
            self.next_shape()
            return frame, None
        return frame, action
