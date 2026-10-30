"""User-test pointing task: select a ring of targets with the pinch click.

Started with ``python -m spectra --usability P01 --filter kalman``. The participant points
with the index finger and pinches to select; the task follows the ISO 9241-9 multidirectional
layout so its throughput is comparable with the literature. Dwell selection is disabled here:
only the pinch click counts, which is what is being measured. When the last target is hit the
mode waits briefly and returns to the shell, which stores the results.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.i18n import t
from spectra.modes.base import ACTION_QUIT, BaseMode, ModeContext
from spectra.ui.text import draw_text, draw_text_centered
from spectra.usability import PointingTask, ring_targets

FINISH_DELAY_SECONDS = 2.0


class UsabilityMode(BaseMode):
    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        radius = max(18.0, min(self.width, self.height) * 0.05)
        self.task = PointingTask(ring_targets(self.width, self.height, radius=radius))
        self.finished_at: float | None = None
        self._flash: tuple[str, float] | None = None

    @property
    def pinch_clicks_enabled(self) -> bool:
        return True

    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        height, width = frame.shape[:2]
        context = self.observe(frame.shape, detection)
        point = self.click_point(context.pointer) if context.clicked else context.pointer
        outcome = self.task.update(point, context.clicked, context.at)
        if outcome:
            self._flash = (outcome, context.at)
            (self.context.sound.success if outcome == "hit" else self.context.sound.error)()

        output = (frame * 0.25).astype(np.uint8)
        for index, target in enumerate(self.task.targets):
            center = (int(target.x), int(target.y))
            active = index == self.task.index
            color = (0, 200, 255) if active else (70, 70, 70)
            cv2.circle(output, center, int(target.radius), color, -1 if active else 2)
        if context.pointer is not None:
            cv2.circle(output, context.pointer, 7, (255, 255, 255), -1)
        if self._flash and context.at - self._flash[1] < 0.3:
            color = (0, 255, 120) if self._flash[0] == "hit" else (0, 0, 255)
            cv2.rectangle(output, (0, 0), (width - 1, height - 1), color, 6)

        done = min(self.task.index, len(self.task.targets))
        draw_text(output, t("usability.task_title"), (14, 28), 0.7, (230, 230, 230), 2)
        draw_text(
            output,
            t("usability.task_progress", done=done, total=len(self.task.targets)),
            (14, 54),
            0.5,
            (180, 200, 255),
        )
        draw_text(output, t("usability.task_hint"), (14, height - 16), 0.45, (190, 190, 190))

        if self.task.finished:
            if self.finished_at is None:
                self.finished_at = context.at
            draw_text_centered(
                output, t("usability.task_done"), width // 2, height // 2, 1.0, (0, 255, 160), 2
            )
            if context.at - self.finished_at >= FINISH_DELAY_SECONDS:
                return output, ACTION_QUIT
        return output, None
