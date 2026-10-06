"""Gesture PIN confirmation shown before a session starts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.i18n import t
from spectra.modes.base import ACTION_QUIT, BaseMode, ModeContext
from spectra.ui.keypad import GesturePinKeypad
from spectra.ui.text import draw_text_centered
from spectra.ui.widgets import dim_frame

ACTION_AUTHENTICATED = "authenticated"

FEEDBACK_SECONDS = 2.0


class PinEntryMode(BaseMode):
    """Collects a 4-digit PIN by dwell and hands it to a verifier callback."""

    def __init__(
        self,
        context: ModeContext,
        verify: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(context)
        self.keypad = GesturePinKeypad(
            self.width, self.height, hover_seconds=context.config.hover_select_secs
        )
        self.verify = verify or (lambda _pin: True)
        self.authenticated = False
        self.locked = False
        self._message = ""
        self._message_at = 0.0
        self.buttons = [
            self.make_button(
                self.width - 148,
                self.height - 54,
                138,
                44,
                t("common.quit"),
                (120, 15, 15),
                value=ACTION_QUIT,
            )
        ]

    def _feedback(self, message: str) -> None:
        self._message = message
        self._message_at = time.time()

    def _submit(self, pin: str) -> None:
        if self.verify(pin):
            self.authenticated = True
            self._feedback(t("keypad.accepted"))
            self.context.sound.success()
            return
        self.keypad.reset()
        self._feedback(t("keypad.rejected"))
        self.context.sound.error()

    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        context = self.observe(frame.shape, detection)
        dim_frame(frame, (10, 10, 18), 0.75)

        if not self.authenticated:
            pin = self.keypad.update(context.pointer)
            if pin is not None:
                self._submit(pin)
        self.keypad.draw(frame)

        if context.pointer is not None:
            cv2.circle(frame, context.pointer, 12, (0, 220, 255), -1)
            cv2.circle(frame, context.pointer, 12, (255, 255, 255), 2)

        if self._message and (time.time() - self._message_at) < FEEDBACK_SECONDS:
            color = (0, 255, 120) if self.authenticated else (0, 80, 255)
            draw_text_centered(
                frame, self._message, self.width // 2, self.height - 90, 0.9, color, 2
            )

        action = self.draw_buttons(frame, context.pointer)
        if self.authenticated:
            return frame, ACTION_AUTHENTICATED
        return frame, action
