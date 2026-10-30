"""Sterile-environment demonstration: browse images without touching anything.

Built for the operating-room scenario in the project: the surgeon cannot touch a mouse or a
screen without breaking asepsis. The same detection stack is re-targeted with the ``sterile``
domain profile (:mod:`spectra.domains`):

* two fingers → next image, three fingers → previous image;
* push the hand towards the camera → zoom in, pull it back → zoom out (a rotation of the
  hand is recognised as rotation and does not zoom);
* open hand → reset the zoom; closed fist → lock/unlock all commands;
* pinch "click" on the on-screen buttons; gestures learned by the classifier can be bound
  to any command as well.

The images are synthetic placeholders generated here: no real exam or patient image is used.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from spectra.detection.hand_detector import DetectionResult
from spectra.domains import DOMAINS, Command, CommandMapper, GestureTokenizer, ImageViewer
from spectra.i18n import t
from spectra.learning.classifier import GestureRecognizer
from spectra.learning.features import pose_features
from spectra.modes.base import BaseMode, ModeContext
from spectra.ui.text import draw_text, draw_text_centered
from spectra.ui.widgets import dim_frame, draw_finger_hud

IMAGE_COUNT = 6
FEEDBACK_SECONDS = 1.5
ACTION_PREVIOUS = "viewer_previous"
ACTION_NEXT_IMAGE = "viewer_next"
VIEWER_ACTIONS = (ACTION_PREVIOUS, ACTION_NEXT_IMAGE)


def demo_image(index: int, size: int = 360) -> np.ndarray:
    """A synthetic grayscale 'exam' so the demo never needs real medical images."""
    rng = np.random.default_rng(index + 7)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size / 2
    radius = np.hypot(xx - center, yy - center)
    image = np.clip(200 - radius * 0.9, 0, 255)
    for _ in range(4 + index):
        cx, cy = rng.uniform(0.25, 0.75, 2) * size
        r = rng.uniform(0.04, 0.12) * size
        image += 70 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    image += rng.normal(0, 6, image.shape)
    gray = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class SterileViewerMode(BaseMode):
    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        profile = DOMAINS["sterile"]
        bindings = {label: Command(command) for label, command in context.learned_bindings.items()}
        self.mapper = CommandMapper(profile.with_learned(bindings) if bindings else profile)
        self.tokenizer = GestureTokenizer()
        self.recognizer = (
            GestureRecognizer(context.gesture_model) if context.gesture_model is not None else None
        )
        self.viewer = ImageViewer(IMAGE_COUNT)
        self.images = [demo_image(i) for i in range(IMAGE_COUNT)]
        self.history: list[Command] = []
        self._feedback = ""
        self._feedback_at = 0.0

        width, height = 150, 48
        y = self.height - height - 70
        self.buttons = [
            self.make_button(
                10,
                y,
                width,
                height,
                t("domain.command.previous"),
                (70, 50, 20),
                value=ACTION_PREVIOUS,
            ),
            self.make_button(
                170,
                y,
                width,
                height,
                t("domain.command.next"),
                (20, 70, 50),
                value=ACTION_NEXT_IMAGE,
            ),
            *self.navigation_buttons(),
        ]

    def execute(self, command: Command) -> None:
        self.viewer.apply(command)
        self.history.append(command)
        self._feedback = t(f"domain.command.{command.value}")
        self._feedback_at = time.time()
        self.context.sound.success()

    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        height, width = frame.shape[:2]
        context = self.observe(frame.shape, detection)
        learned = None
        if self.recognizer is not None:
            features = (
                pose_features(context.landmarks, context.hand.label) if context.hand else None
            )
            learned = self.recognizer.update(features)
        tokens = self.tokenizer.update(
            context.states,
            context.at,
            clicked=False,  # clicks select on-screen buttons below
            depth=context.depth,
            learned=learned,
        )
        for command in self.mapper.update(tokens, context.at):
            if command is Command.LOCK:
                self._feedback = t("domain.locked" if self.mapper.locked else "domain.unlocked")
                self._feedback_at = context.at
            elif command is not Command.SELECT:
                self.execute(command)

        output = np.full_like(frame, 18)
        self._draw_image(output)
        draw_text(output, t("sterile.title"), (14, 28), 0.7, (230, 230, 230), 2)
        draw_text(
            output,
            t(
                "sterile.status",
                index=self.viewer.index + 1,
                total=self.viewer.count,
                zoom=f"{self.viewer.zoom:.1f}",
            ),
            (14, 54),
            0.5,
            (180, 200, 255),
        )
        for i in range(1, 5):
            draw_text(
                output, t(f"sterile.legend_{i}"), (width - 330, 24 + i * 18), 0.4, (170, 170, 170)
            )

        if self.mapper.locked:
            dim_frame(output, (0, 0, 60), 0.45)
            draw_text_centered(
                output, t("domain.locked"), width // 2, height // 2, 1.0, (0, 200, 255), 2
            )
        # While locked only the exit buttons work, so the user can always leave.
        all_buttons = self.buttons
        if self.mapper.locked:
            self.buttons = [b for b in all_buttons if b.value not in VIEWER_ACTIONS]
        action = self.draw_buttons(output, context.pointer)
        self.buttons = all_buttons
        if action == ACTION_PREVIOUS:
            self.execute(Command.PREVIOUS)
            action = None
        elif action == ACTION_NEXT_IMAGE:
            self.execute(Command.NEXT)
            action = None

        if context.pointer is not None:
            cv2.circle(output, context.pointer, 10, (0, 220, 255), -1)
        if self._feedback and time.time() - self._feedback_at < FEEDBACK_SECONDS:
            draw_text_centered(
                output, self._feedback, width // 2, height - 140, 0.9, (0, 255, 160), 2
            )
        if context.states is not None:
            draw_finger_hud(output, context.states, width - 155, height - 22)
        return output, action

    def _draw_image(self, output: np.ndarray) -> None:
        height, width = output.shape[:2]
        image = self.images[self.viewer.index]
        size = image.shape[0]
        crop = max(8, int(size / self.viewer.zoom))
        start = (size - crop) // 2
        view = image[start : start + crop, start : start + crop]
        side = min(height - 160, width - 360)
        if side <= 0:
            return
        view = cv2.resize(view, (side, side), interpolation=cv2.INTER_LINEAR)
        y0 = 70
        x0 = (width - side) // 2
        output[y0 : y0 + side, x0 : x0 + side] = view
        cv2.rectangle(output, (x0, y0), (x0 + side, y0 + side), (90, 90, 90), 1)
        draw_text(output, t("sterile.placeholder"), (x0 + 8, y0 + side - 10), 0.42, (230, 230, 230))
