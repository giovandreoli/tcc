"""Free drawing, driven by the gesture state machine.

Gestures select the *tool* (paint, erase, open the colour menu, pause); colours are
picked by dwelling on a swatch inside the menu. This replaces the original
"finger combination equals colour" mapping, which forced the patient to remember ten
combinations and made every transitional posture paint a stripe.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from spectra.core.canvas import DEFAULT_BRUSH_SIZE, ERASER_SIZE, DrawingCanvas
from spectra.core.trail import Trail
from spectra.detection.hand_detector import DetectionResult
from spectra.gestures.state_machine import GestureState
from spectra.i18n import t
from spectra.modes.base import BaseMode, FrameContext, ModeContext
from spectra.modes.palette import COLOR_ENTRIES
from spectra.ui.button import Button
from spectra.ui.text import draw_text, draw_text_centered
from spectra.ui.widgets import dim_frame, draw_finger_hud, draw_progress_bar

ACTION_CLEAR = "clear"
ACTION_SAVE = "save"
ACTION_UNDO = "undo"

FEEDBACK_SECONDS = 2.5

#: Gestures during which the dwell buttons stay live. While painting or erasing the
#: pointer is busy drawing and must not trip a button it happens to cross.
INTERACTIVE_STATES = (GestureState.IDLE, GestureState.POINTING)


class FreeDrawMode(BaseMode):
    records_metrics = True

    def __init__(self, context: ModeContext) -> None:
        super().__init__(context)
        config = context.config
        self.canvas = DrawingCanvas(self.width, self.height, config.undo_history)
        self.trail = Trail(config.trail_length)
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.color_label = COLOR_ENTRIES[0].label
        self.color_bgr: tuple[int, int, int] = COLOR_ENTRIES[0].color or (0, 0, 255)
        self.previous_point: tuple[int, int] | None = None
        self.drawing = False
        self._feedback_text = ""
        self._feedback_at = 0.0

        button_width, button_height = 112, 42
        y = self.height - button_height - 10
        self.buttons = [
            self.make_button(
                10,
                y,
                button_width,
                button_height,
                t("paint.clear"),
                (80, 20, 20),
                value=ACTION_CLEAR,
            ),
            self.make_button(
                130,
                y,
                button_width,
                button_height,
                t("paint.save"),
                (20, 70, 20),
                value=ACTION_SAVE,
            ),
            self.make_button(
                250,
                y,
                button_width,
                button_height,
                t("paint.undo"),
                (80, 60, 10),
                value=ACTION_UNDO,
            ),
        ]
        self.buttons += self.navigation_buttons()
        self.swatches = self._build_swatches()

    # -------------------------------------------------------------- colour menu
    def _build_swatches(self) -> list[Button]:
        size, gap = 86, 16
        columns = len(COLOR_ENTRIES)
        total = columns * size + (columns - 1) * gap
        x0 = (self.width - total) // 2
        y = self.height // 2 - size // 2
        return [
            self.make_button(
                x0 + i * (size + gap),
                y,
                size,
                size,
                "",
                entry.color or (200, 200, 200),
                value=i,
            )
            for i, entry in enumerate(COLOR_ENTRIES)
        ]

    def _draw_color_menu(self, frame: np.ndarray, pointer: tuple[int, int] | None) -> None:
        dim_frame(frame, (10, 10, 10), 0.7)
        draw_text_centered(
            frame, t("paint.pick_color"), self.width // 2, self.height // 2 - 90, 0.8
        )
        for index, swatch in enumerate(self.swatches):
            if swatch.update_hover(swatch.contains(pointer)):
                entry = COLOR_ENTRIES[index]
                self.color_label = entry.label
                self.color_bgr = entry.color or self.color_bgr
                self._feedback(entry.label)
                self.context.sound.success()
            swatch.draw(frame)
            x, y, width, height = swatch.rect
            draw_text_centered(
                frame, COLOR_ENTRIES[index].label, x + width // 2, y + height + 22, 0.45
            )

    def _reset_swatches(self) -> None:
        for swatch in self.swatches:
            swatch.reset()

    # ----------------------------------------------------------------- feedback
    def _feedback(self, message: str) -> None:
        self._feedback_text = message
        self._feedback_at = time.time()

    # ------------------------------------------------------------------ painting
    def _paint(self, point: tuple[int, int] | None, color: tuple[int, int, int] | None) -> None:
        if point is None:
            self._stop_painting()
            return
        if not self.drawing:
            self.canvas.begin_stroke()
            self.drawing = True
        if self.previous_point is not None:
            size = ERASER_SIZE if color is None else self.brush_size
            self.canvas.draw(self.previous_point, point, color, size)
        self.trail.add(point, color)
        self.previous_point = point

    def _stop_painting(self) -> None:
        self.previous_point = None
        self.drawing = False

    def save(self) -> None:
        path = self.context.config.drawings_dir / f"pintura_{int(time.time())}.png"
        try:
            self.canvas.save(path)
        except OSError:
            self._feedback(t("paint.save_failed"))
            self.context.sound.error()
            return
        self._feedback(t("paint.saved", filename=path.name))
        self.context.sound.success()

    # ------------------------------------------------------------------- process
    def process(self, frame: np.ndarray, detection: DetectionResult) -> tuple[np.ndarray, Any]:
        height, width = frame.shape[:2]
        context = self.observe(frame.shape, detection)
        composite = cv2.addWeighted(self.canvas.image, 0.65, frame, 0.35, 0)

        if context.gesture is GestureState.PAINTING:
            self._paint(context.pointer, self.color_bgr)
        elif context.gesture is GestureState.ERASING:
            self._paint(context.pointer, None)
        else:
            self._stop_painting()

        if context.gesture is not GestureState.COLOR_MENU:
            self._reset_swatches()

        self.trail.draw(composite)
        action = self._draw_controls(composite, context)
        self._draw_hud(composite, context)

        if context.gesture is GestureState.COLOR_MENU:
            self._draw_color_menu(composite, context.pointer)
        elif context.paused:
            dim_frame(composite, (0, 0, 40), 0.6)
            draw_text_centered(
                composite, t("safety.paused"), width // 2, height // 2, 0.9, (0, 220, 255), 2
            )

        self._draw_cursor(composite, context)

        if self._feedback_text and (time.time() - self._feedback_at) < FEEDBACK_SECONDS:
            draw_text_centered(
                composite, self._feedback_text, width // 2, height - 160, 1.0, (0, 220, 255), 3
            )

        if context.states is not None:
            draw_finger_hud(composite, context.states, width - 155, height - 68)

        return composite, self._apply_action(action)

    def _draw_controls(self, frame: np.ndarray, context: FrameContext) -> Any:
        interactive = context.gesture in INTERACTIVE_STATES
        pointer = context.pointer if interactive else None
        if not interactive:
            for button in self.buttons:
                button.reset()
        return self.draw_buttons(frame, pointer)

    def _apply_action(self, action: Any) -> Any:
        if action == ACTION_CLEAR:
            self.canvas.clear()
            self.trail.clear()
            self._feedback(t("paint.canvas_cleared"))
            return None
        if action == ACTION_SAVE:
            self.save()
            return None
        if action == ACTION_UNDO:
            undone = self.canvas.undo()
            self._feedback(t("paint.undone") if undone else t("paint.nothing_to_undo"))
            return None
        return action

    def _draw_cursor(self, frame: np.ndarray, context: FrameContext) -> None:
        if context.pointer is None:
            return
        erasing = context.gesture is GestureState.ERASING
        radius = ERASER_SIZE // 2 if erasing else self.brush_size
        color = (255, 255, 255) if erasing else self.color_bgr
        cv2.circle(frame, context.pointer, radius, color, -1 if not erasing else 2)
        cv2.circle(frame, context.pointer, radius + 2, (20, 20, 20), 2)

    def _draw_hud(self, frame: np.ndarray, context: FrameContext) -> None:
        width = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (width, 50), (15, 15, 15), -1)
        draw_text(frame, t("paint.title"), (10, 22), 0.62)
        cv2.circle(frame, (230, 14), 11, self.color_bgr, -1)
        cv2.circle(frame, (230, 14), 12, (255, 255, 255), 1)
        draw_text(frame, self.color_label, (248, 20), 0.58)
        draw_text(frame, t("paint.brush", size=self.brush_size), (420, 20), 0.48, (180, 180, 180))
        draw_text(
            frame,
            t("paint.gesture", gesture=t(f"gesture.state.{context.gesture.value}")),
            (14, frame.shape[0] - 84),
            0.62,
            (220, 220, 220),
        )
        progress = self.engine.machine.hold_progress
        if progress > 0 and context.gesture is GestureState.IDLE:
            draw_progress_bar(frame, (14, frame.shape[0] - 74), (160, 8), progress)
        for i, key in enumerate(("paint.legend_1", "paint.legend_2")):
            draw_text(frame, t(key), (width - 420, 18 + i * 16), 0.38, (170, 170, 170))
