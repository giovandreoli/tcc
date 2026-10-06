"""Text rendering helpers for OpenCV.

OpenCV only ships Hershey vector fonts, which cannot render accented characters: a
pt-BR string such as ``"Educação"`` would come out mangled. Every on-screen string is
therefore folded to ASCII right before drawing, while the catalog keeps proper pt-BR
spelling for reports and for the therapist panel.
"""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

ColorBGR = tuple[int, int, int]


def ascii_fold(text: str) -> str:
    """Strip diacritics so Hershey fonts can render the string."""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(ascii_fold(text), FONT, scale, thickness)
    return width, height


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.6,
    color: ColorBGR = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, ascii_fold(text), origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_text_centered(
    frame: np.ndarray,
    text: str,
    center_x: int,
    baseline_y: int,
    scale: float = 0.6,
    color: ColorBGR = (255, 255, 255),
    thickness: int = 1,
) -> None:
    width, _ = text_size(text, scale, thickness)
    draw_text(frame, text, (center_x - width // 2, baseline_y), scale, color, thickness)
