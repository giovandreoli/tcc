"""Locate (and if needed download) the MediaPipe hand landmark model."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: Path) -> Path:
    """Return ``model_path``, downloading the model there if it is missing."""
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading hand landmark model to %s", model_path)
    tmp_path = model_path.with_suffix(model_path.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(model_path)
    logger.info("Model download finished")
    return model_path
