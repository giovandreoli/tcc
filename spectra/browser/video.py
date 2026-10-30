"""Camera component: hand tracking in the patient's browser, landmarks to the session.

The component (``camera/``) opens the webcam, shows a mirrored preview and runs the
MediaPipe hand landmarker **inside the browser**, from files served by this package, with
no CDN. Only 21 normalised points per analysed frame are sent back, in small batches.

Delivery is acknowledged: the browser keeps every frame newer than the last timestamp the
server confirmed and resends them with the next batch, so a batch dropped while Streamlit
is busy is not lost; the server ignores frames it has already processed.
"""

from __future__ import annotations

import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.components.v1 import declare_component

from spectra.browser.frames import BatchError, parse_batch
from spectra.browser.session import LandmarkSession
from spectra.config import load_config
from spectra.i18n import t

logger = logging.getLogger(__name__)

COMPONENT_DIR = Path(__file__).with_name("camera")
MODEL_NAME = "hand_landmarker.task"

# Windows registries often map .js to text/plain and lack .wasm, which browsers reject.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")


def ensure_model() -> Path:
    """Copy the bundled hand model next to the component so the browser can fetch it."""
    target = COMPONENT_DIR / MODEL_NAME
    if not target.is_file():
        source = load_config().model_path()
        try:
            if source.is_file():
                shutil.copyfile(source, target)
        except OSError:
            logger.exception("could not place the hand model next to the camera component")
    return target


ensure_model()
_component = declare_component("spectra_camera", path=str(COMPONENT_DIR))


def _ack_key(key: str) -> str:
    return f"{key}_ack_ms"


def camera_labels() -> dict[str, str]:
    keys = {
        name: f"browser.camera_{name}"
        for name in (
            "start",
            "stop",
            "loading",
            "connected",
            "denied",
            "missing",
            "error",
            "unsupported",
            "fps",
        )
    }
    keys["device"] = "browser.select_device"
    return {name: t(key) for name, key in keys.items()}


def handle_value(session: LandmarkSession, value: Any, key: str) -> None:
    """Apply one component value to the session (separated for unit tests)."""
    if not isinstance(value, dict):
        return
    try:
        frames = parse_batch(value.get("frames") or [])
    except BatchError:
        logger.warning("ignored a malformed landmark batch from the browser")
        frames = []
    acknowledged = float(st.session_state.get(_ack_key(key), -1.0))
    fresh = [frame for frame in frames if frame.at_ms > acknowledged]
    if fresh:
        session.process_batch(fresh)
        st.session_state[_ack_key(key)] = fresh[-1].at_ms
    if value.get("error"):
        # Only an error category is ever sent: no device names and no images.
        logger.warning("browser camera reported %s", str(value["error"])[:40])
        if session.runner is not None:
            session.finish()
        return
    if value.get("playing") is False and session.runner is not None:
        session.finish()


def camera_streamer(session: LandmarkSession, key: str) -> None:
    value = _component(
        key=key,
        labels=camera_labels(),
        ack=st.session_state.get(_ack_key(key), -1.0),
        default=None,
    )
    handle_value(session, value, key)


def reset_camera(key: str) -> None:
    st.session_state.pop(_ack_key(key), None)
