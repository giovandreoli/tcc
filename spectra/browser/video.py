"""Local camera preview with a send-only WebRTC analysis connection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import av
import streamlit as st
from aiortc import RTCConfiguration
from streamlit.components.v1 import declare_component
from streamlit_webrtc import VideoProcessorBase, WebRtcMode
from streamlit_webrtc.webrtc import WebRtcWorker

from spectra.browser.session import BrowserSession
from spectra.i18n import t

logger = logging.getLogger(__name__)
_component = declare_component("spectra_camera", path=str(Path(__file__).with_name("camera")))


class ExerciseVideoProcessor(VideoProcessorBase):
    def __init__(self, session: BrowserSession) -> None:
        self.session = session

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.session.process(frame.to_ndarray(format="bgr24"))
        return frame

    def on_ended(self) -> None:
        self.session.finish()


@dataclass
class CameraConnection:
    worker: WebRtcWorker
    offer_id: str
    answer: dict[str, str]


def _state_key(session: BrowserSession) -> str:
    return f"camera_connection_{session.admission.session.id}"


def stop_camera(session: BrowserSession) -> None:
    connection = st.session_state.pop(_state_key(session), None)
    if connection is not None:
        connection.worker.stop()


def _create_worker(session: BrowserSession) -> WebRtcWorker:
    # This adapter targets the pinned streamlit-webrtc 0.63.11 worker API.
    return WebRtcWorker(
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration(iceServers=[]),
        source_video_track=None,
        source_audio_track=None,
        player_factory=None,
        in_recorder_factory=None,
        out_recorder_factory=None,
        video_frame_callback=None,
        audio_frame_callback=None,
        queued_video_frames_callback=None,
        queued_audio_frames_callback=None,
        on_video_ended=None,
        on_audio_ended=None,
        video_processor_factory=lambda: ExerciseVideoProcessor(session),
        audio_processor_factory=None,
        async_processing=True,
        video_receiver_size=1,
        audio_receiver_size=1,
        sendback_video=False,
        sendback_audio=False,
    )


def camera_streamer(session: BrowserSession) -> None:
    connection = st.session_state.get(_state_key(session))
    value = _component(
        key=f"camera_{session.admission.session.id}",
        answer=connection.answer if connection is not None else None,
        labels={
            "start": t("browser.camera_start"),
            "stop": t("browser.camera_stop"),
            "device": t("browser.select_device"),
            "connecting": t("browser.camera_connecting"),
            "connected": t("browser.camera_connected"),
            "denied": t("browser.camera_denied"),
            "missing": t("browser.camera_missing"),
            "error": t("browser.processing_error"),
        },
        default=None,
    )
    if not value:
        return
    if value.get("error"):
        logger.error("local camera connection failed: %s", value["error"])
        session.error_key = "browser.processing_error"
        stop_camera(session)
        session.finish()
        st.error(t("browser.processing_error"))
        return
    if not value.get("playing"):
        stop_camera(session)
        session.finish()
        return
    offer_id = value.get("id")
    offer = value.get("offer")
    if not offer or (connection is not None and connection.offer_id == offer_id):
        return
    if connection is not None:
        raise RuntimeError("a camera connection is already active")
    worker = _create_worker(session)
    try:
        worker.process_offer(offer["sdp"], offer["type"], timeout=15)
        description = worker.pc.localDescription
        if description is None:
            raise RuntimeError("camera negotiation produced no answer")
        st.session_state[_state_key(session)] = CameraConnection(
            worker=worker,
            offer_id=offer_id,
            answer={"id": offer_id, "sdp": description.sdp, "type": description.type},
        )
    except Exception:
        logger.exception("local camera negotiation failed")
        session.error_key = "browser.processing_error"
        worker.stop()
        session.finish()
        raise
    st.rerun()
