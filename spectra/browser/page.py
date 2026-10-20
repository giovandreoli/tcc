"""Patient browser entry point, hosted by the same local Streamlit application."""

from __future__ import annotations

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from spectra.browser.rooms import RoomError, Rooms
from spectra.browser.session import BrowserSession
from spectra.i18n import t
from spectra.service import SpectraService


class ExerciseVideoProcessor(VideoProcessorBase):
    def __init__(self, session: BrowserSession) -> None:
        self.session = session

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = self.session.process(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(image, format="bgr24")


@st.fragment(run_every=2)
def render_status(session: BrowserSession) -> None:
    finished, saved, error_key, exercise, reps, frames = session.snapshot()
    if error_key:
        st.error(t(error_key))
    elif finished:
        if saved:
            st.success(t("browser.saved"))
        else:
            st.warning(t("browser.no_frames"))
    elif frames == 0:
        st.info(t("browser.waiting_camera"))
    else:
        st.info(t("browser.progress", exercise=t(f"physio.exercise.{exercise}"), reps=reps))


def render_patient(service: SpectraService) -> None:
    st.title(t("browser.title"))
    st.warning(t("browser.local_only"))
    st.caption(t("browser.privacy"))
    session: BrowserSession | None = st.session_state.get("browser_session")
    if session is None:
        with st.form("room_login"):
            code = st.text_input(t("browser.code"), value=st.query_params.get("sala", ""))
            pin = st.text_input(t("browser.pin"), type="password", max_chars=4)
            submitted = st.form_submit_button(t("browser.enter"), type="primary")
        if submitted:
            try:
                admission = Rooms(service).admit(code, pin)
            except RoomError as exc:
                st.error(t(str(exc)))
            else:
                st.session_state["browser_session"] = BrowserSession(service.config, admission)
                st.query_params.clear()
                st.query_params["paciente"] = "1"
                st.rerun()
        return

    st.caption(t("browser.camera_hint"))
    finished, _, error_key, _, _, _ = session.snapshot()
    if not finished and not error_key:
        webrtc_streamer(
            key="physio_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": []},
            server_rtc_configuration={"iceServers": []},
            media_stream_constraints={
                "video": {"width": {"ideal": 960}, "height": {"ideal": 720}},
                "audio": False,
            },
            video_processor_factory=lambda: ExerciseVideoProcessor(session),
            on_video_ended=session.finish,
            async_processing=False,
            translations={
                "start": t("browser.camera_start"),
                "stop": t("browser.camera_stop"),
                "select_device": t("browser.select_device"),
                "media_api_not_available": t("browser.camera_unavailable"),
                "device_ask_permission": t("browser.camera_permission"),
                "device_not_available": t("browser.camera_missing"),
                "device_access_denied": t("browser.camera_denied"),
            },
        )
    render_status(session)
    if not finished and st.button(t("browser.finish"), type="primary"):
        session.finish()
        st.rerun()
    if finished and st.button(t("browser.another")):
        del st.session_state["browser_session"]
        st.rerun()
