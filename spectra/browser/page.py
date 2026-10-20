"""Patient browser entry point, hosted by the same local Streamlit application.

Guided flow: enter (code + PIN) → prepare (pain check-in, environment checklist,
exercise preview) → exercise (camera with live progress) → finish (summary and
post-session pain). Spectro, the mascot, explains each step in plain language.
"""

from __future__ import annotations

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from spectra.browser.live import LiveStatus
from spectra.browser.rooms import RoomError, Rooms
from spectra.browser.session import BrowserSession
from spectra.i18n import t
from spectra.service import SpectraService
from spectra.webui import art
from spectra.webui import components as ui

SESSION_KEY = "browser_session"
STAGE_KEY = "browser_stage"
PAIN_AFTER_KEY = "browser_pain_after_saved"
CHECKLIST = ("light", "framing", "distance", "comfort")
STEPS = ("enter", "prepare", "exercise", "finish")


class ExerciseVideoProcessor(VideoProcessorBase):
    def __init__(self, session: BrowserSession) -> None:
        self.session = session

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = self.session.process(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def _pain_label(value: int) -> str:
    return f"{value} — {t(f'ux.pain.band.{art.pain_band(value)}')}"


def _minutes(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _stepper(stage: str) -> None:
    ui.steps([t(f"ux.patient.step.{step}") for step in STEPS], STEPS.index(stage))


def _reset() -> None:
    for key in (SESSION_KEY, STAGE_KEY, PAIN_AFTER_KEY):
        st.session_state.pop(key, None)


def coach(status: LiveStatus) -> tuple[str, str]:
    """Mascot mood and message for the live status."""
    if status.frames == 0:
        return "think", t("browser.waiting_camera")
    if not status.hand_visible:
        return "think", t("ux.patient.coach.show_hand")
    if status.target and status.reps >= status.target:
        return "cheer", t("ux.patient.coach.exercise_done")
    if status.target and status.reps >= status.target / 2:
        return "cheer", t("ux.patient.coach.halfway")
    return "calm", t("ux.patient.coach.keep_going")


# ------------------------------------------------------------------- enter
def _render_enter(service: SpectraService) -> None:
    ui.hero(t("ux.patient.title"), t("ux.patient.subtitle"))
    _stepper("enter")
    ui.guide("hello", t("ux.patient.welcome"), t("ux.mascot.name"))
    with st.form("room_login"):
        code = st.text_input(
            t("browser.code"),
            value=st.query_params.get("sala", ""),
            placeholder="XXXX-XXXX-XXXX",
        )
        pin = st.text_input(t("browser.pin"), type="password", max_chars=4)
        submitted = st.form_submit_button(
            t("browser.enter"), type="primary", use_container_width=True
        )
    if submitted:
        try:
            admission = Rooms(service).admit(code, pin)
        except RoomError as exc:
            st.error(t(str(exc)))
        else:
            st.session_state[SESSION_KEY] = BrowserSession(service.config, admission)
            st.session_state[STAGE_KEY] = "prepare"
            st.query_params.clear()
            st.query_params["paciente"] = "1"
            st.rerun()
    st.caption(t("browser.local_only"))


# ----------------------------------------------------------------- prepare
def _render_prepare(session: BrowserSession) -> None:
    _stepper("prepare")
    ui.guide("hello", t("ux.patient.prepare_intro"), t("ux.mascot.name"))

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(t("ux.pain.question"))
        pain = st.select_slider(
            t("ux.pain.scale"),
            options=list(range(11)),
            value=0,
            format_func=_pain_label,
            key="pain_before",
        )
        st.subheader(t("ux.checklist.title"))
        checks = [st.checkbox(t(f"ux.checklist.{item}")) for item in CHECKLIST]
    with right:
        st.subheader(t("ux.patient.today"))
        for index, name in enumerate(session.admission.exercises, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {t(f'physio.exercise.{name}')}**")
                for line in range(1, 4):
                    st.caption(t(f"physio.instructions.{name}_{line}"))
        st.caption(t("ux.patient.duration", minutes=session.admission.duration_minutes))

    if pain >= art.SEVERE_PAIN:
        ui.guide("care", t("ux.pain.severe_before"), t("ux.pain.severe_title"))
        st.button(t("ux.patient.leave"), type="primary", on_click=_leave, args=(session,))
        return
    ready = all(checks)
    if not ready:
        st.caption(t("ux.checklist.pending"))
    st.button(
        t("ux.patient.ready"),
        type="primary",
        disabled=not ready,
        use_container_width=True,
        on_click=_start_exercise,
        args=(session,),
    )


def _leave(session: BrowserSession) -> None:
    session.finish()
    st.session_state[STAGE_KEY] = "left"


def _start_exercise(session: BrowserSession) -> None:
    session.pain_before = st.session_state["pain_before"]
    st.session_state[STAGE_KEY] = "exercise"


def _finish(session: BrowserSession) -> None:
    session.finish()
    st.session_state[STAGE_KEY] = "finish"


# ---------------------------------------------------------------- exercise
@st.fragment(run_every=2)
def _render_live(session: BrowserSession) -> None:
    status = session.status()
    if status.finished or status.error_key:
        st.session_state[STAGE_KEY] = "finish"
        st.rerun(scope="app")
        return
    mood, message = coach(status)
    ui.guide(mood, message)
    st.html(
        art.progress_ring_html(
            status.time_fraction,
            _minutes(status.limit_seconds - status.elapsed_seconds),
            t("ux.live.remaining"),
        )
    )
    exercise_label = t(
        "ux.live.exercise", current=status.exercise_index + 1, total=status.exercise_total
    )
    ui.stats(
        [
            (exercise_label, t(f"physio.exercise.{status.exercise}"), ""),
            (
                t("ux.live.reps"),
                f"{status.reps}/{status.target or '—'}",
                t("ux.live.total", total=status.total_reps),
            ),
        ]
    )


def _render_exercise(session: BrowserSession) -> None:
    _stepper("exercise")
    camera, panel = st.columns([3, 2], gap="large")
    with camera:
        st.caption(t("browser.camera_hint"))
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
        st.caption(t("browser.privacy"))
    with panel:
        _render_live(session)
        st.button(
            t("browser.finish"),
            type="primary",
            use_container_width=True,
            on_click=_finish,
            args=(session,),
        )


# ------------------------------------------------------------------ finish
def _save_pain_after(service: SpectraService, session: BrowserSession) -> None:
    pain = st.session_state["pain_after_value"]
    service.db.record_pain_after(session.admission.session.id, pain)
    st.session_state[PAIN_AFTER_KEY] = pain


def _render_pain_after(service: SpectraService, session: BrowserSession) -> None:
    saved = st.session_state.get(PAIN_AFTER_KEY)
    if saved is None:
        with st.form("pain_after"):
            st.select_slider(
                t("ux.pain.after_question"),
                options=list(range(11)),
                value=0,
                format_func=_pain_label,
                key="pain_after_value",
            )
            st.form_submit_button(
                t("ux.pain.save"),
                type="primary",
                on_click=_save_pain_after,
                args=(service, session),
            )
        return
    before = session.pain_before or 0
    if saved >= art.SEVERE_PAIN or saved - before >= 3:
        ui.guide("care", t("ux.pain.severe_after"), t("ux.pain.severe_title"))
    else:
        st.success(t("ux.pain.saved"))


def _render_finish(service: SpectraService, session: BrowserSession) -> None:
    _stepper("finish")
    status = session.status()
    if status.error_key:
        ui.guide("care", t(status.error_key))
        st.error(t(status.error_key))
    elif not status.saved:
        ui.guide("think", t("browser.no_frames"))
    else:
        ui.guide("cheer", t("ux.patient.celebrate"), t("ux.patient.celebrate_title"))
        ui.stats(
            [
                (t("ux.summary.time"), _minutes(status.elapsed_seconds), ""),
                (t("ux.summary.reps"), str(status.total_reps), ""),
                (t("ux.summary.exercises"), str(status.exercise_total), ""),
            ]
        )
        _render_pain_after(service, session)
    st.button(t("browser.another"), on_click=_reset)


def render_patient(service: SpectraService) -> None:
    with st.sidebar:
        st.markdown(f"### {t('ux.a11y.title')}")
        ui.accessibility_controls()
    session: BrowserSession | None = st.session_state.get(SESSION_KEY)
    if session is None:
        _render_enter(service)
        return
    stage = st.session_state.get(STAGE_KEY, "prepare")
    if stage == "left":
        ui.guide("care", t("ux.pain.left"), t("ux.mascot.name"))
        st.button(t("browser.another"), on_click=_reset)
        return
    if stage == "finish" or session.status().finished:
        _render_finish(service, session)
    elif stage == "exercise":
        _render_exercise(session)
    else:
        _render_prepare(session)
