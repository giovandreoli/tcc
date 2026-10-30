"""Streamlit view of the gesture learning path, shared by the panel and the patient page."""

from __future__ import annotations

import streamlit as st

from spectra.browser.sinks import TutorialSink
from spectra.browser.video import camera_streamer, reset_camera
from spectra.i18n import t
from spectra.learning.tutorial import LESSON_KEYS
from spectra.webui import art


def _stars(count: int) -> str:
    return "★" * count + "☆" * (3 - count)


def render_tutorial(sink: TutorialSink, camera_key: str) -> None:
    camera, panel = st.columns([2, 1], gap="large")
    with camera:
        _camera(sink, camera_key)
    with panel:
        _status(sink)


@st.fragment
def _camera(sink: TutorialSink, camera_key: str) -> None:
    camera_streamer(sink, camera_key)


@st.fragment(run_every=1)
def _status(sink: TutorialSink) -> None:
    status = sink.status()
    st.caption(
        t("learning.path.step", current=min(status.index + 1, status.total), total=status.total)
    )
    if status.finished:
        st.subheader(t("learning.path.done_title"))
        st.html(art.bubble_html("cheer", t("learning.path.done", stars=status.stars), size=44))
    else:
        st.subheader(t(f"learning.lesson.{status.lesson}.title"))
        message = (
            t(f"learning.lesson.{status.lesson}.how")
            if status.hand_visible
            else t("exercise.cue.show_hand")
        )
        st.html(art.bubble_html("calm" if status.hand_visible else "think", message, size=44))
        st.progress(status.progress)
        if status.target > 1:
            st.caption(t("learning.path.count", count=status.count, target=status.target))
        if status.lesson in ("depth", "rotate") and status.depth is not None:
            st.caption(
                t(
                    "learning.path.depth",
                    scale=f"{status.depth.scale:.2f}",
                    motion=t(f"depth.motion.{status.depth.motion.value}"),
                    tilt=f"{status.depth.tilt_degrees:.0f}",
                )
            )
        if st.button(t("learning.path.skip"), key="tutorial_skip"):
            sink.skip()
            st.rerun(scope="fragment")
    if status.results:
        st.markdown(f"**{t('learning.path.results')}**")
        for result in status.results:
            label = t(f"learning.lesson.{result.key}.title")
            detail = t("learning.path.skipped") if result.skipped else f"{result.seconds:.1f} s"
            st.caption(f"{_stars(result.stars)}  {label} — {detail}")


def reset_tutorial(state_key: str, camera_key: str) -> None:
    st.session_state.pop(state_key, None)
    reset_camera(camera_key)


__all__ = ["LESSON_KEYS", "render_tutorial", "reset_tutorial"]
