"""Streamlit wrappers around :mod:`spectra.webui.art`."""

from __future__ import annotations

import re

import streamlit as st

from spectra.i18n import t
from spectra.webui import art

LARGE_TEXT_KEY = "sp_large_text"
HIGH_CONTRAST_KEY = "sp_high_contrast"
SPLASH_KEY = "sp_splash_shown"
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")


def escape_markdown(text: str) -> str:
    """Render user text literally: names, masked CPFs (``***``) and therapist-written text."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def apply_theme() -> None:
    st.html(
        art.theme_css(
            large_text=bool(st.session_state.get(LARGE_TEXT_KEY)),
            high_contrast=bool(st.session_state.get(HIGH_CONTRAST_KEY)),
        )
    )


def splash_once() -> None:
    """Play the opening animation once per browser session."""
    if st.session_state.get(SPLASH_KEY):
        return
    st.session_state[SPLASH_KEY] = True
    st.html(art.splash_html("SPECTRA", t("ux.splash.tagline")))


def accessibility_controls() -> None:
    st.toggle(t("ux.a11y.large_text"), key=LARGE_TEXT_KEY)
    st.toggle(t("ux.a11y.high_contrast"), key=HIGH_CONTRAST_KEY)


def guide(mood: str, message: str, title: str = "") -> None:
    st.html(art.bubble_html(mood, message, title))


def hero(title: str, subtitle: str, mood: str = "hello") -> None:
    st.html(art.hero_html(title, subtitle, mood))


def stats(cards: list[tuple[str, str, str]]) -> None:
    st.html(art.stat_cards_html(cards))


def steps(labels: list[str], current: int) -> None:
    st.html(art.steps_html(labels, current))
