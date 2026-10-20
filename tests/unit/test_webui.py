"""Web look and feel: pure HTML builders, pain bands, dashboard aggregations and coaching."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spectra.browser.live import LiveStatus
from spectra.i18n import load_catalog
from spectra.storage.models import Session
from spectra.webui import art
from spectra.webui.insights import overview, pain_series


@pytest.mark.parametrize("mood", art.MOODS)
def test_every_mood_renders_an_accessible_svg(mood):
    svg = art.ghost_svg(mood, 80, label="Spectro")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'role="img"' in svg and 'aria-label="Spectro"' in svg


def test_ghost_img_embeds_the_svg_as_a_data_uri():
    import base64
    import re

    html = art.ghost_img("cheer", 64, label="Spectro")
    assert html.startswith("<img") and 'alt="Spectro"' in html and "sp-ghost" in html
    data = re.search(r'src="data:image/svg\+xml;base64,([^"]+)"', html).group(1)
    assert base64.b64decode(data).decode("utf-8") == art.ghost_svg("cheer", 64, label="Spectro")


def test_unknown_mood_falls_back_to_calm():
    assert art.ghost_svg("spooky") == art.ghost_svg("calm")


def test_user_text_is_escaped_in_html():
    html = art.bubble_html("calm", "<script>alert(1)</script>", "<b>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;b&gt;" in html
    assert "<img" not in art.stat_cards_html([("<img>", "1", "")])


@pytest.mark.parametrize(
    ("value", "band"),
    [(0, "none"), (1, "mild"), (3, "mild"), (4, "moderate"), (6, "moderate"), (7, "severe")],
)
def test_pain_bands_follow_the_numeric_scale(value, band):
    assert art.pain_band(value) == band


def test_stepper_marks_done_current_and_next():
    html = art.steps_html(["a", "b", "c"], 1)
    assert html.count("sp-step-done") == 1 and html.count('aria-current="step"') == 1
    assert "sp-step-next" in html


def test_progress_ring_is_clamped():
    assert "--sp-fill:360deg" in art.progress_ring_html(3.0, "00:00", "")
    assert "--sp-fill:0deg" in art.progress_ring_html(-1.0, "10:00", "")


def test_theme_is_offline_and_respects_reduced_motion():
    css = art.theme_css(large_text=True, high_contrast=True)
    assert "prefers-reduced-motion" in css
    assert "http://" not in css and "https://" not in css and "@import" not in css
    assert "font-size:19px" in css and "#3A23D6" in css
    splash = art.splash_html("SPECTRA", "tag")
    assert 'aria-hidden="true"' in splash and "pointer-events" not in splash


def _session(started, minutes, before=None, after=None, ended=True):
    return Session(
        patient_id="p",
        started_at=started,
        ended_at=started + minutes * 60 if ended else None,
        pain_before=before,
        pain_after=after,
    )


def test_overview_counts_only_last_week_finished_sessions():
    now = 10_000_000.0
    sessions = [
        _session(now - 3600, 20, 6, 3),
        _session(now - 7200, 10, 4, 4),
        _session(now - 8 * 24 * 3600, 30, 9, 1),
        _session(now - 60, 5, ended=False),
    ]
    summary = overview(3, 1, sessions, now)
    assert (summary.patients, summary.without_consent) == (3, 1)
    assert summary.sessions_week == 2 and summary.minutes_week == 30
    assert summary.pain_relief_week == pytest.approx(1.5)
    assert overview(0, 0, [], now).pain_relief_week is None


def test_pain_series_skips_sessions_without_scores():
    rows = pain_series([_session(1000, 5, 5, 2), _session(2000, 5), _session(3000, 5, ended=False)])
    assert [(row["before"], row["after"]) for row in rows] == [(5, 2)]


BASE_STATUS = LiveStatus(
    finished=False,
    saved=False,
    error_key=None,
    exercise="open_close",
    reps=0,
    target=10,
    total_reps=0,
    frames=10,
    hand_visible=True,
    elapsed_seconds=0.0,
    limit_seconds=600.0,
    exercise_index=0,
    exercise_total=1,
)


def _status(**changes):
    return replace(BASE_STATUS, **changes)


@pytest.mark.parametrize(
    ("changes", "mood"),
    [
        ({"frames": 0}, "think"),
        ({"hand_visible": False}, "think"),
        ({"reps": 2}, "calm"),
        ({"reps": 5}, "cheer"),
        ({"reps": 10}, "cheer"),
    ],
)
def test_coach_reacts_to_live_status(changes, mood):
    from spectra.browser.page import coach

    assert coach(_status(**changes))[0] == mood


def test_new_interface_keys_exist_in_catalog():
    catalog = load_catalog("pt_BR")
    keys = [
        "ux.splash.tagline",
        "ux.mascot.name",
        *(f"ux.nav.{s}" for s in ("home", "patients", "consent", "session", "history", "privacy")),
        *(f"ux.room.{s}" for s in ("waiting", "active", "completed", "interrupted", "expired")),
        *(f"ux.pain.band.{b}" for b in ("none", "mild", "moderate", "severe")),
        *(f"ux.checklist.{c}" for c in ("light", "framing", "distance", "comfort")),
        *(f"ux.patient.step.{s}" for s in ("enter", "prepare", "exercise", "finish")),
    ]
    assert [key for key in keys if key not in catalog] == []
