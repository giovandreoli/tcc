"""Pure HTML/CSS builders for the web interface (no Streamlit import, unit testable)."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

MOODS = ("hello", "calm", "cheer", "think", "sleep", "care")
SEVERE_PAIN = 8

_INK = "#2E2A55"


def _eyes(mood: str) -> str:
    if mood == "cheer":
        return (
            f'<path d="M40 60 Q46 50 52 60" stroke="{_INK}" stroke-width="4" fill="none"'
            ' stroke-linecap="round"/>'
            f'<path d="M68 60 Q74 50 80 60" stroke="{_INK}" stroke-width="4" fill="none"'
            ' stroke-linecap="round"/>'
        )
    if mood == "sleep":
        return (
            f'<path d="M40 60 Q46 65 52 60" stroke="{_INK}" stroke-width="3.5" fill="none"'
            ' stroke-linecap="round"/>'
            f'<path d="M68 60 Q74 65 80 60" stroke="{_INK}" stroke-width="3.5" fill="none"'
            ' stroke-linecap="round"/>'
        )
    pupil_dy = -3 if mood == "think" else 0
    eyes = (
        f'<ellipse cx="46" cy="{58 + pupil_dy}" rx="6" ry="8" fill="{_INK}"/>'
        f'<ellipse cx="74" cy="{58 + pupil_dy}" rx="6" ry="8" fill="{_INK}"/>'
        f'<circle cx="48" cy="{55 + pupil_dy}" r="2.2" fill="#fff"/>'
        f'<circle cx="76" cy="{55 + pupil_dy}" r="2.2" fill="#fff"/>'
    )
    if mood == "care":
        eyes += (
            f'<path d="M38 46 L52 49" stroke="{_INK}" stroke-width="3" stroke-linecap="round"/>'
            f'<path d="M82 46 L68 49" stroke="{_INK}" stroke-width="3" stroke-linecap="round"/>'
        )
    return eyes


def _mouth(mood: str) -> str:
    if mood == "cheer":
        return f'<path d="M50 74 Q60 90 70 74 Z" fill="{_INK}"/>'
    if mood == "think":
        return f'<circle cx="62" cy="78" r="3.5" fill="{_INK}"/>'
    if mood == "sleep":
        return f'<path d="M54 78 Q60 81 66 78" stroke="{_INK}" stroke-width="3" fill="none"/>'
    if mood == "care":
        return (
            f'<path d="M52 80 Q60 75 68 80" stroke="{_INK}" stroke-width="3" fill="none"'
            ' stroke-linecap="round"/>'
        )
    return (
        f'<path d="M52 75 Q60 84 68 75" stroke="{_INK}" stroke-width="3.5" fill="none"'
        ' stroke-linecap="round"/>'
    )


def _extras(mood: str) -> str:
    if mood == "hello":
        # A small open hand waving: the mascot greets with the gesture the app tracks.
        return (
            '<g class="sp-wave">'
            '<ellipse cx="111" cy="58" rx="9" ry="11" fill="url(#sp-body)" stroke="#CFC6FF"'
            ' stroke-width="2"/>'
            '<path d="M104 50 v-9 M109 48 v-11 M114 48 v-10 M119 51 v-7" stroke="#CFC6FF"'
            ' stroke-width="4" stroke-linecap="round"/>'
            "</g>"
        )
    if mood == "cheer":
        return (
            '<path class="sp-twinkle" d="M14 26 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3z"'
            ' fill="#FFC857"/>'
            '<path class="sp-twinkle sp-delay" d="M104 18 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z"'
            ' fill="#14B8A6"/>'
        )
    if mood == "think":
        return (
            '<text x="98" y="30" font-size="26" font-weight="700" fill="#6D5EF5"'
            ' font-family="Segoe UI, sans-serif">?</text>'
        )
    if mood == "sleep":
        return (
            '<text x="92" y="28" font-size="16" font-weight="700" fill="#8B7CFF"'
            ' font-family="Segoe UI, sans-serif">z</text>'
            '<text x="104" y="16" font-size="12" font-weight="700" fill="#B3A9FF"'
            ' font-family="Segoe UI, sans-serif">z</text>'
        )
    if mood == "care":
        return '<path d="M98 34 q5 9 0 12 q-5 -3 0 -12z" fill="#7CC8F0"/>'
    return ""


_SVG_ANIMATION = (
    "<style>"
    ".sp-wave{transform-origin:104px 66px;animation:w 1.6s ease-in-out infinite}"
    ".sp-twinkle{transform-box:fill-box;transform-origin:center;"
    "animation:k 1.8s ease-in-out infinite}"
    ".sp-delay{animation-delay:.6s}"
    "@keyframes w{0%,100%{transform:rotate(0)}25%{transform:rotate(14deg)}"
    "75%{transform:rotate(-8deg)}}"
    "@keyframes k{0%,100%{transform:scale(.6);opacity:.5}50%{transform:scale(1.1);opacity:1}}"
    "@media (prefers-reduced-motion:reduce){.sp-wave,.sp-twinkle{animation:none}}"
    "</style>"
)


def ghost_svg(mood: str = "calm", size: int = 96, *, label: str = "Spectro") -> str:
    """The friendly ghost: a soft, rounded figure, never spooky."""
    if mood not in MOODS:
        mood = "calm"
    height = round(size * 140 / 130)
    return (
        f'<svg class="sp-ghost" role="img" aria-label="{escape(label)}" width="{size}"'
        f' height="{height}" viewBox="-5 0 135 140" xmlns="http://www.w3.org/2000/svg">'
        f"<title>{escape(label)}</title>{_SVG_ANIMATION}"
        "<defs>"
        '<linearGradient id="sp-body" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E7E2FF"/>'
        "</linearGradient>"
        "</defs>"
        '<ellipse class="sp-shadow" cx="60" cy="134" rx="30" ry="4" fill="#6D5EF5"'
        ' opacity=".14"/>'
        '<path d="M14 62 C14 32 34 10 60 10 C86 10 106 32 106 62 L106 120'
        ' Q94.5 136 83 120 Q71.5 136 60 120 Q48.5 136 37 120 Q25.5 136 14 120 Z"'
        ' fill="url(#sp-body)" stroke="#CFC6FF" stroke-width="2.5"/>'
        f"{_eyes(mood)}"
        '<ellipse cx="36" cy="72" rx="6" ry="3.5" fill="#FF8FB1" opacity=".45"/>'
        '<ellipse cx="84" cy="72" rx="6" ry="3.5" fill="#FF8FB1" opacity=".45"/>'
        f"{_mouth(mood)}{_extras(mood)}</svg>"
    )


def ghost_img(mood: str = "calm", size: int = 96, *, label: str = "Spectro") -> str:
    """The ghost as a data-URI image: Streamlit's HTML sanitizer strips inline SVG."""
    data = base64.b64encode(ghost_svg(mood, size, label=label).encode("utf-8")).decode("ascii")
    height = round(size * 140 / 130)
    return (
        f'<img class="sp-ghost" src="data:image/svg+xml;base64,{data}" alt="{escape(label)}"'
        f' width="{size}" height="{height}"/>'
    )


def bubble_html(mood: str, message: str, title: str = "", *, size: int = 76) -> str:
    """The mascot with a speech bubble. ``message`` and ``title`` are escaped."""
    heading = f'<div class="sp-bubble-title">{escape(title)}</div>' if title else ""
    return (
        f'<div class="sp-guide sp-mood-{escape(mood)}">'
        f'<div class="sp-float">{ghost_img(mood, size)}</div>'
        f'<div class="sp-bubble">{heading}<div>{escape(message)}</div></div></div>'
    )


def hero_html(title: str, subtitle: str, mood: str = "hello") -> str:
    return (
        '<div class="sp-hero">'
        f'<div class="sp-float">{ghost_img(mood, 92)}</div>'
        f'<div><div class="sp-hero-title">{escape(title)}</div>'
        f'<div class="sp-hero-sub">{escape(subtitle)}</div></div></div>'
    )


def stat_cards_html(cards: list[tuple[str, str, str]]) -> str:
    """``cards`` is a list of ``(label, value, hint)``."""
    items = "".join(
        '<div class="sp-stat">'
        f'<div class="sp-stat-label">{escape(label)}</div>'
        f'<div class="sp-stat-value">{escape(value)}</div>'
        f'<div class="sp-stat-hint">{escape(hint)}</div></div>'
        for label, value, hint in cards
    )
    return f'<div class="sp-stats">{items}</div>'


def chip_html(text: str, tone: str) -> str:
    return f'<span class="sp-chip sp-chip-{escape(tone)}">{escape(text)}</span>'


def steps_html(labels: list[str], current: int) -> str:
    """A horizontal stepper; ``current`` is zero based."""
    items = []
    for index, label in enumerate(labels):
        state = "done" if index < current else "now" if index == current else "next"
        marker = "✓" if state == "done" else str(index + 1)
        current_attr = ' aria-current="step"' if state == "now" else ""
        items.append(
            f'<li class="sp-step sp-step-{state}"{current_attr}>'
            f'<span class="sp-step-dot">{marker}</span>'
            f'<span class="sp-step-label">{escape(label)}</span></li>'
        )
    return f'<ol class="sp-steps">{"".join(items)}</ol>'


def progress_ring_html(fraction: float, center: str, caption: str) -> str:
    fraction = max(0.0, min(1.0, fraction))
    return (
        f'<div class="sp-ring" style="--sp-fill:{fraction * 360:.0f}deg">'
        f'<div class="sp-ring-inner"><div class="sp-ring-value">{escape(center)}</div>'
        f'<div class="sp-ring-caption">{escape(caption)}</div></div></div>'
    )


def pain_band(value: int) -> str:
    """Bands of the 0–10 numeric pain scale (EVA/NRS)."""
    if value <= 0:
        return "none"
    if value <= 3:
        return "mild"
    if value <= 6:
        return "moderate"
    return "severe"


def splash_html(title: str, tagline: str) -> str:
    """Opening animation: CSS only, plays once and never blocks clicks."""
    letters = "".join(
        f'<span style="animation-delay:{0.35 + index * 0.07:.2f}s">{escape(letter)}</span>'
        for index, letter in enumerate(title)
    )
    return (
        '<div class="sp-splash" aria-hidden="true">'
        f'<div class="sp-splash-ghost">{ghost_img("hello", 150)}</div>'
        f'<div class="sp-splash-title">{letters}</div>'
        f'<div class="sp-splash-tag">{escape(tagline)}</div>'
        '<div class="sp-splash-bar"><span></span></div></div>'
    )


_BASE_CSS = (Path(__file__).with_name("theme.css")).read_text(encoding="utf-8")

_LARGE_TEXT_CSS = """
<style>
html{ font-size:19px; }
.stApp p, .stApp label, .stApp li{ font-size:1.1rem; }
</style>
"""

_HIGH_CONTRAST_CSS = """
<style>
:root{ --sp-primary:#3A23D6; --sp-primary-2:#3A23D6; --sp-ink:#000; --sp-muted:#23213A;
  --sp-border:#3A3851; --sp-bg:#FFFFFF; }
.stApp{ background:#FFFFFF !important; color:#000; }
.sp-bubble, .sp-stat, [data-testid="stForm"]{ border-width:2px !important; }
.sp-hero-title{ background:none; color:#000; }
</style>
"""


def theme_css(*, large_text: bool = False, high_contrast: bool = False) -> str:
    css = f"<style>{_BASE_CSS}</style>"
    if large_text:
        css += _LARGE_TEXT_CSS
    if high_contrast:
        css += _HIGH_CONTRAST_CSS
    return css
