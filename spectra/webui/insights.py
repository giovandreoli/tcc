"""Small, pure aggregations behind the panel's home dashboard and charts."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass

from spectra.storage.models import Session

WEEK_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class Overview:
    patients: int
    without_consent: int
    sessions_week: int
    minutes_week: int
    pain_relief_week: float | None


def pain_series(sessions: Iterable[Session]) -> list[dict]:
    """One row per finished session that has at least one pain score."""
    return [
        {
            "date": dt.datetime.fromtimestamp(s.started_at),
            "before": s.pain_before,
            "after": s.pain_after,
        }
        for s in sessions
        if s.ended_at is not None and (s.pain_before is not None or s.pain_after is not None)
    ]


def overview(
    patients: int, without_consent: int, sessions: Iterable[Session], now: float
) -> Overview:
    recent = [s for s in sessions if s.ended_at is not None and s.started_at >= now - WEEK_SECONDS]
    deltas = [
        s.pain_before - s.pain_after
        for s in recent
        if s.pain_before is not None and s.pain_after is not None
    ]
    return Overview(
        patients=patients,
        without_consent=without_consent,
        sessions_week=len(recent),
        minutes_week=round(sum(s.duration for s in recent) / 60),
        pain_relief_week=sum(deltas) / len(deltas) if deltas else None,
    )
