"""Tracking quality: how well the camera actually saw the hand.

Every other metric is conditional on this one. A session where the hand was visible in
40% of frames produces a ROM number, but that number is meaningless, and the report must
say so rather than quietly presenting it next to a good session.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Thresholds below which a session is labelled unreliable in the report.
MIN_RELIABLE_PRESENCE = 0.60
MIN_RELIABLE_CONFIDENCE = 0.50


@dataclass(frozen=True)
class TrackingQuality:
    """Presence and confidence statistics for one session."""

    frames_total: int
    frames_with_hand: int
    mean_confidence: float

    @property
    def presence_ratio(self) -> float:
        return self.frames_with_hand / self.frames_total if self.frames_total else 0.0

    @property
    def reliable(self) -> bool:
        return (
            self.frames_total > 0
            and self.presence_ratio >= MIN_RELIABLE_PRESENCE
            and self.mean_confidence >= MIN_RELIABLE_CONFIDENCE
        )

    @property
    def warning_key(self) -> str | None:
        """i18n key explaining *why* the session is unreliable, or ``None``."""
        if self.frames_total == 0:
            return "metrics.quality.no_frames"
        if self.presence_ratio < MIN_RELIABLE_PRESENCE:
            return "metrics.quality.low_presence"
        if self.mean_confidence < MIN_RELIABLE_CONFIDENCE:
            return "metrics.quality.low_confidence"
        return None

    def to_dict(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_with_hand": self.frames_with_hand,
            "mean_confidence": self.mean_confidence,
            "presence_ratio": self.presence_ratio,
            "reliable": self.reliable,
        }
