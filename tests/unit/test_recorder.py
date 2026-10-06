from __future__ import annotations

import pytest

from spectra.metrics.recorder import (
    MIN_RELIABLE_PRESENCE,
    FrameSample,
    SessionRecorder,
    TrackingQuality,
)
from tests.conftest import make_hand

OPEN = (True, True, True, True, True)
FIST = (False, False, False, False, False)


class TestSampling:
    def test_samples_are_throttled_to_the_configured_rate(self):
        recorder = SessionRecorder(rate_hz=10.0)
        for i in range(100):  # 100 frames over 1 second, i.e. 100 fps
            recorder.observe(make_hand(*OPEN), 0.9, now=i * 0.01)
        assert recorder.frames_total == 100
        assert 9 <= len(recorder.samples) <= 12

    def test_the_first_frame_is_always_stored(self):
        recorder = SessionRecorder()
        assert recorder.observe(make_hand(*OPEN), 0.9, now=0.0) is not None

    def test_frames_without_a_hand_are_counted_but_not_stored(self):
        recorder = SessionRecorder()
        recorder.observe(None, now=0.0)
        assert recorder.frames_total == 1
        assert recorder.frames_with_hand == 0
        assert recorder.samples == []

    def test_a_sample_carries_only_derived_features(self):
        recorder = SessionRecorder()
        sample = recorder.observe(make_hand(*OPEN), 0.8, now=0.0)
        assert isinstance(sample, FrameSample)
        assert len(sample.flexion) == 5
        assert len(sample.opposition) == 4
        # no image, no landmark dump: the stored fields are exactly these
        assert set(sample.to_dict()) == {
            "at",
            "flexion",
            "opposition",
            "thumb_abduction",
            "index_tip",
            "palm_size",
            "palm_rotation",
            "palm_openness",
            "confidence",
        }

    def test_flexion_separates_open_from_closed(self):
        recorder = SessionRecorder()
        open_sample = recorder.observe(make_hand(*OPEN), 0.9, now=0.0)
        closed_sample = recorder.observe(make_hand(*FIST), 0.9, now=10.0)
        assert open_sample.flexion[1] < closed_sample.flexion[1]

    def test_reset_clears_counters_and_samples(self):
        recorder = SessionRecorder()
        recorder.observe(make_hand(*OPEN), 0.9, now=0.0)
        recorder.reset()
        assert recorder.samples == []
        assert recorder.frames_total == 0

    def test_window_slices_by_timestamp(self):
        recorder = SessionRecorder(rate_hz=1000.0)
        for i in range(10):
            recorder.observe(make_hand(*OPEN), 0.9, now=float(i))
        assert len(recorder.window(3.0, 5.0)) == 3


class TestTrackingQuality:
    def test_presence_ratio(self):
        recorder = SessionRecorder()
        for i in range(10):
            recorder.observe(make_hand(*OPEN) if i < 7 else None, 0.9, now=i * 0.5)
        assert recorder.quality().presence_ratio == pytest.approx(0.7)

    def test_mean_confidence_averages_frames_with_a_hand(self):
        recorder = SessionRecorder()
        recorder.observe(make_hand(*OPEN), 0.4, now=0.0)
        recorder.observe(make_hand(*OPEN), 0.8, now=1.0)
        recorder.observe(None, now=2.0)
        assert recorder.quality().mean_confidence == pytest.approx(0.6)

    def test_a_well_tracked_session_is_reliable(self):
        quality = TrackingQuality(frames_total=100, frames_with_hand=95, mean_confidence=0.9)
        assert quality.reliable is True

    def test_a_mostly_empty_session_is_unreliable(self):
        quality = TrackingQuality(frames_total=100, frames_with_hand=20, mean_confidence=0.9)
        assert quality.reliable is False
        assert quality.presence_ratio < MIN_RELIABLE_PRESENCE

    def test_a_low_confidence_session_is_unreliable(self):
        quality = TrackingQuality(frames_total=100, frames_with_hand=100, mean_confidence=0.2)
        assert quality.reliable is False

    def test_an_empty_session_is_unreliable(self):
        assert TrackingQuality(0, 0, 0.0).reliable is False

    def test_quality_is_serialisable(self):
        data = TrackingQuality(10, 9, 0.8).to_dict()
        assert data["reliable"] is True
        assert data["presence_ratio"] == pytest.approx(0.9)
