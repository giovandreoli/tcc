"""Metrics tests driven by synthetic landmark sequences with known properties."""

from __future__ import annotations

import math

import pytest

from spectra.guided.scoring import TraceResult
from spectra.guided.shapes import Difficulty
from spectra.metrics.accuracy import RepetitionSummary, summarise_traces
from spectra.metrics.fatigue import (
    amplitude_fatigue,
    fatigue_metrics,
    pace_fatigue,
)
from spectra.metrics.recorder import SessionRecorder, TrajectoryPoint
from spectra.metrics.rom import finger_rom, opposition_rom, wrist_estimate
from spectra.metrics.tracking_quality import TrackingQuality
from spectra.metrics.tremor import TREMOR_BAND_HZ, estimate_sample_rate, tremor_metrics
from tests.conftest import make_hand, rotate_hand

OPEN = (True, True, True, True, True)
FIST = (False, False, False, False, False)


def record(hands, rate_hz: float = 1000.0, step: float = 0.1) -> SessionRecorder:
    """Store every supplied hand as a sample (the rate is set high to disable throttling)."""
    recorder = SessionRecorder(rate_hz=rate_hz)
    for i, hand in enumerate(hands):
        recorder.observe(hand, 0.9, now=i * step)
    return recorder


def open_close_cycle(cycles: int = 3) -> list:
    """A hand repeatedly opening and closing, the canonical physiotherapy movement."""
    hands = []
    for _ in range(cycles):
        hands.extend([make_hand(*OPEN)] * 5)
        hands.extend([make_hand(*FIST)] * 5)
    return hands


class TestFingerRom:
    def test_a_still_open_hand_has_no_range(self):
        roms = finger_rom(record([make_hand(*OPEN)] * 10).samples)
        assert all(rom.span == pytest.approx(0.0) for rom in roms)

    def test_an_open_close_cycle_produces_a_large_range(self):
        roms = finger_rom(record(open_close_cycle()).samples)
        for rom in roms[1:]:  # index to pinky
            assert rom.span > 40.0

    def test_the_result_is_ordered_thumb_to_pinky(self):
        roms = finger_rom(record([make_hand(*OPEN)]).samples)
        assert [rom.finger for rom in roms] == ["thumb", "index", "middle", "ring", "pinky"]

    def test_min_never_exceeds_max(self):
        for rom in finger_rom(record(open_close_cycle()).samples):
            assert rom.minimum <= rom.maximum

    def test_an_empty_recording_is_handled(self):
        roms = finger_rom([])
        assert len(roms) == 5
        assert all(rom.samples == 0 for rom in roms)

    def test_rom_is_serialisable(self):
        data = finger_rom(record([make_hand(*OPEN)]).samples)[0].to_dict()
        assert set(data) == {"finger", "min", "max", "span", "samples"}


class TestOppositionRom:
    def test_a_pinch_lowers_the_index_distance(self):
        from tests.conftest import pinching_hand

        far = record([make_hand(thumb=True, index=True)]).samples
        near = record([pinching_hand(distance=0.005)]).samples
        assert opposition_rom(near)[0].minimum < opposition_rom(far)[0].minimum

    def test_one_entry_per_non_thumb_finger(self):
        roms = opposition_rom(record([make_hand(*OPEN)]).samples)
        assert [rom.finger for rom in roms] == ["index", "middle", "ring", "pinky"]

    def test_an_empty_recording_is_handled(self):
        assert all(rom.samples == 0 for rom in opposition_rom([]))


class TestWristEstimate:
    def test_a_rotating_palm_produces_a_rotation_span(self):
        hands = [rotate_hand(make_hand(*OPEN), angle) for angle in (-30, 0, 30)]
        estimate = wrist_estimate(record(hands).samples)
        assert estimate.rotation_span == pytest.approx(60.0, abs=1.0)

    def test_a_still_palm_has_no_span(self):
        estimate = wrist_estimate(record([make_hand(*OPEN)] * 5).samples)
        assert estimate.rotation_span == pytest.approx(0.0)

    def test_it_is_always_labelled_an_estimate(self):
        estimate = wrist_estimate(record([make_hand(*OPEN)]).samples)
        assert estimate.estimate is True
        assert estimate.to_dict()["estimate"] is True

    def test_an_empty_recording_is_handled(self):
        assert wrist_estimate([]).samples == 0


class TestTremor:
    @staticmethod
    def _trajectory(frequency: float, amplitude: float, rate: float = 30.0, seconds: float = 4.0):
        count = int(rate * seconds)
        return [
            TrajectoryPoint(
                at=i / rate,
                x=0.5 + amplitude * math.sin(2 * math.pi * frequency * i / rate),
                y=0.5,
            )
            for i in range(count)
        ]

    def test_sample_rate_is_recovered(self):
        assert estimate_sample_rate(self._trajectory(1.0, 0.01)) == pytest.approx(30.0, abs=0.5)

    def test_a_tremor_in_band_concentrates_the_power(self):
        metrics = tremor_metrics(self._trajectory(frequency=6.0, amplitude=0.01))
        assert metrics.band_measurable
        assert metrics.band_power_ratio > 0.8
        assert TREMOR_BAND_HZ[0] <= metrics.dominant_frequency <= TREMOR_BAND_HZ[1]

    def test_a_slow_voluntary_movement_has_little_band_power(self):
        metrics = tremor_metrics(self._trajectory(frequency=0.5, amplitude=0.05))
        assert metrics.band_power_ratio < 0.2

    def test_a_larger_tremor_raises_the_jerk(self):
        small = tremor_metrics(self._trajectory(6.0, 0.005)).mean_jerk
        large = tremor_metrics(self._trajectory(6.0, 0.02)).mean_jerk
        assert large > small

    def test_a_still_finger_has_almost_no_jerk(self):
        still = [TrajectoryPoint(at=i / 30, x=0.5, y=0.5) for i in range(120)]
        assert tremor_metrics(still).mean_jerk == pytest.approx(0.0, abs=1e-9)

    def test_a_ten_hertz_recording_cannot_resolve_the_band(self):
        """Nyquist: this is why the trajectory is kept at the full frame rate."""
        metrics = tremor_metrics(self._trajectory(6.0, 0.01, rate=10.0, seconds=6.0))
        assert metrics.band_measurable is False
        assert metrics.band_power_ratio == 0.0

    def test_a_short_recording_is_not_measurable(self):
        assert tremor_metrics(self._trajectory(6.0, 0.01, seconds=0.5)).band_measurable is False

    def test_an_empty_trajectory_is_handled(self):
        metrics = tremor_metrics([])
        assert metrics.samples == 0
        assert metrics.band_measurable is False

    def test_metrics_are_serialisable(self):
        data = tremor_metrics(self._trajectory(6.0, 0.01)).to_dict()
        assert data["band_measurable"] is True


class TestFatigue:
    @staticmethod
    def _declining_amplitude():
        """Amplitude shrinks over time: full cycles first, partial cycles later."""
        hands = [make_hand(*OPEN), make_hand(*FIST)] * 5  # full range
        hands += [make_hand(*OPEN)] * 10  # no movement at all
        return record(hands).samples

    def test_a_shrinking_amplitude_is_detected(self):
        start, end, decline, measurable = amplitude_fatigue(self._declining_amplitude())
        assert measurable
        assert start > end
        assert decline > 0.5

    def test_a_steady_amplitude_shows_no_decline(self):
        samples = record([make_hand(*OPEN), make_hand(*FIST)] * 10).samples
        _start, _end, decline, measurable = amplitude_fatigue(samples)
        assert measurable
        assert decline == pytest.approx(0.0, abs=0.05)

    def test_a_short_recording_is_not_measurable(self):
        assert amplitude_fatigue(record([make_hand(*OPEN)] * 3).samples)[3] is False

    def test_a_slowing_pace_is_detected(self):
        fast = [0.0, 1.0, 2.0, 3.0]
        slow = [10.0, 13.0, 16.0, 19.0, 22.0]
        _start, _end, decline, measurable = pace_fatigue(fast + slow)
        assert measurable
        assert decline > 0

    def test_a_steady_pace_shows_no_decline(self):
        _start, _end, decline, _ = pace_fatigue([float(i) for i in range(12)])
        assert decline == pytest.approx(0.0, abs=0.01)

    def test_too_few_reps_are_not_measurable(self):
        assert pace_fatigue([0.0, 1.0, 2.0])[3] is False

    def test_combined_metrics_report_measurability(self):
        metrics = fatigue_metrics([], [])
        assert metrics.measurable is False
        assert metrics.to_dict()["amplitude_decline"] == 0.0

    def test_warming_up_shows_a_negative_decline(self):
        hands = [make_hand(*OPEN)] * 10 + [make_hand(*OPEN), make_hand(*FIST)] * 5
        _start, _end, decline, _ = amplitude_fatigue(record(hands).samples)
        assert decline < 0


class TestAccuracyAggregation:
    @staticmethod
    def _result(score: float, deviation: float = 0.01, completion: float = 0.9) -> TraceResult:
        return TraceResult(
            shape_key="line",
            difficulty=Difficulty.EASY,
            samples=100,
            mean_deviation=deviation,
            max_deviation=deviation * 2,
            inside_ratio=0.9,
            completion=completion,
            duration=10.0,
            score=score,
        )

    def test_an_empty_session_summarises_to_zeros(self):
        assert summarise_traces([]).attempts == 0

    def test_empty_attempts_are_ignored(self):
        empty = TraceResult("line", Difficulty.EASY, 0, 0, 0, 0, 0, 0, 0)
        assert summarise_traces([empty]).attempts == 0

    def test_best_and_mean_scores(self):
        summary = summarise_traces([self._result(40.0), self._result(80.0)])
        assert summary.best_score == 80.0
        assert summary.mean_score == 60.0

    def test_max_deviation_is_the_worst_across_attempts(self):
        summary = summarise_traces(
            [self._result(50, deviation=0.01), self._result(50, deviation=0.05)]
        )
        assert summary.max_deviation == pytest.approx(0.10)

    def test_duration_is_summed(self):
        assert summarise_traces([self._result(50), self._result(50)]).total_duration == 20.0

    def test_summary_is_serialisable(self):
        assert "mean_completion" in summarise_traces([self._result(50)]).to_dict()


class TestRepetitionSummary:
    def test_pace_is_reps_per_minute(self):
        assert RepetitionSummary("open_close", 10, 8, 60.0).pace_rpm == pytest.approx(10.0)

    def test_completion_is_capped_at_one(self):
        assert RepetitionSummary("open_close", 20, 8, 60.0).completion == 1.0

    def test_a_zero_duration_has_no_pace(self):
        assert RepetitionSummary("open_close", 5, 8, 0.0).pace_rpm == 0.0

    def test_summary_is_serialisable(self):
        assert RepetitionSummary("open_close", 5, 8, 30.0).to_dict()["exercise"] == "open_close"


class TestQualityGating:
    def test_an_unreliable_session_explains_why(self):
        assert TrackingQuality(100, 10, 0.9).warning_key == "metrics.quality.low_presence"
        assert TrackingQuality(100, 100, 0.1).warning_key == "metrics.quality.low_confidence"
        assert TrackingQuality(0, 0, 0.0).warning_key == "metrics.quality.no_frames"

    def test_a_good_session_has_no_warning(self):
        assert TrackingQuality(100, 95, 0.9).warning_key is None

    def test_metrics_are_still_computed_for_a_bad_session(self):
        """The numbers exist; the report is responsible for flagging them."""
        recorder = SessionRecorder()
        recorder.observe(make_hand(*OPEN), 0.1, now=0.0)
        for i in range(1, 20):
            recorder.observe(None, now=float(i))
        assert recorder.quality().reliable is False
        assert finger_rom(recorder.samples)[1].samples == 1
