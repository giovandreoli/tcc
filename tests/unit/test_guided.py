from __future__ import annotations

import math

import pytest

from spectra.guided.scoring import TraceResult, TraceScorer
from spectra.guided.shapes import (
    DIFFICULTY_TOLERANCE,
    SHAPE_ORDER,
    SHAPES,
    CircleShape,
    Difficulty,
    LineShape,
    SpiralShape,
    ZigzagShape,
    build_path,
    get_shape,
    point_segment_distance,
)


class TestPointSegmentDistance:
    def test_projection_falls_inside_the_segment(self):
        assert point_segment_distance((0.5, 1.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_projection_clamps_to_the_start(self):
        assert point_segment_distance((-2.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(2.0)

    def test_projection_clamps_to_the_end(self):
        assert point_segment_distance((3.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(2.0)

    def test_degenerate_segment_is_a_point(self):
        assert point_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(5.0)


class TestShapes:
    @pytest.mark.parametrize("key", SHAPE_ORDER)
    @pytest.mark.parametrize("difficulty", list(Difficulty))
    def test_samples_stay_inside_the_unit_square(self, key, difficulty):
        points = SHAPES[key].sample(difficulty, samples=120)
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)

    @pytest.mark.parametrize("key", SHAPE_ORDER)
    def test_requested_sample_count_is_honoured(self, key):
        assert len(SHAPES[key].sample(Difficulty.EASY, samples=57)) == 57

    def test_only_the_circle_is_closed(self):
        assert CircleShape().closed is True
        assert LineShape().closed is False

    def test_line_tilts_with_difficulty(self):
        easy = LineShape().sample(Difficulty.EASY, 2)
        hard = LineShape().sample(Difficulty.HARD, 2)
        assert easy[0][1] == pytest.approx(easy[-1][1])
        assert hard[0][1] != pytest.approx(hard[-1][1])

    def test_zigzag_gains_peaks_with_difficulty(self):
        shape = ZigzagShape()
        assert shape.peaks(Difficulty.EASY) < shape.peaks(Difficulty.HARD)

    def test_spiral_gains_turns_with_difficulty(self):
        shape = SpiralShape()
        assert shape.turns(Difficulty.EASY) < shape.turns(Difficulty.HARD)

    def test_circle_samples_sit_on_the_radius(self):
        points = CircleShape().sample(Difficulty.EASY, 64)
        radii = {round(math.dist(p, (0.5, 0.5)), 6) for p in points}
        assert len(radii) == 1

    def test_zigzag_samples_are_evenly_spaced(self):
        """Even arc-length spacing; chords across a corner are shorter by geometry."""
        points = ZigzagShape().sample(Difficulty.EASY, 120)
        steps = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
        assert max(steps) / min(steps) < 2.0

    def test_unknown_shape_falls_back_to_the_line(self):
        assert get_shape("triangle").key == "line"


class TestTargetPath:
    def test_tolerance_tightens_with_difficulty(self):
        tolerances = [build_path("line", level).tolerance for level in Difficulty]
        assert tolerances == sorted(tolerances, reverse=True)
        assert tolerances[0] == DIFFICULTY_TOLERANCE[Difficulty.EASY]

    def test_a_point_on_the_path_has_zero_distance(self):
        path = build_path("line", Difficulty.EASY)
        assert path.distance_to(path.points[10]) == pytest.approx(0.0, abs=1e-9)

    def test_distance_grows_away_from_the_path(self):
        path = build_path("line", Difficulty.EASY)
        near = path.distance_to((0.5, 0.52))
        far = path.distance_to((0.5, 0.9))
        assert near < far

    def test_inside_follows_the_tolerance(self):
        path = build_path("line", Difficulty.EASY)
        assert path.inside((0.5, 0.5 + path.tolerance * 0.5))
        assert not path.inside((0.5, 0.5 + path.tolerance * 2))

    def test_closed_paths_include_the_wrap_around_segment(self):
        circle = build_path("circle", Difficulty.EASY)
        line = build_path("line", Difficulty.EASY)
        assert len(list(circle.segments())) == len(circle.points)
        assert len(list(line.segments())) == len(line.points) - 1

    def test_nearest_index_tracks_progress(self):
        path = build_path("line", Difficulty.EASY)
        assert path.nearest_index(path.points[0]) == 0
        assert path.nearest_index(path.points[-1]) == len(path.points) - 1

    def test_to_pixels_scales_to_the_frame(self):
        pixels = build_path("line", Difficulty.EASY).to_pixels(1280, 720)
        assert all(0 <= x <= 1280 and 0 <= y <= 720 for x, y in pixels)

    def test_path_length_is_positive(self):
        assert build_path("spiral", Difficulty.HARD).length > 0


class TestScoring:
    def test_an_untouched_attempt_scores_zero(self):
        result = TraceScorer(build_path("line", Difficulty.EASY)).result()
        assert result.is_empty
        assert result.score == 0.0

    def test_a_perfect_trace_scores_one_hundred(self):
        path = build_path("line", Difficulty.EASY)
        scorer = TraceScorer(path)
        for i, point in enumerate(path.points):
            scorer.add(point, now=i * 0.05)
        result = scorer.result()
        assert result.completion == pytest.approx(1.0)
        assert result.mean_deviation == pytest.approx(0.0, abs=1e-9)
        assert result.score == pytest.approx(100.0)
        assert result.inside_ratio == pytest.approx(1.0)

    def test_half_the_path_gives_about_half_the_completion(self):
        path = build_path("line", Difficulty.EASY)
        scorer = TraceScorer(path)
        half = len(path.points) // 2
        for i, point in enumerate(path.points[:half]):
            scorer.add(point, now=i * 0.05)
        assert scorer.result().completion == pytest.approx(0.5, abs=0.02)

    def test_a_far_trace_scores_zero_on_accuracy(self):
        path = build_path("line", Difficulty.HARD)
        scorer = TraceScorer(path)
        for i, (x, _y) in enumerate(path.points):
            scorer.add((x, 0.95), now=i * 0.05)
        result = scorer.result()
        assert result.inside_ratio == 0.0
        assert result.completion == 0.0
        assert result.score == 0.0

    def test_max_deviation_records_the_worst_sample(self):
        path = build_path("line", Difficulty.EASY)
        scorer = TraceScorer(path)
        scorer.add(path.points[0], now=0.0)
        scorer.add((path.points[0][0], 0.95), now=0.1)
        assert scorer.result().max_deviation > scorer.result().mean_deviation

    def test_duration_spans_first_to_last_sample(self):
        path = build_path("line", Difficulty.EASY)
        scorer = TraceScorer(path)
        scorer.add(path.points[0], now=10.0)
        scorer.add(path.points[1], now=13.5)
        assert scorer.result().duration == pytest.approx(3.5)

    def test_a_harder_corridor_lowers_the_score_for_the_same_trace(self):
        offset = 0.04
        scores = []
        for level in (Difficulty.EASY, Difficulty.HARD):
            path = build_path("line", level)
            scorer = TraceScorer(path)
            for i, (x, y) in enumerate(path.points):
                scorer.add((x, y + offset), now=i * 0.05)
            scores.append(scorer.result().score)
        assert scores[0] > scores[1]

    def test_reset_clears_everything(self):
        path = build_path("line", Difficulty.EASY)
        scorer = TraceScorer(path)
        scorer.add(path.points[0], now=0.0)
        scorer.reset()
        assert scorer.samples == 0
        assert scorer.completion == 0.0

    def test_result_is_serialisable(self):
        path = build_path("circle", Difficulty.MEDIUM)
        scorer = TraceScorer(path)
        scorer.add(path.points[0], now=0.0)
        data = scorer.result().to_dict()
        assert data["shape"] == "circle"
        assert data["difficulty"] == "medium"
        assert set(data) == {
            "shape",
            "difficulty",
            "samples",
            "mean_deviation",
            "max_deviation",
            "inside_ratio",
            "completion",
            "duration",
            "score",
        }

    def test_result_is_immutable(self):
        result = TraceScorer(build_path("line", Difficulty.EASY)).result()
        with pytest.raises(AttributeError):
            result.score = 99.0  # type: ignore[misc]

    def test_scoring_is_resolution_independent(self):
        """Normalised coordinates mean a trace scores the same at any frame size."""
        path = build_path("zigzag", Difficulty.MEDIUM)
        results: list[TraceResult] = []
        for size in (480, 1080):
            scorer = TraceScorer(path)
            for i, (x, y) in enumerate(path.points):
                # round-trip through pixel coordinates at this resolution
                px, py = int(x * size), int(y * size)
                scorer.add((px / size, py / size), now=i * 0.05)
            results.append(scorer.result())
        assert results[0].score == pytest.approx(results[1].score, abs=0.5)
