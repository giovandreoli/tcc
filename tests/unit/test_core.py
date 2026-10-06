from __future__ import annotations

import numpy as np

from spectra.core.canvas import DrawingCanvas
from spectra.core.fps import FpsCounter
from spectra.core.smoothing import ExponentialSmoother, smooth_point
from spectra.core.trail import Trail


class TestSmoothing:
    def test_first_point_passes_through(self):
        assert smooth_point(None, (10, 20)) == (10, 20)

    def test_output_moves_towards_the_new_point(self):
        assert smooth_point((0, 0), (100, 100), alpha=0.5) == (50, 50)

    def test_smoother_resets_on_none(self):
        smoother = ExponentialSmoother(alpha=0.5)
        smoother.update((100, 100))
        assert smoother.update(None) is None
        assert smoother.update((10, 10)) == (10, 10)


class TestCanvas:
    def test_starts_white(self):
        canvas = DrawingCanvas(20, 10)
        assert canvas.image.shape == (10, 20, 3)
        assert (canvas.image == 255).all()

    def test_draw_changes_pixels(self):
        canvas = DrawingCanvas(40, 40)
        canvas.draw((5, 5), (30, 30), (0, 0, 255), 3)
        assert (canvas.image != 255).any()

    def test_undo_restores_the_previous_snapshot(self):
        canvas = DrawingCanvas(40, 40)
        canvas.begin_stroke()
        canvas.draw((5, 5), (30, 30), (0, 0, 255), 3)
        assert canvas.undo() is True
        assert (canvas.image == 255).all()

    def test_undo_on_empty_history_is_false(self):
        assert DrawingCanvas(10, 10).undo() is False

    def test_undo_history_is_bounded(self):
        canvas = DrawingCanvas(10, 10, undo_history=2)
        for _ in range(5):
            canvas.begin_stroke()
        assert canvas.undo() and canvas.undo()
        assert canvas.undo() is False

    def test_clear_is_undoable(self):
        canvas = DrawingCanvas(40, 40)
        canvas.draw((5, 5), (30, 30), (0, 0, 255), 3)
        canvas.clear()
        assert (canvas.image == 255).all()
        assert canvas.undo() is True
        assert (canvas.image != 255).any()

    def test_eraser_paints_white(self):
        canvas = DrawingCanvas(80, 80)
        canvas.image[:] = 0
        canvas.draw((40, 40), (40, 40), None)
        assert (canvas.image == 255).any()

    def test_save_creates_the_parent_directory(self, tmp_path):
        canvas = DrawingCanvas(20, 20)
        path = canvas.save(tmp_path / "nested" / "drawing.png")
        assert path.is_file()


class TestTrail:
    def test_length_is_bounded(self):
        trail = Trail(max_length=3)
        for i in range(10):
            trail.add((i, i), (0, 0, 255))
        assert len(trail) == 3

    def test_clear_empties_the_trail(self):
        trail = Trail()
        trail.add((1, 1), None)
        trail.clear()
        assert len(trail) == 0

    def test_faded_points_are_not_drawn(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        trail = Trail(fade_seconds=0.1)
        trail.add((25, 25), (0, 0, 255), now=0.0)
        trail.draw(frame, now=10.0)
        assert not frame.any()

    def test_fresh_points_are_drawn(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        trail = Trail()
        trail.add((25, 25), (0, 0, 255), now=0.0)
        trail.draw(frame, now=0.0)
        assert frame.any()


class TestFpsCounter:
    def test_value_updates_after_the_window(self):
        counter = FpsCounter(window_seconds=1.0)
        counter.tick(now=0.0)
        for i in range(1, 31):
            counter.tick(now=i / 30)
        assert 25 <= counter.value <= 35
