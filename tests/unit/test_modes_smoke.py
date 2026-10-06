from __future__ import annotations

import numpy as np
import pytest

from spectra.gestures.state_machine import GestureState
from spectra.guided.shapes import Difficulty
from spectra.modes.base import ACTION_QUIT, AppMode, ModeContext
from spectra.modes.edu_colors import EduColorsMode
from spectra.modes.edu_count import EduCountMode
from spectra.modes.free_draw import FreeDrawMode
from spectra.modes.guided_draw import GuidedDrawMode
from spectra.modes.menu import MenuMode
from spectra.modes.physio import PhysioMode
from spectra.ui.sound import SoundPlayer
from tests.conftest import make_detection

WIDTH, HEIGHT = 640, 480
MODES = [MenuMode, FreeDrawMode, GuidedDrawMode, EduColorsMode, EduCountMode, PhysioMode]

POINT = (False, True, False, False, False)
PAINT = (False, True, True, False, False)
ERASE = (False, True, True, True, False)
OPEN = (True, True, True, True, True)


@pytest.fixture
def context(config) -> ModeContext:
    return ModeContext(config=config, width=WIDTH, height=HEIGHT, sound=SoundPlayer(enabled=False))


def blank_frame() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def run_gesture(mode, pattern, clock, seconds: float = 1.0, step: float = 0.05):
    """Feed one gesture to a mode for ``seconds`` of simulated time."""
    detection = make_detection(pattern)
    output = action = None
    for _ in range(max(1, int(seconds / step))):
        clock.advance(step)
        output, action = mode.process(blank_frame(), detection)
    return output, action


class TestSmoke:
    @pytest.mark.parametrize("mode_class", MODES)
    def test_process_runs_headless_without_a_hand(self, mode_class, context, empty_detection):
        output, action = mode_class(context).process(blank_frame(), empty_detection)
        assert output.shape == (HEIGHT, WIDTH, 3)
        assert action is None

    @pytest.mark.parametrize("mode_class", MODES)
    def test_process_runs_headless_with_a_hand(self, mode_class, context):
        output, _ = mode_class(context).process(blank_frame(), make_detection(POINT))
        assert output.shape == (HEIGHT, WIDTH, 3)

    @pytest.mark.parametrize("mode_class", MODES)
    def test_every_mode_offers_a_way_out(self, mode_class, context):
        assert ACTION_QUIT in {button.value for button in mode_class(context).buttons}

    def test_menu_exposes_every_implemented_mode(self, context):
        values = {b.value for b in MenuMode(context).buttons if isinstance(b.value, AppMode)}
        assert values == {
            AppMode.FREE_DRAW,
            AppMode.GUIDED_DRAW,
            AppMode.EDU_COLORS,
            AppMode.EDU_COUNT,
            AppMode.PHYSIO,
        }

    def test_mode_titles_are_translated(self):
        assert AppMode.FREE_DRAW.title == "Pintura Livre"
        assert AppMode.GUIDED_DRAW.title == "Desenho Guiado"


class TestFreeDraw:
    def test_the_paint_gesture_paints(self, context, clock):
        mode = FreeDrawMode(context)
        run_gesture(mode, PAINT, clock)
        assert (mode.canvas.image != 255).any()

    def test_pointing_does_not_paint(self, context, clock):
        mode = FreeDrawMode(context)
        run_gesture(mode, POINT, clock)
        assert (mode.canvas.image == 255).all()

    def test_a_transitional_posture_leaves_no_mark(self, context, clock):
        """Opening the hand crosses the paint and erase patterns on the way."""
        mode = FreeDrawMode(context)
        for pattern in (POINT, PAINT, ERASE, OPEN):
            run_gesture(mode, pattern, clock, seconds=0.1, step=0.05)
        assert (mode.canvas.image == 255).all()

    def test_the_open_hand_opens_the_colour_menu(self, context, clock):
        mode = FreeDrawMode(context)
        run_gesture(mode, OPEN, clock)
        assert mode.engine.state is GestureState.COLOR_MENU

    def test_erasing_restores_white(self, context, clock):
        mode = FreeDrawMode(context)
        mode.canvas.image[:] = 0
        run_gesture(mode, ERASE, clock)
        assert (mode.canvas.image == 255).any()

    def test_buttons_are_inert_while_painting(self, context, clock):
        mode = FreeDrawMode(context)
        run_gesture(mode, PAINT, clock, seconds=3.0)
        assert all(button.progress == 0.0 for button in mode.buttons)

    def test_save_writes_into_the_data_directory(self, context):
        FreeDrawMode(context).save()
        assert len(list(context.config.drawings_dir.glob("*.png"))) == 1


class TestGuidedDraw:
    def test_tracing_the_target_scores(self, context, clock):
        mode = GuidedDrawMode(context, "line", Difficulty.EASY)
        for point in mode.path.points[:40]:
            mode.scorer.add(point, clock.advance(0.05))
        assert mode.scorer.completion > 0

    def test_finishing_stores_a_result_and_resets(self, context):
        mode = GuidedDrawMode(context, "line", Difficulty.EASY)
        mode.scorer.add(mode.path.points[0], now=0.0)
        result = mode.finish()
        assert mode.results == [result]
        assert mode.scorer.samples == 0
        assert mode.attempt == 2

    def test_next_shape_cycles_and_resets_the_attempt(self, context):
        mode = GuidedDrawMode(context, "line", Difficulty.EASY)
        mode.restart()
        mode.next_shape()
        assert mode.shape_key == "zigzag"
        assert mode.attempt == 1

    def test_changing_difficulty_rebuilds_the_path(self, context):
        mode = GuidedDrawMode(context, "line", Difficulty.EASY)
        easy_tolerance = mode.path.tolerance
        mode.load_challenge(difficulty=Difficulty.HARD)
        assert mode.path.tolerance < easy_tolerance

    def test_only_the_paint_gesture_records_samples(self, context, clock):
        mode = GuidedDrawMode(context, "line", Difficulty.EASY)
        run_gesture(mode, POINT, clock)
        assert mode.scorer.samples == 0

    def test_the_paint_gesture_records_samples(self, context, clock):
        mode = GuidedDrawMode(context, "line", Difficulty.EASY)
        run_gesture(mode, PAINT, clock)
        assert mode.scorer.samples > 0


class TestPhysioMetricHooks:
    def test_the_mode_feeds_the_shared_recorder(self, context, clock):
        mode = PhysioMode(context)
        run_gesture(mode, OPEN, clock)
        assert context.recorder.frames_total > 0
        assert context.recorder.samples

    def test_quizzes_do_not_pollute_the_recording(self, context, clock):
        mode = EduCountMode(context)
        run_gesture(mode, OPEN, clock)
        assert context.recorder.frames_total == 0

    def test_each_exercise_run_exposes_a_time_window(self, context):
        mode = PhysioMode(context)
        start = mode.progress.started_at
        mode._next_exercise()
        finished = mode.history[-1]
        assert finished.window[0] == start
        assert finished.ended_at is not None
        assert finished.duration >= 0

    def test_export_includes_the_duration_column(self, context, tmp_path):
        mode = PhysioMode(context)
        mode.progress.add_rep()
        text = mode.export_csv(tmp_path / "exports").read_text(encoding="utf-8")
        assert "duration_s" in text
