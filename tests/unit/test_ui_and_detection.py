from __future__ import annotations

import numpy as np
import pytest

from spectra.detection.hand_detector import DetectedHand, DetectionResult, handedness_label
from spectra.detection.landmarks import FINGER_CHAINS, HAND_CONNECTIONS, LANDMARK_COUNT
from spectra.gestures.features import FingerStates
from spectra.modes.quiz import QuizScore, ResultBanner
from spectra.ui.sound import SoundPlayer
from spectra.ui.text import ascii_fold, draw_text, text_size
from spectra.ui.widgets import dim_frame, draw_finger_hud, draw_hand_landmarks, draw_progress_bar
from tests.conftest import make_hand


@pytest.fixture
def mediapipe_stub(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    import spectra.detection.hand_detector as detector_module

    result = SimpleNamespace(
        hand_landmarks=[make_hand(index=True)],
        handedness=[[SimpleNamespace(category_name="Right", score=0.9)]],
    )
    detector = Mock()
    detector.detect.return_value = result
    detector.detect_for_video.return_value = result
    create = Mock(return_value=detector)
    vision = SimpleNamespace(
        HandLandmarkerOptions=SimpleNamespace,
        HandLandmarker=SimpleNamespace(create_from_options=create),
        RunningMode=SimpleNamespace(IMAGE="image", VIDEO="video"),
    )
    python = SimpleNamespace(BaseOptions=SimpleNamespace, vision=vision)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(Image=SimpleNamespace, ImageFormat=SimpleNamespace(SRGB=1)),
    )
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", SimpleNamespace(python=python))
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", python)
    monkeypatch.setattr(detector_module, "ensure_model", lambda path: path)
    return detector, create


def test_video_detector_uses_tracking_and_strictly_increasing_timestamps(
    mediapipe_stub, monkeypatch, tmp_path
):
    import spectra.detection.hand_detector as detector_module

    detector, create = mediapipe_stub
    clock = iter((10_000_000, 10_000_000, 9_000_000, 14_000_000))
    monkeypatch.setattr(detector_module, "monotonic_ns", lambda: next(clock))
    with detector_module.HandDetector(
        tmp_path / "model.task", video_mode=True, num_hands=1
    ) as hand:
        for _ in range(4):
            result = hand.detect(np.zeros((540, 960, 3), dtype=np.uint8))
            assert result.primary.label == "Right"
            assert result.primary.score == 0.9
    options = create.call_args.args[0]
    assert options.running_mode == "video" and options.num_hands == 1
    assert [call.args[1] for call in detector.detect_for_video.call_args_list] == [10, 11, 12, 14]
    detector.detect.assert_not_called()
    detector.close.assert_called_once()


def test_default_detector_keeps_image_mode(mediapipe_stub, tmp_path):
    from spectra.detection.hand_detector import HandDetector

    detector, create = mediapipe_stub
    with HandDetector(tmp_path / "model.task") as hand:
        hand.detect(np.zeros((540, 960, 3), dtype=np.uint8))
    assert create.call_args.args[0].running_mode == "image"
    assert create.call_args.args[0].num_hands == 2
    detector.detect.assert_called_once()
    detector.detect_for_video.assert_not_called()


class TestAsciiFold:
    def test_accents_are_stripped_for_hershey_fonts(self):
        assert ascii_fold("Educação: Médio") == "Educacao: Medio"

    def test_plain_ascii_is_untouched(self):
        assert ascii_fold("Menu Principal") == "Menu Principal"

    def test_text_size_uses_the_folded_string(self):
        assert text_size("Sessão", 0.6) == text_size("Sessao", 0.6)


class TestSound:
    def test_disabled_player_is_silent(self):
        assert SoundPlayer(enabled=False).success() is False

    def test_player_is_silent_when_winsound_is_missing(self):
        import spectra.ui.sound as sound_module

        if sound_module.winsound is None:
            assert SoundPlayer(enabled=True).error() is False


class TestWidgets:
    def test_drawing_helpers_do_not_crash(self):
        frame = np.zeros((200, 320, 3), dtype=np.uint8)
        draw_text(frame, "Olá", (10, 20))
        draw_finger_hud(frame, FingerStates(True, False, True, False, True), 10, 100)
        draw_progress_bar(frame, (10, 150), (100, 10), 0.5)
        draw_hand_landmarks(frame, make_hand(index=True))
        assert frame.any()

    def test_progress_bar_clamps_out_of_range_values(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_progress_bar(frame, (0, 0), (100, 10), 5.0)
        draw_progress_bar(frame, (0, 20), (100, 10), -3.0)

    def test_dim_frame_blends_towards_the_colour(self):
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        dim_frame(frame, (0, 0, 0), 0.5)
        assert frame.mean() < 200


class TestDetectionResult:
    def test_primary_is_none_without_hands(self):
        assert DetectionResult().primary is None

    def test_for_label_picks_the_matching_hand(self):
        left = DetectedHand(make_hand(index=True), "Left")
        right = DetectedHand(make_hand(index=True), "Right")
        result = DetectionResult((left, right))
        assert result.for_label("Right") is right
        assert result.for_label(None) is left
        assert result.for_label("Unknown") is None

    def test_handedness_label_is_tolerant(self):
        assert handedness_label(None) == "Right"
        assert handedness_label([]) == "Right"


class TestLandmarkConstants:
    def test_connections_reference_valid_indices(self):
        indices = {i for pair in HAND_CONNECTIONS for i in pair}
        assert max(indices) == LANDMARK_COUNT - 1

    def test_there_is_one_chain_per_finger(self):
        assert len(FINGER_CHAINS) == 5
        assert all(len(chain) == 4 for chain in FINGER_CHAINS)


class TestQuizScore:
    def test_streak_grows_and_resets(self):
        score = QuizScore()
        score.register(True)
        score.register(True)
        assert score.streak == 2
        score.register(False)
        assert score.streak == 0
        assert score.best_streak == 2
        assert (score.correct, score.total) == (2, 3)

    def test_banner_expires(self):
        banner = ResultBanner(duration=0.0)
        banner.show("ok", True)
        assert banner.visible is False
