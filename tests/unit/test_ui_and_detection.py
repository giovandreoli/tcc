from __future__ import annotations

import numpy as np

from spectra.detection.hand_detector import DetectedHand, DetectionResult, handedness_label
from spectra.detection.landmarks import FINGER_CHAINS, HAND_CONNECTIONS, LANDMARK_COUNT
from spectra.gestures.features import FingerStates
from spectra.modes.quiz import QuizScore, ResultBanner
from spectra.ui.sound import SoundPlayer
from spectra.ui.text import ascii_fold, draw_text, text_size
from spectra.ui.widgets import dim_frame, draw_finger_hud, draw_hand_landmarks, draw_progress_bar
from tests.conftest import make_hand


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
