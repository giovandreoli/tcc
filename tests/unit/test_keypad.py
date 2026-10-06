from __future__ import annotations

import numpy as np
import pytest

from spectra.modes.base import ModeContext
from spectra.modes.pin_entry import ACTION_AUTHENTICATED, PinEntryMode
from spectra.ui.keypad import ACTION_BACKSPACE, ACTION_CLEAR, GesturePinKeypad
from spectra.ui.sound import SoundPlayer
from tests.conftest import make_detection

WIDTH, HEIGHT = 800, 600
POINT = (False, True, False, False, False)


@pytest.fixture
def keypad() -> GesturePinKeypad:
    return GesturePinKeypad(WIDTH, HEIGHT, hover_seconds=1.0)


@pytest.fixture
def context(config) -> ModeContext:
    return ModeContext(config=config, width=WIDTH, height=HEIGHT, sound=SoundPlayer(enabled=False))


class TestKeypadLayout:
    def test_there_are_twelve_keys(self, keypad):
        assert len(keypad.keys) == 12

    def test_every_digit_is_present(self, keypad):
        values = {key.value for key in keypad.keys}
        assert {str(d) for d in range(10)} <= values
        assert {ACTION_CLEAR, ACTION_BACKSPACE} <= values

    def test_keys_stay_inside_the_frame(self, keypad):
        for key in keypad.keys:
            x, y, width, height = key.rect
            assert x >= 0 and x + width <= WIDTH
            assert y >= 0 and y + height <= HEIGHT

    def test_keys_do_not_overlap(self, keypad):
        rects = [key.rect for key in keypad.keys]
        for i, (x1, y1, w1, h1) in enumerate(rects):
            for x2, y2, w2, h2 in rects[i + 1 :]:
                assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1


class TestKeypadEntry:
    def test_digits_accumulate(self, keypad):
        for digit in "123":
            keypad.press(digit)
        assert keypad.entry == "123"
        assert keypad.complete is False

    def test_the_fourth_digit_completes_the_pin(self, keypad):
        for digit in "1234":
            result = keypad.press(digit)
        assert result == "1234"
        assert keypad.complete

    def test_entry_stops_at_the_pin_length(self, keypad):
        for digit in "123456":
            keypad.press(digit)
        assert keypad.entry == "1234"

    def test_backspace_removes_the_last_digit(self, keypad):
        keypad.press("1")
        keypad.press("2")
        keypad.press(ACTION_BACKSPACE)
        assert keypad.entry == "1"

    def test_backspace_on_an_empty_entry_is_harmless(self, keypad):
        keypad.press(ACTION_BACKSPACE)
        assert keypad.entry == ""

    def test_clear_empties_the_entry(self, keypad):
        for digit in "123":
            keypad.press(digit)
        keypad.press(ACTION_CLEAR)
        assert keypad.entry == ""

    def test_the_entry_is_masked_for_display(self, keypad):
        keypad.press("1")
        keypad.press("2")
        assert keypad.masked == "••__"

    def test_reset_clears_everything(self, keypad):
        keypad.press("1")
        keypad.reset()
        assert keypad.entry == ""
        assert keypad.submitted is None


class TestKeypadDwell:
    def test_dwelling_on_a_key_enters_it(self, keypad):
        key = next(k for k in keypad.keys if k.value == "7")
        x, y, width, height = key.rect
        centre = (x + width // 2, y + height // 2)
        keypad.update(centre, now=0.0)
        keypad.update(centre, now=1.0)
        assert keypad.entry == "7"

    def test_a_brief_hover_enters_nothing(self, keypad):
        key = next(k for k in keypad.keys if k.value == "7")
        x, y, width, height = key.rect
        centre = (x + width // 2, y + height // 2)
        keypad.update(centre, now=0.0)
        keypad.update(centre, now=0.3)
        assert keypad.entry == ""

    def test_no_pointer_enters_nothing(self, keypad):
        keypad.update(None, now=0.0)
        keypad.update(None, now=5.0)
        assert keypad.entry == ""

    def test_dwelling_through_four_keys_returns_the_pin(self, keypad):
        now = 0.0
        result = None
        for digit in "1234":
            key = next(k for k in keypad.keys if k.value == digit)
            x, y, width, height = key.rect
            centre = (x + width // 2, y + height // 2)
            keypad.update(centre, now=now)
            now += 1.0
            result = keypad.update(centre, now=now) or result
            now += 0.1
        assert result == "1234"

    def test_drawing_does_not_crash(self, keypad):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        keypad.press("1")
        keypad.draw(frame)
        assert frame.any()


class TestPinEntryMode:
    def _frame(self) -> np.ndarray:
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    def test_it_renders_headless(self, context):
        mode = PinEntryMode(context)
        output, action = mode.process(self._frame(), make_detection(POINT))
        assert output.shape == (HEIGHT, WIDTH, 3)
        assert action is None

    def test_a_correct_pin_authenticates(self, context):
        mode = PinEntryMode(context, verify=lambda pin: pin == "1234")
        for digit in "1234":
            mode.keypad.press(digit)
        mode._submit(mode.keypad.entry)
        assert mode.authenticated
        _output, action = mode.process(self._frame(), make_detection(POINT))
        assert action == ACTION_AUTHENTICATED

    def test_a_wrong_pin_clears_the_entry(self, context):
        mode = PinEntryMode(context, verify=lambda pin: pin == "1234")
        for digit in "0000":
            mode.keypad.press(digit)
        mode._submit(mode.keypad.entry)
        assert mode.authenticated is False
        assert mode.keypad.entry == ""

    def test_the_patient_can_always_quit(self, context):
        from spectra.modes.base import ACTION_QUIT

        assert ACTION_QUIT in {button.value for button in PinEntryMode(context).buttons}

    def test_the_pin_is_never_shown_in_clear(self, context):
        mode = PinEntryMode(context)
        for digit in "1234":
            mode.keypad.press(digit)
        assert "1234" not in mode.keypad.masked
