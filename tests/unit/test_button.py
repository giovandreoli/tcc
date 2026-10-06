from __future__ import annotations

import numpy as np

from spectra.ui.button import Button


def make_button(**kwargs) -> Button:
    return Button(100, 100, 120, 40, "Teste", hover_seconds=1.0, value="ok", **kwargs)


def test_contains_accepts_points_within_the_margin():
    button = make_button()
    assert button.contains((160, 120))
    assert button.contains((92, 95))  # inside the forgiving hit margin
    assert not button.contains((10, 10))
    assert not button.contains(None)


def test_dwell_fires_once_after_the_hold_time():
    button = make_button()
    assert button.update_hover(True, now=0.0) is False
    assert button.update_hover(True, now=0.5) is False
    assert button.progress == 0.5
    assert button.update_hover(True, now=1.0) is True
    # selection resets the timer, so the next frame does not fire again
    assert button.update_hover(True, now=1.1) is False


def test_leaving_the_button_resets_progress():
    button = make_button()
    button.update_hover(True, now=0.0)
    button.update_hover(True, now=0.6)
    button.update_hover(False, now=0.7)
    assert button.progress == 0.0
    assert button.hover_start is None
    assert button.hovering is False


def test_progress_is_clamped_to_one():
    button = make_button()
    button.update_hover(True, now=0.0)
    button.update_hover(True, now=0.99)
    assert 0.0 < button.progress <= 1.0


def test_draw_does_not_crash_and_touches_the_frame():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    button = make_button()
    button.update_hover(True, now=0.0)
    button.update_hover(True, now=0.5)
    button.draw(frame)
    assert frame.any()
