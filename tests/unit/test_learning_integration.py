"""Integration of the learning features: desktop modes, storage, browser sinks and panel."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spectra.browser.frames import BrowserFrame
from spectra.browser.sinks import GestureCollector, PinchCalibrationSink, TutorialSink
from spectra.detection.hand_detector import DetectedHand, DetectionResult
from spectra.domains import Command
from spectra.evaluation.synthetic import HandPose, project
from spectra.gestures.click import PinchThresholds
from spectra.learning import GestureClassifier
from spectra.modes.base import ACTION_QUIT, ModeContext
from spectra.modes.free_draw import FreeDrawMode, depth_brush_size
from spectra.modes.sterile import SterileViewerMode, demo_image
from spectra.modes.usability import UsabilityMode
from spectra.service import PatientForm, SpectraService
from spectra.storage.db import Database
from spectra.storage.security import Cipher
from spectra.ui.sound import SoundPlayer
from tests.conftest import make_detection, make_hand, pinching_hand

WIDTH, HEIGHT = 640, 480
TWO = (False, True, True, False, False)


@pytest.fixture
def context(config) -> ModeContext:
    return ModeContext(config=config, width=WIDTH, height=HEIGHT, sound=SoundPlayer(enabled=False))


@pytest.fixture
def service(config):
    from cryptography.fernet import Fernet

    instance = SpectraService(Database(config.db_path, Cipher(Fernet.generate_key())), config)
    yield instance
    instance.close()


@pytest.fixture
def patient(service):
    return service.create_patient(PatientForm(name="Ana Teste", cpf="52998224725", pin="1234"))


def frame() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def detection(landmarks, label="Right") -> DetectionResult:
    return DetectionResult((DetectedHand(landmarks, label, 0.9),))


def run(mode, det, clock, seconds=1.0, step=0.05):
    output = action = None
    for _ in range(max(1, int(seconds / step))):
        clock.advance(step)
        output, action = mode.process(frame(), det)
    return output, action


def _touch(finger: str):
    from spectra.detection.landmarks import Landmark

    tips = {"middle": 12, "ring": 16, "pinky": 20}
    hand = make_hand(index=True, middle=True, ring=True, pinky=True)
    tip = hand[tips[finger]]
    hand[4] = Landmark(tip.x, tip.y, 0.0)
    return hand


# ------------------------------------------------------------- desktop modes
def test_pinch_click_selects_a_button_without_dwelling(context, clock):
    mode = FreeDrawMode(context)
    clear = next(button for button in mode.buttons if button.value == "clear")
    x, y, w, h = clear.rect
    mode.canvas.image[:] = 0
    pointer_hand = make_hand(index=True)
    tip_x, tip_y = (x + w / 2) / WIDTH, (y + h / 2) / HEIGHT
    dx, dy = tip_x - pointer_hand[8].x, tip_y - pointer_hand[8].y
    from spectra.detection.landmarks import Landmark

    def shifted(hand):
        return [Landmark(p.x + dx, p.y + dy, p.z) for p in hand]

    run(mode, detection(shifted(pointer_hand)), clock, seconds=0.6)
    assert clear.progress < 1.0
    pinch = shifted(pinching_hand(0.0))
    mode.process(frame(), detection(pinch))
    clock.advance(0.05)
    mode.process(frame(), detection(pinch))
    assert (mode.canvas.image == 255).all()  # cleared by the click, not by dwelling


def test_pinch_click_is_disabled_when_the_profile_paints_with_a_pinch(context):
    from spectra.gestures.profiles import SIMPLIFIED_PROFILE

    context.gesture_profile = SIMPLIFIED_PROFILE
    assert not FreeDrawMode(context).pinch_clicks_enabled
    context.config.pinch_click = False
    context.gesture_profile = __import__(
        "spectra.gestures.profiles", fromlist=["x"]
    ).STANDARD_PROFILE
    assert not FreeDrawMode(context).pinch_clicks_enabled


def test_touching_a_finger_with_the_thumb_changes_the_colour(context, clock):
    mode = FreeDrawMode(context)
    run(mode, detection(_touch("ring")), clock, seconds=0.6)
    assert mode.color_label == "Verde"
    run(mode, detection(make_hand(True, True, True, True, True)), clock, seconds=0.3)
    run(mode, detection(_touch("pinky")), clock, seconds=0.6)
    assert mode.color_label == "Azul"


def test_depth_controls_the_brush_but_rotation_does_not(context, clock):
    context.config.depth_method = "area_aspect"
    mode = FreeDrawMode(context)
    mode.depth.image_aspect = WIDTH / HEIGHT
    run(mode, detection(project(HandPose())), clock, seconds=1.0)
    base = mode.brush_size
    run(mode, detection(project(HandPose(distance_cm=35))), clock, seconds=1.0)
    assert mode.brush_size > base
    near = mode.brush_size
    run(mode, detection(project(HandPose(distance_cm=35, roll=50))), clock, seconds=1.0)
    assert mode.brush_size == near
    assert depth_brush_size(8, 0.1) == 2 and depth_brush_size(8, 10) == 36


def test_depth_brush_can_be_turned_off(context, clock):
    context.config.depth_brush = False
    mode = FreeDrawMode(context)
    run(mode, detection(project(HandPose(distance_cm=35))), clock, seconds=1.5)
    assert mode.brush_size == mode.base_brush


def test_sterile_viewer_is_driven_by_gestures_and_lock(context, clock):
    mode = SterileViewerMode(context)
    run(mode, make_detection(TWO), clock, seconds=1.0)
    assert mode.viewer.index == 1
    run(mode, make_detection((False, True, True, True, False)), clock, seconds=1.0)
    assert mode.viewer.index == 0
    run(mode, make_detection((False,) * 5), clock, seconds=1.0)
    assert mode.mapper.locked
    run(mode, make_detection(TWO), clock, seconds=1.0)
    assert mode.viewer.index == 0  # locked
    output, _ = run(mode, make_detection((False,) * 5), clock, seconds=0.1)
    assert output.shape == (HEIGHT, WIDTH, 3)


def test_sterile_viewer_zooms_by_depth(context, clock):
    mode = SterileViewerMode(context)
    mode.depth.image_aspect = WIDTH / HEIGHT
    run(mode, detection(project(HandPose())), clock, seconds=1.0)
    run(mode, detection(project(HandPose(distance_cm=36))), clock, seconds=1.0)
    assert mode.viewer.zoom > 1.0
    assert Command.ZOOM_IN in mode.history


def test_sterile_viewer_obeys_learned_gestures(context, clock):
    from spectra.learning.features import pose_features

    model = GestureClassifier()
    for _ in range(6):
        model.add("ok", pose_features(make_hand(True, False, True, True, True)))
        model.add("nada", pose_features(make_hand(True, False, False, False, False)))
    context.gesture_model = model.fit()
    context.learned_bindings = {"ok": "next"}
    mode = SterileViewerMode(context)
    run(mode, make_detection((True, False, True, True, True)), clock, seconds=1.0)
    assert mode.viewer.index == 1


def test_demo_images_are_synthetic_and_distinct():
    first, second = demo_image(0, 64), demo_image(1, 64)
    assert first.shape == (64, 64, 3) and not np.array_equal(first, second)


def test_usability_task_records_clicks_and_ends(context, clock):
    mode = UsabilityMode(context)
    for target in mode.task.targets:
        tip = make_hand(index=True)[8]
        from spectra.detection.landmarks import Landmark

        dx, dy = target.x / WIDTH - tip.x, target.y / HEIGHT - tip.y
        pointing = [Landmark(p.x + dx, p.y + dy) for p in make_hand(index=True)]
        pinching = [Landmark(p.x + dx, p.y + dy) for p in pinching_hand(0.0)]
        run(mode, detection(pointing), clock, seconds=0.5)
        run(mode, detection(pinching), clock, seconds=0.1)
    assert mode.task.finished
    _, action = run(mode, detection(make_hand(index=True)), clock, seconds=2.5)
    assert action == ACTION_QUIT
    assert mode.task.summary()["selections"] == len(mode.task.targets) - 1


def test_app_stores_the_usability_result(config, monkeypatch):
    from spectra.app import SpectraApp

    app = SpectraApp(config, usability_code="P07")
    app.build_modes(WIDTH, HEIGHT)
    assert app.mode.value == "usability"
    summary = app.finish_usability()
    assert summary["completed"] is False
    with SpectraService.open(config) as service:
        [result] = service.db.list_usability_results("P07")
    assert result["kind"] == "pointing" and result["condition"] == "kalman"
    assert SpectraApp(config).finish_usability() is None


def test_app_loads_exported_gestures(config):
    from spectra.app import SpectraApp, load_learned_gestures
    from spectra.learning.features import pose_features

    assert load_learned_gestures(config.gesture_model_path) == (None, {})
    config.gesture_model_path.parent.mkdir(parents=True, exist_ok=True)
    config.gesture_model_path.write_text("{broken", encoding="utf-8")
    assert load_learned_gestures(config.gesture_model_path) == (None, {})
    model = GestureClassifier()
    for _ in range(5):
        model.add("a", pose_features(make_hand(True, True, True, True, True)))
        model.add("b", pose_features(make_hand()))
    data = {"model": model.fit().to_dict(), "bindings": {"a": "next"}}
    config.gesture_model_path.write_text(json.dumps(data), encoding="utf-8")
    app = SpectraApp(config)
    app.build_modes(WIDTH, HEIGHT)
    assert app.modes[app.mode].context.learned_bindings == {"a": "next"}


# ------------------------------------------------------------------ storage
def test_gesture_model_is_saved_exported_and_part_of_lgpd_export(service, patient):
    from spectra.learning.features import pose_features

    model = GestureClassifier()
    for _ in range(5):
        model.add("joinha", pose_features(make_hand(thumb=True)))
        model.add("aberta", pose_features(make_hand(True, True, True, True, True)))
    service.save_gesture_model(patient.id, model.fit(), {"joinha": "next", "aberta": ""})
    loaded, bindings = service.load_gesture_model(patient.id)
    assert loaded.counts() == {"joinha": 5, "aberta": 5} and bindings == {"joinha": "next"}
    path = service.export_gesture_model(patient.id)
    assert json.loads(path.read_text(encoding="utf-8"))["bindings"] == {"joinha": "next"}
    assert service.db.export_patient(patient.id)["gesture_model"] is not None
    service.db.delete_patient(patient.id)
    assert service.load_gesture_model(patient.id) == (None, {})
    with pytest.raises(ValueError):
        service.export_gesture_model(patient.id)


def test_pinch_calibration_is_saved_with_the_finger_calibration(service, patient):
    service.save_pinch_calibration(patient.id, PinchThresholds(0.2, 0.3, True))
    assert service.calibration_for(patient.id).pinch == PinchThresholds(0.2, 0.3, True)


def test_participants_and_results_are_stored_and_deleted(service):
    service.db.add_participant("P01", True, "destro")
    service.db.add_usability_result("P01", "sus", "", {"score": 72.5})
    service.db.add_usability_result("P09", "pointing", "kalman", {"selections": 0})
    codes = [p["code"] for p in service.db.list_participants()]
    assert codes == ["P01", "P09"]
    assert service.db.list_usability_results("P01")[0]["payload"] == {"score": 72.5}
    service.db.delete_participant("P01")
    assert [r["code"] for r in service.db.list_usability_results()] == ["P09"]


# -------------------------------------------------------------------- sinks
def _frames(hands, start=0.0):
    return [
        BrowserFrame(start + i * 100.0, None if h is None else DetectedHand(h, "Right", 0.9))
        for i, h in enumerate(hands)
    ]


def test_tutorial_sink_feeds_the_learning_path(monkeypatch, clock):
    sink = TutorialSink(hand_label="Right")
    assert sink.runner is None
    sink.process_batch(_frames([make_hand(True, True, True, True, True)] * 25))
    assert sink.runner is sink and sink.frames == 25
    assert sink.status().results[0].key == "show_hand"
    sink.skip()
    assert sink.status().index == 2
    sink.finish()
    assert sink.finished
    left = TutorialSink(hand_label="Left")
    left.process_batch(_frames([make_hand()] * 3))
    assert not left.status().hand_visible


def test_gesture_collector_records_then_predicts(monkeypatch):
    import spectra.browser.sinks as sinks

    now = [1000.0]
    monkeypatch.setattr(sinks.time, "time", lambda: now[0])
    collector = GestureCollector()
    for label, hand in (
        ("aberta", make_hand(True, True, True, True, True)),
        ("punho", make_hand()),
    ):
        collector.record(label, seconds=1.0, countdown=0.5)
        assert collector.recording()
        for _ in range(20):
            now[0] += 0.1
            collector.process_batch(_frames([hand], start=now[0] * 1000))
        assert not collector.recording()
    assert collector.model.counts() == {"aberta": 10, "punho": 10}
    collector.model.fit()
    collector.start_predicting()
    collector.process_batch(_frames([make_hand()], start=now[0] * 1000 + 50))
    assert collector.prediction.label == "punho"
    collector.process_batch(_frames([None], start=now[0] * 1000 + 99))
    assert collector.prediction is None and not collector.hand_visible


def test_pinch_calibration_sink_learns_thresholds(monkeypatch):
    sink = PinchCalibrationSink(hand_label="Right")
    sink.capture("apart")
    sink.process_batch(_frames([pinching_hand(0.12)] * 5))
    sink.capture("pinch")
    sink.process_batch(_frames([pinching_hand(0.02)] * 5, start=1000))
    assert sink.ratio is not None
    assert sink.thresholds().calibrated
    with pytest.raises(ValueError):
        sink.capture("other")


# -------------------------------------------------------------------- panel
def _learning_app(service):
    from streamlit.testing.v1 import AppTest

    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.therapist_panel.learning import render_learning\n"
        "render_learning(st.session_state['_svc'])"
    )
    ui.session_state["_svc"] = service
    return ui


def test_learning_section_renders_every_tab(service, patient):
    service.db.add_participant("P01", True)
    service.db.add_usability_result("P01", "sus", "", {"score": 70.0})
    for code, kalman, exponential in (("P01", 1.0, 1.3), ("P02", 1.1, 1.5), ("P03", 0.9, 1.2)):
        for condition, mt in (("kalman", kalman), ("exponential", exponential)):
            service.db.add_usability_result(
                code,
                "pointing",
                condition,
                {
                    "selections": 8,
                    "movement_time_s": mt,
                    "throughput_bps": 2 / mt,
                    "error_rate": 0.1,
                },
            )
    ui = _learning_app(service)
    ui.run(timeout=30)
    assert not ui.exception
    labels = [tab.label for tab in ui.tabs]
    assert labels[0] == "🎓 Trilha de gestos" and len(labels) == 5
    assert any("--usability P01 --filter kalman" in code.value for code in ui.code)
    assert any("p" in str(frame.value.columns) for frame in ui.dataframe)


def test_learning_panel_saves_a_sus_questionnaire(service):
    service.db.add_participant("P02", True)
    ui = _learning_app(service)
    ui.run(timeout=30)
    for slider in ui.select_slider:
        slider.set_value(4)
    next(b for b in ui.button if b.label == "Salvar respostas do SUS").click().run(timeout=30)
    assert not ui.exception
    [result] = service.db.list_usability_results("P02")
    assert result["kind"] == "sus" and result["payload"]["score"] == 50.0  # all "agree"


def test_learning_panel_runs_the_algorithm_evaluation(service):
    ui = _learning_app(service)
    ui.run(timeout=30)
    next(b for b in ui.button if b.label == "Executar avaliação").click().run(timeout=60)
    assert not ui.exception
    assert len(ui.dataframe) >= 2


def test_patient_can_practice_gestures_before_exercising(service, monkeypatch):
    from unittest.mock import Mock

    from streamlit.testing.v1 import AppTest

    import spectra.browser.learning_ui as learning_ui
    import spectra.therapist_panel.app as panel_module
    from spectra.browser.rooms import Rooms
    from spectra.service import SessionPlan

    patient = service.create_patient(PatientForm(name="Bia", cpf="11144477735", pin="4321"))
    service.record_consent(patient.id, "Bia")
    monkeypatch.setattr(panel_module, "get_service", lambda: service)
    monkeypatch.setattr(learning_ui, "camera_streamer", Mock())
    code = Rooms(service).create(
        SessionPlan(patient_id=patient.id, duration_minutes=5, exercises=("open_close",))
    )
    ui = AppTest.from_string("from spectra.therapist_panel.app import main\nmain()")
    ui.query_params["sala"] = code
    ui.run(timeout=30)
    ui.text_input[1].input("4321")
    ui.button[0].click().run(timeout=30)
    next(b for b in ui.button if b.label == "Treinar os gestos antes (opcional)").click().run(
        timeout=30
    )
    assert not ui.exception
    assert ui.session_state["browser_stage"] == "learn"
    assert any("Lição 1 de 8" in caption.value for caption in ui.caption)
    next(b for b in ui.button if b.label == "Voltar para a preparação").click().run(timeout=30)
    assert ui.session_state["browser_stage"] == "prepare"
