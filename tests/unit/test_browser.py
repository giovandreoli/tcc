"""Local room admission and browser video lifecycle without hardware access."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spectra.browser.rooms import LEASE_SECONDS, ROOM_LIFETIME_SECONDS, RoomError, Rooms
from spectra.browser.session import BrowserSession
from spectra.detection.hand_detector import DetectionResult
from spectra.modes.base import AppMode
from spectra.service import PatientForm, SessionPlan, SpectraService
from spectra.storage.db import ConsentRequiredError, Database
from spectra.storage.security import Cipher
from tests.conftest import make_detection


@pytest.fixture(autouse=True)
def room_clock(monkeypatch, clock):
    monkeypatch.setattr("spectra.browser.rooms.utc_now", clock)


@pytest.fixture
def service(config):
    # Avoid OS credential storage in tests.
    from cryptography.fernet import Fernet

    database = Database(config.db_path, Cipher(Fernet.generate_key()))
    instance = SpectraService(database, config)
    yield instance
    instance.close()


@pytest.fixture
def plan(service):
    patient = service.create_patient(
        PatientForm(name="Test Patient", cpf="52998224725", pin="1234")
    )
    service.record_consent(patient.id, "Test Participant")
    return SessionPlan(
        patient_id=patient.id,
        duration_minutes=5,
        exercises=("open_close",),
    )


def test_room_requires_consent(service, plan):
    consent = service.db.active_consent(plan.patient_id)
    service.db.revoke_consent(consent.id)
    with pytest.raises(ConsentRequiredError):
        Rooms(service).create(plan)


def test_room_requires_patient_pin(service, plan):
    with service.db.transaction() as connection:
        connection.execute("UPDATE patient SET pin_hash = NULL WHERE id = ?", (plan.patient_id,))
    with pytest.raises(RoomError, match="pin_required"):
        Rooms(service).create(plan)


@pytest.mark.parametrize("exercises", [(), ("circle",), ("unknown",)])
def test_browser_rejects_unsupported_exercises(service, plan, exercises):
    with pytest.raises(RoomError, match="invalid_plan"):
        Rooms(service).create(replace(plan, exercises=exercises))
    assert service.db.list_sessions(plan.patient_id) == []


def test_code_is_hashed_and_pin_is_required(service, plan):
    rooms = Rooms(service)
    code = rooms.create(plan)
    assert len(code) == 14
    with service.db.transaction() as connection:
        row = connection.execute("SELECT * FROM browser_room").fetchone()
    assert code.replace("-", "") not in tuple(row)
    with pytest.raises(RoomError, match="invalid_pin"):
        rooms.admit(code, "9999")
    admission = rooms.admit(code.lower().replace("-", " "), "1234")
    assert admission.session.patient_id == plan.patient_id
    assert admission.exercises == ("open_close",)
    assert admission.duration_minutes == 5


def test_invalid_code_has_no_patient_information(service):
    with pytest.raises(RoomError, match="invalid_room"):
        Rooms(service).admit("NOT-A-ROOM", "1234")


def test_room_entry_expires(service, plan, clock):
    rooms = Rooms(service)
    code = rooms.create(plan)
    clock.advance(ROOM_LIFETIME_SECONDS + 1)
    with pytest.raises(RoomError, match="invalid_room"):
        rooms.admit(code, "1234")


def test_wrong_pin_uses_existing_account_lockout(service, plan):
    rooms = Rooms(service)
    code = rooms.create(plan)
    for _ in range(5):
        with pytest.raises(RoomError, match="invalid_pin"):
            rooms.admit(code, "9999")
    with pytest.raises(RoomError, match="locked"):
        rooms.admit(code, "1234")


def test_only_one_browser_can_claim_room(service, plan):
    rooms = Rooms(service)
    code = rooms.create(plan)
    rooms.admit(code, "1234")
    with pytest.raises(RoomError, match="in_use"):
        rooms.admit(code, "1234")


def test_interrupted_admission_can_be_reclaimed_but_old_claim_is_invalid(service, plan, clock):
    rooms = Rooms(service)
    code = rooms.create(plan)
    old = rooms.admit(code, "1234")
    clock.advance(LEASE_SECONDS + 1)
    new = rooms.admit(code, "1234")
    with pytest.raises(RoomError, match="in_use"):
        rooms.heartbeat(old)
    rooms.heartbeat(new)


def test_completed_room_cannot_be_reused(service, plan):
    rooms = Rooms(service)
    code = rooms.create(plan)
    admission = rooms.admit(code, "1234")
    rooms.complete(admission)
    with pytest.raises(RoomError, match="invalid_room"):
        rooms.admit(code, "1234")


def test_revoked_consent_blocks_entry_and_ongoing_session(service, plan):
    rooms = Rooms(service)
    code = rooms.create(plan)
    admission = rooms.admit(code, "1234")
    consent = service.db.active_consent(plan.patient_id)
    service.db.revoke_consent(consent.id)
    with pytest.raises(RoomError, match="invalid_room"):
        rooms.heartbeat(admission)
    with pytest.raises(RoomError, match="invalid_room"):
        rooms.admit(code, "1234")


def test_patient_deletion_removes_room(service, plan):
    rooms = Rooms(service)
    code = rooms.create(plan)
    service.db.delete_patient(plan.patient_id)
    with pytest.raises(RoomError, match="invalid_room"):
        rooms.admit(code, "1234")


class FakeDetector:
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.result = DetectionResult()
        self.options = kwargs

    def detect(self, frame):
        self.last_frame = frame.copy()
        return self.result

    def close(self):
        self.closed = True


@pytest.fixture
def browser(service, plan, monkeypatch):
    import spectra.browser.session as browser_module

    monkeypatch.setattr(
        browser_module.SpectraService, "open", classmethod(lambda cls, config=None: service)
    )
    monkeypatch.setattr(browser_module, "HandDetector", FakeDetector)
    rooms = Rooms(service)
    admission = rooms.admit(rooms.create(plan), "1234")
    return BrowserSession(service.config, admission)


def test_browser_reuses_exercises_without_camera_or_window(browser, clock):
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    output = browser.process(frame)
    assert output.shape == frame.shape
    assert browser._app.mode is AppMode.PHYSIO
    assert browser._app._pin_mode is None
    assert browser._app.guard.limit_minutes == 5
    mode = browser._app.modes[AppMode.PHYSIO]
    assert [exercise.value for exercise in mode.EXERCISES] == ["open_close"]
    assert browser._app.recorder.frames_total == 1
    assert browser._detector.options["video_mode"] is True
    assert browser._detector.options["num_hands"] == 1


def test_browser_counts_open_close_with_a_partial_fist(browser, clock):
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    browser.process(frame)
    detector = browser._detector

    for pattern in ((True,) * 5, (True, False, False, False, False)):
        detector.result = make_detection(pattern)
        for _ in range(8):
            clock.advance(0.1)
            browser.process(frame)

    assert browser.status().reps == 1


def test_browser_keeps_camera_pixels_free_of_desktop_controls(browser):
    frame = np.full((720, 960, 3), (70, 100, 140), dtype=np.uint8)
    output = browser.process(frame)
    np.testing.assert_array_equal(output, frame)
    assert browser._app.modes[AppMode.PHYSIO].context.camera_overlay is False


@pytest.mark.parametrize("label", ["Left", "Right"])
def test_browser_detects_anatomical_hand_from_unmirrored_camera_pixels(
    browser, service, clock, label
):
    from spectra.storage.models import Hand

    session = replace(browser.admission.session, hand=Hand(label))
    browser.admission = replace(browser.admission, session=session)
    # Attach_session loads this value from storage.
    with service.db.transaction() as connection:
        connection.execute("UPDATE session SET hand = ? WHERE id = ?", (label, session.id))
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:, :100] = (20, 80, 160)
    browser.process(frame)
    for pattern in ((True,) * 5, (False,) * 5):
        browser._detector.result = make_detection(pattern, label=label)
        for _ in range(8):
            clock.advance(0.1)
            browser.process(frame)
    np.testing.assert_array_equal(browser._detector.last_frame, frame)
    assert browser.status().hand_visible
    assert browser.status().reps == 1


def test_browser_rest_prompt_remains_available_without_video_overlays(browser, clock, monkeypatch):
    from spectra.core.session_guard import SessionGuard

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    browser.process(frame)
    browser._app.guard = SessionGuard(limit_minutes=5, rest_every_minutes=1)
    browser._app.guard.start()
    monkeypatch.setattr("spectra.browser.session.time.monotonic", lambda: clock())
    clock.advance(61)
    output = browser.process(frame)
    np.testing.assert_array_equal(output, frame)
    assert browser.status().safety_key == "safety.rest"
    clock.advance(9)
    assert browser.status().safety_key is None


def test_browser_next_button_preserves_recorded_reps(browser, clock):
    browser.admission = replace(browser.admission, exercises=("open_close", "finger_touch"))
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    browser.process(frame)
    for pattern in ((True,) * 5, (False,) * 5):
        browser._detector.result = make_detection(pattern)
        for _ in range(8):
            clock.advance(0.1)
            browser.process(frame)
    assert browser.status().reps == 1
    browser.next_exercise()
    status = browser.status()
    assert status.exercise == "finger_touch"
    assert status.exercise_index == 1 and status.reps == 0
    assert status.total_reps == 1
    browser.finish()
    with pytest.raises(RuntimeError, match="inactive"):
        browser.next_exercise()


def test_browser_finishes_once_and_saves_existing_report_metrics(browser, service, clock):
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    browser.process(frame)
    detector = browser._detector
    for pattern in [(True,) * 5, (False,) * 5]:
        detector.result = make_detection(pattern)
        for _ in range(15):
            clock.advance(0.1)
            browser.process(frame)
    browser.finish()
    browser.finish()
    assert browser.saved
    assert detector.closed
    assert service.db.get_session(browser.admission.session.id).ended_at is not None
    assert service.db.list_exercise_runs(browser.admission.session.id)
    assert browser.process(frame).shape == frame.shape


def test_camera_not_started_releases_room_without_saving(browser, service):
    browser.finish()
    assert browser.finished and not browser.saved
    with service.db.transaction() as connection:
        row = connection.execute("SELECT status FROM browser_room").fetchone()
    assert row["status"] == "waiting"
    assert service.db.get_session(browser.admission.session.id).ended_at is None


def test_browser_time_limit_finishes_session(browser, service, clock):
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    browser.process(frame)
    clock.advance(301)
    browser.process(frame)
    assert browser.finished and browser.saved
    assert service.db.get_session(browser.admission.session.id).started_at == 1000


def test_save_failure_is_not_reported_as_success(browser, monkeypatch):
    browser.process(np.zeros((720, 960, 3), dtype=np.uint8))

    def fail(**kwargs):
        raise RuntimeError("test write failure")

    monkeypatch.setattr(browser._app, "finish_session", fail)
    with pytest.raises(RuntimeError, match="test write failure"):
        browser.finish()
    assert browser.finished and not browser.saved
    assert browser.snapshot()[2] == "browser.save_error"
    browser.finish()  # No second write or success-shaped retry after a partial failure.


def test_processing_error_is_exposed_not_hidden(browser, monkeypatch):
    def fail(frame):
        raise RuntimeError("test initialization failure")

    monkeypatch.setattr(browser, "_start", fail)
    with pytest.raises(RuntimeError, match="test initialization"):
        browser.process(np.zeros((720, 960, 3), dtype=np.uint8))
    assert browser.snapshot()[2] == "browser.processing_error"


def test_video_processor_processes_frames_and_saves_when_track_ends(browser):
    import av

    from spectra.browser.video import ExerciseVideoProcessor

    processor = ExerciseVideoProcessor(browser)
    image = np.zeros((720, 960, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
    output = processor.recv(frame)
    assert output is frame
    assert output.to_ndarray(format="bgr24").shape == image.shape
    assert browser.status().frames == 1
    assert browser.status().target == 8
    processor.on_ended()
    assert browser.finished and browser.saved
    processor.on_ended()


def test_patient_ui_admits_by_code_and_pin_without_therapist_login(service, plan, monkeypatch):
    from unittest.mock import Mock

    from streamlit.testing.v1 import AppTest

    import spectra.browser.page as page_module
    import spectra.therapist_panel.app as panel_module

    monkeypatch.setattr(panel_module, "get_service", lambda: service)
    streamer = Mock()
    monkeypatch.setattr(page_module, "camera_streamer", streamer)
    code = Rooms(service).create(plan)
    ui = AppTest.from_string("from spectra.therapist_panel.app import main\nmain()")
    ui.query_params["sala"] = code
    ui.run()
    assert not ui.exception
    assert ui.text_input[0].value == code
    ui.text_input[1].input("9999")
    ui.button[0].click().run()
    assert ui.error[0].value == "PIN incorreto."
    ui.text_input[1].input("1234")
    ui.button[0].click().run()
    assert not ui.exception
    assert ui.session_state["browser_session"].admission.exercises == ("open_close",)
    assert ui.query_params == {"paciente": ["1"]}
    assert ui.session_state["browser_stage"] == "prepare"
    ready = _button(ui, "Estou pronto(a), abrir a câmera")
    assert ready.disabled  # The environment checklist gates the camera.
    for checkbox in ui.checkbox:
        checkbox.check()
    ui.run()
    _button(ui, "Estou pronto(a), abrir a câmera").click().run()
    assert not ui.exception
    assert ui.session_state["browser_stage"] == "exercise"
    assert ui.session_state["browser_session"].pain_before == 0
    assert _button(ui, "Próximo exercício").disabled
    assert any(expander.label == "Como fazer este exercício" for expander in ui.expander)
    assert streamer.call_args.args[0] is ui.session_state["browser_session"]
    _button(ui, "Finalizar e salvar sessão").click().run()
    assert not ui.exception
    assert ui.session_state["browser_session"].finished


def _button(ui, label):
    return next(button for button in ui.button if button.label == label)


def test_direct_preview_worker_is_sendonly_and_does_not_encode_video(browser, monkeypatch):
    from unittest.mock import Mock

    from streamlit_webrtc import WebRtcMode

    import spectra.browser.video as video

    worker = Mock()
    monkeypatch.setattr(video, "WebRtcWorker", worker)
    video._create_worker(browser)
    options = worker.call_args.kwargs
    assert options["mode"] is WebRtcMode.SENDONLY
    assert options["async_processing"] is True
    assert options["video_receiver_size"] == 1
    assert not options["sendback_video"] and not options["sendback_audio"]
    assert options["rtc_configuration"].iceServers == []
    assert not any(
        options.get(key)
        for key in ("on_video_ended", "video_frame_callback", "queued_video_frames_callback")
    )
    assert options["video_processor_factory"]().session is browser


def test_camera_signals_answer_once_and_cleans_up_worker(browser, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from streamlit.testing.v1 import AppTest

    import spectra.browser.video as video

    worker = Mock()
    worker.pc.localDescription = SimpleNamespace(sdp="answer-sdp", type="answer")
    create = Mock(return_value=worker)
    component = Mock(
        return_value={
            "playing": True,
            "id": "offer-1",
            "offer": {"sdp": "offer-sdp", "type": "offer"},
        }
    )
    monkeypatch.setattr(video, "_component", component)
    monkeypatch.setattr(video, "_create_worker", create)
    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.browser.video import camera_streamer\n"
        "camera_streamer(st.session_state['test_session'])"
    )
    ui.session_state["test_session"] = browser
    ui.run()
    assert not ui.exception
    create.assert_called_once_with(browser)
    worker.process_offer.assert_called_once_with("offer-sdp", "offer", timeout=15)
    assert component.call_args.kwargs["answer"] == {
        "id": "offer-1",
        "sdp": "answer-sdp",
        "type": "answer",
    }
    component.return_value = {"playing": False}
    ui.run()
    assert not ui.exception
    worker.stop.assert_called_once()
    assert browser.finished and not browser.saved


@pytest.mark.parametrize("error", ["NotAllowedError", "ConnectionTimeout"])
def test_camera_error_finishes_without_saving_an_empty_session(browser, monkeypatch, error):
    from unittest.mock import Mock

    from streamlit.testing.v1 import AppTest

    import spectra.browser.video as video

    monkeypatch.setattr(video, "_component", Mock(return_value={"playing": False, "error": error}))
    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.browser.video import camera_streamer\n"
        "camera_streamer(st.session_state['test_session'])"
    )
    ui.session_state["test_session"] = browser
    ui.run()
    assert not ui.exception
    assert ui.error
    assert browser.finished and not browser.saved
    assert browser.error_key == "browser.processing_error"


def test_camera_negotiation_failure_stops_worker_and_surfaces_error(browser, monkeypatch):
    from unittest.mock import Mock

    from streamlit.testing.v1 import AppTest

    import spectra.browser.video as video

    worker = Mock()
    worker.process_offer.side_effect = RuntimeError("test negotiation failure")
    monkeypatch.setattr(video, "_create_worker", Mock(return_value=worker))
    monkeypatch.setattr(
        video,
        "_component",
        Mock(return_value={"playing": True, "id": "offer", "offer": {"sdp": "", "type": "offer"}}),
    )
    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.browser.video import camera_streamer\n"
        "camera_streamer(st.session_state['test_session'])"
    )
    ui.session_state["test_session"] = browser
    ui.run()
    assert ui.exception[0].message == "test negotiation failure"
    worker.stop.assert_called_once()
    assert browser.finished and not browser.saved
    assert browser.error_key == "browser.processing_error"


def test_severe_pain_before_session_blocks_exercise_and_releases_room(service, plan, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import spectra.therapist_panel.app as panel_module

    monkeypatch.setattr(panel_module, "get_service", lambda: service)
    code = Rooms(service).create(plan)
    ui = AppTest.from_string("from spectra.therapist_panel.app import main\nmain()")
    ui.query_params["sala"] = code
    ui.run()
    ui.text_input[1].input("1234")
    ui.button[0].click().run()
    ui.select_slider(key="pain_before").set_value(9).run()
    assert not ui.exception
    assert all(button.label != "Estou pronto(a), abrir a câmera" for button in ui.button)
    _button(ui, "Encerrar sem exercitar").click().run()
    assert ui.session_state["browser_stage"] == "left"
    with service.db.transaction() as connection:
        assert connection.execute("SELECT status FROM browser_room").fetchone()[0] == "waiting"


def test_pain_scores_are_saved_with_the_session(browser, service, clock):
    browser.pain_before = 6
    browser.process(np.zeros((720, 960, 3), dtype=np.uint8))
    browser.finish()
    session_id = browser.admission.session.id
    assert service.db.get_session(session_id).pain_before == 6
    assert service.db.record_pain_after(session_id, 3)
    assert not service.db.record_pain_after(session_id, 1)  # Recorded once only.
    assert service.db.get_session(session_id).pain_after == 3
    with pytest.raises(ValueError):
        service.db.record_pain_after(session_id, 11)


def test_pain_after_requires_a_finished_session(browser, service):
    assert not service.db.record_pain_after(browser.admission.session.id, 2)


def test_live_status_is_published_for_the_therapist_monitor(browser, clock):
    from spectra.browser.live import live_status

    session_id = browser.admission.session.id
    assert live_status(session_id).frames == 0
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    browser.process(frame)
    browser._detector.result = make_detection((True,) * 5)
    clock.advance(30)
    browser.process(frame)
    status = live_status(session_id)
    assert status.frames == 2 and status.hand_visible
    assert status.exercise == "open_close" and status.exercise_total == 1
    assert status.limit_seconds == 300 and 0 < status.time_fraction < 1
    assert live_status("unknown") is None


def test_live_status_only_marks_the_treated_hand_as_visible(browser):
    browser.process(np.zeros((720, 960, 3), dtype=np.uint8))
    browser._detector.result = make_detection((True,) * 5, label="Left")
    browser.process(np.zeros((720, 960, 3), dtype=np.uint8))
    assert browser.status().hand_visible is False


def test_room_overview_derives_status_without_secrets(service, plan, clock):
    rooms = Rooms(service)
    code = rooms.create(plan)
    [room] = rooms.overview()
    assert room.status == "waiting" and room.patient_id == plan.patient_id
    assert code not in repr(room)
    rooms.admit(code, "1234")
    assert rooms.overview()[0].status == "active"
    clock.advance(LEASE_SECONDS + 1)
    assert rooms.overview()[0].status == "interrupted"
    clock.advance(ROOM_LIFETIME_SECONDS)
    assert rooms.overview()[0].status == "interrupted"
    clock.advance(13 * 3600)
    assert rooms.overview() == []


def test_therapist_home_shows_dashboard_and_room_monitor(service, plan):
    from streamlit.testing.v1 import AppTest

    Rooms(service).create(plan)
    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.therapist_panel.app import render_home\n"
        "render_home(st.session_state['_test_service'])"
    )
    ui.session_state["_test_service"] = service
    ui.session_state["therapist_name"] = "Ana Souza"
    ui.run()
    assert not ui.exception
    html = "".join(element.proto.body for element in ui.get("html"))
    assert "Olá, Ana!" in html
    assert "Aguardando paciente" in html
    assert "sp-ghost" in html


def test_panel_creates_browser_room_and_button_instead_of_command(service, plan):
    from streamlit.testing.v1 import AppTest

    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.therapist_panel.app import render_session\n"
        "render_session(st.session_state['_test_service'])"
    )
    ui.session_state["_test_service"] = service
    ui.run()
    assert not ui.exception
    assert ui.radio[0].value == "browser"
    ui.button[0].click().run()
    assert not ui.exception
    assert len(ui.session_state["browser_room_code"]) == 14
    assert ui.code[0].value == ui.session_state["browser_room_code"]
    assert ui.get("link_button")[0].proto.url.startswith("?paciente=1&sala=")
    assert all("python -m spectra" not in success.value for success in ui.success)
