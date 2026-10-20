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

    def detect(self, frame):
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

    def fail():
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


def test_patient_ui_admits_by_code_and_pin_without_therapist_login(service, plan, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import spectra.browser.page as page_module
    import spectra.therapist_panel.app as panel_module

    monkeypatch.setattr(panel_module, "get_service", lambda: service)
    monkeypatch.setattr(page_module, "webrtc_streamer", lambda **kwargs: None)
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
    ui.button[0].click().run()
    assert not ui.exception
    assert ui.session_state["browser_session"].finished


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
