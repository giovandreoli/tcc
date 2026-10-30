"""Room admission and browser sessions driven by landmark batches, without hardware."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spectra.browser.frames import BrowserFrame
from spectra.browser.rooms import LEASE_SECONDS, ROOM_LIFETIME_SECONDS, RoomError, Rooms
from spectra.browser.session import BrowserSession
from spectra.detection.hand_detector import DetectedHand
from spectra.exercises.catalog import ExerciseSpec, builtin_spec
from spectra.service import PatientForm, SessionPlan, SpectraService
from spectra.storage.db import ConsentRequiredError, Database
from spectra.storage.security import Cipher
from tests.conftest import make_hand


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
    assert admission.exercises == (builtin_spec("open_close"),)
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


# ------------------------------------------------------------------ sessions
OPEN = (True,) * 5
FIST = (False,) * 5


class Feeder:
    """Sends landmark batches the way the browser does, with a browser-side clock."""

    def __init__(self, session, clock, label="Right"):
        self.session = session
        self.clock = clock
        self.label = label
        self.at_ms = 0.0

    def frame(self, pattern=None, label=None):
        self.clock.advance(0.1)
        self.at_ms += 100.0
        if pattern is None:
            return BrowserFrame(self.at_ms, None)
        hand_label = label or self.label
        return BrowserFrame(
            self.at_ms, DetectedHand(make_hand(*pattern, label=hand_label), hand_label, 0.9)
        )

    def hold(self, pattern, frames=8, **kwargs):
        self.session.process_batch([self.frame(pattern, **kwargs) for _ in range(frames)])


def _patch_service(monkeypatch, service):
    import spectra.browser.session as browser_module

    monkeypatch.setattr(
        browser_module.SpectraService, "open", classmethod(lambda cls, config=None: service)
    )


@pytest.fixture
def browser(service, plan, monkeypatch):
    _patch_service(monkeypatch, service)
    rooms = Rooms(service)
    admission = rooms.admit(rooms.create(plan), "1234")
    return BrowserSession(service.config, admission)


@pytest.fixture
def feed(browser, clock):
    return Feeder(browser, clock)


def test_browser_session_starts_on_first_batch_without_any_camera(browser, feed):
    assert browser.runner is None
    feed.hold(None, frames=1)
    assert browser.runner is not None
    assert [spec.key for spec in browser.specs] == ["open_close"]
    assert browser.guard.limit_minutes == 5
    assert browser.status().frames == 1
    assert browser.status().exercise_title == "Abrir e Fechar a Mão"


def test_browser_counts_open_close_with_a_partial_fist(browser, feed):
    feed.hold(OPEN)
    feed.hold((True, False, False, False, False))
    assert browser.status().reps == 1


@pytest.mark.parametrize("label", ["Left", "Right"])
def test_browser_counts_only_the_anatomical_hand_of_the_session(browser, feed, label):
    browser.hand_label = label
    feed.label = label
    feed.hold(OPEN)
    feed.hold(FIST)
    assert browser.status().hand_visible
    assert browser.status().reps == 1


def test_live_status_only_marks_the_treated_hand_as_visible(browser, feed):
    feed.hold(OPEN, frames=2, label="Left")
    assert browser.status().hand_visible is False
    assert browser.status().frames == 2


def test_browser_rest_prompt_is_exposed_beside_the_video(browser, feed, clock, monkeypatch):
    from spectra.core.session_guard import SessionGuard

    monkeypatch.setattr("spectra.browser.session.time.monotonic", lambda: clock())
    feed.hold(None, frames=1)
    browser.guard = SessionGuard(limit_minutes=5, rest_every_minutes=1)
    browser.guard.start()
    clock.advance(61)
    feed.hold(None, frames=1)
    assert browser.status().safety_key == "safety.rest"
    clock.advance(9)
    assert browser.status().safety_key is None


def test_browser_next_button_preserves_recorded_reps(browser, feed):
    browser.specs = (builtin_spec("open_close"), builtin_spec("finger_touch"))
    feed.hold(OPEN)
    feed.hold(FIST)
    assert browser.status().reps == 1
    browser.next_exercise()
    status = browser.status()
    assert status.exercise == "finger_touch"
    assert status.exercise_index == 1 and status.reps == 0
    assert status.total_reps == 1
    browser.finish()
    with pytest.raises(RuntimeError, match="inactive"):
        browser.next_exercise()


def test_browser_finishes_once_and_saves_report_metrics(browser, service, feed):
    feed.hold(OPEN, frames=15)
    feed.hold(FIST, frames=15)
    browser.finish()
    browser.finish()
    assert browser.saved
    session_id = browser.admission.session.id
    assert service.db.get_session(session_id).ended_at is not None
    [run] = service.db.list_exercise_runs(session_id)
    assert run.name == "open_close" and run.reps == 1
    assert {metric.metric for metric in service.db.list_metrics(session_id)} >= {"rom", "quality"}
    feed.hold(OPEN, frames=1)  # Ignored once finished.
    assert browser.status().frames == 30


def test_custom_exercise_is_counted_and_reported_with_its_title(service, plan, monkeypatch, clock):
    from spectra.exercises import Movement
    from spectra.reports.builder import build_session_report

    _patch_service(monkeypatch, service)
    spec = service.save_custom_exercise(
        plan.patient_id,
        ExerciseSpec(
            "", Movement.FINGER_LIFT, title="Levantar o médio", finger="middle", target_reps=2
        ),
    )
    rooms = Rooms(service)
    code = rooms.create(replace(plan, exercises=(spec.key,)))
    session = BrowserSession(service.config, rooms.admit(code, "1234"))
    feed = Feeder(session, clock)
    for _ in range(2):
        feed.hold((False, False, True, False, False))
        feed.hold(FIST)
    assert session.status().reps == 2
    assert session.status().exercise_title == "Levantar o médio"
    session.finish()
    [run] = service.db.list_exercise_runs(session.admission.session.id)
    assert run.name == spec.key
    assert run.payload["title"] == "Levantar o médio"
    assert run.payload["details"]["finger"] == "middle"
    report = build_session_report(service.db, session.admission.session.id)
    assert report.exercises[0].label() == "Levantar o médio"


def test_room_keeps_a_snapshot_of_the_exercise_after_it_is_archived(service, plan):
    from spectra.exercises import Movement

    spec = service.save_custom_exercise(
        plan.patient_id, ExerciseSpec("", Movement.OPEN_CLOSE, title="Punho firme", hold_seconds=2)
    )
    rooms = Rooms(service)
    code = rooms.create(replace(plan, exercises=(spec.key,)))
    assert service.archive_custom_exercise(plan.patient_id, spec.key)
    admission = rooms.admit(code, "1234")
    assert admission.exercises[0].title == "Punho firme"
    assert admission.exercises[0].hold_seconds == 2


def test_room_rejects_another_patients_custom_exercise(service, plan):
    from spectra.exercises import Movement

    other = service.create_patient(PatientForm(name="Other", cpf="11144477735", pin="4321"))
    spec = service.save_custom_exercise(
        other.id, ExerciseSpec("", Movement.OPEN_CLOSE, title="Do outro paciente")
    )
    with pytest.raises(RoomError, match="invalid_plan"):
        Rooms(service).create(replace(plan, exercises=(spec.key,)))


def test_camera_not_started_releases_room_without_saving(browser, service):
    browser.finish()
    assert browser.finished and not browser.saved
    with service.db.transaction() as connection:
        row = connection.execute("SELECT status FROM browser_room").fetchone()
    assert row["status"] == "waiting"
    assert service.db.get_session(browser.admission.session.id).ended_at is None


def test_browser_time_limit_finishes_session(browser, service, feed, clock):
    feed.hold(None, frames=1)
    started = service.db.get_session(browser.admission.session.id).started_at
    clock.advance(301)
    feed.hold(None, frames=1)
    assert browser.finished and browser.saved
    assert started == pytest.approx(1000.1)


def test_save_failure_is_not_reported_as_success(browser, service, feed, monkeypatch):
    feed.hold(None, frames=1)

    def fail(*args, **kwargs):
        raise RuntimeError("test write failure")

    monkeypatch.setattr(service, "finish_session", fail)
    with pytest.raises(RuntimeError, match="test write failure"):
        browser.finish()
    assert browser.finished and not browser.saved
    assert browser.status().error_key == "browser.save_error"
    browser.finish()  # No second write or success-shaped retry after a partial failure.


def test_processing_error_is_exposed_not_hidden(browser, feed, monkeypatch):
    def fail(at):
        raise RuntimeError("test initialization failure")

    monkeypatch.setattr(browser, "_on_start", fail)
    with pytest.raises(RuntimeError, match="test initialization"):
        feed.hold(None, frames=1)
    assert browser.status().error_key == "browser.processing_error"


def test_pain_scores_are_saved_with_the_session(browser, service, feed):
    browser.pain_before = 6
    feed.hold(None, frames=1)
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


def test_live_status_is_published_for_the_therapist_monitor(browser, feed, clock):
    from spectra.browser.live import live_status

    session_id = browser.admission.session.id
    assert live_status(session_id).frames == 0
    feed.hold(None, frames=1)
    clock.advance(30)
    feed.hold(OPEN, frames=1)
    status = live_status(session_id)
    assert status.frames == 2 and status.hand_visible
    assert status.exercise == "open_close" and status.exercise_total == 1
    assert status.limit_seconds == 300 and 0 < status.time_fraction < 1
    assert status.cue == "Agora feche a mão em punho."
    assert live_status("unknown") is None


# --------------------------------------------------------- camera component
def _component_app(monkeypatch, value):
    from unittest.mock import Mock

    from streamlit.testing.v1 import AppTest

    import spectra.browser.video as video

    component = Mock(return_value=value)
    monkeypatch.setattr(video, "_component", component)
    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.browser.video import camera_streamer\n"
        "camera_streamer(st.session_state['test_session'], 'cam')"
    )
    return ui, component


def _wire(frames):
    rows = []
    for frame in frames:
        if frame.hand is None:
            rows.append([frame.at_ms, None, 0, None])
        else:
            coordinates = [v for point in frame.hand.landmarks for v in (point.x, point.y, point.z)]
            rows.append([frame.at_ms, frame.hand.label, frame.hand.score, coordinates])
    return rows


def test_camera_batches_are_acknowledged_and_never_processed_twice(browser, feed, monkeypatch):
    frames = [feed.frame(OPEN) for _ in range(8)] + [feed.frame(FIST) for _ in range(8)]
    ui, component = _component_app(monkeypatch, {"playing": True, "frames": _wire(frames)})
    ui.session_state["test_session"] = browser
    ui.run()
    assert not ui.exception
    assert browser.status().frames == 16 and browser.status().reps == 1
    ui.run()  # The browser resends unacknowledged frames; duplicates are ignored.
    assert browser.status().frames == 16
    assert component.call_args.kwargs["ack"] == frames[-1].at_ms
    component.return_value = {"playing": False, "frames": []}
    ui.run()
    assert browser.finished and browser.saved


def test_malformed_batch_is_rejected_without_touching_the_session(browser, monkeypatch):
    bad = {"playing": True, "frames": [[1, "Right", 0.9, [0.5] * 10]]}
    ui, _ = _component_app(monkeypatch, bad)
    ui.session_state["test_session"] = browser
    ui.run()
    assert not ui.exception
    assert browser.runner is None and not browser.finished


@pytest.mark.parametrize("error", ["NotAllowedError", "NotFoundError"])
def test_camera_error_before_any_frame_keeps_the_room_for_a_retry(
    browser, service, monkeypatch, error
):
    ui, _ = _component_app(monkeypatch, {"playing": False, "error": error, "frames": []})
    ui.session_state["test_session"] = browser
    ui.run()
    assert not ui.exception
    assert not browser.finished and not browser.saved
    with service.db.transaction() as connection:
        assert connection.execute("SELECT status FROM browser_room").fetchone()[0] == "active"


def test_camera_component_assets_are_local_and_packaged():
    from spectra.browser.video import COMPONENT_DIR, ensure_model

    html = (COMPONENT_DIR / "index.html").read_text(encoding="utf-8")
    script = (COMPONENT_DIR / "camera.js").read_text(encoding="utf-8")
    assert "http" not in html
    assert "cdn" not in script.lower() and "googleapis" not in script
    assert (COMPONENT_DIR / "vendor" / "vision_bundle.js").is_file()
    assert (COMPONENT_DIR / "vendor" / "wasm" / "vision_wasm_internal.wasm").is_file()
    assert ensure_model().is_file()


# ---------------------------------------------------------------- patient UI
def _button(ui, label):
    return next(button for button in ui.button if button.label == label)


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
    assert ui.session_state["browser_session"].admission.exercises == (builtin_spec("open_close"),)
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


def test_internet_requests_only_ever_reach_the_patient_page(service, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import spectra.therapist_panel.app as panel_module

    monkeypatch.setattr(panel_module, "get_service", lambda: service)
    monkeypatch.setattr(panel_module, "_request_is_local", lambda: False)
    ui = AppTest.from_string("from spectra.therapist_panel.app import main\nmain()")
    ui.session_state["therapist_id"] = "someone"  # Even a stolen panel session gets nothing.
    ui.run()
    assert not ui.exception
    labels = [field.label for field in ui.text_input]
    assert labels == ["Código da sala", "PIN do paciente (4 dígitos)"]
    assert not ui.sidebar.radio


# ------------------------------------------------------------- therapist UI
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


def _session_screen(service):
    from streamlit.testing.v1 import AppTest

    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from spectra.therapist_panel.app import render_session\n"
        "render_session(st.session_state['_test_service'])"
    )
    ui.session_state["_test_service"] = service
    return ui


def test_panel_creates_browser_room_and_offers_internet_access(service, plan):
    ui = _session_screen(service)
    ui.run()
    assert not ui.exception
    assert ui.radio[0].value == "browser"
    assert any("internet" in markdown.value for markdown in ui.markdown)
    _button(ui, "Iniciar sessão").click().run()
    assert not ui.exception
    assert len(ui.session_state["browser_room_code"]) == 14
    assert ui.code[0].value == ui.session_state["browser_room_code"]
    assert ui.get("link_button")[0].proto.url.startswith("?paciente=1&sala=")
    assert all("python -m spectra" not in success.value for success in ui.success)


def test_panel_session_lists_the_patients_custom_exercises(service, plan):
    from spectra.exercises import Movement

    spec = service.save_custom_exercise(
        plan.patient_id, ExerciseSpec("", Movement.PINCH, title="Pinça com o anelar", finger="ring")
    )
    ui = _session_screen(service)
    ui.run()
    assert not ui.exception
    assert "Pinça com o anelar" in ui.multiselect[0].options
    assert spec.key not in ui.multiselect[0].value
