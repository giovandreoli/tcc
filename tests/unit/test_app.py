"""Shell-level tests: argument parsing, session attachment and the safety guard.

The camera loop itself is not tested (it is a thin wrapper over OpenCV); everything the
shell *decides* is.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectra.__main__ import build_parser, main
from spectra.app import MODE_FACTORIES, SpectraApp
from spectra.modes.base import ACTION_MENU, ACTION_QUIT, AppMode
from spectra.modes.physio import PhysioMode
from spectra.service import PatientForm, SessionPlan, SpectraService
from spectra.storage.db import Database
from spectra.storage.models import Hand, Sex
from spectra.storage.security import Cipher, KeyStore
from tests.conftest import make_detection

VALID_CPF = "529.982.247-25"
WIDTH, HEIGHT = 640, 480
POINT = (False, True, False, False, False)


@pytest.fixture
def app(config) -> SpectraApp:
    instance = SpectraApp(config)
    instance.build_modes(WIDTH, HEIGHT)
    return instance


@pytest.fixture
def service(config) -> SpectraService:
    cipher = Cipher.from_key_store(KeyStore(config.key_path))
    instance = SpectraService(db=Database(config.db_path, cipher), config=config)
    yield instance
    instance.close()


def blank_frame() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


class TestArgumentParsing:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.session is None
        assert args.hand is None
        assert args.no_sound is False

    def test_session_is_accepted(self):
        assert build_parser().parse_args(["--session", "abc"]).session == "abc"

    def test_hand_is_restricted(self):
        assert build_parser().parse_args(["--hand", "Left"]).hand == "Left"
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--hand", "Middle"])

    def test_version_exits_cleanly(self):
        with pytest.raises(SystemExit) as info:
            build_parser().parse_args(["--version"])
        assert info.value.code == 0

    def test_a_missing_camera_returns_a_failure_code(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPECTRA_DATA_DIR", str(tmp_path / "Spectra"))
        assert main(["--camera", "99"]) == 1


class TestModeRouting:
    def test_every_mode_has_a_factory(self):
        assert set(MODE_FACTORIES) == {
            AppMode.MENU,
            AppMode.FREE_DRAW,
            AppMode.GUIDED_DRAW,
            AppMode.EDU_COLORS,
            AppMode.EDU_COUNT,
            AppMode.PHYSIO,
        }

    def test_the_app_starts_on_the_menu(self, app):
        assert app.mode is AppMode.MENU

    def test_quit_stops_the_loop(self, app):
        assert app.handle_action(ACTION_QUIT) is False

    def test_a_mode_action_switches_mode(self, app):
        app.handle_action(AppMode.PHYSIO)
        assert app.mode is AppMode.PHYSIO

    def test_returning_to_the_menu_rebuilds_it(self, app):
        before = app.modes[AppMode.MENU]
        app.handle_action(ACTION_MENU)
        assert app.modes[AppMode.MENU] is not before
        assert app.mode is AppMode.MENU

    def test_an_unknown_action_keeps_running(self, app):
        assert app.handle_action("something-else") is True

    def test_rendering_a_frame_works_headless(self, app):
        output, action = app.render_frame(blank_frame(), make_detection(POINT))
        assert output.shape == (HEIGHT, WIDTH, 3)
        assert action is None


class TestSharedSessionState:
    def test_every_mode_shares_one_recorder(self, app):
        recorders = {id(mode.context.recorder) for mode in app.modes.values()}
        assert len(recorders) == 1
        assert next(iter(recorders)) == id(app.recorder)

    def test_the_recorder_survives_a_mode_change(self, app, clock):
        app.handle_action(AppMode.PHYSIO)
        for _ in range(5):
            clock.advance(0.1)
            app.render_frame(blank_frame(), make_detection(POINT))
        before = app.recorder.frames_total
        app.handle_action(ACTION_MENU)
        assert app.recorder.frames_total == before > 0

    def test_the_guard_limits_are_taken_from_the_config(self, config):
        config.session_limit_minutes = 7
        config.rest_prompt_minutes = 2
        instance = SpectraApp(config)
        assert instance.guard.limit_minutes == 7
        assert instance.guard.rest_every_minutes == 2

    def test_the_session_ends_when_the_limit_expires(self, app, clock):
        app.guard.start(now=clock.now)
        clock.advance(app.guard.limit_seconds + 1)
        app.render_frame(blank_frame(), make_detection(POINT))
        assert app.guard.expired is True


class TestStandaloneMode:
    def test_nothing_is_attached_without_a_session_id(self, app):
        assert app.attached is False

    def test_finishing_is_a_no_op_when_detached(self, app):
        app.finish_session()  # must not raise

    def test_no_pin_is_required_when_detached(self, app):
        assert app._pin_mode is None


class TestAttachedSession:
    def _prepare(self, service) -> str:
        patient = service.create_patient(
            PatientForm(name="Maria da Silva", cpf=VALID_CPF, sex=Sex.FEMALE, pin="1234")
        )
        service.record_consent(patient.id, "Maria da Silva")
        session = service.start_session(SessionPlan(patient_id=patient.id, hand=Hand.LEFT))
        return session.id

    def test_attaching_applies_the_session_hand(self, service, config):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        assert app.attached
        assert app.hand_label == "Left"
        app.finish_session()

    def test_an_unknown_session_is_refused(self, config):
        app = SpectraApp(config, session_id="does-not-exist")
        with pytest.raises(RuntimeError):
            app.attach_session()

    def test_a_pin_screen_guards_an_attached_session(self, service, config):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        app.build_modes(WIDTH, HEIGHT)
        assert app._pin_mode is not None
        assert app.verify_pin("1234") is True
        assert app.verify_pin("0000") is False
        app.finish_session()

    def test_the_pin_screen_is_rendered_before_any_mode(self, service, config):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        app.build_modes(WIDTH, HEIGHT)
        output, _ = app.render_frame(blank_frame(), make_detection(POINT))
        assert output.any()
        assert app.recorder.frames_total == 0  # nothing recorded before the PIN
        app.finish_session()

    def test_finishing_persists_the_metrics(self, service, config, clock):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        app.build_modes(WIDTH, HEIGHT)
        app._pin_mode = None
        app.handle_action(AppMode.PHYSIO)
        for _ in range(10):
            clock.advance(0.1)
            app.render_frame(blank_frame(), make_detection(POINT))
        app.finish_session()

        with SpectraService.open(config) as reopened:
            stored = reopened.db.get_session(session_id)
            assert stored.ended_at is not None
            assert stored.tracking_quality["frames_total"] > 0
            assert {m.metric for m in reopened.db.list_metrics(session_id)} >= {"rom", "quality"}

    def test_exercise_runs_with_repetitions_are_collected(self, service, config):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        app.build_modes(WIDTH, HEIGHT)
        physio = app.modes[AppMode.PHYSIO]
        assert isinstance(physio, PhysioMode)
        physio.progress.add_rep()
        assert [run.name for run in app.collect_runs()] == ["open_close"]
        app.finish_session()

    def test_runs_without_repetitions_are_not_recorded(self, service, config):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        app.build_modes(WIDTH, HEIGHT)
        assert app.collect_runs() == []
        app.finish_session()

    def test_guided_attempts_are_collected(self, service, config):
        session_id = self._prepare(service)
        service.close()
        app = SpectraApp(config, session_id=session_id)
        app.attach_session()
        app.build_modes(WIDTH, HEIGHT)
        guided = app.modes[AppMode.GUIDED_DRAW]
        guided.scorer.add(guided.path.points[0], now=0.0)
        guided.finish()
        assert len(app.collect_traces()) == 1
        app.finish_session()
