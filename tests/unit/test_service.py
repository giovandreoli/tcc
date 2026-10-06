from __future__ import annotations

from datetime import date, timedelta

import pytest

from spectra.consent import load_consent
from spectra.guided.scoring import TraceResult
from spectra.guided.shapes import Difficulty
from spectra.metrics.analysis import RunWindow, analyze_run, analyze_session
from spectra.metrics.recorder import SessionRecorder
from spectra.service import PatientForm, SessionPlan, SpectraService
from spectra.storage.db import ConsentRequiredError, Database, DuplicateCPFError
from spectra.storage.models import AuditAction, Hand, Sex
from spectra.storage.security import Cipher, KeyStore
from tests.conftest import make_hand

VALID_CPF = "529.982.247-25"
VALID_CPF_2 = "168.995.350-09"
OPEN = (True, True, True, True, True)
FIST = (False, False, False, False, False)


@pytest.fixture
def service(config) -> SpectraService:
    cipher = Cipher.from_key_store(KeyStore(config.key_path))
    database = Database(config.db_path, cipher)
    instance = SpectraService(db=database, config=config)
    yield instance
    instance.close()


def valid_form(**overrides) -> PatientForm:
    defaults = {
        "name": "Maria da Silva",
        "cpf": VALID_CPF,
        "birth_date": date(1990, 5, 20),
        "sex": Sex.FEMALE,
        "affected_hand": Hand.LEFT,
        "pin": "1234",
    }
    defaults.update(overrides)
    return PatientForm(**defaults)


def recorded_session(cycles: int = 6) -> SessionRecorder:
    """A recorder fed an open/close cycle at 30 fps."""
    recorder = SessionRecorder(rate_hz=10.0)
    now = 0.0
    for _ in range(cycles):
        for hand in [make_hand(*OPEN)] * 15 + [make_hand(*FIST)] * 15:
            recorder.observe(hand, 0.9, now=now)
            now += 1 / 30
    return recorder


class TestPatientFormValidation:
    def test_a_complete_form_is_valid(self):
        assert valid_form().validate() == []

    def test_the_name_is_required(self):
        assert "panel.error.name_required" in valid_form(name="  ").validate()

    def test_an_invalid_cpf_is_caught_before_the_database(self):
        assert "panel.error.cpf_invalid" in valid_form(cpf="111.111.111-11").validate()

    def test_a_short_pin_is_caught(self):
        assert "panel.error.pin_invalid" in valid_form(pin="12").validate()

    def test_an_empty_pin_is_allowed(self):
        assert valid_form(pin="").validate() == []

    def test_future_dates_are_rejected(self):
        tomorrow = date.today() + timedelta(days=1)
        assert "panel.error.birth_date_future" in valid_form(birth_date=tomorrow).validate()
        assert "panel.error.injury_date_future" in valid_form(injury_date=tomorrow).validate()

    def test_the_form_maps_onto_a_patient(self):
        patient = valid_form().to_patient()
        assert patient.name == "Maria da Silva"
        assert patient.affected_hand is Hand.LEFT


class TestPatientCreation:
    def test_a_valid_form_creates_a_patient(self, service):
        patient = service.create_patient(valid_form())
        assert service.db.get_patient(patient.id, audit=False) is not None

    def test_an_invalid_form_never_reaches_the_database(self, service):
        with pytest.raises(ValueError):
            service.create_patient(valid_form(cpf="111.111.111-11"))
        assert service.db.list_patients() == []

    def test_a_duplicate_cpf_is_surfaced(self, service):
        service.create_patient(valid_form())
        with pytest.raises(DuplicateCPFError):
            service.create_patient(valid_form(name="Outra"))

    def test_the_pin_is_set_when_provided(self, service):
        patient = service.create_patient(valid_form(pin="4321"))
        assert service.db.authenticate_patient(patient.id, "4321").authenticated


class TestConsentFlow:
    def test_a_new_patient_has_no_consent(self, service):
        patient = service.create_patient(valid_form())
        assert service.has_consent(patient.id) is False

    def test_recording_consent_stores_the_current_version_and_hash(self, service):
        patient = service.create_patient(valid_form())
        document = service.current_consent_document()
        consent = service.record_consent(patient.id, "Maria da Silva")
        assert consent.version == document.version
        assert consent.document_hash == document.document_hash
        assert service.has_consent(patient.id)

    def test_the_consent_text_is_loaded_from_a_file(self):
        document = load_consent("pt_BR")
        assert "Termo de Consentimento" in document.text
        assert len(document.document_hash) == 64

    def test_an_unknown_locale_falls_back_to_portuguese(self):
        assert "Termo de Consentimento" in load_consent("de_DE").text

    def test_the_hash_changes_with_the_text(self):
        from spectra.consent import ConsentDocument

        assert ConsentDocument("1", "a").document_hash != ConsentDocument("1", "b").document_hash

    def test_a_session_is_refused_without_consent(self, service):
        patient = service.create_patient(valid_form())
        with pytest.raises(ConsentRequiredError):
            service.start_session(SessionPlan(patient_id=patient.id))

    def test_a_session_is_allowed_after_consent(self, service):
        patient = service.create_patient(valid_form())
        service.record_consent(patient.id, "Maria")
        assert service.start_session(SessionPlan(patient_id=patient.id)).id


class TestSessionPlan:
    def test_the_standard_profile_has_no_warnings(self):
        assert SessionPlan(patient_id="x", gesture_profile_name="standard").profile_warnings() == []

    def test_the_simplified_profile_has_no_errors(self):
        warnings = SessionPlan(patient_id="x", gesture_profile_name="simplified").profile_warnings()
        assert not any(issue.is_error for issue in warnings)

    def test_an_unknown_profile_falls_back_to_standard(self):
        assert (
            SessionPlan(patient_id="x", gesture_profile_name="nope").gesture_profile().name
            == "standard"
        )


class TestCalibration:
    def test_calibration_is_stored_with_the_patient(self, service):
        patient = service.create_patient(valid_form())
        profile = service.calibrate(patient.id, [make_hand(*OPEN)] * 3, [make_hand(*FIST)] * 3)
        assert profile.is_calibrated
        assert service.calibration_for(patient.id).thresholds == profile.thresholds

    def test_an_uncalibrated_patient_gets_the_defaults(self, service):
        patient = service.create_patient(valid_form())
        assert service.calibration_for(patient.id).is_calibrated is False

    def test_unreadable_stored_calibration_falls_back(self, service):
        patient = service.create_patient(valid_form())
        patient.calibration = {"thresholds": "garbage"}
        service.db.update_patient(patient)
        assert service.calibration_for(patient.id).is_calibrated is False


class TestSessionAnalysis:
    def test_a_session_produces_every_metric_family(self):
        metrics = analyze_session(recorded_session())
        assert set(metrics) >= {"quality", "rom", "opposition", "wrist", "tremor", "fatigue"}

    def test_guided_attempts_add_the_accuracy_metric(self):
        trace = TraceResult("line", Difficulty.EASY, 100, 0.01, 0.02, 0.9, 0.95, 10.0, 80.0)
        assert "accuracy" in analyze_session(recorded_session(), traces=[trace])

    def test_a_run_is_sliced_from_the_session_window(self):
        recorder = recorded_session()
        window = RunWindow(name="open_close", started_at=0.0, ended_at=1.0)
        assert len(recorder.window(0.0, 1.0)) < len(recorder.samples)
        assert "rom" in analyze_run(recorder, window)

    def test_an_empty_recording_still_produces_payloads(self):
        metrics = analyze_session(SessionRecorder())
        assert metrics["quality"]["frames_total"] == 0
        assert len(metrics["rom"]["fingers"]) == 5

    def test_payloads_are_plain_json_types(self):
        import json

        json.dumps(analyze_session(recorded_session()))


class TestFinishSession:
    def _started(self, service):
        patient = service.create_patient(valid_form())
        service.record_consent(patient.id, "Maria")
        return patient, service.start_session(SessionPlan(patient_id=patient.id))

    def test_metrics_and_samples_are_persisted(self, service):
        _patient, session = self._started(service)
        runs = [
            RunWindow(
                name="open_close",
                started_at=0.0,
                ended_at=12.0,
                reps=6,
                target=8,
                rep_times=(1.0, 3.0, 5.0, 7.0, 9.0, 11.0),
            )
        ]
        service.finish_session(session, recorded_session(), runs=runs)

        assert len(service.db.list_exercise_runs(session.id)) == 1
        metrics = {m.metric for m in service.db.list_metrics(session.id)}
        assert {"rom", "tremor", "fatigue", "quality"} <= metrics
        assert service.db.list_samples(session.id)

    def test_tracking_quality_is_stored_on_the_session(self, service):
        _patient, session = self._started(service)
        service.finish_session(session, recorded_session())
        assert service.db.get_session(session.id).tracking_quality["frames_total"] > 0

    def test_pain_scores_and_notes_are_stored(self, service):
        _patient, session = self._started(service)
        service.finish_session(session, recorded_session(), pain_before=7, pain_after=4, notes="ok")
        stored = service.db.get_session(session.id)
        assert (stored.pain_before, stored.pain_after) == (7, 4)
        assert stored.notes == "ok"

    def test_sample_persistence_can_be_skipped(self, service):
        _patient, session = self._started(service)
        service.finish_session(session, recorded_session(), persist_samples=False)
        assert service.db.list_samples(session.id) == []

    def test_a_finished_session_can_be_reported(self, service):
        _patient, session = self._started(service)
        service.finish_session(session, recorded_session())
        report, generated = service.generate_reports(session.id)
        assert generated.pdf_path.exists()
        assert generated.txt_path.exists()
        assert report.session_id == session.id

    def test_demo_mode_is_honoured_by_the_report(self, service):
        _patient, session = self._started(service)
        service.finish_session(session, recorded_session())
        report, _ = service.generate_reports(session.id, demo_mode=True)
        assert report.patient.display_name == "M.D.S."


class TestShareAndBackup:
    def test_exporting_without_a_share_dir_fails_loudly(self, service):
        service.config.share_dir = None
        with pytest.raises(ValueError):
            service.export_reports_to_share(None, "patient")

    def test_a_backup_is_written_and_audited(self, service, tmp_path):
        service.create_patient(valid_form())
        result = service.backup("senha-forte", destination=tmp_path / "share")
        assert result.path.exists()
        assert service.db.audit.count([AuditAction.BACKUP_CREATED]) == 1

    def test_the_backup_does_not_contain_readable_data(self, service, tmp_path):
        service.create_patient(valid_form())
        result = service.backup("senha-forte", destination=tmp_path / "share")
        assert b"Maria da Silva" not in result.path.read_bytes()
