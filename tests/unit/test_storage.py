from __future__ import annotations

import hashlib
import json
import threading
from datetime import date

import pytest

from spectra.storage.backup import BackupError, create_backup, restore_backup
from spectra.storage.db import (
    AccountLockedError,
    ConsentRequiredError,
    Database,
    DuplicateCPFError,
    StorageError,
)
from spectra.storage.models import (
    AuditAction,
    Consent,
    ExerciseRun,
    Hand,
    MetricSummary,
    Patient,
    Session,
    Sex,
)
from spectra.storage.security import MAX_FAILED_ATTEMPTS, Cipher, KeyStore

VALID_CPF = "529.982.247-25"
VALID_CPF_2 = "168.995.350-09"


@pytest.fixture
def db(tmp_path) -> Database:
    cipher = Cipher.from_key_store(KeyStore(tmp_path / "secrets" / "master.key"))
    database = Database(tmp_path / "spectra.db", cipher)
    yield database
    database.close()


def make_patient(**overrides) -> Patient:
    defaults = {
        "name": "Maria da Silva",
        "cpf": VALID_CPF,
        "birth_date": date(1990, 5, 20),
        "sex": Sex.FEMALE,
        "phone": "11 99999-0000",
        "address": "Rua das Flores, 10",
        "dominant_hand": Hand.RIGHT,
        "affected_hand": Hand.LEFT,
        "diagnosis_cid": "S62.5",
        "contraindications": "Evitar carga axial",
        "notes": "Pós-operatório de 6 semanas.",
    }
    defaults.update(overrides)
    return Patient(**defaults)


def with_consent(db: Database, patient: Patient) -> Consent:
    text = "TCLE v1 - texto completo"
    return db.record_consent(
        Consent(
            patient_id=patient.id,
            version="1.0",
            document_hash=hashlib.sha256(text.encode()).hexdigest(),
            signed_by=patient.name,
        )
    )


class TestPatientCrud:
    def test_create_and_read_round_trip(self, db):
        created = db.create_patient(make_patient(), pin="1234")
        loaded = db.get_patient(created.id)
        assert loaded.name == "Maria da Silva"
        assert loaded.cpf == "52998224725"
        assert loaded.address == "Rua das Flores, 10"
        assert loaded.notes.startswith("Pós-operatório")
        assert loaded.affected_hand is Hand.LEFT
        assert loaded.birth_date == date(1990, 5, 20)

    def test_an_invalid_cpf_is_rejected(self, db):
        with pytest.raises(StorageError):
            db.create_patient(make_patient(cpf="111.111.111-11"))

    def test_a_duplicate_cpf_is_rejected(self, db):
        db.create_patient(make_patient())
        with pytest.raises(DuplicateCPFError):
            db.create_patient(make_patient(name="Outra Pessoa"))

    def test_lookup_by_cpf_works_without_storing_it_in_clear(self, db):
        created = db.create_patient(make_patient())
        assert db.find_patient_by_cpf(VALID_CPF).id == created.id

    def test_an_unknown_cpf_returns_none(self, db):
        assert db.find_patient_by_cpf(VALID_CPF_2) is None

    def test_update_persists_changes(self, db):
        patient = db.create_patient(make_patient())
        patient.notes = "Alta clínica"
        db.update_patient(patient)
        assert db.get_patient(patient.id).notes == "Alta clínica"

    def test_listing_is_ordered_by_name(self, db):
        db.create_patient(make_patient(name="Zuleica", cpf=VALID_CPF))
        db.create_patient(make_patient(name="Ana", cpf=VALID_CPF_2))
        assert [p.name for p in db.list_patients()] == ["Ana", "Zuleica"]

    def test_initials_are_used_as_a_pseudonym(self, db):
        patient = db.create_patient(make_patient())
        assert patient.initials() == "M.D.S."
        assert patient.display_name(demo_mode=True) == "M.D.S."
        assert patient.display_name(demo_mode=False) == "Maria da Silva"

    def test_age_is_computed_from_the_birth_date(self, db):
        patient = make_patient(birth_date=date(2000, 1, 1))
        assert patient.age_at(date(2024, 1, 1)) == 24
        assert patient.age_at(date(2023, 12, 31)) == 23


class TestEncryptionAtRest:
    def test_sensitive_columns_are_not_readable_in_the_file(self, db, tmp_path):
        db.create_patient(make_patient())
        db.close()
        raw = (tmp_path / "spectra.db").read_bytes()
        for secret in (b"52998224725", b"Rua das Flores", b"11 99999-0000", b"Evitar carga"):
            assert secret not in raw

    def test_the_name_is_stored_in_clear_by_design(self, db, tmp_path):
        """The therapist must be able to find a patient; the name is not encrypted."""
        db.create_patient(make_patient())
        db.close()
        assert b"Maria da Silva" in (tmp_path / "spectra.db").read_bytes()

    def test_a_pin_is_never_stored_in_clear(self, db, tmp_path):
        db.create_patient(make_patient(), pin="4321")
        db.close()
        assert b"4321" not in (tmp_path / "spectra.db").read_bytes()


class TestPinAndLockout:
    def test_a_correct_pin_authenticates(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        assert db.authenticate_patient(patient.id, "1234").authenticated

    def test_a_wrong_pin_is_rejected(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        assert not db.authenticate_patient(patient.id, "0000").authenticated

    def test_the_pin_must_be_four_digits(self, db):
        with pytest.raises(StorageError):
            db.create_patient(make_patient(), pin="12")

    def test_repeated_failures_lock_the_account(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        for _ in range(MAX_FAILED_ATTEMPTS):
            state = db.authenticate_patient(patient.id, "0000", now=1000.0)
        assert state.locked
        with pytest.raises(AccountLockedError):
            db.authenticate_patient(patient.id, "1234", now=1001.0)

    def test_the_lock_expires(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        for _ in range(MAX_FAILED_ATTEMPTS):
            db.authenticate_patient(patient.id, "0000", now=1000.0)
        assert db.authenticate_patient(patient.id, "1234", now=100_000.0).authenticated

    def test_a_success_clears_the_counter(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        db.authenticate_patient(patient.id, "0000")
        db.authenticate_patient(patient.id, "1234")
        assert db.get_patient(patient.id, audit=False).failed_attempts == 0

    def test_changing_the_pin_unlocks_the_account(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        for _ in range(MAX_FAILED_ATTEMPTS):
            db.authenticate_patient(patient.id, "0000", now=1000.0)
        db.set_patient_pin(patient.id, "5678")
        assert db.authenticate_patient(patient.id, "5678", now=1001.0).authenticated


class TestTherapistAuth:
    def test_login_round_trip(self, db):
        db.create_therapist("Dra. Ana", "ana@example.org", "uma-senha-longa")
        assert db.authenticate_therapist("ana@example.org", "uma-senha-longa").authenticated

    def test_the_email_is_case_insensitive(self, db):
        db.create_therapist("Dra. Ana", "Ana@Example.org", "uma-senha-longa")
        assert db.get_therapist_by_email("ana@example.org") is not None

    def test_an_unknown_email_does_not_authenticate(self, db):
        assert not db.authenticate_therapist("nobody@example.org", "x").authenticated

    def test_repeated_failures_lock_the_therapist(self, db):
        db.create_therapist("Dra. Ana", "ana@example.org", "uma-senha-longa")
        for _ in range(MAX_FAILED_ATTEMPTS):
            db.authenticate_therapist("ana@example.org", "errada", now=1000.0)
        with pytest.raises(AccountLockedError):
            db.authenticate_therapist("ana@example.org", "uma-senha-longa", now=1001.0)


class TestConsentGate:
    def test_a_session_without_consent_is_refused(self, db):
        patient = db.create_patient(make_patient())
        with pytest.raises(ConsentRequiredError):
            db.start_session(Session(patient_id=patient.id))

    def test_a_session_with_consent_is_allowed(self, db):
        patient = db.create_patient(make_patient())
        consent = with_consent(db, patient)
        session = db.start_session(Session(patient_id=patient.id))
        assert session.consent_id == consent.id

    def test_a_revoked_consent_blocks_new_sessions(self, db):
        patient = db.create_patient(make_patient())
        consent = with_consent(db, patient)
        db.revoke_consent(consent.id)
        with pytest.raises(ConsentRequiredError):
            db.start_session(Session(patient_id=patient.id))

    def test_the_consent_records_a_version_and_a_document_hash(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        active = db.active_consent(patient.id)
        assert active.version == "1.0"
        assert len(active.document_hash) == 64
        assert active.accepted_at > 0

    def test_the_most_recent_consent_wins(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        newest = db.record_consent(
            Consent(patient_id=patient.id, version="2.0", document_hash="a" * 64)
        )
        assert db.active_consent(patient.id).id == newest.id


class TestSessionData:
    def test_a_full_session_round_trips(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        session = db.start_session(
            Session(patient_id=patient.id, hand=Hand.LEFT, gesture_profile_name="simplified")
        )
        run = db.add_exercise_run(
            ExerciseRun(session_id=session.id, kind="physio", name="open_close", reps=8, target=8)
        )
        db.add_metric(
            MetricSummary(
                session_id=session.id, exercise_run_id=run.id, metric="rom", payload={"span": 70}
            )
        )
        db.add_samples(session.id, [{"at": 1.0, "flexion": [0, 1, 2, 3, 4]}])
        session.tracking_quality = {"reliable": True}
        session.notes = "Boa adesão"
        db.finish_session(session)

        loaded = db.get_session(session.id)
        assert loaded.hand is Hand.LEFT
        assert loaded.notes == "Boa adesão"
        assert loaded.tracking_quality == {"reliable": True}
        assert loaded.ended_at is not None
        assert len(db.list_exercise_runs(session.id)) == 1
        assert db.list_metrics(session.id)[0].payload["span"] == 70
        assert len(db.list_samples(session.id)) == 1

    def test_session_notes_are_encrypted(self, db, tmp_path):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        session = db.start_session(Session(patient_id=patient.id))
        session.notes = "anotação confidencial"
        db.finish_session(session)
        db.close()
        assert "anotação confidencial".encode() not in (tmp_path / "spectra.db").read_bytes()

    def test_sessions_are_listed_in_order(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        first = db.start_session(Session(patient_id=patient.id, started_at=10.0))
        second = db.start_session(Session(patient_id=patient.id, started_at=20.0))
        assert [s.id for s in db.list_sessions(patient.id)] == [first.id, second.id]


class TestDataSubjectRights:
    def test_export_contains_the_decrypted_record(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        session = db.start_session(Session(patient_id=patient.id))
        db.add_samples(session.id, [{"at": 1.0}])
        export = db.export_patient(patient.id)
        assert export["patient"]["cpf"] == "52998224725"
        assert export["patient"]["address"] == "Rua das Flores, 10"
        assert len(export["consents"]) == 1
        assert len(export["sessions"]) == 1
        assert len(export["sessions"][0]["samples"]) == 1

    def test_the_export_filename_uses_the_uuid_not_the_cpf(self, db, tmp_path):
        patient = db.create_patient(make_patient())
        path = db.export_patient_file(patient.id, tmp_path / "exports")
        assert patient.id in path.name
        assert "52998224725" not in path.name
        assert json.loads(path.read_text(encoding="utf-8"))["patient"]["id"] == patient.id

    def test_exporting_an_unknown_patient_fails(self, db):
        with pytest.raises(StorageError):
            db.export_patient("does-not-exist")

    def test_anonymisation_strips_identifiers_but_keeps_measurements(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        session = db.start_session(Session(patient_id=patient.id))
        db.add_samples(session.id, [{"at": 1.0}])

        db.anonymize_patient(patient.id)
        anonymous = db.get_patient(patient.id, audit=False)
        assert anonymous.is_anonymized
        assert anonymous.cpf == ""
        assert anonymous.address == ""
        assert anonymous.notes == ""
        assert anonymous.birth_date is None
        assert "Anônimo" in anonymous.name
        assert len(db.list_samples(session.id)) == 1

    def test_an_anonymised_cpf_can_be_registered_again(self, db):
        patient = db.create_patient(make_patient())
        db.anonymize_patient(patient.id)
        assert db.create_patient(make_patient(name="Outra")) is not None

    def test_deletion_cascades_to_every_child_row(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        session = db.start_session(Session(patient_id=patient.id))
        db.add_samples(session.id, [{"at": 1.0}])

        db.delete_patient(patient.id)
        assert db.get_patient(patient.id, audit=False) is None
        assert db.list_sessions(patient.id) == []
        assert db.list_samples(session.id) == []


class TestAuditTrail:
    def test_creation_access_export_and_deletion_are_logged(self, db):
        patient = db.create_patient(make_patient())
        db.get_patient(patient.id)
        db.export_patient(patient.id)
        db.delete_patient(patient.id)
        actions = [entry.action for entry in db.audit.entries(subject_id=patient.id)]
        assert AuditAction.PATIENT_CREATED in actions
        assert AuditAction.PATIENT_VIEWED in actions
        assert AuditAction.PATIENT_EXPORTED in actions
        assert AuditAction.PATIENT_DELETED in actions

    def test_the_audit_log_stores_no_personal_data(self, db):
        patient = db.create_patient(make_patient())
        db.export_patient(patient.id)
        for entry in db.audit.entries():
            assert "Maria" not in entry.detail
            assert "52998224725" not in (entry.detail or "")
            assert entry.subject_id in (patient.id, None)

    def test_failed_logins_are_logged(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        db.authenticate_patient(patient.id, "0000")
        assert db.audit.count([AuditAction.LOGIN_FAILURE]) == 1

    def test_a_lockout_is_logged(self, db):
        patient = db.create_patient(make_patient(), pin="1234")
        for _ in range(MAX_FAILED_ATTEMPTS):
            db.authenticate_patient(patient.id, "0000", now=1000.0)
        assert db.audit.count([AuditAction.ACCOUNT_LOCKED]) == 1

    def test_consent_and_session_events_are_logged(self, db):
        patient = db.create_patient(make_patient())
        with_consent(db, patient)
        db.start_session(Session(patient_id=patient.id))
        assert db.audit.count([AuditAction.CONSENT_RECORDED]) == 1
        assert db.audit.count([AuditAction.SESSION_STARTED]) == 1

    def test_entries_are_newest_first(self, db):
        patient = db.create_patient(make_patient())
        db.get_patient(patient.id)
        entries = db.audit.entries(subject_id=patient.id)
        assert entries[0].at >= entries[-1].at


class TestBackup:
    def test_round_trip(self, db, tmp_path):
        db.create_patient(make_patient())
        db.close()
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "relatorio.txt").write_text("conteúdo", encoding="utf-8")

        result = create_backup(
            tmp_path / "share",
            "senha-forte",
            db_path=tmp_path / "spectra.db",
            extra_dirs=(reports,),
        )
        assert result.path.exists()
        assert result.size_bytes > 0

        restored = restore_backup(result.path, "senha-forte", tmp_path / "restored")
        assert "manifest.json" in restored
        assert (tmp_path / "restored" / "db" / "spectra.db").exists()
        assert (tmp_path / "restored" / "reports" / "relatorio.txt").exists()

    def test_a_wrong_password_is_rejected(self, db, tmp_path):
        db.close()
        result = create_backup(tmp_path / "share", "certa", db_path=tmp_path / "spectra.db")
        with pytest.raises(BackupError):
            restore_backup(result.path, "errada", tmp_path / "restored")

    def test_the_backup_is_not_readable_in_clear(self, db, tmp_path):
        db.create_patient(make_patient())
        db.close()
        result = create_backup(tmp_path / "share", "senha", db_path=tmp_path / "spectra.db")
        assert b"Maria da Silva" not in result.path.read_bytes()

    def test_a_password_is_mandatory(self, tmp_path):
        with pytest.raises(BackupError):
            create_backup(tmp_path / "share", "")

    def test_a_foreign_file_is_rejected(self, tmp_path):
        bogus = tmp_path / "bogus.spectra"
        bogus.write_bytes(b"not a backup")
        with pytest.raises(BackupError):
            restore_backup(bogus, "senha", tmp_path / "restored")


class TestThreading:
    """Streamlit shares one cached service across script threads."""

    def test_the_database_is_usable_from_another_thread(self, db):
        created = db.create_patient(make_patient())
        results: list = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                results.append([p.id for p in db.list_patients()])
                db.audit.count()
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert errors == []
        assert results == [[created.id]]

    def test_writes_from_another_thread_are_visible_to_the_main_thread(self, db):
        thread = threading.Thread(target=lambda: db.create_patient(make_patient()))
        thread.start()
        thread.join()
        assert len(db.list_patients()) == 1
