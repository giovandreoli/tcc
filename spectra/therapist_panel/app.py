"""Streamlit therapist panel.

Every operation delegates to :class:`spectra.service.SpectraService`, which is unit
tested; this module is widgets and layout only, and is excluded from coverage.

Run it with::

    streamlit run spectra/therapist_panel/app.py
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from spectra.config import load_config
from spectra.gestures.profiles import PRESETS
from spectra.guided.shapes import SHAPE_ORDER, Difficulty
from spectra.i18n import set_locale, t
from spectra.service import PatientForm, SessionPlan, SpectraService
from spectra.storage.db import AccountLockedError, ConsentRequiredError, DuplicateCPFError
from spectra.storage.models import Hand, Sex
from spectra.storage.security import mask_cpf

PAGE_TITLE = "SPECTRA"


# --------------------------------------------------------------------- session
@st.cache_resource
def get_service() -> SpectraService:
    config = load_config()
    set_locale(config.locale)
    return SpectraService.open(config)


def _therapist_id() -> str | None:
    return st.session_state.get("therapist_id")


def _demo_mode() -> bool:
    return bool(st.session_state.get("demo_mode", False))


def _patient_label(patient) -> str:
    name = patient.display_name(_demo_mode())
    return f"{name} — {mask_cpf(patient.cpf)}"


# ----------------------------------------------------------------------- login
def render_login(service: SpectraService) -> None:
    st.title(PAGE_TITLE)
    st.caption(t("panel.login.subtitle"))

    if not service.db.list_patients() and st.session_state.get("first_run") is None:
        st.info(t("panel.login.first_run"))

    with st.form("login"):
        email = st.text_input(t("panel.login.email"))
        password = st.text_input(t("panel.login.password"), type="password")
        submitted = st.form_submit_button(t("panel.login.submit"))
    if submitted:
        try:
            state = service.db.authenticate_therapist(email, password)
        except AccountLockedError:
            st.error(t("panel.login.locked"))
            return
        if state.authenticated:
            therapist = service.db.get_therapist_by_email(email)
            st.session_state["therapist_id"] = therapist.id
            st.session_state["therapist_name"] = therapist.name
            st.rerun()
        else:
            st.error(t("panel.login.failed", remaining=state.remaining_attempts))

    with st.expander(t("panel.login.create_account")):
        with st.form("create_therapist"):
            name = st.text_input(t("panel.login.name"))
            new_email = st.text_input(t("panel.login.email"), key="new_email")
            new_password = st.text_input(t("panel.login.password"), type="password", key="new_pw")
            created = st.form_submit_button(t("panel.login.create"))
        if created:
            if len(new_password) < 8:
                st.error(t("panel.error.password_short"))
            else:
                service.db.create_therapist(name, new_email, new_password)
                st.success(t("panel.login.created"))


# -------------------------------------------------------------------- patients
def render_patients(service: SpectraService) -> None:
    st.header(t("panel.section.patients"))
    patients = service.db.list_patients()

    if patients:
        st.dataframe(
            [
                {
                    t("report.field.patient"): p.display_name(_demo_mode()),
                    t("report.field.cpf"): mask_cpf(p.cpf),
                    t("report.field.affected_hand"): p.affected_hand.value,
                    t("report.field.diagnosis"): p.diagnosis_cid,
                    t("panel.field.consent"): "✔" if service.has_consent(p.id) else "✖",
                }
                for p in patients
            ],
            use_container_width=True,
        )
    else:
        st.info(t("panel.patients.empty"))

    with st.expander(t("panel.patients.new"), expanded=not patients):
        with st.form("new_patient"):
            columns = st.columns(2)
            form = PatientForm(
                name=columns[0].text_input(t("report.field.patient")),
                cpf=columns[1].text_input(t("report.field.cpf")),
                birth_date=columns[0].date_input(
                    t("report.field.age"), value=None, min_value=dt.date(1900, 1, 1)
                ),
                sex=Sex(
                    columns[1].selectbox(
                        t("report.field.sex"),
                        [s.value for s in Sex],
                        format_func=lambda v: t(f"patient.sex.{v}"),
                    )
                ),
                phone=columns[0].text_input(t("panel.field.phone")),
                address=columns[1].text_input(t("panel.field.address")),
                dominant_hand=Hand(
                    columns[0].selectbox(t("report.field.dominant_hand"), [h.value for h in Hand])
                ),
                affected_hand=Hand(
                    columns[1].selectbox(t("report.field.affected_hand"), [h.value for h in Hand])
                ),
                diagnosis_cid=columns[0].text_input(t("report.field.diagnosis")),
                injury_date=columns[1].date_input(
                    t("panel.field.injury_date"), value=None, min_value=dt.date(1900, 1, 1)
                ),
                referring_professional=st.text_input(t("panel.field.referring")),
                contraindications=st.text_area(t("panel.field.contraindications")),
                notes=st.text_area(t("panel.field.notes")),
                pin=st.text_input(t("panel.field.pin"), max_chars=4),
            )
            submitted = st.form_submit_button(t("panel.patients.save"))
        if submitted:
            problems = form.validate()
            if problems:
                for key in problems:
                    st.error(t(key))
                return
            try:
                service.create_patient(form, actor_id=_therapist_id())
            except DuplicateCPFError:
                st.error(t("panel.error.cpf_duplicate"))
                return
            st.success(t("panel.patients.saved"))
            st.rerun()


# --------------------------------------------------------------------- consent
def render_consent(service: SpectraService) -> None:
    st.header(t("panel.section.consent"))
    patients = service.db.list_patients()
    if not patients:
        st.info(t("panel.patients.empty"))
        return

    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    document = service.current_consent_document()
    active = service.db.active_consent(patient.id)

    if active:
        st.success(t("panel.consent.active", version=active.version))
        if st.button(t("panel.consent.revoke")):
            service.db.revoke_consent(active.id, actor_id=_therapist_id())
            st.rerun()
    else:
        st.warning(t("panel.consent.missing"))

    with st.expander(t("panel.consent.read"), expanded=not active):
        st.markdown(document.text)
    st.caption(t("panel.consent.hash", hash=document.document_hash[:16]))

    agreed = st.checkbox(t("panel.consent.agree"))
    signer = st.text_input(t("panel.consent.signed_by"), value=patient.name)
    if st.button(t("panel.consent.record"), disabled=not agreed):
        service.record_consent(patient.id, signer, actor_id=_therapist_id())
        st.success(t("panel.consent.recorded"))
        st.rerun()


# --------------------------------------------------------------------- session
def render_session(service: SpectraService) -> None:
    st.header(t("panel.section.session"))
    patients = service.db.list_patients()
    if not patients:
        st.info(t("panel.patients.empty"))
        return

    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    columns = st.columns(3)
    plan = SessionPlan(
        patient_id=patient.id,
        therapist_id=_therapist_id(),
        hand=Hand(
            columns[0].selectbox(
                t("report.field.hand"),
                [h.value for h in Hand],
                index=1 if patient.affected_hand is Hand.RIGHT else 0,
            )
        ),
        gesture_profile_name=columns[1].selectbox(
            t("report.field.gesture_profile"),
            list(PRESETS),
            format_func=lambda name: t(f"gesture.profile.{name}"),
        ),
        difficulty=columns[2].selectbox(
            t("panel.field.difficulty"),
            [d.value for d in Difficulty],
            format_func=lambda v: t(f"guided.difficulty.{v}"),
        ),
        duration_minutes=st.slider(t("panel.field.duration"), 5, 40, 20),
        exercises=tuple(
            st.multiselect(
                t("panel.field.exercises"),
                ["open_close", "finger_touch", "finger_wave", *SHAPE_ORDER],
                default=["open_close", "finger_touch", "finger_wave"],
            )
        ),
        demo_mode=_demo_mode(),
    )

    warnings = plan.profile_warnings()
    if warnings:
        st.warning(t("panel.session.profile_warnings"))
        for issue in warnings:
            actions = ", ".join(t(f"gesture.action.{a.value}") for a in issue.actions)
            st.caption(f"• {t(issue.message_key, actions=actions, detail=issue.detail)}")

    calibration = service.calibration_for(patient.id)
    if calibration.is_calibrated:
        st.info(t("panel.session.calibrated", samples=calibration.samples_used))
        if calibration.uncalibrated:
            st.caption(
                t("gesture.calibration.fallback", fingers=", ".join(calibration.uncalibrated))
            )
    else:
        st.caption(t("panel.session.not_calibrated"))

    if st.button(t("panel.session.start"), type="primary"):
        try:
            session = service.start_session(plan)
        except ConsentRequiredError:
            st.error(t("panel.session.consent_required"))
            return
        st.session_state["active_session"] = session.id
        st.success(t("panel.session.started", command=f"python -m spectra --session {session.id}"))


# --------------------------------------------------------------------- history
def render_history(service: SpectraService) -> None:
    st.header(t("panel.section.history"))
    patients = service.db.list_patients()
    if not patients:
        st.info(t("panel.patients.empty"))
        return

    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    sessions = service.db.list_sessions(patient.id)
    if not sessions:
        st.info(t("panel.history.empty"))
        return

    st.dataframe(
        [
            {
                t("report.field.date"): dt.datetime.fromtimestamp(s.started_at).strftime(
                    "%d/%m/%Y %H:%M"
                ),
                t("report.field.duration"): f"{s.duration / 60:.0f} min",
                t("report.field.hand"): s.hand.value,
                t("panel.field.quality"): "✔" if s.tracking_quality.get("reliable") else "⚠",
            }
            for s in sessions
        ],
        use_container_width=True,
    )

    session = st.selectbox(
        t("panel.history.session"),
        sessions,
        format_func=lambda s: dt.datetime.fromtimestamp(s.started_at).strftime("%d/%m/%Y %H:%M"),
    )
    report, generated = None, None
    if st.button(t("panel.history.generate")):
        report, generated = service.generate_reports(
            session.id, demo_mode=_demo_mode(), actor_id=_therapist_id()
        )
        st.session_state["last_report"] = (
            report.session_id,
            str(generated.pdf_path),
            str(generated.txt_path),
        )

    stored = st.session_state.get("last_report")
    if stored and stored[0] == session.id:
        _, pdf_path, txt_path = stored
        columns = st.columns(2)
        with open(pdf_path, "rb") as handle:
            columns[0].download_button(
                t("panel.history.download_pdf"), handle.read(), file_name=pdf_path.split("/")[-1]
            )
        with open(txt_path, "rb") as handle:
            columns[1].download_button(
                t("panel.history.download_txt"), handle.read(), file_name=txt_path.split("/")[-1]
            )

        if service.config.share_dir and st.button(t("panel.history.export_share")):
            from pathlib import Path

            from spectra.reports import GeneratedReport

            service.export_reports_to_share(
                GeneratedReport(Path(pdf_path), Path(txt_path)),
                patient.id,
                actor_id=_therapist_id(),
            )
            st.success(t("panel.history.exported"))

    # --- evolution charts ------------------------------------------------
    if len(sessions) > 1 and (report := _safe_report(service, sessions[-1].id)):
        for key, points in report.trends.items():
            st.subheader(t(key))
            st.line_chart({t(key): [point.value for point in points]})


def _safe_report(service: SpectraService, session_id: str):
    try:
        from spectra.reports.builder import build_session_report

        return build_session_report(service.db, session_id, demo_mode=_demo_mode())
    except ValueError:
        return None


# ------------------------------------------------------------------ privacy
def render_privacy(service: SpectraService) -> None:
    st.header(t("panel.section.privacy"))
    patients = service.db.list_patients()
    if patients:
        patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
        columns = st.columns(3)
        if columns[0].button(t("panel.privacy.export")):
            path = service.db.export_patient_file(
                patient.id, service.config.exports_dir, actor_id=_therapist_id()
            )
            st.success(t("panel.privacy.exported", path=path.name))
        if columns[1].button(t("panel.privacy.anonymize")):
            service.db.anonymize_patient(patient.id, actor_id=_therapist_id())
            st.success(t("panel.privacy.anonymized"))
            st.rerun()
        if columns[2].button(t("panel.privacy.delete")):
            if st.session_state.get("confirm_delete") == patient.id:
                service.db.delete_patient(patient.id, actor_id=_therapist_id())
                st.session_state.pop("confirm_delete", None)
                st.rerun()
            else:
                st.session_state["confirm_delete"] = patient.id
                st.warning(t("panel.privacy.confirm_delete"))

    st.divider()
    st.subheader(t("panel.privacy.backup"))
    st.caption(t("panel.privacy.backup_hint"))
    password = st.text_input(t("panel.privacy.backup_password"), type="password")
    if st.button(t("panel.privacy.backup_run"), disabled=not password):
        result = service.backup(password, actor_id=_therapist_id())
        st.success(t("panel.privacy.backup_done", path=result.path.name))

    st.divider()
    st.subheader(t("panel.privacy.audit"))
    entries = service.db.audit.entries(limit=50)
    st.dataframe(
        [
            {
                t("report.field.date"): dt.datetime.fromtimestamp(e.at).strftime("%d/%m/%Y %H:%M"),
                t("panel.field.action"): t(f"audit.action.{e.action.value}"),
                t("panel.field.subject"): (e.subject_id or "—")[:8],
            }
            for e in entries
        ],
        use_container_width=True,
    )


# ------------------------------------------------------------------------ main
def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🖐", layout="wide")
    service = get_service()

    if not _therapist_id():
        render_login(service)
        return

    with st.sidebar:
        st.title(PAGE_TITLE)
        st.caption(st.session_state.get("therapist_name", ""))
        st.toggle(t("panel.demo_mode"), key="demo_mode", help=t("panel.demo_mode_help"))
        section = st.radio(
            t("panel.navigation"),
            ("patients", "consent", "session", "history", "privacy"),
            format_func=lambda key: t(f"panel.section.{key}"),
        )
        if st.button(t("panel.logout")):
            st.session_state.clear()
            st.rerun()
        st.caption(t("panel.data_dir", path=str(service.config.data_dir)))

    {
        "patients": render_patients,
        "consent": render_consent,
        "session": render_session,
        "history": render_history,
        "privacy": render_privacy,
    }[section](service)


if __name__ == "__main__":
    main()
