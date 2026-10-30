"""Streamlit therapist panel.

Every operation delegates to :class:`spectra.service.SpectraService`, which is unit
tested; this module is widgets and layout only, and is excluded from coverage.

Run it with::

    streamlit run spectra/therapist_panel/app.py
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from spectra.browser.rooms import RoomError, Rooms
from spectra.config import load_config
from spectra.exercises.catalog import (
    BUILTIN_KEYS,
    DIRECTIONS,
    FINGERS,
    MAX_HOLD_SECONDS,
    MAX_REPS,
    MIN_REPS,
    MOVEMENT_PARAMS,
    TOLERANCES,
    ExerciseSpec,
    Movement,
)
from spectra.gestures.profiles import PRESETS
from spectra.guided.shapes import SHAPE_ORDER, Difficulty
from spectra.i18n import set_locale, t
from spectra.remote import (
    Tunnel,
    TunnelError,
    download_cloudflared,
    find_cloudflared,
    is_local_request,
    patient_link,
)
from spectra.service import (
    DOCUMENT_CATEGORIES,
    DocumentForm,
    PatientForm,
    SessionPlan,
    SpectraService,
)
from spectra.storage.db import AccountLockedError, ConsentRequiredError, DuplicateCPFError
from spectra.storage.models import Hand, Sex
from spectra.storage.security import mask_cpf
from spectra.webui import art
from spectra.webui import components as ui
from spectra.webui.insights import overview, pain_series

PAGE_TITLE = "SPECTRA"
SECTIONS = (
    "home",
    "patients",
    "consent",
    "exercises",
    "documents",
    "session",
    "learning",
    "history",
    "privacy",
)
ROOM_TONES = {
    "waiting": "waiting",
    "active": "active",
    "completed": "completed",
    "interrupted": "expired",
    "expired": "expired",
}


# --------------------------------------------------------------------- session
@st.cache_resource
def get_service() -> SpectraService:
    config = load_config()
    set_locale(config.locale)
    return SpectraService.open(config)


@st.cache_resource
def get_tunnel_holder() -> dict[str, Tunnel | None]:
    """One tunnel per panel process, shared by every browser tab of the therapist."""
    return {"tunnel": None}


def _active_tunnel() -> Tunnel | None:
    tunnel = get_tunnel_holder()["tunnel"]
    return tunnel if tunnel is not None and tunnel.running and tunnel.url else None


def _request_is_local() -> bool:
    try:
        return is_local_request(dict(st.context.headers))
    except Exception:
        # Without request headers we cannot prove the request is local: fail closed.
        return False


def _therapist_id() -> str | None:
    return st.session_state.get("therapist_id")


def _demo_mode() -> bool:
    return bool(st.session_state.get("demo_mode", False))


def _patient_label(patient) -> str:
    name = patient.display_name(_demo_mode())
    return f"{name} — {mask_cpf(patient.cpf)}"


_md = ui.escape_markdown


# ----------------------------------------------------------------------- login
def render_login(service: SpectraService) -> None:
    ui.hero(PAGE_TITLE, t("panel.login.subtitle"))
    if not service.db.list_patients() and st.session_state.get("first_run") is None:
        ui.guide("hello", t("panel.login.first_run"), t("ux.mascot.name"))
    else:
        ui.guide("hello", t("ux.login.welcome_back"), t("ux.mascot.name"))

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


# ------------------------------------------------------------------------ home
def render_home(service: SpectraService) -> None:
    name = st.session_state.get("therapist_name", "")
    ui.hero(t("ux.home.greeting", name=name.split(" ")[0] if name else ""), t("ux.home.subtitle"))
    patients = service.db.list_patients()
    without_consent = [p for p in patients if not service.has_consent(p.id)]
    sessions = [s for p in patients for s in service.db.list_sessions(p.id)]
    summary = overview(len(patients), len(without_consent), sessions, dt.datetime.now().timestamp())
    relief = summary.pain_relief_week
    ui.stats(
        [
            (t("ux.home.patients"), str(summary.patients), t("ux.home.patients_hint")),
            (
                t("ux.home.sessions_week"),
                str(summary.sessions_week),
                t("ux.home.minutes_week", minutes=summary.minutes_week),
            ),
            (
                t("ux.home.pain_relief"),
                "—" if relief is None else f"{relief:+.1f}",
                t("ux.home.pain_relief_hint"),
            ),
            (t("ux.home.pending_consent"), str(summary.without_consent), t("ux.home.consent_hint")),
        ]
    )

    if not patients:
        ui.guide("hello", t("ux.home.tip_first_patient"), t("ux.mascot.name"))
    elif without_consent:
        ui.guide("think", t("ux.home.tip_consent", count=len(without_consent)), t("ux.mascot.name"))
    elif summary.sessions_week == 0:
        ui.guide("calm", t("ux.home.tip_new_session"), t("ux.mascot.name"))
    else:
        ui.guide("cheer", t("ux.home.tip_good_week"), t("ux.mascot.name"))

    st.subheader(t("ux.home.monitor"))
    render_room_monitor(service, {p.id: p for p in patients})


@st.fragment(run_every=3)
def render_room_monitor(service: SpectraService, patients: dict) -> None:
    from spectra.browser.live import live_status

    rooms = Rooms(service).overview()
    if not rooms:
        ui.guide("sleep", t("ux.home.no_rooms"))
        return
    for room in rooms:
        patient = patients.get(room.patient_id)
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.markdown(f"**{_md(_patient_label(patient)) if patient else '—'}**")
            left.caption(
                t(
                    "ux.home.room_created",
                    time=dt.datetime.fromtimestamp(room.created_at).strftime("%d/%m %H:%M"),
                )
            )
            right.html(art.chip_html(t(f"ux.room.{room.status}"), ROOM_TONES[room.status]))
            status = live_status(room.session_id) if room.status == "active" else None
            if status is not None and status.frames:
                right.caption(
                    t(
                        "ux.home.live",
                        exercise=_md(status.exercise_title),
                        reps=status.reps,
                        total=status.total_reps,
                        hand=t(
                            "ux.home.hand_ok" if status.hand_visible else "ux.home.hand_missing"
                        ),
                    )
                )
                right.progress(status.time_fraction)


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
        ui.guide("sleep", t("ux.empty.patients"))

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
        ui.guide("sleep", t("ux.empty.patients"))
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


# ------------------------------------------------------------------- exercises
def render_exercises(service: SpectraService) -> None:
    st.header(t("panel.section.exercises"))
    patients = service.db.list_patients()
    if not patients:
        ui.guide("sleep", t("ux.empty.patients"))
        return
    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    ui.guide("think", t("exercise.panel.intro"), t("ux.mascot.name"))
    flash = st.session_state.pop("exercise_flash", None)
    if flash:
        st.success(t("exercise.panel.saved", title=flash))

    custom = service.custom_exercises(patient.id)
    st.subheader(t("exercise.panel.list"))
    if not custom:
        st.caption(t("exercise.panel.empty"))
    for spec in custom:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{_md(spec.display_title())}** · {_md(spec.movement.title)}")
            left.caption(_md(spec.summary()))
            for line in spec.display_instructions():
                left.caption(_md(line))
            if right.button(t("exercise.panel.archive"), key=f"archive_{spec.key}"):
                service.archive_custom_exercise(patient.id, spec.key, actor_id=_therapist_id())
                st.rerun()

    with st.expander(t("exercise.panel.new"), expanded=not custom):
        _render_exercise_form(service, patient)

    st.subheader(t("exercise.panel.test"))
    st.caption(t("exercise.panel.test_hint"))
    options = service.available_exercises(patient.id)
    chosen = st.selectbox(
        t("exercise.panel.test_choose"),
        options,
        format_func=lambda spec: spec.display_title(),
        key=f"practice_choice_{patient.id}",
    )
    _render_practice(chosen, patient.affected_hand.value, service.calibration_for(patient.id))


def _render_exercise_form(service: SpectraService, patient) -> None:
    movement = st.selectbox(
        t("exercise.field.movement"),
        list(Movement),
        format_func=lambda item: item.title,
        key="new_exercise_movement",
    )
    st.caption(movement.description)
    accepted = MOVEMENT_PARAMS[movement]
    with st.form("new_exercise", clear_on_submit=True):
        title = st.text_input(t("exercise.field.title"), max_chars=60)
        reps = st.slider(t("exercise.field.reps"), MIN_REPS, MAX_REPS, 8)
        values: dict = {}
        if "finger" in accepted:
            values["finger"] = st.selectbox(
                t("exercise.field.finger"), FINGERS, format_func=lambda key: t(f"finger.{key}")
            )
        if "tolerance" in accepted:
            values["tolerance"] = st.select_slider(
                t("exercise.field.tolerance"),
                TOLERANCES,
                value="normal",
                format_func=lambda key: t(f"exercise.tolerance.{key}"),
                help=t("exercise.field.tolerance_help"),
            )
        if "direction" in accepted:
            values["direction"] = st.radio(
                t("exercise.field.direction"),
                DIRECTIONS,
                format_func=lambda key: t(f"exercise.direction.{key}"),
                horizontal=True,
            )
        if "isolated" in accepted:
            values["isolated"] = st.checkbox(
                t("exercise.field.isolated"), help=t("exercise.field.isolated_help")
            )
        if "hold_seconds" in accepted:
            values["hold_seconds"] = st.slider(
                t("exercise.field.hold"), 0.0, MAX_HOLD_SECONDS, 0.0, step=0.5, format="%.1f s"
            )
        instructions = st.text_area(
            t("exercise.field.instructions"), help=t("exercise.field.instructions_help")
        )
        submitted = st.form_submit_button(t("exercise.panel.save"), type="primary")
    if submitted:
        spec = ExerciseSpec(
            key="",
            movement=movement,
            title=title,
            instructions=tuple(line for line in instructions.splitlines() if line.strip()),
            target_reps=reps,
            **values,
        )
        try:
            saved = service.save_custom_exercise(patient.id, spec, actor_id=_therapist_id())
        except ValueError as exc:
            _show_problems(exc)
            return
        st.session_state["exercise_flash"] = saved.display_title()
        st.rerun()


def _show_problems(exc: ValueError) -> None:
    problems = exc.args[0] if exc.args and isinstance(exc.args[0], list) else []
    for key in problems:
        st.error(t(key))


def _render_practice(spec: ExerciseSpec, hand_label: str, calibration) -> None:
    from spectra.browser.session import LandmarkSession
    from spectra.browser.video import reset_camera

    state_key = f"practice_{spec.key}_{hand_label}"
    session = st.session_state.get(state_key)
    if session is None or session.finished:
        session = LandmarkSession((spec,), hand_label=hand_label, calibration=calibration)
        st.session_state[state_key] = session
    camera, panel = st.columns([2, 1], gap="large")
    with camera:
        _practice_camera(session, state_key)
    with panel:
        _practice_status(session)
        if st.button(t("exercise.panel.test_reset"), key=f"{state_key}_reset"):
            st.session_state.pop(state_key, None)
            reset_camera(f"{state_key}_camera")
            st.rerun()


@st.fragment
def _practice_camera(session, state_key: str) -> None:
    from spectra.browser.video import camera_streamer

    camera_streamer(session, f"{state_key}_camera")


@st.fragment(run_every=1)
def _practice_status(session) -> None:
    status = session.status()
    st.metric(t("ux.live.reps"), f"{status.reps}/{status.target}")
    st.progress(min(1.0, status.reps / status.target) if status.target else 0.0)
    if status.frames:
        st.html(art.bubble_html("calm", status.cue or t("exercise.cue.show_hand"), size=40))
        if status.hold_fraction > 0:
            st.progress(status.hold_fraction, text=t("ux.live.hold"))
        st.caption(t("exercise.panel.test_frames", frames=status.frames))
    else:
        st.caption(t("browser.waiting_camera"))


# ------------------------------------------------------------------- documents
def render_documents(service: SpectraService) -> None:
    st.header(t("panel.section.documents"))
    patients = service.db.list_patients()
    if not patients:
        ui.guide("sleep", t("ux.empty.patients"))
        return
    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    st.caption(t("document.privacy"))

    write, upload = st.tabs([t("document.tab.write"), t("document.tab.upload")])
    with write, st.form("write_report", clear_on_submit=True):
        category = st.selectbox(
            t("document.field.category"),
            DOCUMENT_CATEGORIES,
            format_func=lambda key: t(f"document.category.{key}"),
        )
        title = st.text_input(t("document.field.title"), value=t("document.default_title"))
        issued = st.date_input(t("document.field.issued_on"), value=dt.date.today())
        body = st.text_area(t("document.field.body"), height=260, help=t("document.body_help"))
        if st.form_submit_button(t("document.save"), type="primary"):
            _save_document(
                service,
                patient.id,
                DocumentForm(category=category, title=title, issued_on=issued, body=body),
            )
    with upload, st.form("upload_report", clear_on_submit=True):
        category = st.selectbox(
            t("document.field.category"),
            DOCUMENT_CATEGORIES,
            index=1,
            format_func=lambda key: t(f"document.category.{key}"),
        )
        title = st.text_input(t("document.field.title"))
        issued = st.date_input(t("document.field.issued_on"), value=None)
        attached = st.file_uploader(
            t("document.field.file"),
            type=["pdf", "png", "jpg", "jpeg"],
            help=t("document.file_help"),
        )
        notes = st.text_area(t("document.field.notes"))
        if st.form_submit_button(t("document.attach"), type="primary"):
            _save_document(
                service,
                patient.id,
                DocumentForm(
                    category=category,
                    title=title,
                    issued_on=issued,
                    body=notes,
                    filename=attached.name if attached else "",
                    content=attached.getvalue() if attached else None,
                ),
            )

    st.subheader(t("document.list"))
    documents = service.db.list_documents(patient.id)
    if not documents:
        ui.guide("sleep", t("document.empty"))
    for document in documents:
        _render_document(service, document)


def _save_document(service: SpectraService, patient_id: str, form: DocumentForm) -> None:
    try:
        service.add_document(patient_id, form, actor_id=_therapist_id())
    except ValueError as exc:
        _show_problems(exc)
        return
    st.success(t("document.saved"))


def _render_document(service: SpectraService, document) -> None:
    issued = (
        document.issued_on.strftime("%d/%m/%Y")
        if document.issued_on
        else dt.datetime.fromtimestamp(document.created_at).strftime("%d/%m/%Y")
    )
    category = t(f"document.category.{document.category}")
    parts = [document.title] + ([category] if category != document.title else []) + [issued]
    with st.expander(" · ".join(parts)):
        if document.has_file:
            st.caption(t("document.file_info", name=document.filename, kb=document.size // 1024))
        if document.body:
            st.text(document.body)
        prepared = st.session_state.get("prepared_document")
        columns = st.columns(2)
        if prepared and prepared[0] == document.id:
            _, data, name, media = prepared
            columns[0].download_button(t("document.download"), data, file_name=name, mime=media)
        elif columns[0].button(t("document.prepare"), key=f"prepare_{document.id}"):
            # Decrypting is an audited access, so it happens on request, not on every render.
            if document.has_file:
                data = service.db.read_document_file(document.id, actor_id=_therapist_id())
                name, media = document.filename or f"{document.id}.bin", document.media_type
            else:
                data = service.written_report_pdf(document.id, actor_id=_therapist_id())
                name, media = f"laudo_{document.id[:8]}.pdf", "application/pdf"
            st.session_state["prepared_document"] = (document.id, data, name, media)
            st.rerun()
        if columns[1].button(t("document.delete"), key=f"delete_{document.id}"):
            if st.session_state.get("confirm_document") == document.id:
                service.db.delete_document(document.id, actor_id=_therapist_id())
                st.session_state.pop("confirm_document", None)
                st.session_state.pop("prepared_document", None)
                st.rerun()
            else:
                st.session_state["confirm_document"] = document.id
                st.warning(t("document.confirm_delete"))


# --------------------------------------------------------------------- session
def render_session(service: SpectraService) -> None:
    st.header(t("panel.section.session"))
    patients = service.db.list_patients()
    if not patients:
        ui.guide("sleep", t("ux.empty.patients"))
        return

    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    browser_mode = (
        st.radio(
            t("browser.delivery"),
            ("browser", "desktop"),
            format_func=lambda value: t(f"browser.delivery_{value}"),
            horizontal=True,
        )
        == "browser"
    )
    if browser_mode:
        render_remote_access(service)
    available = service.available_exercises(patient.id)
    titles = {spec.key: spec.display_title() for spec in available}
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
                list(titles) if browser_mode else [*BUILTIN_KEYS, *SHAPE_ORDER],
                default=list(BUILTIN_KEYS),
                format_func=lambda key: titles.get(key, key),
                key=f"exercises_{browser_mode}_{patient.id}",
                help=t("panel.session.exercises_help"),
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
            if browser_mode:
                code = Rooms(service).create(plan)
                st.session_state["browser_room_code"] = code
                st.session_state["browser_room_patient"] = patient.id
            else:
                session = service.start_session(plan)
                st.session_state["active_session"] = session.id
                st.success(
                    t("panel.session.started", command=f"python -m spectra --session {session.id}")
                )
        except ConsentRequiredError:
            st.error(t("panel.session.consent_required"))
            return
        except RoomError as exc:
            st.error(t(str(exc)))
            return
    if (
        browser_mode
        and st.session_state.get("browser_room_code")
        and st.session_state.get("browser_room_patient") == patient.id
    ):
        code = st.session_state["browser_room_code"]
        ui.guide("cheer", t("browser.created"), t("ux.mascot.name"))
        st.code(code, language=None)
        st.link_button(t("browser.open_patient"), f"?paciente=1&sala={code}", type="primary")
        tunnel = _active_tunnel()
        if tunnel is not None and tunnel.url:
            st.markdown(f"**{t('remote.patient_link')}**")
            st.code(patient_link(tunnel.url, code), language=None)
            st.caption(t("remote.patient_link_hint"))
        st.caption(t("browser.code_hint"))
        st.caption(t("ux.home.monitor_hint"))


def render_remote_access(service: SpectraService) -> None:
    """Turn the free Cloudflare tunnel on or off for patients at home."""
    holder = get_tunnel_holder()
    tunnel = _active_tunnel()
    with st.container(border=True):
        st.markdown(f"**{t('remote.title')}**")
        if tunnel is not None:
            st.success(t("remote.active", url=tunnel.url))
            st.caption(t("remote.active_hint"))
            if st.button(t("remote.stop")):
                tunnel.stop()
                holder["tunnel"] = None
                st.rerun()
            return
        st.caption(t("remote.explain"))
        binary = find_cloudflared(service.config.tools_dir)
        if binary is None:
            st.caption(t("remote.missing"))
            if st.button(t("remote.download")):
                with st.spinner(t("remote.downloading")):
                    try:
                        download_cloudflared(service.config.tools_dir)
                    except TunnelError as exc:
                        st.error(t(str(exc)))
                        return
                st.rerun()
            return
        if st.button(t("remote.start"), type="primary"):
            port = int(st.get_option("server.port") or 8501)
            candidate = Tunnel(binary, port)
            with st.spinner(t("remote.starting")):
                try:
                    candidate.start()
                except TunnelError as exc:
                    st.error(t(str(exc)))
                    return
            holder["tunnel"] = candidate
            st.rerun()


# --------------------------------------------------------------------- history
def render_history(service: SpectraService) -> None:
    st.header(t("panel.section.history"))
    patients = service.db.list_patients()
    if not patients:
        ui.guide("sleep", t("ux.empty.patients"))
        return

    patient = st.selectbox(t("report.field.patient"), patients, format_func=_patient_label)
    sessions = service.db.list_sessions(patient.id)
    if not sessions:
        ui.guide("sleep", t("ux.empty.history"))
        return

    def pain(value: int | None) -> str:
        return "—" if value is None else str(value)

    st.dataframe(
        [
            {
                t("report.field.date"): dt.datetime.fromtimestamp(s.started_at).strftime(
                    "%d/%m/%Y %H:%M"
                ),
                t("report.field.duration"): f"{s.duration / 60:.0f} min",
                t("report.field.hand"): s.hand.value,
                t("panel.field.quality"): "✔" if s.tracking_quality.get("reliable") else "⚠",
                t("ux.history.pain_before"): pain(s.pain_before),
                t("ux.history.pain_after"): pain(s.pain_after),
            }
            for s in sessions
        ],
        use_container_width=True,
    )
    pain_rows = pain_series(sessions)
    if pain_rows:
        import pandas as pd

        st.subheader(t("ux.history.pain_chart"))
        before, after = t("ux.history.pain_before"), t("ux.history.pain_after")
        frame = pd.DataFrame(pain_rows).rename(columns={"before": before, "after": after})
        st.line_chart(frame, x="date", y=[before, after])
        st.caption(t("ux.history.pain_chart_hint"))

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
    st.set_page_config(page_title=PAGE_TITLE, page_icon="👻", layout="wide")
    service = get_service()
    ui.apply_theme()

    # Requests arriving through the internet tunnel only ever see the patient page.
    if not _request_is_local():
        from spectra.browser.page import render_patient

        render_patient(service)
        return

    ui.splash_once()

    if st.query_params.get("paciente") == "1" or "sala" in st.query_params:
        from spectra.browser.page import render_patient

        render_patient(service)
        return

    if not _therapist_id():
        render_login(service)
        return

    with st.sidebar:
        st.html(
            '<div class="sp-guide">'
            f'<div class="sp-float">{art.ghost_img("calm", 54)}</div>'
            f'<div><div class="sp-hero-title" style="font-size:1.6rem">{PAGE_TITLE}</div>'
            "</div></div>"
        )
        st.caption(st.session_state.get("therapist_name", ""))
        section = st.radio(
            t("panel.navigation"),
            SECTIONS,
            format_func=lambda key: t(f"ux.nav.{key}"),
        )
        st.divider()
        st.toggle(t("panel.demo_mode"), key="demo_mode", help=t("panel.demo_mode_help"))
        with st.expander(t("ux.a11y.title")):
            ui.accessibility_controls()
        if st.button(t("panel.logout"), use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.caption(t("panel.data_dir", path=str(service.config.data_dir)))

    {
        "home": render_home,
        "patients": render_patients,
        "consent": render_consent,
        "exercises": render_exercises,
        "documents": render_documents,
        "session": render_session,
        "learning": render_learning,
        "history": render_history,
        "privacy": render_privacy,
    }[section](service)


def render_learning(service: SpectraService) -> None:
    from spectra.therapist_panel.learning import render_learning as render

    render(service)


if __name__ == "__main__":
    main()
