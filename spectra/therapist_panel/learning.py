"""Panel section "Aprendizado": learning path, learned gestures, click calibration,
user tests and the reproducible algorithm evaluation.

Widgets only; the logic lives in :mod:`spectra.learning`, :mod:`spectra.usability`,
:mod:`spectra.evaluation` and :mod:`spectra.browser.sinks`, which are unit tested.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from spectra.browser.learning_ui import render_tutorial, reset_tutorial
from spectra.browser.sinks import GestureCollector, PinchCalibrationSink, TutorialSink
from spectra.browser.video import camera_streamer
from spectra.domains import Command
from spectra.i18n import t
from spectra.learning.classifier import TrainingError
from spectra.service import SpectraService
from spectra.usability import (
    SUS_ITEMS,
    compare_filters,
    counterbalanced_order,
    describe,
    flat_rows,
    participant_rows,
    results_csv,
    sus_adjective,
    sus_score,
)
from spectra.webui import art
from spectra.webui import components as ui

HANDS = ("Right", "Left")


def _actor() -> str | None:
    return st.session_state.get("therapist_id")


def _patient(service: SpectraService, key: str):
    from spectra.therapist_panel.app import _patient_label

    patients = service.db.list_patients()
    if not patients:
        ui.guide("sleep", t("ux.empty.patients"))
        return None
    return st.selectbox(t("report.field.patient"), patients, format_func=_patient_label, key=key)


def render_learning(service: SpectraService) -> None:
    st.header(t("panel.section.learning"))
    st.caption(t("learning.intro"))
    path, gestures, click, tests, evaluation = st.tabs(
        [
            t("learning.tab.path"),
            t("learning.tab.gestures"),
            t("learning.tab.click"),
            t("learning.tab.tests"),
            t("learning.tab.evaluation"),
        ]
    )
    with path:
        _tab_path(service)
    with gestures:
        _tab_gestures(service)
    with click:
        _tab_click(service)
    with tests:
        _tab_tests(service)
    with evaluation:
        _tab_evaluation()


# ------------------------------------------------------------------- path
def _tab_path(service: SpectraService) -> None:
    ui.guide("hello", t("learning.path.intro"), t("ux.mascot.name"))
    left, right = st.columns(2)
    hand = left.radio(
        t("report.field.hand"),
        HANDS,
        horizontal=True,
        format_func=lambda h: t(f"report.hand.{h.lower()}"),
        key="path_hand",
    )
    participants = [p["code"] for p in service.db.list_participants()]
    code = right.selectbox(
        t("learning.path.participant"),
        ["", *participants],
        format_func=lambda c: c or t("learning.path.no_participant"),
        key="path_participant",
    )
    state_key, camera_key = f"path_sink_{hand}", f"path_camera_{hand}"
    sink = st.session_state.get(state_key)
    if sink is None:
        sink = TutorialSink(hand_label=hand, finger_colors=service.config.finger_colors)
        st.session_state[state_key] = sink
    render_tutorial(sink, camera_key)
    columns = st.columns(2)
    if columns[0].button(t("learning.path.restart"), key="path_restart"):
        reset_tutorial(state_key, camera_key)
        st.rerun()
    status = sink.status()
    if code and status.results and columns[1].button(t("learning.path.save"), key="path_save"):
        service.db.add_usability_result(
            code, "tutorial", "", {"lessons": [r.to_dict() for r in status.results]}
        )
        st.success(t("learning.path.saved", code=code))


# --------------------------------------------------------------- gestures
def _tab_gestures(service: SpectraService) -> None:
    ui.guide("think", t("learning.ai.intro"), t("ux.mascot.name"))
    patient = _patient(service, "ai_patient")
    if patient is None:
        return
    state_key, camera_key = f"ai_collector_{patient.id}", f"ai_camera_{patient.id}"
    collector: GestureCollector | None = st.session_state.get(state_key)
    if collector is None:
        model, bindings = service.load_gesture_model(patient.id)
        collector = GestureCollector(model)
        st.session_state[state_key] = collector
        st.session_state[f"{state_key}_bindings"] = bindings
    bindings: dict[str, str] = st.session_state[f"{state_key}_bindings"]

    camera, panel = st.columns([2, 1], gap="large")
    with camera:
        _ai_camera(collector, camera_key)
    with panel:
        label = st.text_input(t("learning.ai.label"), max_chars=40, key="ai_label")
        if st.button(t("learning.ai.record"), disabled=not label.strip(), type="primary"):
            collector.record(label)
        _ai_status(collector)

    counts = collector.model.counts()
    if counts:
        removable = st.selectbox(t("learning.ai.remove_label"), sorted(counts), key="ai_remove")
        if st.button(t("learning.ai.remove")):
            collector.model.remove_label(removable)
            bindings.pop(removable, None)
            st.rerun()

    st.subheader(t("learning.ai.train_title"))
    st.caption(t("learning.ai.train_hint"))
    if st.button(t("learning.ai.train")):
        try:
            collector.model.fit()
            st.session_state[f"{state_key}_evaluation"] = collector.model.evaluate()
        except TrainingError as exc:
            st.error(t(str(exc)))
    evaluation = st.session_state.get(f"{state_key}_evaluation")
    if evaluation is not None and collector.model.trained:
        _show_evaluation(evaluation)
        st.subheader(t("learning.ai.bind_title"))
        st.caption(t("learning.ai.bind_hint"))
        commands = ["", *[c.value for c in Command]]
        for gesture in collector.model.labels:
            current = bindings.get(gesture, "")
            choice = st.selectbox(
                gesture,
                commands,
                index=commands.index(current) if current in commands else 0,
                format_func=lambda c: t(f"domain.command.{c}") if c else t("learning.ai.unbound"),
                key=f"ai_bind_{patient.id}_{gesture}",
            )
            bindings[gesture] = choice
        columns = st.columns(2)
        live = columns[0].toggle(t("learning.ai.live"), key=f"ai_live_{patient.id}")
        if live and not collector.predicting:
            collector.start_predicting()
        elif not live and collector.predicting:
            collector.predicting = False
        if columns[1].button(t("learning.ai.save"), type="primary"):
            service.save_gesture_model(patient.id, collector.model, bindings, actor_id=_actor())
            path = service.export_gesture_model(patient.id)
            st.success(t("learning.ai.saved", path=path.name))


@st.fragment
def _ai_camera(collector: GestureCollector, camera_key: str) -> None:
    camera_streamer(collector, camera_key)


@st.fragment(run_every=0.5)
def _ai_status(collector: GestureCollector) -> None:
    import time

    now = time.time()
    if collector.recording(now):
        if now < collector.record_from:
            st.info(t("learning.ai.get_ready", label=collector.label))
        else:
            st.warning(t("learning.ai.recording", label=collector.label))
    if collector.predicting:
        prediction = collector.prediction
        if prediction is None or prediction.label is None:
            st.metric(t("learning.ai.prediction"), t("learning.ai.unknown"))
        else:
            st.metric(
                t("learning.ai.prediction"),
                prediction.label,
                t("learning.ai.confidence", value=f"{prediction.confidence:.0%}"),
                delta_color="off",
            )
    counts = collector.model.counts()
    if counts:
        st.caption(t("learning.ai.examples"))
        for label, count in sorted(counts.items()):
            st.caption(f"• {label}: {count}")
    elif not collector.hand_visible:
        st.caption(t("browser.waiting_camera"))


def _show_evaluation(evaluation) -> None:
    import pandas as pd

    columns = st.columns(3)
    columns[0].metric(t("learning.ai.accuracy"), f"{evaluation.accuracy:.0%}")
    columns[1].metric(t("learning.ai.rejected"), str(evaluation.rejected))
    columns[2].metric(t("learning.ai.classes"), str(len(evaluation.labels)))
    st.caption(t("learning.ai.loo_hint"))
    frame = pd.DataFrame(evaluation.confusion).T.rename(columns={"?": t("learning.ai.unknown")})
    st.dataframe(frame, use_container_width=True)


# ------------------------------------------------------------------ click
def _tab_click(service: SpectraService) -> None:
    ui.guide("think", t("learning.click.intro"), t("ux.mascot.name"))
    patient = _patient(service, "click_patient")
    if patient is None:
        return
    hand = patient.affected_hand.value
    state_key, camera_key = f"click_sink_{patient.id}", f"click_camera_{patient.id}"
    sink = st.session_state.get(state_key)
    if sink is None:
        sink = PinchCalibrationSink(hand_label=hand)
        st.session_state[state_key] = sink
    camera, panel = st.columns([2, 1], gap="large")
    with camera:
        _click_camera(sink, camera_key)
    with panel:
        if st.button(t("learning.click.apart")):
            sink.capture("apart")
        if st.button(t("learning.click.pinch")):
            sink.capture("pinch")
        _click_status(sink)
    current = service.calibration_for(patient.id).pinch
    if current is not None:
        st.caption(t("learning.click.current", on=f"{current.on:.2f}", off=f"{current.off:.2f}"))
    if st.button(t("learning.click.save"), type="primary"):
        thresholds = sink.thresholds()
        if not thresholds.calibrated:
            st.warning(t("learning.click.too_close"))
            return
        service.save_pinch_calibration(patient.id, thresholds)
        st.success(
            t("learning.click.saved", on=f"{thresholds.on:.2f}", off=f"{thresholds.off:.2f}")
        )


@st.fragment
def _click_camera(sink: PinchCalibrationSink, camera_key: str) -> None:
    camera_streamer(sink, camera_key)


@st.fragment(run_every=0.5)
def _click_status(sink: PinchCalibrationSink) -> None:
    import time

    if sink.phase and time.time() < sink.until:
        st.warning(t(f"learning.click.capturing_{sink.phase}"))
    if sink.ratio is not None:
        st.metric(t("learning.click.ratio"), f"{sink.ratio:.2f}")
    st.caption(
        t(
            "learning.click.samples",
            apart=len(sink.samples["apart"]),
            pinch=len(sink.samples["pinch"]),
        )
    )


# ------------------------------------------------------------------ tests
def _next_code(existing: list[str]) -> str:
    numbers = [int("".join(c for c in code if c.isdigit()) or 0) for code in existing]
    return f"P{(max(numbers) if numbers else 0) + 1:02d}"


def _tab_tests(service: SpectraService) -> None:
    import pandas as pd

    ui.guide("think", t("usability.intro"), t("ux.mascot.name"))
    with st.expander(t("usability.protocol_title"), expanded=False):
        for step in range(1, 7):
            st.markdown(f"{step}\\. {t(f'usability.protocol_{step}')}")

    participants = service.db.list_participants()
    codes = [p["code"] for p in participants]
    with st.form("new_participant", clear_on_submit=True):
        st.markdown(f"**{t('usability.new_participant')}**")
        code = st.text_input(t("usability.code"), value=_next_code(codes), max_chars=8)
        consent = st.checkbox(t("usability.consent"))
        notes = st.text_input(t("usability.notes"), help=t("usability.notes_help"))
        if st.form_submit_button(t("usability.add")):
            clean = code.strip().upper()
            if not clean.isalnum():
                st.error(t("usability.code_invalid"))
            elif not consent:
                st.error(t("usability.consent_required"))
            else:
                service.db.add_participant(clean, True, notes.strip())
                st.rerun()

    if not participants:
        return
    selected = st.selectbox(t("usability.participant"), codes, key="tests_participant")
    first, second = counterbalanced_order(selected)
    st.markdown(f"**{t('usability.run_title')}**")
    st.caption(t("usability.run_hint"))
    st.code(
        f"scripts\\spectra.bat --usability {selected} --filter {first}\n"
        f"scripts\\spectra.bat --usability {selected} --filter {second}",
        language=None,
    )

    with st.form(f"sus_{selected}", clear_on_submit=True):
        st.markdown(f"**{t('usability.sus_title')}**")
        st.caption(t("usability.sus_hint"))
        answers = [
            st.select_slider(
                f"{i}. {t(f'usability.sus.{i}')}",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda v: t(f"usability.likert.{v}"),
            )
            for i in range(1, SUS_ITEMS + 1)
        ]
        if st.form_submit_button(t("usability.sus_save")):
            score = sus_score(answers)
            service.db.add_usability_result(
                selected, "sus", "", {"answers": answers, "score": score}
            )
            st.success(t("usability.sus_saved", score=f"{score:.1f}"))

    results = service.db.list_usability_results()
    rows = participant_rows(results)
    if not rows:
        return
    st.subheader(t("usability.results_title"))
    st.dataframe(pd.DataFrame(rows).set_index("participant"), use_container_width=True)
    sus_values = [row["sus"] for row in rows if "sus" in row]
    if sus_values:
        summary = describe(sus_values)
        st.metric(
            t("usability.sus_mean"),
            f"{summary['mean']:.1f}",
            t(sus_adjective(summary["mean"])),
            delta_color="off",
        )
    comparison = compare_filters(rows)
    lines = []
    for metric, result in comparison.items():
        if result is None:
            continue
        lines.append(
            {
                t("usability.metric"): t(f"usability.metric_{metric}"),
                "n": result.n,
                t("usability.difference"): round(result.mean_difference, 3),
                "t": round(result.t, 2),
                "p": round(result.p, 4),
                "dz": round(result.effect_size_dz, 2),
            }
        )
    if lines:
        st.markdown(f"**{t('usability.comparison_title')}**")
        st.dataframe(lines, use_container_width=True)
        st.caption(t("usability.comparison_hint"))
    st.download_button(
        t("usability.csv"),
        results_csv(flat_rows(results)),
        file_name=f"spectra_testes_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
    )
    if st.button(t("usability.delete", code=selected)):
        if st.session_state.get("confirm_participant") == selected:
            service.db.delete_participant(selected)
            st.session_state.pop("confirm_participant", None)
            st.rerun()
        else:
            st.session_state["confirm_participant"] = selected
            st.warning(t("usability.confirm_delete"))


# ------------------------------------------------------------- evaluation
@st.cache_data(show_spinner=False)
def _evaluation_tables():
    from dataclasses import asdict

    from spectra.evaluation import evaluate_depth, evaluate_smoothing

    return [asdict(r) for r in evaluate_smoothing()], [asdict(r) for r in evaluate_depth()]


def _tab_evaluation() -> None:
    import pandas as pd

    ui.guide("think", t("learning.eval.intro"), t("ux.mascot.name"))
    if not st.button(t("learning.eval.run"), type="primary") and not st.session_state.get(
        "eval_ran"
    ):
        st.caption(t("learning.eval.cli"))
        return
    st.session_state["eval_ran"] = True
    with st.spinner(t("learning.eval.running")):
        smoothing, depth = _evaluation_tables()
    st.subheader(t("learning.eval.filters_title"))
    frame = pd.DataFrame(smoothing)
    pivot = frame.pivot(index="trajectory", columns="filter", values="rmse_px")
    st.caption(t("learning.eval.rmse"))
    st.bar_chart(pivot, stack=False)
    st.dataframe(frame.round(2), use_container_width=True, hide_index=True)
    st.caption(t("learning.eval.filters_hint"))
    st.subheader(t("learning.eval.depth_title"))
    depth_frame = pd.DataFrame(depth)
    st.caption(t("learning.eval.depth_error"))
    st.bar_chart(
        depth_frame.pivot(index="scenario", columns="method", values="mean_abs_error"),
        stack=False,
    )
    st.dataframe(depth_frame.round(3), use_container_width=True, hide_index=True)
    st.caption(t("learning.eval.depth_hint"))
    st.html(art.bubble_html("cheer", t("learning.eval.cli"), size=40))


__all__ = ["render_learning"]
