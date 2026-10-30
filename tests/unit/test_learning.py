"""Learning features: Kalman filter, depth (Z), pinch click, finger colours, gesture
classifier, learning path, application domains, user tests and the evaluation."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from spectra.core.kalman import (
    ExponentialPointFilter,
    KalmanPointFilter,
    RawPointFilter,
    make_point_filter,
)
from spectra.detection.landmarks import Landmark
from spectra.domains import (
    DOMAINS,
    Command,
    CommandMapper,
    DomainProfile,
    GestureTokenizer,
    ImageViewer,
    posture_of,
)
from spectra.evaluation import evaluate_depth, evaluate_smoothing, markdown_report
from spectra.evaluation.synthetic import IMAGE_ASPECT, HandPose, project
from spectra.gestures.calibration import CalibrationProfile
from spectra.gestures.click import PinchClick, PinchThresholds, calibrate_pinch
from spectra.gestures.depth import (
    DepthEstimator,
    DepthMethod,
    DepthReading,
    Motion,
    estimate,
    palm_geometry,
)
from spectra.gestures.features import FingerStates
from spectra.gestures.finger_colors import FingerColorSelector
from spectra.learning import GestureClassifier, GestureRecognizer, TrainingError, pose_features
from spectra.learning.tutorial import LESSON_KEYS, TutorialEngine, stars_for
from spectra.usability import (
    PointingTask,
    compare_filters,
    counterbalanced_order,
    describe,
    flat_rows,
    paired_t_test,
    participant_rows,
    results_csv,
    ring_targets,
    sus_adjective,
    sus_score,
    t_two_sided_p,
)
from tests.conftest import make_hand, pinching_hand, rotate_hand

OPEN = (True, True, True, True, True)
FIST = (False, False, False, False, False)


# ------------------------------------------------------------------- Kalman
def _noisy_circle(seconds=3.0, noise=4.0, seed=1):
    rng = random.Random(seed)
    times = [i / 30 for i in range(int(seconds * 30))]
    truth = [(320 + 150 * math.cos(1.6 * t), 240 + 150 * math.sin(1.6 * t)) for t in times]
    noisy = [(x + rng.gauss(0, noise), y + rng.gauss(0, noise)) for x, y in truth]
    return times, truth, noisy


def _rmse(points, truth):
    return math.sqrt(
        sum(math.dist(p, q) ** 2 for p, q in zip(points, truth, strict=True)) / len(truth)
    )


def test_kalman_tracks_a_moving_pointer_better_than_raw_and_exponential():
    times, truth, noisy = _noisy_circle()
    errors = {}
    for kind in ("raw", "exponential", "kalman"):
        smoother = make_point_filter(kind)
        errors[kind] = _rmse(
            [smoother.update_float(p, t) for p, t in zip(noisy, times, strict=True)], truth
        )
    assert errors["kalman"] < errors["raw"] < errors["exponential"]


def test_kalman_steadies_a_still_pointer():
    rng = random.Random(3)
    noisy = [(100 + rng.gauss(0, 4), 100 + rng.gauss(0, 4)) for _ in range(90)]
    smoother = KalmanPointFilter()
    filtered = [smoother.update_float(point, i / 30) for i, point in enumerate(noisy)]
    raw_error = sum(math.dist(p, (100, 100)) for p in noisy[30:]) / 60
    kalman_error = sum(math.dist(p, (100, 100)) for p in filtered[30:]) / 60
    assert kalman_error < raw_error * 0.7


def test_kalman_restarts_after_a_gap_and_on_missing_pointer():
    smoother = KalmanPointFilter()
    smoother.update((0, 0), now=0.0)
    smoother.update((10, 0), now=1 / 30)
    assert smoother.update((500, 500), now=5.0) == (500, 500)  # gap > 0.5 s
    assert smoother.update(None, now=5.1) is None
    assert smoother.update((7, 9), now=5.2) == (7, 9)


def test_kalman_estimates_velocity_and_handles_repeated_timestamps():
    smoother = KalmanPointFilter()
    for i in range(30):
        smoother.update((i * 10, 0), now=i / 30)
    assert smoother.velocity[0] == pytest.approx(300, rel=0.1)
    smoother.update((300, 0), now=29 / 30)  # same timestamp: falls back to 1/30 s


def test_filter_factory_and_validation():
    assert isinstance(make_point_filter("exponential"), ExponentialPointFilter)
    assert isinstance(make_point_filter("raw"), RawPointFilter)
    raw = make_point_filter("raw")
    assert raw.update((3, 4)) == (3, 4) and raw.update_float(None) is None
    raw.reset()
    with pytest.raises(ValueError):
        make_point_filter("median")
    with pytest.raises(ValueError):
        KalmanPointFilter(measurement_std=0)
    with pytest.raises(ValueError):
        ExponentialPointFilter(alpha=0)
    exponential = ExponentialPointFilter(0.5)
    exponential.update((0, 0))
    assert exponential.update((10, 10)) == (5, 5)


# -------------------------------------------------------------------- depth
REFERENCE = palm_geometry(project(HandPose()), IMAGE_ASPECT)


def _reading(pose, method):
    return estimate(palm_geometry(project(pose), IMAGE_ASPECT), REFERENCE, method)


@pytest.mark.parametrize("method", list(DepthMethod))
@pytest.mark.parametrize("distance", [35, 50, 80])
def test_every_method_measures_pure_translation(method, distance):
    reading = _reading(HandPose(distance_cm=distance), method)
    assert reading.relative_distance == pytest.approx(distance / 50, rel=0.03)
    assert reading.motion is not Motion.ROTATION
    assert reading.distance_cm(50) == pytest.approx(distance, rel=0.03)


def test_area_alone_mistakes_rotation_for_distance():
    reading = _reading(HandPose(roll=50), DepthMethod.AREA)
    assert reading.relative_distance > 1.2
    assert reading.motion is Motion.TRANSLATION


@pytest.mark.parametrize("pose", [HandPose(roll=50), HandPose(pitch=40), HandPose(70, roll=40)])
def test_aspect_ratio_separates_rotation_from_distance(pose):
    reading = _reading(pose, DepthMethod.AREA_ASPECT)
    assert reading.motion is Motion.ROTATION
    assert reading.relative_distance == pytest.approx(pose.distance_cm / 50, rel=0.04)
    assert reading.tilt_degrees > 20


def test_wrist_middle_length_is_stable_under_roll_but_not_pitch():
    roll = _reading(HandPose(roll=50), DepthMethod.AREA_LENGTH)
    assert roll.motion is Motion.ROTATION
    assert roll.relative_distance == pytest.approx(1.0, abs=0.03)
    pitch = _reading(HandPose(pitch=40), DepthMethod.AREA_LENGTH)
    assert abs(pitch.relative_distance - 1.0) > 0.1  # the documented limitation


def test_geometry_is_independent_of_in_plane_rotation():
    upright = palm_geometry(project(HandPose()), IMAGE_ASPECT)
    turned = palm_geometry(project(HandPose(yaw=60)), IMAGE_ASPECT)
    assert turned.aspect == pytest.approx(upright.aspect, rel=0.02)
    assert turned.area == pytest.approx(upright.area, rel=0.02)


def test_finger_flexion_does_not_change_the_palm_polygon():
    open_hand = palm_geometry(make_hand(*OPEN))
    fist = palm_geometry(make_hand(*FIST))
    assert fist.area == pytest.approx(open_hand.area)


def test_depth_estimator_captures_a_reference_then_smooths():
    estimator = DepthEstimator(image_aspect=IMAGE_ASPECT, reference_frames=3, smoothing=1.0)
    assert estimator.update(None) is None
    for _ in range(2):
        assert estimator.update(project(HandPose())) is None
    assert estimator.update(project(HandPose())).scale == pytest.approx(1.0)
    assert estimator.calibrated
    closer = estimator.update(project(HandPose(distance_cm=40)))
    assert closer.scale == pytest.approx(1.25, rel=0.03)
    estimator.recalibrate()
    assert not estimator.calibrated and estimator.reading is None


def test_degenerate_hand_is_ignored():
    estimator = DepthEstimator(reference_frames=1)
    flat = [Landmark(0.5, 0.5)] * 21
    assert estimator.update(flat) is None
    assert palm_geometry(flat).area == 0.0


# -------------------------------------------------------------------- click
def test_pinch_click_fires_once_per_press_with_hysteresis():
    click = PinchClick()
    apart = make_hand(thumb=True, index=True)
    assert not click.update(apart, 0.0)
    assert click.update(pinching_hand(0.0), 0.1)
    assert not click.update(pinching_hand(0.0), 0.2)  # still pressed
    assert not click.update(pinching_hand(0.11), 0.3)  # between on and off: no release
    assert click.pressed
    click.update(apart, 0.4)
    assert click.update(pinching_hand(0.0), 0.8)
    assert click.clicks == 2
    assert not click.update(None, 0.9) and not click.pressed


def test_pinch_click_refractory_period():
    click = PinchClick()
    assert click.update(pinching_hand(0.0), 0.0)
    click.update(make_hand(thumb=True, index=True), 0.05)
    assert not click.update(pinching_hand(0.0), 0.1)


def test_pinch_click_is_independent_of_distance_to_camera():
    near = pinching_hand(0.02)
    far = [Landmark(0.5 + (p.x - 0.5) * 0.4, 0.5 + (p.y - 0.5) * 0.4) for p in near]
    assert PinchClick().update(near, 0.0) and PinchClick().update(far, 0.0)


def test_pinch_calibration_learns_the_users_range():
    apart = [pinching_hand(0.12)] * 5
    pinched = [pinching_hand(0.03)] * 5
    thresholds = calibrate_pinch(apart, pinched)
    assert thresholds.calibrated
    click = PinchClick(thresholds)
    assert not click.update(pinching_hand(0.09), 0.0)
    assert click.update(pinching_hand(0.04), 0.1)
    assert not calibrate_pinch(apart, apart).calibrated
    assert not calibrate_pinch([], pinched).calibrated
    with pytest.raises(ValueError):
        PinchThresholds(0.5, 0.4)


def test_pinch_calibration_round_trips_in_the_patient_profile():
    profile = CalibrationProfile(pinch=PinchThresholds(0.2, 0.3, True))
    restored = CalibrationProfile.from_dict(profile.to_dict())
    assert restored.pinch == profile.pinch
    assert CalibrationProfile.from_dict({"pinch": profile.pinch.to_dict()}).pinch.on == 0.2


# ------------------------------------------------------------ finger colours
def _touch(finger: str) -> list[Landmark]:
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    hand = make_hand(index=True, middle=True, ring=True, pinky=True)
    tip = hand[tips[finger]]
    hand[4] = Landmark(tip.x, tip.y, 0.0)
    return hand


def test_finger_colours_select_once_per_touch_after_a_short_hold():
    selector = FingerColorSelector()
    assert selector.update(_touch("ring"), 0.0) is None
    assert selector.progress(0.15) == pytest.approx(0.5)
    assert selector.update(_touch("ring"), 0.31) == "ring"
    assert selector.update(_touch("ring"), 0.6) is None  # must release first
    selector.update(make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True), 0.7)
    selector.update(_touch("pinky"), 0.8)
    assert selector.update(_touch("pinky"), 1.2) == "pinky"


def test_the_index_is_left_for_the_click():
    selector = FingerColorSelector()
    selector.update(_touch("index"), 0.0)
    assert selector.update(_touch("index"), 1.0) is None


def test_finger_colour_bindings_are_validated():
    with pytest.raises(ValueError):
        FingerColorSelector({"thumb": "color.red"})
    assert FingerColorSelector({}).update(_touch("ring"), 0.0) is None
    selector = FingerColorSelector()
    assert selector.update(None, 0.0) is None and selector.progress(0.0) == 0.0


# --------------------------------------------------------------- classifier
def _features(pattern, jitter=0.0, rng=None, label="Right", angle=0.0):
    rng = rng or random.Random(0)
    hand = rotate_hand(make_hand(*pattern, label=label), angle)
    noisy = [Landmark(p.x + rng.gauss(0, jitter), p.y + rng.gauss(0, jitter)) for p in hand]
    return pose_features(noisy, label)


def _trained(samples=8):
    rng = random.Random(4)
    model = GestureClassifier()
    for label, pattern in (
        ("joinha", (True, False, False, False, False)),
        ("dois", (False, True, True, False, False)),
        ("aberta", OPEN),
    ):
        for _ in range(samples):
            model.add(label, _features(pattern, 0.004, rng))
    return model.fit()


def test_features_are_invariant_to_position_size_rotation_and_side():
    base = pose_features(make_hand(*OPEN))
    moved = pose_features([Landmark(p.x * 0.5 + 0.2, p.y * 0.5 + 0.1) for p in make_hand(*OPEN)])
    rotated = pose_features(rotate_hand(make_hand(*OPEN), 40))
    mirrored = pose_features(make_hand(*OPEN, label="Left"), "Left")
    for other in (moved, rotated, mirrored):
        assert np.allclose(base, other, atol=1e-6)


def test_classifier_learns_and_generalises_to_unseen_rotations():
    model = _trained()
    assert model.predict(_features((False, True, True, False, False), angle=30)).label == "dois"
    assert model.predict(_features(OPEN, label="Left", angle=-20)).label == "aberta"
    evaluation = model.evaluate()
    assert evaluation.accuracy == 1.0
    assert evaluation.confusion["dois"]["dois"] == 8
    assert set(evaluation.per_label.values()) == {1.0}


def test_classifier_rejects_poses_it_was_not_taught():
    model = _trained()
    unknown = model.predict(_features((False, False, False, True, True)))
    assert unknown.label is None and unknown.distance > model.threshold


def test_classifier_requires_enough_data_and_round_trips():
    model = GestureClassifier()
    with pytest.raises(TrainingError, match="labels"):
        model.fit()
    for _ in range(3):
        model.add("a", _features(OPEN))
        model.add("b", _features(FIST))
    with pytest.raises(TrainingError, match="samples"):
        model.fit()
    with pytest.raises(TrainingError, match="untrained"):
        model.predict(_features(OPEN))
    with pytest.raises(TrainingError, match="label"):
        model.add(" ", _features(OPEN))
    with pytest.raises(TrainingError, match="features"):
        model.add("a", (1.0, 2.0))
    trained = _trained()
    restored = GestureClassifier.from_dict(trained.to_dict())
    assert restored.counts() == trained.counts() and restored.threshold == trained.threshold
    with pytest.raises(TrainingError, match="version"):
        GestureClassifier.from_dict({"version": 99})
    trained.remove_label("dois")
    assert "dois" not in trained.labels and not trained.trained


def test_recognizer_emits_an_event_once_the_gesture_is_stable():
    recognizer = GestureRecognizer(_trained(), window=5, min_votes=4)
    features = _features((False, True, True, False, False))
    events = [recognizer.update(features) for _ in range(6)]
    assert events.count("dois") == 1 and events[:3] == [None, None, None]
    assert recognizer.update(None) is None
    recognizer.reset()
    assert recognizer.current is None


# ------------------------------------------------------------ learning path
class Driver:
    def __init__(self, engine):
        self.engine = engine
        self.at = 0.0

    def feed(self, landmarks, seconds=1.2, step=1 / 15):
        done = False
        for _ in range(int(seconds / step)):
            self.at += step
            done = self.engine.feed(landmarks, self.at) or done
        return done


def test_learning_path_is_completed_with_real_gestures():
    engine = TutorialEngine(image_aspect=IMAGE_ASPECT)
    drive = Driver(engine)
    assert engine.status().lesson == "show_hand" and not engine.status().finished
    assert drive.feed(make_hand(*OPEN), 2.0)
    assert drive.feed(make_hand(*OPEN))
    assert drive.feed(make_hand(*FIST))
    assert drive.feed(make_hand(index=True))
    for _ in range(3):
        drive.feed(make_hand(thumb=True, index=True), 0.4)
        drive.feed(pinching_hand(0.0), 0.4)
    assert engine.status().lesson == "finger_colors"
    for finger in ("middle", "ring", "pinky"):
        drive.feed(_touch(finger), 0.6)
        drive.feed(make_hand(*OPEN), 0.3)
    assert engine.status().lesson == "depth"
    drive.feed(project(HandPose()), 1.0)
    drive.feed(project(HandPose(distance_cm=36)), 1.0)
    assert engine.status().count == 1
    drive.feed(project(HandPose()), 1.0)
    assert engine.status().lesson == "rotate"
    drive.feed(project(HandPose()), 1.0)
    drive.feed(project(HandPose(distance_cm=72)), 1.5)  # moving away is not rotating
    assert engine.status().lesson == "rotate"
    drive.feed(project(HandPose(roll=50)), 2.0)
    status = engine.status()
    assert status.finished and status.progress == 1.0
    assert [r.key for r in status.results] == list(LESSON_KEYS)
    assert all(r.stars >= 1 for r in status.results) and status.stars >= 8


def test_learning_path_can_skip_lessons_and_scores_stars():
    engine = TutorialEngine()
    engine.skip()
    assert engine.results[0].skipped and engine.results[0].stars == 0
    assert engine.status().lesson == "open_hand"
    assert (stars_for(5), stars_for(20), stars_for(90)) == (3, 2, 1)
    Driver(engine).feed(None, 0.5)
    assert not engine.status().hand_visible
    assert engine.results[0].to_dict()["skipped"] is True


# ------------------------------------------------------------------ domains
def test_postures_ignore_the_thumb():
    assert posture_of(FingerStates(True, True, True, False, False)) == "two"
    assert posture_of(FingerStates(False, False, False, False, True)) is None
    assert posture_of(None) is None


def test_tokenizer_emits_postures_once_and_depth_without_rotation():
    tokenizer = GestureTokenizer(hold_seconds=0.5)
    two = FingerStates(False, True, True, False, False)
    assert tokenizer.update(two, 0.0) == []
    assert tokenizer.update(two, 0.6) == ["two"]
    assert tokenizer.update(two, 1.0) == []
    near = DepthReading(1.3, Motion.TRANSLATION, DepthMethod.AREA_ASPECT)
    turned = DepthReading(1.3, Motion.ROTATION, DepthMethod.AREA_ASPECT, 40)
    assert tokenizer.update(None, 1.1, depth=turned) == []
    assert tokenizer.update(None, 1.2, depth=near) == ["push"]
    assert tokenizer.update(None, 1.3, depth=near) == []
    back = DepthReading(1.0, Motion.STEADY, DepthMethod.AREA_ASPECT)
    far = DepthReading(0.8, Motion.TRANSLATION, DepthMethod.AREA_ASPECT)
    tokenizer.update(None, 1.4, depth=back)
    assert tokenizer.update(None, 1.5, depth=far) == ["pull"]
    tokenizer.update(None, 1.6, depth=back)
    assert tokenizer.update(None, 1.7, clicked=True, learned="joinha") == ["click", "ml:joinha"]
    tokenizer.reset()


def test_sterile_domain_lock_blocks_everything_but_unlock():
    mapper = CommandMapper(DOMAINS["sterile"])
    assert mapper.update(["two"], 0.0) == [Command.NEXT]
    assert mapper.update(["two"], 0.3) == []  # cooldown
    assert mapper.update(["fist"], 1.0) == [Command.LOCK] and mapper.locked
    assert mapper.update(["two", "push"], 2.0) == []
    assert mapper.update(["fist"], 3.0) == [Command.LOCK] and not mapper.locked
    assert mapper.update(["unbound"], 4.0) == []


def test_domains_are_remappable_and_validated():
    profile = DomainProfile.from_dict(
        {"name": "custom", "bindings": {"open_hand": "next", "ml:ok": "select"}}
    )
    assert profile.bindings["ml:ok"] is Command.SELECT
    assert DomainProfile.from_dict(profile.to_dict()) == profile
    learned = DOMAINS["sterile"].with_learned({"joinha": Command.NEXT})
    assert learned.bindings["ml:joinha"] is Command.NEXT
    with pytest.raises(ValueError):
        DomainProfile("x", {"wave": Command.NEXT})
    with pytest.raises(ValueError):
        DomainProfile("x", {"fist": Command.LOCK, "two": Command.LOCK})


def test_image_viewer_commands():
    viewer = ImageViewer(count=3)
    assert viewer.apply(Command.PREVIOUS) and viewer.index == 2
    assert viewer.apply(Command.ZOOM_IN) and viewer.zoom == 1.25
    assert viewer.apply(Command.NEXT) and viewer.index == 0 and viewer.zoom == 1.0
    assert not viewer.apply(Command.ZOOM_OUT)
    for _ in range(20):
        viewer.apply(Command.ZOOM_IN)
    assert viewer.zoom == 4.0
    assert viewer.apply(Command.RESET_VIEW) and not viewer.apply(Command.SELECT)


# --------------------------------------------------------------- usability
def test_sus_scoring_and_adjectives():
    assert sus_score([5, 1] * 5) == 100.0
    assert sus_score([1, 5] * 5) == 0.0
    assert sus_score([3] * 10) == 50.0
    assert sus_score([4, 2, 4, 2, 4, 2, 4, 2, 4, 2]) == 75.0
    assert sus_adjective(90).endswith("excellent") and sus_adjective(75).endswith("good")
    assert sus_adjective(60).endswith("ok") and sus_adjective(45).endswith("poor")
    assert sus_adjective(20).endswith("awful")
    with pytest.raises(ValueError):
        sus_score([3] * 9)


def test_counterbalancing_alternates_filter_order():
    assert counterbalanced_order("P01") == ("kalman", "exponential")
    assert counterbalanced_order("P02") == ("exponential", "kalman")
    assert counterbalanced_order("X") == ("kalman", "exponential")


def test_iso_ring_targets_alternate_across_the_circle():
    targets = ring_targets(640, 480, count=9)
    assert len(targets) == 9
    first, second = targets[0], targets[1]
    assert math.dist((first.x, first.y), (second.x, second.y)) > 300
    with pytest.raises(ValueError):
        ring_targets(640, 480, count=8)


def test_pointing_task_measures_time_errors_and_throughput():
    targets = ring_targets(640, 480, count=5)
    task = PointingTask(targets)
    at = 0.0
    assert task.update((0, 0), True, at) == "miss"
    for target in targets:
        at += 0.8
        task.update((target.x - 50, target.y), False, at)
        assert task.update((target.x, target.y), True, at + 0.2) == "hit"
    assert task.finished and task.current is None
    assert task.update((0, 0), True, at + 1) is None
    summary = task.summary()
    assert summary["selections"] == 4
    assert summary["movement_time_s"] == pytest.approx(0.8)
    assert summary["errors"] == 0 and summary["throughput_bps"] > 0
    assert PointingTask(targets).summary() == {"selections": 0}


def test_statistics_match_reference_values():
    assert t_two_sided_p(2.0, 10) == pytest.approx(0.0734, abs=1e-3)
    assert t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    assert math.isnan(t_two_sided_p(1.0, 0))
    comparison = paired_t_test([1.0, 1.2, 0.9, 1.1], [1.4, 1.5, 1.3, 1.6])
    assert comparison.mean_difference == pytest.approx(-0.4)
    assert comparison.p < 0.01 and comparison.effect_size_dz < 0
    assert paired_t_test([1.0], [2.0]) is None
    assert paired_t_test([1, 2], [0, 1]).p == 0.0
    with pytest.raises(ValueError):
        paired_t_test([1], [1, 2])
    assert describe([]) == {"n": 0}
    assert describe([2.0, 4.0])["sd"] == pytest.approx(math.sqrt(2))


def test_results_are_aggregated_per_participant_and_exported():
    pointing = {"selections": 4, "movement_time_s": 1.0, "throughput_bps": 2.0, "error_rate": 0.1}
    results = [
        {
            "code": "P01",
            "kind": "pointing",
            "condition": "kalman",
            "payload": pointing,
            "created_at": 1,
        },
        {
            "code": "P01",
            "kind": "pointing",
            "condition": "exponential",
            "payload": {**pointing, "movement_time_s": 1.4},
            "created_at": 2,
        },
        {
            "code": "P01",
            "kind": "sus",
            "condition": "",
            "payload": {"score": 80.0},
            "created_at": 3,
        },
        {
            "code": "P01",
            "kind": "tutorial",
            "condition": "",
            "payload": {"lessons": [{"seconds": 5, "stars": 3}]},
            "created_at": 4,
        },
        {
            "code": "P02",
            "kind": "pointing",
            "condition": "kalman",
            "payload": {"selections": 0},
            "created_at": 5,
        },
    ]
    rows = participant_rows(results)
    assert rows[0]["kalman_mt"] == 1.0 and rows[0]["exponential_mt"] == 1.4
    assert rows[0]["sus"] == 80.0 and rows[0]["tutorial_s"] == 5
    assert compare_filters(rows)["mt"] is None  # a single pair cannot be tested
    csv_text = results_csv(flat_rows(results))
    assert csv_text.splitlines()[0].startswith("participant,kind,condition")
    assert len(csv_text.splitlines()) == 6


# --------------------------------------------------------------- evaluation
def test_evaluation_reproduces_the_documented_ranking():
    smoothing = {(r.trajectory, r.filter): r for r in evaluate_smoothing()}
    for trajectory in ("circle", "figure_eight"):
        assert smoothing[(trajectory, "kalman")].rmse_px < smoothing[(trajectory, "raw")].rmse_px
        assert (
            smoothing[(trajectory, "kalman")].rmse_px
            < smoothing[(trajectory, "exponential")].rmse_px
        )
        assert (
            smoothing[(trajectory, "kalman")].jitter_px
            < smoothing[(trajectory, "raw")].jitter_px / 2
        )
    depth = {(r.scenario, r.method): r for r in evaluate_depth(repeats=3)}
    assert (
        depth[("roll", "area_aspect")].mean_abs_error < depth[("roll", "area")].mean_abs_error / 3
    )
    assert (
        depth[("pitch", "area_aspect")].mean_abs_error
        < depth[("pitch", "area_length")].mean_abs_error
    )
    assert depth[("translation", "area")].rotation_detected == 0.0


def test_markdown_report_has_both_tables():
    report = markdown_report()
    assert "| circle | kalman |" in report and "| roll | area_aspect |" in report
