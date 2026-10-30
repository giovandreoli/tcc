"""Gesture learning path: short lessons that teach the touchless interface step by step.

New users of a touchless interface do not know what the system can see. The path introduces
one concept per lesson, in increasing difficulty, and checks each one automatically from the
landmarks: showing the hand, opening and closing it, pointing, the pinch "click", choosing
colours with different fingers, moving towards the camera (depth) and, finally, turning the
hand *without* moving it, which shows that rotation is not mistaken for depth.

Each lesson records how long it took. Those times are a direct **learnability** measure for
the user tests (how fast people learn the gestures), and stars give the patient feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectra.detection.landmarks import HandLandmarks
from spectra.gestures.calibration import CalibrationProfile, FingerStateEstimator
from spectra.gestures.click import PinchClick, PinchThresholds
from spectra.gestures.depth import DepthEstimator, DepthMethod, DepthReading, Motion
from spectra.gestures.features import FingerStates
from spectra.gestures.finger_colors import DEFAULT_FINGER_COLORS, FingerColorSelector

STAR_SECONDS = (15.0, 40.0)
NEAR_SCALE = 1.25
BACK_SCALE = 1.10
ROTATION_TILT = 25.0
ROTATION_DEPTH_TOLERANCE = 0.15


@dataclass(frozen=True)
class Observation:
    landmarks: HandLandmarks | None
    states: FingerStates | None
    clicked: bool
    color_finger: str | None
    depth: DepthReading | None
    at: float


class Lesson:
    key = ""
    target = 1

    def __init__(self) -> None:
        self.count = 0
        self.held_since: float | None = None
        self.phase = 0

    def start(self, engine: TutorialEngine) -> None:
        """Hook: called when the lesson becomes active."""

    def hold(self, condition: bool, at: float, seconds: float) -> bool:
        if not condition:
            self.held_since = None
            return False
        if self.held_since is None:
            self.held_since = at
        return at - self.held_since + 1e-9 >= seconds

    def progress(self, at: float) -> float:
        return min(1.0, self.count / self.target)

    def update(self, obs: Observation) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError


class HoldLesson(Lesson):
    """Done once a posture is held continuously for ``seconds``."""

    seconds = 1.0

    def accept(self, obs: Observation) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def progress(self, at: float) -> float:
        if self.held_since is None:
            return 0.0
        return min(1.0, (at - self.held_since) / self.seconds)

    def update(self, obs: Observation) -> bool:
        return self.hold(obs.landmarks is not None and self.accept(obs), obs.at, self.seconds)


class ShowHand(HoldLesson):
    key = "show_hand"
    seconds = 1.5

    def accept(self, obs: Observation) -> bool:
        return True


class OpenHand(HoldLesson):
    key = "open_hand"

    def accept(self, obs: Observation) -> bool:
        return obs.states is not None and obs.states.extended_count >= 4


class Fist(HoldLesson):
    key = "fist"

    def accept(self, obs: Observation) -> bool:
        return obs.states is not None and obs.states.extended_count <= 1


class Point(HoldLesson):
    key = "point"

    def accept(self, obs: Observation) -> bool:
        states = obs.states
        return states is not None and states.index and not (states.middle or states.ring)


class Click(Lesson):
    key = "click"
    target = 3

    def update(self, obs: Observation) -> bool:
        self.count += obs.clicked
        return self.count >= self.target


class FingerColors(Lesson):
    key = "finger_colors"

    def __init__(self) -> None:
        super().__init__()
        self.done: set[str] = set()

    def start(self, engine: TutorialEngine) -> None:
        self.target = len(engine.colors.bindings)

    def update(self, obs: Observation) -> bool:
        if obs.color_finger:
            self.done.add(obs.color_finger)
        self.count = len(self.done)
        return self.count >= self.target


class _SettledDepthLesson(Lesson):
    """Captures the depth reference only after the hand has settled into the new lesson,
    so frames still moving from the previous lesson do not pollute it."""

    settle_seconds = 0.6

    def start(self, engine: TutorialEngine) -> None:
        self.engine = engine
        self.started: float | None = None
        self.settled = False

    def settled_reading(self, obs: Observation) -> DepthReading | None:
        if self.started is None:
            self.started = obs.at
        if not self.settled:
            if obs.at - self.started < self.settle_seconds:
                return None
            self.engine.depth.recalibrate()
            self.settled = True
            return None
        return obs.depth


class Depth(_SettledDepthLesson):
    """Bring the hand closer to the camera, then back."""

    key = "depth"
    target = 2

    def update(self, obs: Observation) -> bool:
        reading = self.settled_reading(obs)
        if reading is None:
            return False
        if self.phase == 0 and self.hold(reading.scale >= NEAR_SCALE, obs.at, 0.4):
            self.phase, self.count, self.held_since = 1, 1, None
        elif self.phase == 1 and self.hold(reading.scale <= BACK_SCALE, obs.at, 0.4):
            self.count = 2
            return True
        return False


class RotateWithoutDepth(_SettledDepthLesson):
    """Turn the palm without approaching: the interface must not read it as depth."""

    key = "rotate"
    seconds = 0.7

    def progress(self, at: float) -> float:
        if self.held_since is None:
            return 0.0
        return min(1.0, (at - self.held_since) / self.seconds)

    def update(self, obs: Observation) -> bool:
        reading = self.settled_reading(obs)
        rotated = (
            reading is not None
            and reading.motion is Motion.ROTATION
            and reading.tilt_degrees >= ROTATION_TILT
            and abs(reading.relative_distance - 1.0) <= ROTATION_DEPTH_TOLERANCE
        )
        return self.hold(rotated, obs.at, self.seconds)


LESSONS: tuple[type[Lesson], ...] = (
    ShowHand,
    OpenHand,
    Fist,
    Point,
    Click,
    FingerColors,
    Depth,
    RotateWithoutDepth,
)
LESSON_KEYS = tuple(lesson.key for lesson in LESSONS)


@dataclass(frozen=True)
class LessonResult:
    key: str
    seconds: float
    stars: int
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "seconds": round(self.seconds, 2),
            "stars": self.stars,
            "skipped": self.skipped,
        }


def stars_for(seconds: float) -> int:
    fast, slow = STAR_SECONDS
    return 3 if seconds <= fast else 2 if seconds <= slow else 1


@dataclass(frozen=True)
class TutorialStatus:
    lesson: str | None
    index: int
    total: int
    progress: float
    count: int
    target: int
    hand_visible: bool
    results: tuple[LessonResult, ...] = ()
    depth: DepthReading | None = None

    @property
    def finished(self) -> bool:
        return self.lesson is None

    @property
    def stars(self) -> int:
        return sum(result.stars for result in self.results)


@dataclass
class TutorialEngine:
    """Runs the lessons over a stream of timestamped landmarks."""

    calibration: CalibrationProfile | None = None
    pinch: PinchThresholds | None = None
    finger_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FINGER_COLORS))
    lessons: tuple[type[Lesson], ...] = LESSONS
    image_aspect: float = 16 / 9

    def __post_init__(self) -> None:
        self.estimator = FingerStateEstimator(self.calibration)
        self.click = PinchClick(self.pinch)
        self.colors = FingerColorSelector(self.finger_colors)
        self.depth = DepthEstimator(DepthMethod.AREA_ASPECT, self.image_aspect, reference_frames=10)
        self.index = 0
        self.results: list[LessonResult] = []
        self.current: Lesson | None = self.lessons[0]() if self.lessons else None
        self.started_at: float | None = None
        self.last_at = 0.0
        self.hand_visible = False
        self.reading: DepthReading | None = None

    def _begin(self, at: float) -> None:
        if self.index >= len(self.lessons):
            self.current = None
            return
        self.current = self.lessons[self.index]()
        self.started_at = at
        self.current.start(self)

    def _complete(self, at: float, skipped: bool = False) -> None:
        if self.current is None or self.started_at is None:
            return
        seconds = max(0.0, at - self.started_at)
        self.results.append(
            LessonResult(self.current.key, seconds, 0 if skipped else stars_for(seconds), skipped)
        )
        self.index += 1
        self._begin(at)

    def feed(self, landmarks: HandLandmarks | None, at: float) -> bool:
        """Account for one frame; returns ``True`` when a lesson was completed."""
        self.last_at = at
        if self.started_at is None and self.current is not None:
            self.started_at = at
            self.current.start(self)
        if landmarks is None:
            self.estimator.reset()
            states = None
        else:
            states = self.estimator.update(landmarks)
        self.hand_visible = landmarks is not None
        clicked = self.click.update(landmarks, at)
        color = self.colors.update(landmarks, at)
        self.reading = self.depth.update(landmarks)
        if self.current is None:
            return False
        observation = Observation(landmarks, states, clicked, color, self.reading, at)
        if self.current.update(observation):
            self._complete(at)
            return True
        return False

    def skip(self) -> None:
        if self.started_at is None and self.current is not None:
            self.started_at = self.last_at
        self._complete(self.last_at, skipped=True)

    def status(self) -> TutorialStatus:
        lesson = self.current
        return TutorialStatus(
            lesson=lesson.key if lesson else None,
            index=min(self.index, len(self.lessons)),
            total=len(self.lessons),
            progress=lesson.progress(self.last_at) if lesson else 1.0,
            count=lesson.count if lesson else 0,
            target=lesson.target if lesson else 0,
            hand_visible=self.hand_visible,
            results=tuple(self.results),
            depth=self.reading,
        )
