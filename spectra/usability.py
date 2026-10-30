"""User tests: the measurements that become the results chapter of the thesis.

Three instruments, all standard in human-computer interaction:

* **System Usability Scale (SUS)**: ten statements answered from 1 (strongly disagree) to
  5 (strongly agree), scored 0-100 (Brooke, 1996). Odd items are positive, even items
  negative. Adjectives follow Bangor, Kortum and Miller (2009).
* **Pointing task**: a ring of targets selected in the ISO 9241-9 alternating order with the
  pinch click. Each selection records movement time and errors; with Fitts' index of
  difficulty ``ID = log2(D/W + 1)`` the **throughput** (bits/s) summarises speed and accuracy.
* **Learnability**: seconds per lesson of the gesture learning path.

The pointing task runs once per pointer filter (Kalman, exponential), in counterbalanced
order, so each participant is their own control. Comparisons use the paired t-test, with the
p-value computed here (no SciPy) from the regularised incomplete beta function.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

SUS_ITEMS = 10
CONDITIONS = ("kalman", "exponential")


# ----------------------------------------------------------------------- SUS
def sus_score(answers: Sequence[int]) -> float:
    """Score 10 answers (1-5) on the 0-100 SUS scale."""
    if len(answers) != SUS_ITEMS or any(not 1 <= int(a) <= 5 for a in answers):
        raise ValueError("SUS needs ten answers between 1 and 5")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


def sus_adjective(score: float) -> str:
    """i18n key of the adjective rating (Bangor et al., 2009)."""
    if score >= 85.5:
        return "usability.sus_band.excellent"
    if score >= 72.6:
        return "usability.sus_band.good"
    if score >= 52.1:
        return "usability.sus_band.ok"
    if score >= 39.2:
        return "usability.sus_band.poor"
    return "usability.sus_band.awful"


def counterbalanced_order(code: str) -> tuple[str, str]:
    """Alternate the filter order by participant number (P01 Kalman first, P02 exponential)."""
    digits = "".join(character for character in code if character.isdigit())
    number = int(digits) if digits else 1
    return CONDITIONS if number % 2 == 1 else CONDITIONS[::-1]


# ------------------------------------------------------------ pointing task
@dataclass(frozen=True)
class Target:
    x: float
    y: float
    radius: float


def ring_targets(
    width: int, height: int, count: int = 9, radius: float = 34.0, spread: float = 0.36
) -> list[Target]:
    """ISO 9241-9 multidirectional layout: targets alternate across the ring."""
    if count % 2 == 0:
        raise ValueError("the ISO ring needs an odd number of targets")
    cx, cy = width / 2, height / 2
    ring = spread * min(width, height)
    positions = [
        (
            cx + ring * math.cos(2 * math.pi * i / count),
            cy + ring * math.sin(2 * math.pi * i / count),
        )
        for i in range(count)
    ]
    order = [(i * (count + 1) // 2) % count for i in range(count)]
    return [Target(positions[i][0], positions[i][1], radius) for i in order]


@dataclass(frozen=True)
class Selection:
    target: int
    seconds: float
    errors: int
    distance: float
    width: float

    @property
    def index_of_difficulty(self) -> float:
        return math.log2(self.distance / self.width + 1)

    @property
    def throughput(self) -> float:
        return self.index_of_difficulty / self.seconds if self.seconds > 0 else 0.0


@dataclass
class PointingTask:
    """Pure state machine of the pointing task; the mode only draws and feeds it."""

    targets: list[Target]
    index: int = 0
    shown_at: float | None = None
    errors: int = 0
    selections: list[Selection] = field(default_factory=list)
    path: float = 0.0
    _last_pointer: tuple[float, float] | None = None

    @property
    def finished(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def current(self) -> Target | None:
        return None if self.finished else self.targets[self.index]

    def update(self, pointer: tuple[float, float] | None, clicked: bool, at: float) -> str | None:
        """Feed one frame; returns ``hit``, ``miss`` or ``None``."""
        if self.finished:
            return None
        if self.shown_at is None:
            self.shown_at = at
        if pointer is not None and self._last_pointer is not None:
            self.path += math.dist(pointer, self._last_pointer)
        self._last_pointer = pointer
        if not clicked or pointer is None:
            return None
        target = self.targets[self.index]
        if math.dist(pointer, (target.x, target.y)) > target.radius:
            self.errors += 1
            return "miss"
        # The first target has no movement before it: it only starts the clock.
        if self.index > 0:
            previous = self.targets[self.index - 1]
            self.selections.append(
                Selection(
                    target=self.index,
                    seconds=at - self.shown_at,
                    errors=self.errors,
                    distance=math.dist((previous.x, previous.y), (target.x, target.y)),
                    width=2 * target.radius,
                )
            )
        self.index += 1
        self.shown_at = at
        self.errors = 0
        return "hit"

    def summary(self) -> dict:
        selections = self.selections
        if not selections:
            return {"selections": 0}
        return {
            "selections": len(selections),
            "movement_time_s": statistics.mean(s.seconds for s in selections),
            "errors": sum(s.errors for s in selections),
            "error_rate": sum(1 for s in selections if s.errors) / len(selections),
            "throughput_bps": statistics.mean(s.throughput for s in selections),
            "path_px": self.path,
            "per_target": [asdict(s) for s in selections],
        }


# --------------------------------------------------------------- statistics
def describe(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes, Lentz)."""
    tiny, eps = 1e-30, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    )
    front = math.exp(log_front)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value of Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        return float("nan")
    return incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean_difference: float
    t: float
    p: float
    effect_size_dz: float


def paired_t_test(first: Sequence[float], second: Sequence[float]) -> PairedComparison | None:
    """Paired t-test of ``first - second``; ``None`` with fewer than two pairs."""
    if len(first) != len(second):
        raise ValueError("paired samples need the same length")
    differences = [a - b for a, b in zip(first, second, strict=True)]
    n = len(differences)
    if n < 2:
        return None
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    if sd == 0:
        return PairedComparison(n, mean, math.inf if mean else 0.0, 0.0 if mean else 1.0, 0.0)
    t = mean / (sd / math.sqrt(n))
    return PairedComparison(n, mean, t, t_two_sided_p(t, n - 1), mean / sd)


# ----------------------------------------------------------------- aggregation
def participant_rows(results: Iterable[dict]) -> list[dict]:
    """One row per participant: pointing per filter, SUS and learning-path time."""
    rows: dict[str, dict] = {}
    for result in results:
        row = rows.setdefault(result["code"], {"participant": result["code"]})
        payload = result["payload"]
        if result["kind"] == "pointing" and payload.get("selections"):
            prefix = result["condition"]
            row[f"{prefix}_mt"] = payload["movement_time_s"]
            row[f"{prefix}_tp"] = payload["throughput_bps"]
            row[f"{prefix}_err"] = payload["error_rate"]
        elif result["kind"] == "sus":
            row["sus"] = payload["score"]
        elif result["kind"] == "tutorial":
            row["tutorial_s"] = sum(lesson["seconds"] for lesson in payload["lessons"])
            row["tutorial_stars"] = sum(lesson["stars"] for lesson in payload["lessons"])
    return [rows[code] for code in sorted(rows)]


def flat_rows(results: Iterable[dict]) -> list[dict]:
    """Long-format rows for the CSV export."""
    rows = []
    for result in results:
        payload = result["payload"]
        rows.append(
            {
                "participant": result["code"],
                "kind": result["kind"],
                "condition": result["condition"],
                "movement_time_s": payload.get("movement_time_s", ""),
                "throughput_bps": payload.get("throughput_bps", ""),
                "error_rate": payload.get("error_rate", ""),
                "sus": payload.get("score", ""),
                "tutorial_seconds": (
                    sum(lesson["seconds"] for lesson in payload["lessons"])
                    if result["kind"] == "tutorial"
                    else ""
                ),
                "created_at": result["created_at"],
            }
        )
    return rows


def compare_filters(rows: Sequence[dict]) -> dict[str, PairedComparison | None]:
    """Paired Kalman-minus-exponential comparisons over participants who did both."""
    comparisons: dict[str, PairedComparison | None] = {}
    for metric in ("mt", "tp", "err"):
        pairs = [
            (row[f"kalman_{metric}"], row[f"exponential_{metric}"])
            for row in rows
            if f"kalman_{metric}" in row and f"exponential_{metric}" in row
        ]
        comparisons[metric] = (
            paired_t_test([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
        )
    return comparisons


# ----------------------------------------------------------------------- CSV
CSV_COLUMNS = (
    "participant",
    "kind",
    "condition",
    "movement_time_s",
    "throughput_bps",
    "error_rate",
    "sus",
    "tutorial_seconds",
    "created_at",
)


def results_csv(rows: Iterable[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
