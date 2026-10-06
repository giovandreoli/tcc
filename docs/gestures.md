# Gesture engine

How SPECTRA turns 21 hand landmarks into an interaction state. Everything described here is
pure logic under `spectra/gestures/` and `spectra/core/session_guard.py`, with no camera, no
OpenCV window and no MediaPipe dependency, so all of it is unit tested headless.

## 1. Pipeline

```
landmarks ──► features ──► calibration ──► profile ──► state machine ──► GestureState
              (angles)     (hysteresis +   (pattern →   (hold time +
                            majority vote)  action)      pause toggle)
```

| Stage | Module | Output |
|---|---|---|
| Features | `features.py` | joint angles, abduction and pinch ratios |
| Calibration | `calibration.py` | `FingerStates` (5 booleans) |
| Profile | `profiles.py` | `GestureAction` or `None` |
| State machine | `state_machine.py` | `GestureState` |

## 2. Finger states

### Index to pinky — joint angles

For each finger the engine averages the two interphalangeal angles (at PIP and at DIP). The
result is `180°` for a straight finger and drops towards `0°` as the finger curls.

Angles are **invariant to rotation and to scale**, which the original tip-versus-joint comparison
(`tip.y < pip.y`) was not: tilting the hand sideways made the old heuristic report a closed fist.

### Thumb — abduction

The thumb's interphalangeal angle hardly changes between an open hand and a fist. The engine
instead measures the distance from the thumb tip to the index MCP, normalised by palm size
(wrist to middle MCP):

| Posture | Typical ratio |
|---|---|
| Thumb out | > 0.70 |
| Thumb tucked | < 0.30 |

### Hysteresis

Each finger has two thresholds. A finger is only reported extended once its signal crosses `on`,
and only released once it falls below `off`. The dead band between them stops the flag from
flickering while the patient holds a borderline posture.

Defaults (uncalibrated):

| Finger | Signal | `on` | `off` |
|---|---|---|---|
| Thumb | abduction ratio | 0.45 | 0.35 |
| Index to pinky | mean IP angle | 150° | 120° |

### Temporal smoothing

The last three frames vote; the majority wins. At 30 fps that is ~100 ms of latency, below the
perception threshold, and it absorbs the single-frame dropouts MediaPipe produces when a finger
briefly occludes another.

## 3. Per-patient calibration

`build_profile(open_samples, closed_samples)` records the patient holding an open hand and then a
fist, takes the **median** signal of each posture per finger (medians, not means, so a stray frame
cannot drag the threshold), and derives:

```
on  = closed + 0.60 * (open - closed)
off = closed + 0.40 * (open - closed)
```

If a finger's two postures are too close together (less than 25° for a finger, or 0.12 of palm
size for the thumb), that finger keeps the default thresholds and its name is added to
`CalibrationProfile.uncalibrated`. The therapist panel shows that list: a patient who cannot
extend the ring finger should be given a profile that does not require it, not silently
mis-detected thresholds.

The profile is serialisable (`to_dict` / `from_dict`) and is stored with the patient record.

## 4. Gesture profiles

A profile is a table of `GestureAction -> Trigger`. A `Trigger` constrains the finger pattern,
the pinch state, or both. Any of the 32 finger combinations may be used.

### `standard`

| Action | Gesture |
|---|---|
| `POINT` | index |
| `PAINT` | index + middle |
| `ERASE` | index + middle + ring |
| `COLOR_MENU` | open hand |
| `PAUSE` | fist |

Colours are then chosen inside the menu by dwelling on a swatch with the index finger.

### `simplified`

For low dexterity. Painting becomes a pinch, which needs no independent finger control, and the
hold time is raised to 600 ms.

| Action | Gesture |
|---|---|
| `POINT` | index, not pinching |
| `PAINT` | pinch (any finger pattern) |
| `ERASE` | index + middle, not pinching |
| `COLOR_MENU` | open hand, not pinching |
| `PAUSE` | fist, not pinching |

### `custom`

Built with `build_custom_profile()`, which returns the profile together with its validation
issues. Overlapping triggers are resolved by **specificity** (a trigger constraining both fingers
and pinch beats one constraining only fingers), never by declaration order, so a profile loaded
from the database behaves identically to one built in memory.

## 5. Validation

`validate_profile()` returns a list of `ProfileIssue`. Each issue carries a severity and an i18n
key (`gesture.issue.<code>`) that the therapist panel renders in pt-BR.

| Code | Severity | Meaning |
|---|---|---|
| `missing_action` | error | An action has no gesture bound to it. |
| `duplicate_trigger` | error | Two actions share the exact same gesture. |
| `ambiguous_patterns` | warning | Two gestures differ by a single finger. |
| `ring_without_middle` | warning | Requires the ring finger while the middle is down. The two share tendons; isolating the ring finger is hard even for healthy hands and often impossible after an injury. |
| `ring_and_pinky` | warning | Ring + pinky alone, a low-reliability combination. |
| `thumb_dependent` | warning | A sparse pattern that hinges on thumb state, the least reliable signal from a 2D webcam. |

The shipped 1 / 2 / 3-finger ladder (`fist`, `index`, `index+middle`, `index+middle+ring`,
`index+middle+ring+pinky`, open hand) is **exempt** from `ambiguous_patterns`: those patterns
differ by one finger on purpose. Any other one-finger-apart pair still warns.

## 6. State machine

States: `IDLE`, `POINTING`, `PAINTING`, `ERASING`, `COLOR_MENU`, `PAUSED`.

A matched action must be held continuously for `hold_seconds` (default 400 ms) before the state
changes. This is what makes transitional postures harmless: opening a fist passes through
"index", "index+middle" and "index+middle+ring" on the way, and without a hold time the patient
would paint and erase a stripe every time they opened their hand.

Changing gesture restarts the timer, so the hold always measures a *stable* posture.

### Losing the hand

A hand that disappears for less than `release_seconds` (150 ms) is ignored, so a single dropped
frame does not interrupt a brush stroke. Beyond that the machine falls back to `IDLE`.

**A lost hand never resumes a paused session.** A patient who leaves the frame while paused comes
back still paused and must deliberately repeat the gesture.

### Pause

Pause is a toggle, and it must be re-armed: after pausing, the pause gesture has to be *released*
before it can resume. Without that, holding a fist would oscillate between paused and idle every
400 ms. While paused, every other gesture is ignored.

## 7. Safety and ergonomics

`SessionGuard` (in `spectra/core/`) owns the non-gesture half of patient safety:

- a hard session limit (default 20 minutes);
- a rest prompt every `rest_every_minutes` (default 5), emitted **once** per interval so the UI
  does not nag;
- `remaining()` for the on-screen countdown.

Together with the easy pause gesture (a fist, the lowest-effort posture a hand can hold) these
cover the fatigue and overuse risks of unsupervised repetition.

## 8. Known limits

- All angles are measured on the **2D image plane**. A finger pointing at the camera is
  foreshortened and will read as more flexed than it is. See `docs/metrics.md`.
- MediaPipe provides no forearm landmarks, so any wrist measure is an estimate derived from the
  palm plane and is labelled as such everywhere, including in reports.
- Thumb abduction is sensitive to palm rotation; a hand held edge-on to the camera degrades it.
  The tracking-quality metric flags such sessions as unreliable.
