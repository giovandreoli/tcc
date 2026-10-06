# Metrics

What SPECTRA measures, how each number is defined, and — most importantly — what each one
cannot tell you.

> **All angular measurements come from a single 2D webcam. They are approximations and are
> not a substitute for goniometry.** No metric in this document has been validated against a
> clinical instrument. Section 8 describes how such a validation would be run.

Everything here is computed from **landmarks only**. No video frame and no image of the
patient is ever stored (the painting PNG, which contains no patient image, is the sole
exception).

## 1. What is recorded

`spectra/metrics/recorder.py` reduces each frame to numbers and keeps them in two streams:

| Stream | Rate | Contents | Why |
|---|---|---|---|
| `samples` | 10 Hz | flexion angles, opposition distances, thumb abduction, palm orientation, fingertip position, confidence | Voluntary hand movement is well under 5 Hz, so 10 Hz is ample and cuts stored volume by two thirds versus 30 fps. |
| `trajectory` | full frame rate (~30 Hz) | fingertip `(t, x, y)` only | Tremor lives at 4-12 Hz. By Nyquist a 10 Hz recording **cannot** see it; 30 fps gives a 15 Hz ceiling. Three floats per frame is cheap. |

`frames_total` and `frames_with_hand` are counted on **every** frame, including the ones not
stored, so the presence ratio stays honest.

## 2. Tracking quality — the gate on everything else

`spectra/metrics/tracking_quality.py`

| Field | Definition |
|---|---|
| `presence_ratio` | `frames_with_hand / frames_total` |
| `mean_confidence` | mean MediaPipe handedness score over frames with a hand |
| `reliable` | `presence_ratio >= 0.60` **and** `mean_confidence >= 0.50` |

A session that fails either threshold still produces every other metric, but
`warning_key` names the reason and the report prints that warning next to the numbers.

**Read this first.** A ROM of 15° from a session where the hand was visible 30% of the time
means "the camera rarely saw the hand", not "the patient cannot move".

## 3. Range of motion (essential)

`spectra/metrics/rom.py`

### Per finger

For each finger the **flexion angle** is `180° - mean(angle at PIP, angle at DIP)`, so
`0°` is a perfectly straight finger and larger values mean more curl. Over a window the
report gives `min`, `max` and `span`.

- `span` is the clinically interesting number: how much the patient actually moved.
- `min` approximates active extension; `max` approximates active flexion.

**Limits.** The angle is measured in the image plane. A finger pointing towards the camera
is foreshortened and reads as *more* flexed than it is. The effect is largest when the palm
faces the camera and the fingers curl towards the lens, which is exactly the posture of a
fist — so `max` is systematically optimistic. Keep the camera roughly perpendicular to the
plane of movement and compare sessions recorded in the same setup.

### Thumb opposition (recommended)

Distance from the thumb tip to each fingertip, divided by palm size. Lower is better; the
reported value is the **best** (smallest) distance reached. Palm normalisation makes it
independent of the patient's distance from the camera and of hand size, so it is comparable
across sessions and across patients.

### Wrist — **ESTIMATE ONLY**

MediaPipe Hands provides **no forearm landmarks**. There is no anatomical reference against
which to measure a wrist angle, so SPECTRA measures the palm instead:

| Proxy | Definition | Approximates |
|---|---|---|
| `palm_rotation` | in-plane angle of the wrist → middle-MCP vector | radial/ulnar deviation, *only if the forearm is still* |
| `palm_openness` | area of the (wrist, index-MCP, pinky-MCP) triangle over palm size squared | pronation/supination, as a foreshortening proxy |

Both are labelled "estimativa" everywhere they appear, including in the PDF and TXT reports.
Neither should be quoted as a wrist range of motion. If the patient rotates their forearm,
`palm_rotation` changes without the wrist moving at all.

## 4. Tracing accuracy (essential)

`spectra/guided/scoring.py`, aggregated by `spectra/metrics/accuracy.py`

All distances are in **normalised units** (fractions of the frame), which makes the score
independent of screen resolution.

| Field | Definition |
|---|---|
| `mean_deviation` | mean distance from each sample to the target path |
| `max_deviation` | the single worst sample |
| `inside_ratio` | fraction of samples within the tolerance corridor |
| `completion` | fraction of the target's waypoints the pointer passed within the corridor |
| `duration` | first to last sample |
| `score` | `100 * (0.6 * completion + 0.4 * accuracy)` |

`accuracy` is `1.0` on the line, falling linearly to `0.0` at twice the corridor width.
Completion is weighted higher because finishing the movement matters more clinically than
tracing it beautifully.

**Limits.** The score mixes motor control with the patient's understanding of the task. A
low first attempt often means "did not understand yet", not "cannot move". Compare the best
attempt of a session, not the first.

## 5. Repetitions and pace (essential)

`spectra/metrics/accuracy.py` (`RepetitionSummary`), counted by the detectors in
`spectra/modes/physio.py`.

- `reps` — completed cycles, counted on a state transition (open → closed, pinch rising
  edge, full index→pinky wave), never on a single frame.
- `pace_rpm` — `reps / duration * 60`.
- `completion` — `reps / target`, capped at 1.

**Limits.** A repetition is only counted when the detector sees both ends of the cycle. A
patient who cannot fully close their hand will score zero repetitions on "open and close"
even though they are working hard. Read `reps` together with the ROM span.

## 6. Smoothness and tremor (recommended)

`spectra/metrics/tremor.py`

| Field | Definition |
|---|---|
| `mean_speed` | mean magnitude of fingertip velocity |
| `mean_jerk` | mean magnitude of the third derivative of position |
| `band_power_ratio` | fraction of spectral power inside 4-12 Hz, DC removed |
| `dominant_frequency` | frequency of the highest-power bin |
| `band_measurable` | `False` when the window is shorter than 32 samples or slower than 24 Hz |

The signal is mean-centred and Hann-windowed before the FFT; the DC bin is zeroed so that
voluntary movement does not drown the oscillation.

**Limits.** `band_measurable` is not a formality. Below 24 Hz the band is aliased and the
ratio is meaningless, so the code refuses to report it rather than returning a plausible
wrong number. Landmark jitter from the detector also lands in this band: a high
`band_power_ratio` in a session with poor tracking quality is far more likely to be camera
noise than patient tremor.

## 7. Fatigue (recommended)

`spectra/metrics/fatigue.py`

The first third of a window is compared with the last third:

```
decline = (start - end) / max(start, end)      # in [-1, 1]
```

Positive means the patient got worse over the window; negative means they warmed up.
Dividing by `max(start, end)` keeps the value bounded and stays defined when the patient
began with no measurable movement.

- `amplitude_decline` — from the mean per-finger flexion span (index to pinky; the thumb is
  excluded because its signal is the noisiest).
- `pace_decline` — from repetitions per minute.

`measurable` is `False` below 6 samples or 6 repetitions, where thirds are too small to
compare.

**Limits.** Fatigue and loss of interest look identical to this metric. A decline measured
in a 3-minute session says little; the intended use is within a single exercise of 8 or more
repetitions.

## 8. Towards a validation study

The data model is deliberately shaped so that a goniometer comparison can be bolted on
without changing anything:

- Every metric is stored as a **summary row per exercise run**, with the exercise identifier
  and the time window, so an external measurement taken right after a run can be joined to
  it by `(session_id, exercise_run_id)`.
- Angles are stored in degrees in the same convention a goniometer uses (0 = straight), so
  the comparison needs no transformation.
- `TrackingQuality` is stored alongside, so poor sessions can be excluded from the analysis
  rather than silently biasing it.

A minimal study: for each participant and each finger, record active flexion with a
goniometer immediately after a SPECTRA session, then compare per-finger `max` flexion with
a Bland-Altman plot and report the limits of agreement. The expected result is a systematic
positive bias from foreshortening (section 3) plus a spread driven by camera angle.

## 9. Optional metrics (not yet implemented)

Reserved in the schema, to be added if the field work calls for them: pain before/after
(VAS 0-10), hold time at end range, and reaction time.
