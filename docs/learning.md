# Learning features and evaluation

This document maps the thesis proposal to the implementation, states the method behind
each part, and records the reproducible results. Regenerate every table with:

```powershell
python -m spectra.evaluation
```

| Proposal | Implementation |
|---|---|
| Touchless HCI on a plain 2-D webcam, MediaPipe landmarks | `detection/`, `browser/camera/` (MediaPipe in the browser) |
| "Click" = Euclidean thumb–index distance | `gestures/click.py` (`PinchClick`), used by every on-screen button |
| Depth (Z) from the hand polygon area, disambiguated from rotation | `gestures/depth.py` — three estimators compared below |
| Modular, re-mappable gesture → command layer for different domains | `domains.py` (rehabilitation, sterile environment, drawing; JSON-loadable) |
| Fingers mapped to colours in spatial drawing | `gestures/finger_colors.py`, free-drawing mode |
| Kalman filter to remove jitter from the trajectory | `core/kalman.py`, default pointer filter |
| User tests | `usability.py`, desktop pointing task, panel "Testes com usuários" |
| *Machine learning*: the system learns the user's gestures | `learning/classifier.py` (k-NN with rejection), panel "Gestos personalizados" |
| *Learning*: the user learns the interface | `learning/tutorial.py` (8-lesson gesture path), panel and patient page |

## 1. Kalman filter for the pointer trajectory

The pointer (index fingertip) is filtered by a **constant-velocity Kalman filter**, one
independent 2-state filter per image axis. With state `[p, v]` and frame interval `dt` (from
frame timestamps, so dropped frames are handled):

```
predict:  p ← p + v·dt
          P ← F P Fᵀ + Q,   F = [[1, dt], [0, 1]],
                            Q = σa² · [[dt⁴/4, dt³/2], [dt³/2, dt²]]
correct:  K = P Hᵀ (H P Hᵀ + σm²)⁻¹,   H = [1, 0]
          [p, v] ← [p, v] + K (z − p);   P ← (I − K H) P
```

`σm` (measurement noise) and `σa` (acceleration noise) set the trade-off between steadiness
and lag. A sweep on the synthetic trajectories (below) selected `σm = 8 px` and
`σa = 900 px/s²`: jitter close to the exponential average with roughly a third of its error.
Gaps longer than 0.5 s restart the filter instead of extrapolating. The original exponential
average and a raw (unfiltered) pointer remain selectable (`pointer_filter` in `config.json`,
`--filter` on the command line) so they can be compared in the user tests.

**Results** (synthetic trajectories, 30 fps, Gaussian noise σ = 4 px; jitter = mean norm of the
second difference of the output; *still error* = error while the true pointer is still):

| Trajectory | Filter | RMSE (px) | Jitter (px) | Still error (px) |
|---|---|---:|---:|---:|
| circle | raw | 5.86 | 13.46 | — |
| circle | exponential | 14.72 | 3.27 | — |
| circle | kalman | 3.96 | 3.65 | — |
| figure_eight | raw | 5.86 | 13.46 | — |
| figure_eight | exponential | 14.52 | 3.27 | — |
| figure_eight | kalman | 4.52 | 3.65 | — |
| hold_then_stroke | raw | 5.86 | 13.37 | 4.76 |
| hold_then_stroke | exponential | 10.14 | 3.29 | 1.94 |
| hold_then_stroke | kalman | 5.33 | 3.71 | 2.38 |

Reading: the exponential average removes jitter only by lagging (its RMSE is 2.5× the raw
error on moving strokes). The Kalman filter cuts jitter by ~73% *and* has the lowest error on
moving strokes. On a perfectly still pointer the exponential average is slightly steadier; the
user tests decide whether that is perceptible.

## 2. Pinch "click"

`ratio = ‖thumb_tip − index_tip‖ / ‖wrist − middle_MCP‖`. Normalising by the palm length makes
the threshold independent of the distance to the camera. Two thresholds (`on = 0.35`,
`off = 0.45` by default) give hysteresis; a click is the press edge, with a 0.25 s refractory
period. **Per-user calibration** (panel → *Calibrar clique*) records the user's own
"fingers apart" and "pinch" ratios and places the thresholds at 35% and 55% of that range, so
users with limited thumb opposition can click. When the gesture profile already uses the pinch
(the simplified profile paints with it), the click is disabled to avoid conflicts. A click lands
on the last pointer position seen within 0.4 s, because the index fingertip bends while pinching.

## 3. Depth (Z) from a single 2-D camera

The **palm polygon** (wrist, thumb CMC and the four MCP joints) is used because it is almost
rigid: flexing the fingers does not change it. Under a pinhole camera, a translation in depth
scales every length by `1/d` and the area by `1/d²`. A reference is captured at a comfortable
position; every estimate is **relative** to it (`scale = 1`; `scale > 1` = closer).

The ambiguity stated in the proposal is real: tilting the hand shrinks the projected area by
`cos θ`, which area alone reads as "moved away". Both complementary metrics from the proposal
are implemented and compared:

- **`area_aspect`** — aspect ratio `width/height` of the polygon, measured along the hand's own
  axis (wrist → middle MCP), so in-plane rotation does not change it. Translation keeps it
  constant; a tilt about one palm axis changes it by `cos θ`. If it changed by more than 12%,
  the motion is classified as rotation and the area is corrected:
  `scale = √(area/area₀) / √(cos θ)` with `cos θ = min(r, 1/r)` and `r = aspect/aspect₀`.
- **`area_length`** — wrist → middle-MCP length, which a roll about the hand's long axis
  (pronation/supination) leaves unchanged. If `√(area/area₀)` and `length/length₀` disagree by
  more than 10%, the motion is a rotation and the length scale is used.

**Results** (synthetic 3-D hand projected by a 60° FOV camera; 20 noisy repetitions per pose,
σ = 1.5 px; error = |estimated − true relative distance| / true):

| Scenario | Method | Mean error | Max error | Classified as rotation |
|---|---|---:|---:|---:|
| translation 35–80 cm | area | 1.6% | 7.4% | 0% |
| translation 35–80 cm | area_aspect | 1.6% | 7.2% | 0% |
| translation 35–80 cm | area_length | 1.5% | 8.3% | 0% |
| roll 0–60° | area | 13.9% | 51.9% | 0% |
| roll 0–60° | area_aspect | 2.0% | 7.6% | 54% |
| roll 0–60° | area_length | 2.1% | 7.1% | 44% |
| pitch 0–50° | area | 6.2% | 23.8% | 0% |
| pitch 0–50° | area_aspect | 2.2% | 12.2% | 38% |
| pitch 0–50° | area_length | 12.7% | 51.1% | 28% |
| roll + translation | area | 14.0% | 54.9% | 0% |
| roll + translation | area_aspect | 2.1% | 9.1% | 56% |
| roll + translation | area_length | 2.6% | 14.5% | 44% |

Reading: all three measure pure translation equally well. Area alone confuses rotation with
distance (up to 52% error). The wrist–middle distance fixes roll but **not** pitch (flexing the
wrist foreshortens exactly that segment). The aspect ratio fixes both, so `area_aspect` is the
default. "Classified as rotation" below 100% in the rotation scenarios is expected: the 0° and
10° poses are (almost) not rotated. Limits: single-axis tilt model, flat palm, and a reference
that must be captured with the palm facing the camera.

Uses of depth: free drawing (closer = thicker stroke; rotation does not change it), the sterile
viewer (push = zoom in, pull = zoom out) and the learning path.

## 4. Domains: re-mapping gestures to commands

Detection produces **tokens** (`open_hand`, `fist`, `point`, `two`, `three`, `click`, `push`,
`pull`, `ml:<learned label>`); a **domain profile** maps tokens to commands with a per-command
cooldown. Presets: `rehab`, `sterile`, `drawing`; profiles round-trip to JSON. The sterile
profile adds a **lock** (closed fist) that blocks every other command, so an accidental
movement near the screen cannot change the image during a procedure. The desktop *Ambiente
estéril* mode demonstrates it with synthetic images only.

## 5. The system learns the user: gesture classifier

- **Features** (`learning/features.py`): landmarks translated to the wrist, rotated so the
  wrist → middle-MCP axis points up, divided by the palm length, left hands mirrored, plus the
  five finger-extension angles (45 values). Invariant to position, size, in-plane rotation and
  side — verified by tests.
- **Model**: distance-weighted k-NN (k = 3). Chosen over a neural network because it needs tens
  of examples, trains instantly on any laptop, adapts when examples are added and its
  decisions are explainable.
- **Rejection**: a pose farther than `1.5 × P95` of the in-class nearest-neighbour distances is
  reported as unknown instead of forcing a class.
- **Evaluation**: leave-one-out accuracy and confusion matrix in the panel (the rejection
  threshold is fitted on all examples, a small optimistic bias that is acceptable for a
  per-user model and is stated here).
- **Online use**: a majority vote over the last 7 predictions (≥ 5 agreeing) produces one event
  per gesture onset. Learned gestures can be bound to any command and are exported to the
  desktop application (`<data_dir>/models/gestures.json`).
- **Privacy**: only the 45 normalised numbers per example are stored, per patient, encrypted
  database, included in the LGPD export and deleted with the patient.

## 6. The user learns the interface: gesture learning path

Eight automatically checked lessons, one concept each: show the hand, open, fist, point, three
pinch clicks, the three finger colours, bring the hand closer and back (depth), and turn the
palm **without** approaching (the interface must report rotation, not depth). Each lesson
records its time and gives 1–3 stars (≤ 15 s, ≤ 40 s, slower); lessons can be skipped. It is
available to the therapist (panel) and optionally to the patient before exercising, and its
times are a **learnability** measure for the user tests.

## 7. User-test protocol

1. Register an anonymous participant code (P01, P02…) after the research TCLE is signed.
2. Learning path, saved under the participant code (learnability).
3. Pointing task (ISO 9241-9 ring of 9 targets, pinch click only) twice, once per filter, in
   counterbalanced order: odd codes start with Kalman, even codes with the exponential average.
   `scripts\spectra.bat --usability P01 --filter kalman`
4. System Usability Scale (10 items, 0–100 score, Bangor adjective bands).
5. Optional free observation in spatial drawing and the sterile viewer.
6. The panel shows per-participant rows, SUS mean and a **paired t-test** (Kalman − exponential)
   for movement time, throughput (`ID/MT`, `ID = log₂(D/W + 1)`) and error rate, with the effect
   size `dz`; the CSV export feeds the results chapter. The p-value uses the regularised
   incomplete beta function (no SciPy), validated against reference values in the tests.

With the usual 8–15 participants, report effect sizes and confidence alongside p-values.
