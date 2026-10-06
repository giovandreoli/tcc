# Architecture

Why SPECTRA is split the way it is, and what each boundary buys.

## 1. The constraint that shapes everything

The application runs on **Windows with a webcam**, but it is developed and tested on
**Linux/WSL with neither**. Every design decision below follows from that: the parts that
touch hardware are small, isolated and thin, and the parts that hold the logic worth testing
are pure.

```
┌──────────────────────────────────────────────────────────────────┐
│  Hardware / platform (NOT unit tested, deliberately thin)        │
│  detection/camera.py · detection/hand_detector.py · ui/sound.py  │
│  app.py (cv2 window) · therapist_panel/app.py (Streamlit)        │
└───────────────────────────┬──────────────────────────────────────┘
                            │ plain data (landmarks, points, floats)
┌───────────────────────────▼──────────────────────────────────────┐
│  Pure logic (538 unit tests, headless)                           │
│  gestures/ · core/ · guided/ · metrics/ · reports/builder+txt    │
│  storage/ · service.py                                           │
└──────────────────────────────────────────────────────────────────┘
```

The rule: **nothing below the line may import a camera, open a window or need a GPU.**
MediaPipe is imported lazily inside `HandDetector.__init__` so even importing the package
works without the vision stack installed.

## 2. Layers

| Layer | Package | Responsibility | Depends on |
|---|---|---|---|
| Configuration | `config.py` | data directory, paths, tunables | — |
| Localisation | `i18n/`, `locales/` | pt-BR catalog, dotted keys | — |
| Detection | `detection/` | camera, MediaPipe, landmark constants | config |
| Gestures | `gestures/` | angles, calibration, profiles, state machine | detection (types only) |
| Core | `core/` | canvas, trail, smoothing, session guard | — |
| Guided | `guided/` | target shapes, trace scoring | — |
| Metrics | `metrics/` | recorder, ROM, tremor, fatigue, accuracy, quality | gestures, guided |
| Storage | `storage/` | SQLite, encryption, audit, backup | — |
| Reports | `reports/` | report object, TXT, PDF, charts | metrics, storage |
| Service | `service.py` | use cases shared by both apps | all of the above |
| Modes | `modes/` | interactive screens | gestures, core, guided, ui |
| Shells | `app.py`, `therapist_panel/` | camera loop; Streamlit UI | service, modes |

Dependencies point **downwards only**. `metrics/` never imports `modes/`; `storage/` never
imports `reports/`.

## 3. Data flow of one session

```
therapist panel          patient app                       storage
──────────────────────────────────────────────────────────────────────────
create patient ─────────────────────────────────────────► patient
record TCLE ────────────────────────────────────────────► consent
start session ──────────────────────────────────────────► session (needs consent)
     │
     └─ session id ──► PinEntryMode (PIN by dwell)
                            │
                            ▼
                       Camera ─► HandDetector ─► DetectionResult
                                                      │
                                       BaseMode.observe() once per frame
                                       ├─► GestureEngine ─► GestureState
                                       └─► SessionRecorder (10 Hz + full-rate tip)
                                                      │
                            Mode reacts to the state (paint / erase / menu / pause)
                            │
                            ▼ on exit
                       analyze_session() ──────────────► metric_summary, sample,
                                                          exercise_run
     ◄──── history ──────────────────────────────────────┘
generate report ─► build_session_report ─► SessionReport ─► PDF + TXT ─► share_dir
```

## 4. The five boundaries that matter

### `BaseMode.observe()` — one gesture step per frame

Every mode calls it exactly once and receives a `FrameContext` with the pointer, the finger
states, the gesture state and the landmarks. The earlier design let each mode call
`pointer()` and `read_states()` separately, which could advance the hysteresis estimator
twice in a single frame. One call makes that impossible.

### `DetectionResult` — one hand per session

Detection returns `DetectedHand` objects with a handedness label. `for_label()` implements
the "one hand per session, the other is ignored" rule in one place rather than in every mode.

### `SessionRecorder` — derived features, never frames

The seam where pixels stop. Above it there are images; below it there are only numbers. This
is what makes "no video is ever stored" a structural property rather than a promise.

### `SpectraService` — use cases, not widgets

Streamlit scripts are hard to test. Everything the panel *does* lives in the service, which
is unit tested; the panel is widgets and layout, and is excluded from coverage.

### `SessionReport` — masking happens once

Demo mode and CPF masking are applied when the report object is built. A renderer cannot leak
a name it was never given, so a future third output format cannot reintroduce the leak.

## 5. Where state lives

| State | Owner | Lifetime |
|---|---|---|
| Finger hysteresis, pinch | `GestureEngine` | per mode |
| Gesture state, hold timers | `GestureStateMachine` | per mode |
| Canvas and undo history | `DrawingCanvas` | per free-draw instance |
| Samples and trajectory | `SessionRecorder` | **shared across modes**, per session |
| Session limit, rest prompts | `SessionGuard` | per session |
| Patients, sessions, metrics | SQLite | permanent |

The recorder and the guard are built once by the shell and passed into `ModeContext`, so
moving between modes does not restart the recording or the clock.

## 6. Testing strategy

- **Synthetic hands.** `tests/conftest.py` builds anatomically plausible 21-landmark hands
  for any finger pattern, plus `rotate_hand` to prove the angle features are
  orientation-independent.
- **A fake clock.** The `clock` fixture freezes `time.time`, so hold times, dwell timers and
  rest prompts are deterministic instead of flaky.
- **Headless rendering.** Modes are driven over `numpy` frames; OpenCV draws into arrays
  without ever opening a window.
- **Properties, not snapshots.** Tests assert behaviour ("a transitional posture leaves no
  mark", "a 10 Hz recording cannot resolve the tremor band") rather than pixel output.

Not unit tested, by decision: the camera, the MediaPipe call and the Streamlit script. All
three are thin wrappers over a third-party API, and testing them would mean testing the mock.

## 7. Extension points

| To add… | Touch |
|---|---|
| A language | `locales/<locale>.json`; set `locale` in the config |
| A gesture profile | `gestures/profiles.py`, or build one from the panel |
| A target shape | subclass `Shape` in `guided/shapes.py`, add to `SHAPE_ORDER` |
| A metric | a function in `metrics/`, a line in `analyze_session`, a row in `reports/builder.py` — **no database migration**, payloads are JSON |
| An exercise | a detector in `modes/physio.py` plus three instruction strings |
| An output format | a renderer reading `SessionReport`; masking is already applied |
