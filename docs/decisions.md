# Decision log

Decisions delegated to the implementation during the SPECTRA refactor.
Format: decision, then the reason. Written in English per the project language rule
(code, comments and technical docs in English; user-visible text in pt-BR).

## Phase 0 — Hygiene

**D-001 — Python 3.10 to 3.12, targeting 3.12.**
`mediapipe==0.10.21` publishes Windows wheels up to CPython 3.12 and none for 3.13.
3.12 is the common denominator between the author's Windows machine and the WSL dev box.

**D-002 — pytest configured in `pyproject.toml`, no `pytest.ini`.**
The brief asked for both, but `pytest.ini` takes precedence over `pyproject.toml`; having both
would hide the real configuration. A single source of truth (`[tool.pytest.ini_options]`) matches
the fixed decision to configure black/isort/ruff/pytest from `pyproject.toml`.

**D-003 — Coverage configured in `.coveragerc`.**
The fixed decision about `pyproject.toml` does not mention coverage, and the author's other
repositories use a standalone `.coveragerc`.

**D-004 — `hand_landmarker.task` stays tracked.**
The file is already in Git history, so removing it would not shrink the repository, and keeping it
lets the app run offline. `.gitignore` excludes every other `*.task` (copies downloaded into the
data directory) with an explicit exception for the root copy.

**D-005 — `requirements.txt` (runtime) split from `requirements-dev.txt` (lint/test).**
The headless test machine needs neither MediaPipe nor a webcam, so CI never installs the vision
stack.

**D-006 — Runtime dependencies pinned with `==`.**
Reproducibility is a thesis requirement: the examiner must be able to rebuild the environment.
`numpy` uses a range (`>=1.26,<2.2`) because it is a shared transitive dependency.

## Phase 1 — Package split

**D-007 — On-screen text is ASCII-folded in `spectra/ui/text.py`.**
OpenCV only ships Hershey vector fonts, which cannot render accents: `"Educação"` would come out
mangled. The catalog keeps correct pt-BR (needed for reports and the panel) and the drawing layer
strips diacritics at the last moment. Rejected alternative: embedding a TrueType font via Pillow,
which costs per-frame performance and adds a dependency to the patient app.

**D-008 — Detection returns `DetectionResult`/`DetectedHand` instead of two parallel lists.**
The original code passed `lm_list` and `hd_list` around and re-derived handedness in every mode.
An object with `for_label()` directly implements the fixed "one hand per session" decision.

**D-009 — MediaPipe is imported lazily, inside `HandDetector.__init__`.**
Every other module — and the whole test suite — can then be imported on Linux/WSL without the
vision stack installed.

**D-010 — `winsound` isolated behind `SoundPlayer` in `spectra/ui/sound.py`.**
Off Windows the import fails and every beep becomes a no-op, with no `try/except` scattered
through the code.

**D-011 — `AppMode` and `PhysioExercise` enum values are i18n keys.**
`AppMode.PHYSIO.value == "physio"` resolves `mode.physio` in the catalog, removing a parallel
enum-to-label map and guaranteeing that any new mode needs a translation.

**D-012 — `logging` instead of `print`; `ESC` as an emergency exit.**
The panel and the tests need to capture these messages. Patient interaction stays 100% gestural;
`ESC` exists only as a safety interrupt for the therapist.

**D-013 — "Guided draw" kept out of the menu until phase 3.**
Phase 1 is a pure code move, so the menu entry would be a dead button. The enum member
(`AppMode.GUIDED_DRAW`) already exists.

**D-014 — Outputs written to the data directory, never to the working directory.**
`pintura_*.png` goes to `<data_dir>/drawings` and the physiotherapy CSV to `<data_dir>/exports`.
This was the cause of the stray PNG in the repository root.

## Phase 2 — Gesture engine

**D-015 — Finger extension read from the mean of the two interphalangeal angles.**
A single angle (at PIP) is noisy when MediaPipe jitters one landmark; averaging PIP and DIP is
cheap and markedly more stable. Angles are invariant to rotation and scale, which the old
tip-versus-joint comparison was not. `tests/unit/test_features.py` keeps the old heuristic around
purely to demonstrate that it breaks on a tilted hand.

**D-016 — The thumb uses abduction, not an angle.**
A thumb's interphalangeal angle barely changes between an open hand and a fist (~173° vs ~143° on
synthetic hands, well inside the noise). Palm-normalised distance from the thumb tip to the index
MCP separates the two postures by roughly a factor of three.

**D-017 — All ratios normalised by palm size (wrist to middle MCP).**
Makes every threshold independent of how far the patient sits from the camera, so a single
calibration survives a change of chair.

**D-018 — Hysteresis thresholds derived as fractions of the patient's own open/closed range.**
`on = closed + 0.60 * range`, `off = closed + 0.40 * range`. The 20% dead band is wider than the
observed landmark jitter while still feeling responsive. A finger whose two postures are too
similar falls back to the defaults and is listed in `uncalibrated`, so the therapist is told
rather than silently given bad thresholds.

**D-019 — Temporal smoothing by majority vote over 3 frames, not an average.**
Finger states are boolean; a majority filter removes isolated dropouts without the half-open
states an average would produce, and 3 frames is ~100 ms at 30 fps, below the perception
threshold.

**D-020 — Hold time of 400 ms by default, 600 ms for the simplified profile.**
Inside the 300-500 ms band the brief allows. The simplified profile targets low dexterity, where
postures are reached more slowly, so it gets the longer window.

**D-021 — Pause is a toggle that must be re-armed.**
After a pause the gesture must be released before it can resume, otherwise holding a fist would
oscillate between paused and idle. Losing the hand never resumes a paused session: a patient who
drops out of frame must deliberately come back.

**D-022 — Overlapping triggers resolved by specificity, not declaration order.**
In the simplified profile "index only" and "pinch" can match the same frame; the pinch trigger is
more constrained and wins. Order-independence keeps serialised custom profiles deterministic.

**D-023 — The deliberate 1/2/3-finger ladder is exempt from the ambiguity warning.**
Those patterns differ by one finger *by design*. Warning about the shipped default would train the
therapist to ignore warnings. The exemption is an explicit allow-list (`PROGRESSIVE_PATTERNS`), so
any other one-finger-apart pair still warns.

**D-024 — A float tolerance (`HOLD_EPSILON`) on hold comparisons.**
Frame timestamps are floats; `0.76 - 0.36` is `0.3999999999999999`, which would silently drop an
exactly-held gesture. One microsecond of slack is far below any perceivable duration.

**D-025 — `SessionGuard` lives in `spectra/core`, not in `spectra/gestures`.**
The session time limit and rest prompts are not gesture concerns: the guided-draw and
physiotherapy modes need them too, and the reports will record them.

## Phase 3 — Modes

**D-026 — Educational modes kept, not merged into guided drawing.**
They serve a different purpose: the colour and counting quizzes are *engagement and
proprioception* exercises with their own scoring, while guided drawing is an *assessment*. They
also still use finger combinations directly, which is exactly the skill they train. Merging them
would have produced one mode with two unrelated scoring models.

**D-027 — The free-draw palette moved from finger combinations to a dwell colour menu.**
Ten memorised combinations are a poor fit for a patient with limited dexterity, and every
transitional posture painted a stripe. Gestures now choose the *tool* (paint, erase, menu, pause)
and colour is chosen by dwelling on a swatch. The combination table survives in `palette.py`
because the colour quiz still teaches it.

**D-028 — Shapes defined in normalised coordinates, scored in normalised units.**
A score must be comparable between sessions even if the patient's laptop is plugged into a
different monitor. `test_scoring_is_resolution_independent` pins this down.

**D-029 — Difficulty changes both the corridor and the shape.**
Only narrowing the tolerance would make "hard" a test of camera noise. Harder levels also tilt
the line, shrink the circle, add zigzag peaks and add spiral turns, so the movement itself gets
harder, not just the measurement.

**D-030 — Score weights: 60% completion, 40% accuracy.**
Clinically, finishing the movement matters more than tracing it beautifully; a patient who
completes a rough circle has achieved more range than one who draws 20% of a perfect one. The
accuracy component reaches zero at twice the corridor width.

**D-031 — Dwell buttons go inert outside `IDLE` and `POINTING`.**
While painting, the pointer sweeps the whole screen and would otherwise trip "clear" or "quit"
on its way past. Their dwell timers are also reset, so crossing a button never leaves it
part-charged.

**D-032 — `BaseMode.observe()` runs the gesture engine and the recorder exactly once per frame.**
The previous design had each mode call `pointer()` and `read_states()` separately, which meant
the hysteresis estimator could be advanced twice in one frame. A single `FrameContext` makes
double-stepping impossible.

**D-033 — Metric recording is opt-in per mode (`records_metrics`).**
Only free draw, guided draw and physiotherapy feed the recorder; the menu and the quizzes do not,
so time spent navigating does not dilute the session's tracking-quality statistics.

**D-034 — The recorder stores derived features at 10 Hz, never frames.**
10 Hz is well above the bandwidth of voluntary hand movement (a fast tremor is 4-12 Hz, and the
Nyquist limit still holds at 10 Hz for the amplitude envelope we measure) while cutting stored
volume by two thirds versus 30 fps. Tracking quality is counted over *every* frame so the
presence ratio stays honest.

**D-035 — Each exercise run records a time window (`started_at`, `ended_at`).**
The metrics engine slices the shared session recording by window, so modes never need to know
anything about ROM, tremor or fatigue.

## Phase 4 — Metrics engine

**D-036 — The fingertip trajectory is recorded at the full frame rate, separately from the
10 Hz feature samples.**
Tremor lives at 4-12 Hz; by Nyquist a 10 Hz recording cannot resolve it at all. The trajectory
costs three floats per frame, so keeping it at 30 Hz is nearly free while the heavy feature
samples stay throttled. `test_a_ten_hertz_recording_cannot_resolve_the_band` pins the reasoning
into the test suite.

**D-037 — `band_measurable` is returned instead of a plausible wrong number.**
When the window is too short or the rate too low, the tremor ratio is reported as not
measurable rather than computed from aliased data. A clinical tool that silently returns
nonsense is worse than one that says it does not know.

**D-038 — Wrist measures are two explicit palm proxies, never a "wrist angle".**
MediaPipe has no forearm landmarks, so there is no anatomical reference for a wrist angle.
`palm_rotation` and `palm_openness` are named after what they actually measure, and
`WristEstimate.estimate` is `True` at the type level so no report can forget the label.

**D-039 — Fatigue compares thirds, using a symmetric bounded decline.**
First-versus-last repetition is too noisy; thirds average that out. `(start - end) /
max(start, end)` stays in `[-1, 1]` and remains defined when the patient started from no
measurable movement, which plain `(start - end) / start` does not.

**D-040 — The thumb is excluded from the fatigue amplitude.**
It is the noisiest of the five signals (D-016), so including it would mostly add variance.

**D-041 — Metrics are computed even for unreliable sessions; the report does the flagging.**
Returning `None` would force every consumer to special-case it. Instead `TrackingQuality`
travels with the numbers and carries a `warning_key` explaining exactly which threshold failed.

**D-042 — `TrackingQuality` lives in its own module, imported by the recorder.**
It is the gate on every other metric and is consumed by reports and the panel; burying it in
`recorder.py` would have made that dependency invisible.

**D-043 — NumPy is used for the tremor spectrum only.**
Everything else in `metrics/` is plain Python, so the modules stay importable and fast. NumPy is
already a transitive dependency of OpenCV, so this adds nothing to the install.

## Phase 5 — Storage and LGPD

**D-044 — SQLite through stdlib `sqlite3`, no ORM.**
The schema is nine tables and the queries are simple. An ORM would add a dependency, a
migration tool and a layer between the code and the encryption boundary, which is exactly the
place that must stay obvious.

**D-045 — Encryption with Fernet (`cryptography`), not raw AES.**
Fernet is authenticated (AES-128-CBC + HMAC-SHA256) and versioned, so a tampered field fails
loudly instead of decrypting to garbage. Hand-rolling AES-GCM would be more code for no gain.

**D-046 — scrypt from `hashlib`, not argon2.**
Memory-hard and in the standard library, so the patient app gains no dependency. `argon2-cffi`
is marginally stronger but needs a compiled wheel on Windows for a 4-digit PIN whose real
protection is the lockout, not the KDF. `maxmem` is passed explicitly because OpenSSL's default
32 MB cap rejects n=2^15.

**D-047 — CPF lookup by HMAC, not by plain SHA-256.**
The CPF space is 10^11, which a plain digest makes brute-forceable in minutes. The HMAC key is
derived from the master key with a domain separator so it cannot decrypt anything.

**D-048 — The patient name is not encrypted; the CPF is.**
The therapist must search patients by name, and encrypting it would mean decrypting every row
per search. The risk is accepted explicitly, documented in `docs/lgpd.md`, and mitigated by demo
mode. The CPF, which is the actual national identifier, gets the full treatment.

**D-049 — The consent gate is a database invariant, not a UI check.**
`start_session` raises `ConsentRequiredError` when no active consent exists. A UI-level check
could be bypassed by a future script, a test fixture or the panel; this one cannot.

**D-050 — Anonymisation is offered alongside deletion, and preferred.**
A participant who withdraws usually wants to stop being identifiable, not to destroy the study.
`anonymize_patient` strips every identifier and keeps the measurements; `delete_patient`
cascades everything away. Both are available and both are audited.

**D-051 — The audit log stores UUIDs only.**
A log holding names or CPFs would be a second, unencrypted copy of the sensitive data — a
common way for an audit control to become the breach.

**D-052 — Keyring first, 0600 file fallback second, with a warning.**
Windows Credential Manager is the intended store. Headless Linux and CI have no backend, and
failing hard there would make the test suite unrunnable. The fallback logs a warning and its
weaker guarantee is documented.

**D-053 — Backups are zipped then encrypted, not encrypted zips.**
`zipfile` can only *read* encrypted archives and its legacy ZipCrypto is broken. Sealing the
whole archive with Fernet, keyed by scrypt from a password, is both stronger and simpler.

**D-054 — `SecurityError` and persistence errors share a `StorageError` root.**
A caller guarding "save this patient" should not need to know whether the refusal came from CPF
validation or from the database.

**D-055 — Exports are named `patient_<uuid>.json`.**
Filenames leak: they appear in sync clients, backup indexes and shoulder-surfing. The UUID is
the only identifier that ever reaches a filename.

## Phase 7 — Reports (built before the panel)

**D-056 — Reports implemented before the therapist panel, inverting the stated order.**
The panel's "generate and download PDF/TXT" button is one of its features; building it first
would have required a stub that then had to be rewritten. The phase numbering in the commit
history reflects the dependency, not the brief's ordering.

**D-057 — One `SessionReport` object, two renderers.**
PDF and TXT read the same frozen dataclass, so they can never disagree about a value. The
object is pure data, which means the *content* of a report is unit tested without generating a
single file.

**D-058 — Masking happens in the builder, never in the renderers.**
Demo mode and CPF masking are applied when the object is constructed. A renderer cannot leak a
name it was never given, so adding a third output format later cannot reintroduce the leak.

**D-059 — ReportLab for the PDF.**
Pure Python, no system dependency (unlike WeasyPrint's GTK stack on Windows), and its Platypus
flowables handle the table-heavy layout a clinical report needs. Rejected: fpdf2 (weaker
tables), HTML-to-PDF (extra runtime to install on the patient's machine).

**D-060 — Matplotlib on the `Agg` backend, returning PNG bytes.**
`Agg` needs no display server, so charts render headless and in CI. Returning bytes rather than
writing files means no patient-derived image is ever left in a temporary directory.

**D-061 — Chart labels are ASCII-folded; PDF body text is not.**
Matplotlib's default font stack is unpredictable across machines, so folding labels avoids
tofu boxes in the thesis. ReportLab's Helvetica renders pt-BR accents correctly, so the body
keeps proper spelling.

**D-062 — `estimate` is a field on `MetricRow`, not a naming convention.**
Every renderer reads the same flag, so a metric cannot be shown without its "estimativa" label
by forgetting to type it. The wrist rows and all 2D-derived angles set it.

**D-063 — Report filenames are `sessao_<session-uuid>`, and the PDF subject is the UUID.**
Filenames and document metadata both leak into sync clients and file browsers; neither ever
carries a name or a CPF.

**D-064 — An unmeasurable metric renders as an em dash, not as zero.**
`band_measurable=False` becomes "—" so a therapist is never shown a 0% tremor that actually
means "could not be measured".

**D-065 — Only three values are compared between sessions.**
Mean ROM, best guided-drawing score and amplitude decline. Comparing every metric would produce
a wall of numbers where the regression that matters is invisible.

## Phase 6 — Therapist panel and session flow

**D-066 — Streamlit, with every operation behind `SpectraService`.**
Streamlit gives a usable clinical panel in one file with no front-end build, no server to
deploy and no JavaScript — the right trade for a thesis. Its weakness is that the script
itself is hard to test, so the panel holds **no** business logic: validation, consent, session
lifecycle, reports and backup all live in `spectra/service.py`, which is unit tested, and
`therapist_panel/` is excluded from coverage. Rejected: Flask/FastAPI plus a template layer
(more code, more surface), Tkinter (dated, poor charting), Electron (absurd for this).

**D-067 — `PatientForm` validates before anything reaches the database.**
Validation returns a list of i18n keys rather than raising on the first problem, so the panel
can show every error at once instead of making the therapist fix them one at a time.

**D-068 — The consent text lives in `spectra/locales/tcle_pt_BR.md`, not in code.**
Changing the wording becomes a reviewable diff, and the stored SHA-256 then genuinely proves
which version a participant saw. The file carries explicit `<<DEFINIR: ...>>` placeholders for
the researcher and advisor contacts, so an unfilled term is visible rather than silently wrong.

**D-069 — The gesture keypad is for the PIN only; CPF login is by keyboard.**
Four digits by dwell takes about 5 seconds at 1.2 s per key; eleven digits would take over a
minute and is a usability failure. The therapist opens the session from the panel and the
patient confirms with the PIN; the standalone CPF + PIN login exists for the keyboard path.

**D-070 — The PIN is masked on screen as it is typed.**
The patient app runs on a shared screen, often with the therapist and others present.

**D-071 — `analyze_session` / `analyze_run` sit in `metrics/`, not in `service.py`.**
They are the seam between pure analysis and persistence. Keeping them as functions of plain
data means the entire metric pipeline is tested without a database and without a camera.

**D-072 — Metric payloads are stored as JSON blobs.**
A new metric needs no migration, and the reports already read payloads defensively with
`.get(...)`. The cost is no SQL aggregation over metrics; with one patient cohort that is not a
real constraint.

**D-073 — Sample persistence is capped and can be switched off.**
`MAX_SAMPLES_PERSISTED` bounds a runaway session, and `persist_samples=False` lets a therapist
keep only the summaries. Summaries are what the reports and the evolution charts actually read.

**D-074 — Demo mode is session state in the panel, not a stored setting.**
It is a presentation toggle, not a property of the data. It flows into the report builder per
call, so one click masks the screen and the next generated PDF alike.

**D-075 — Two `.bat` launchers under `scripts/`, which activate the venv and pause on error.**
A double-clicked `.bat` that fails closes instantly and shows nothing; pausing on error is the
difference between a usable launcher and an unexplained flash.

## Phase 8 — Documentation and polish

**D-076 — The patient app can attach to a panel-created session (`--session <uuid>`).**
Without it the panel's instruction would have been a lie: it tells the therapist to run that
exact command. Attaching loads the session's hand and gesture profile and the patient's
calibration, requires the PIN, and on exit computes and persists every metric.

**D-077 — Nothing is recorded before the PIN is accepted.**
The recorder only runs once `PinEntryMode` hands control over, so a stranger standing in front
of the camera cannot contribute frames to a patient's session.

**D-078 — Standalone mode persists nothing.**
`python -m spectra` with no session is free exploration: no database, no consent needed, no
stored data. It is what gets demonstrated, and what the author uses while developing.

**D-079 — The recorder and the session guard are owned by the shell, not by the modes.**
They are passed into `ModeContext`, so moving between painting and physiotherapy neither
restarts the recording nor resets the session clock.

**D-080 — Dependency pins corrected to the versions actually installed and tested.**
The initial pins were written from memory and were wrong (`numpy<2.2` while 2.5.3 was in use).
Only MediaPipe and OpenCV remain unverified pins, since the headless suite does not install
them; that is stated in the file itself rather than left implicit.

**D-081 — `mkdocs.yaml` added but the build is not wired into CI.**
The author's other repositories use MkDocs, and the four documents are already written in
Markdown. There is no CI in this repository to publish it from, so it stays a local
convenience.

**D-082 — Screenshots are placeholders with explicit paths.**
`docs/images/` exists with the exact filenames the README expects, so adding the real captures
before submission is a drop-in, not an edit of the README.

**D-083 — The camera, the MediaPipe call and the Streamlit script are not unit tested.**
All three are thin wrappers over a third-party API; testing them would test the mock. They are
excluded in `.coveragerc` so the figure reflects what is actually verified: **92.6% overall**,
with the pure-logic packages (`gestures`, `metrics`, `storage`, `guided`, `reports`) between
94% and 100%.
