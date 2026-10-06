# LGPD, privacy and research ethics

How SPECTRA handles personal data, which protections are implemented, and — equally
important — where the limits are.

> **This document is not legal advice.** SPECTRA is an undergraduate research prototype,
> not a certified medical device and not a commercial product. Before collecting data from
> real participants, confirm the plan with the thesis advisor.

## 1. What is collected, and why

LGPD requires *purpose limitation* and *data minimisation*: collect only what serves a
declared purpose. Every field below has a clinical or measurement purpose; anything that
did not, was not added.

| Field | Purpose | Encrypted at rest |
|---|---|---|
| Name | Therapist must identify the patient | No (searchable; see §4) |
| CPF | Unique identification, patient login | **Yes**, plus a keyed hash for lookup |
| Birth date | Age is a covariate for range of motion | No |
| Sex | Clinical covariate | No |
| Phone, address | Contact and follow-up | **Yes** |
| Dominant hand, affected hand | Determines which hand a session treats | No |
| Diagnosis (CID), injury/surgery date | Clinical context for the evolution | No |
| Referring professional | Clinical traceability | No |
| Contraindications | **Patient safety**: limits which exercises may run | **Yes** |
| Clinical notes | Therapist's free text | **Yes** |
| Gesture profile, calibration | Adapts the interface to the patient's ability | No |
| 4-digit PIN | Patient authentication | Hashed (scrypt), never recoverable |

### What is deliberately **not** collected

- **No video and no image of the patient is ever stored.** Frames are processed in memory
  and discarded; only derived numbers (joint angles, distances, fingertip coordinates)
  are persisted. The painting PNG is the only image output and contains no image of a
  person.
- No audio.
- No biometric template. MediaPipe landmarks are geometric measurements of a hand pose at
  one instant and are not used, nor usable, for identification.
- No location data, no device identifiers, no analytics, no network telemetry. The
  application makes exactly one outbound request in its lifetime: downloading the
  MediaPipe model, if it is missing.

## 2. Consent (TCLE) — enforced, not documented

A consent record is **required before any session can be created**. This is a database
invariant, not a convention:

```python
db.start_session(Session(patient_id=patient.id))
# ConsentRequiredError: the patient has no active consent (TCLE) on record
```

Each consent stores the version, the acceptance timestamp, the signer and a **SHA-256 hash
of the exact consent text the patient saw**, so it can later be proven which wording was
agreed to. Consent is revocable; revoking it blocks new sessions immediately.

## 3. Encryption at rest

- **Algorithm**: Fernet (AES-128-CBC with HMAC-SHA256 authentication), from the
  `cryptography` package. Authenticated encryption, so a tampered field fails loudly
  instead of decrypting to garbage.
- **Key storage**: the OS credential store through `keyring` — Windows Credential Manager
  on the target platform. The key is never in the repository and never in the database.
- **Documented fallback**: where no keyring backend exists (headless Linux, CI), the key is
  written to `<data_dir>/secrets/master.key` with `0600` permissions and a warning is
  logged. **This is a weaker guarantee**: anyone who can read the data directory can read
  the key. It exists so development and testing work; it should not be the configuration
  used with real participants.
- `secrets/` is in `.gitignore`, as is every `*.key`.

## 4. CPF handling

The CPF is the highest-risk field in the database, so it gets four separate rules:

1. **Validated** — both check digits are verified, and the repeated-digit case
   (`111.111.111-11`) is rejected.
2. **Encrypted at rest** — the plaintext never touches disk.
3. **Searchable without being reversible** — lookup goes through an HMAC-SHA256 keyed with
   a key derived from the master key, held in `cpf_hash`. A plain SHA-256 would be useless
   here: the CPF space is only 10^11, which is brute-forceable in minutes. The lookup key
   is derived so that it cannot decrypt anything.
4. **Never leaves the database layer in the clear** — not in a filename (exports are named
   `patient_<uuid>.json`), not in a log line, not in an exception message
   (`InvalidCPFError` deliberately carries no value), and masked as `***.456.789-**` in
   every report.

The patient **name** is intentionally *not* encrypted: the therapist must be able to search
for a patient, and encrypting it would mean decrypting every row on every search. The risk
is accepted and documented; demo mode (§7) masks it on screen and in reports.

## 5. Authentication

| Account | Credential | Hash |
|---|---|---|
| Therapist | email + password | scrypt (n=2^15, r=8, p=1) |
| Patient | CPF + 4-digit PIN | scrypt, same parameters |

scrypt is memory-hard, so a stolen database cannot be attacked with GPUs the way a plain
SHA-256 could. It is in the Python standard library, so this adds no dependency.

**A 4-digit PIN is weak by construction** — 10,000 possibilities. The compensating control
is a **lockout after 5 failed attempts**, for 5 minutes, applied to both account types and
written to the audit log. The PIN is a convenience credential for a patient who must enter
it by gesture; it is not what protects the data. The encryption key and the lockout are.

## 6. Audit log

Every access, export and deletion of patient data is recorded: creation, update, read,
export, anonymisation, deletion, consent recorded/revoked, session started/finished,
report generated/exported, backup created, login success/failure, account locked.

**The audit log stores UUIDs only** — never a name, never a CPF — so the log is not a second
copy of the sensitive data.

## 7. Data subject rights

| Right | Command |
|---|---|
| Access / portability | `db.export_patient_file(patient_id, directory)` → JSON with every decrypted field, consents, sessions, metrics and samples |
| Erasure | `db.delete_patient(patient_id)` → cascades to sessions, runs, metrics and samples |
| Anonymisation | `db.anonymize_patient(patient_id)` → strips name, CPF, phone, address, birth date, notes, contraindications and PIN, **keeping the measurements** |

Anonymisation is the option to prefer when a participant withdraws but agreed that
anonymous measurements may be kept for the study: it honours the withdrawal without
destroying the research data. Erasure is available when they want everything gone.

### Demo mode

A toggle in the therapist panel replaces patient names with initials (`M.D.S.`) on screen
and in the PDF and TXT reports. It exists so the thesis can be presented with real data on
a projector without exposing anyone.

## 8. Retention

**No automatic deletion is implemented.** Retention is therefore a procedural control, and
the author is responsible for it:

- Keep identified data only while the study runs plus the period the advisor and, if
  applicable, the ethics committee require.
- At the end of the study, **anonymise** every participant rather than keeping identified
  records "just in case". One command, `anonymize_patient`, per participant.
- The consent text must state the retention period that was actually agreed.

## 9. Backups and the shared folder

The live SQLite database stays on **local disk**. It must never be placed inside a
cloud-synced folder: file sync and SQLite's locking corrupt each other, and a sync client
would also copy the plaintext-searchable columns to a third party.

What goes to the shared folder is a **backup**: a zip of the database and the reports,
sealed with Fernet using a key derived from a password with scrypt. A leaked backup file is
not a leaked patient record.

> **Requirement for the shared folder (`share_dir`).** It must have **restricted access** —
> named people only, no "anyone with the link", and access reviewed when the study ends.
> The link is referenced in the README as a placeholder and must never be committed.

Python's `zipfile` can only *read* encrypted archives, and its legacy ZipCrypto is broken,
which is why the archive is zipped first and encrypted afterwards.

## 10. Research ethics

The participants are the author's colleagues. That makes them **human research subjects**,
and their consent is not a formality:

- A colleague is in an **asymmetric relationship** with the researcher. Participation must
  be visibly voluntary, declining must carry no cost, and withdrawal must be possible at
  any time without explanation.
- A colleague is also easily re-identifiable from a small sample. Prefer anonymisation over
  retention, and use demo mode for any public presentation.
- The TCLE must state, in plain language: what is collected, that no video is recorded, the
  purpose, how long data is kept, who can access it, and how to withdraw.

> **The author must check with the thesis advisor whether approval from an ethics committee
> (CEP/CONEP, via Plataforma Brasil) is required before collecting data.** Research with
> human subjects in Brazil generally requires it; whether an undergraduate thesis with
> colleagues falls under an exception is a decision for the advisor and the institution, not
> for this document.

## 11. Known limitations

- The file-based key fallback (§3) is weaker than the OS credential store.
- Patient names are not encrypted (§4).
- No automatic retention enforcement (§8).
- No full-disk encryption is assumed; on Windows, enabling BitLocker is strongly
  recommended and is outside what the application can control.
- The audit log is append-only by convention, not by database permission: anyone with
  write access to the SQLite file can alter it.
- SPECTRA is **not** a certified medical device. Its measurements are estimates (see
  `docs/metrics.md`) and must not be used as the sole basis for a clinical decision.
