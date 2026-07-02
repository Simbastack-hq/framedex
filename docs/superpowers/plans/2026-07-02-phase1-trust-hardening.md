# Phase 1 implementation plan — Trust hardening

Traces to [the PRD](../specs/2026-07-02-trust-xmp-burst-prd.md) §1. One PR.
TDD: each task writes the failing test first, then the code. All tests hermetic
(no ffmpeg, no network, no insightface) — pure functions + tmp sqlite + mocked
subprocess.

**Corrections folded in from the code-verification pass (not in the raw PRD):**
- §1.7 touches **query.py + master_index.py only** — `trip_summary.py` never
  reads the `path` field (verified; issue #14 also names only these two). The
  real failure is a **TypeError crash on a non-string `path`**, plus the silent
  fallback; fix both.
- §1.5 drops `--max-turns 8` — **the flag does not exist** in the installed
  `claude` CLI (`claude --help` shows `--max-budget-usd`, no `--max-turns`).
  Tool lockdown (`dontAsk` + scoped `Read`) is the real safety mechanism; the
  turn cap was belt-and-suspenders and would hard-error every vision call.
- §1.4 fix normalizes **both** the filter comparison (150-153) and the sort key
  (155). Primary crash is the sort key; the filter only crashes if the user
  passes an aware `--since/--until`. `_parse_date` returns naive-or-aware
  depending on input, so normalize on read regardless.

---

## Task 1 — `atomic_write_text` helper + wire into 4 writers (§1.1)

- **Test first** (`tests/test_pipeline.py`): monkeypatch `os.replace` to raise
  after the temp file is written → assert (a) target absent/unchanged, (b) a
  `.tmp` file remains, (c) no partial target. Success path: correct content,
  temp cleaned up.
- **Code** (`pipeline.py`): add
  ```python
  def atomic_write_text(path: Path, text: str) -> None:
      tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
      tmp.write_text(text)
      os.replace(tmp, path)
  ```
  Same-dir temp (os.replace atomic only within a fs), dot-prefixed (discovery
  skips hidden), pid-suffixed (concurrent runs can't collide).
- Swap `write_text` → `atomic_write_text` at: `serialize_sidecar` (pipeline.py
  :555), `master_index.py:103` (_INDEX.json), `master_index.py:253` (_INDEX.md),
  `trip_summary.py:321` (_folder-summary.md).

## Task 2 — faces-before-sidecar + zero-face write (§1.2)

The sidecar is the resume marker; it must be written **last**.
- **images.py** (`process_one_image`, ~543-548): compute
  `sidecar = sidecar_path_override or pipeline.sidecar_path(image)` first, then
  `if ctx.face_conn is not None: face_db.write_faces(...)`, then
  `serialize_sidecar(sidecar, ...)`. Drop the `and detected_faces` gate — detection
  was *attempted* whenever `face_conn` is set (preview exists past the early
  no_preview return), so a zero-face re-run must still reach the DELETE.
- **index_videos.py** (`process_one_video`, ~1057-1073): same reorder. Compute
  the sidecar path, `if ctx.face_conn is not None and frames:` write faces (note:
  detection here is gated on `frames` too, line 1037 — mirror that exact
  predicate), then `write_sidecar(..., sidecar_path_override=sidecar)`.
- **Test**: a fake ctx with a real tmp sqlite conn + monkeypatched
  `detect_faces_in_frames`. First run detects 1 face → row present. Second run
  (same file, detection now returns `[]`) → assert the stale row is gone. Assert
  write order via a spy that records the sequence of (write_faces, serialize).

## Task 3 — face-DB idempotency (§1.3) + first `tests/test_face_db.py`

- **`write_faces`**: 
  1. `SELECT DISTINCT cluster_id FROM faces WHERE video_path=? OR sidecar_path=?`
     → `old_ids`.
  2. `DELETE FROM faces WHERE video_path=? OR sidecar_path=?`.
  3. INSERT each face; upsert the cluster row **without** `member_count+1`
     (INSERT … ON CONFLICT DO UPDATE SET last_seen_at=excluded.last_seen_at).
  4. `UPDATE clusters SET member_count=(SELECT COUNT(*) FROM faces WHERE
     faces.cluster_id=clusters.cluster_id) WHERE cluster_id IN (old ∪ new)`.
  5. `DELETE FROM clusters WHERE member_count=0 AND person_name IS NULL`
     (named clusters with 0 members survive — a label is user work).
- **Schema**: add `CREATE INDEX IF NOT EXISTS idx_faces_sidecar ON
  faces(sidecar_path)` to `SCHEMA_SQL` (migration-free; runs on every open_db).
- **Test** (hermetic, hand-built `DetectedFace`, no insightface): write→rewrite
  same file → face-row count + member_count stable; changed sidecar_path w/ same
  video_path (and vice-versa) still dedupes; orphan unnamed cluster reaped;
  named zero-member cluster retained.

## Task 4 — fdx-photos tz normalization (§1.4)

- **photos.py**: add `_as_aware(dt) -> dt.astimezone() if dt.tzinfo is None else
  dt`; import `timezone`. Apply to both sides of the `since`/`until` comparisons
  (150-153) and the sort key: `key=lambda a: (_as_aware(a.date) if a.date else
  datetime.min.replace(tzinfo=timezone.utc), a.uuid)`.
- **Test** (`tests/test_photos.py`): aware asset + naive `--since`; naive asset +
  aware bound; mixed aware/undated sort — all currently TypeError, must pass.

## Task 5 — lock down `claude -p` transport (§1.5)

- **pipeline.py** (`describe_frames_cli`): replace `--permission-mode
  bypassPermissions` with `--permission-mode dontAsk` + `--allowedTools Read`.
  No `--max-turns` (absent from CLI). Keep the `is_permission_denied` guard — a
  denied Read surfaces as a permission-denied text/rc and becomes a retryable
  `vision_error`.
- **Live smoke test — DONE, findings folded in.** Ran real `claude -p` calls:
  - `dontAsk` + `Read` → frame Read **allowed**, benign Bash `touch` **denied**
    (marker file never created). ✓
  - `dontAsk` + `Read(<dir>/**)` (and `Read(<dir>)`, `Read(<dir>/*)`, `**/*`) →
    **all reads denied** — the CLI does not honor path-scoped Read allowlists.
  - `default` + `Read` also works, but `dontAsk` is the explicit "deny, never
    prompt" mode (doesn't rely on headless-prompt-becomes-deny), so we keep it.
  Conclusion: ship **unscoped `--allowedTools Read`** (the PRD's pre-approved
  fallback). Residual: the agent can read other local files but cannot execute
  or write, and its only output is the local sidecar — a far smaller surface
  than the prior Bash access. Documented in README.
- **Test** (`tests/test_pipeline.py`): pure argv assertion (subprocess already
  mocked) — built command contains `--permission-mode dontAsk`, a `Read`-scoped
  `--allowedTools`, and does **not** contain `bypassPermissions`.

## Task 6 — kill the silent-defaults sidecar (§1.6) — guard-only (Codex: cut option B)

Codex plan review confirmed: ship the minimal guard fix, **not** the
`VisionResponse` dataclass. B only additionally fixes an inverse false-positive
(legit response starting with `[`) that causes a harmless *retry*, not state
corruption — not worth ~40 lines across 3 transports, and it would break
`parse_vision_response`'s signature + existing test.
- The load-bearing bug: the persist guard `if not structured and
  description.startswith("[")` writes a **defaults-only sidecar** (rating:review,
  all-unclear) whenever the model returns text with no parseable YAML fence that
  doesn't start with `[`. That file is then never revisited.
- **Fix**: change both guards (images.py:527, index_videos.py:1053) to skip
  whenever `not structured` — a parsed-but-empty response is as unusable as a
  transport error and must retry. Print a loud, distinguishing stderr line:
  transport error (`description` starts with `[`) → `vision call failed: …`;
  else → `vision response had no parsable YAML block — will retry next run`.
  Return `skipped_reason="vision_error"` in both cases.
- **Test**: response with prose but no ```yaml fence → no sidecar written,
  `vision_error` returned (the regression that catches the silent bug). Keep the
  existing `parse_vision_response` sentinel test passing (signature unchanged).

## Task 7 — fail-loud malformed `path` (§1.7, closes #14)

Scope: **query.py + master_index.py** only.
- Guard where `path` is read: if `path` is missing, **not a str, or
  blank/whitespace-only** (Codex), print a stderr warning naming the sidecar +
  reason, **skip** the record, and increment a counter. End of run prints
  `skipped N sidecars with unusable path`.
  - query.py: guard the resolve at 259-261 and the emit at 277 (today: silent
    fallback to `_sidecar_path`; non-str → TypeError).
  - master_index.py: guard 82-84 / 89 (today: `""` → trip `(unknown)`; non-str →
    TypeError).
- Warn-and-skip (issue #14's decided approach) so one bad sidecar can't kill a
  whole query/index run.
- **Test**: sidecar dicts with missing `path`, `path: 123`, `path: null` →
  skipped + counted + stderr line; valid records still emitted.

## Task 8 — docs + CHANGELOG (repo convention: docs in the same PR)

- **README** reliability/backend section: atomic-resumable writes; the
  `claude -p` lockdown (`dontAsk` + scoped `Read`) and the **minimum `claude`
  CLI version** that supports `--permission-mode dontAsk`; note that stale
  `.<name>.<pid>.tmp` files from a killed run are inert and safe to delete.
- **docs/troubleshooting.md**: retry semantics — a vision response with no
  parseable YAML fence is retried on the next run (loud stderr line), not
  silently written.
- **CHANGELOG**: one entry summarizing the seven trust fixes.

---

## Verify before done
- `ruff check` + `ruff format --check` + `mypy` (per pyproject).
- Full `pytest` (existing + new suites).
- The §1.5 live `claude -p` smoke check, recorded in the PR body.

## Not in this PR (deferred)
- Sidecar schema change (#4). fdx-summary backend rework. Nominatim
  timeout-caching nit.
- `VisionResponse` typed result (Codex + I cut it — guard fix suffices).
- fsync durability on atomic writes (threat model is Ctrl-C/crash, not
  power-loss; a re-run heals).
- Active stale-temp "notice" from the discovery walkers — documented as
  safe-to-delete instead.
