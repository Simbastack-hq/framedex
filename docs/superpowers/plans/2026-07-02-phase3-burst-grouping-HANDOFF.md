# Handoff — framedex trust/XMP/burst work (Phase 3 + pending Codex re-reviews)

Read this first in a fresh session. It captures the whole 3-phase effort, exactly
where it stands, and a ready-to-execute Phase 3 plan. Source of truth for the
design is the committed PRD:
[docs/superpowers/specs/2026-07-02-trust-xmp-burst-prd.md](../specs/2026-07-02-trust-xmp-burst-prd.md).

## Where we are

| Phase | State | Branch | PR |
|---|---|---|---|
| 1 — Trust hardening | **Done, CI green** | `feature/phase1-trust-hardening` | #21 (open, not merged) |
| 2 — `fdx-xmp` Lightroom export | **Implemented, 210 tests green, ruff+mypy clean** | `feature/phase2-fdx-xmp` (stacked on P1) | opening now |
| 3 — Burst + RAW/JPEG grouping | **Not started** | `feature/phase3-burst-grouping` (stack on P2) | — |

Stacked branches: P2 branched off P1, P3 branches off P2. Each PR targets the
prior branch; GitHub auto-retargets to `main` as they merge in order. Merge order
is 1 → 2 → 3. **Nothing is merged yet** — the user asked for PRs created, not
merged.

Working ledger of every Codex finding + verdict: `REVIEW.md` at the worktree root
(not committed, intentionally). Per-phase plans in `docs/superpowers/plans/`.

## PENDING — run when Codex resets (usage limit hit ~this session)

The user's workflow requires Codex verification on every phase. Two Codex passes
were deferred by a rate limit and MUST still run:
1. **Phase 2 code review.** `git diff feature/phase1-trust-hardening...HEAD` on
   `feature/phase2-fdx-xmp`. An *interim* independent (non-Codex, opus) review was
   run in its place — its findings are triaged in `REVIEW.md` (Round 4). Re-run
   the real Codex pass and reconcile.
2. **Phase 3 plan review AND code review** (normal codex-ship gates).

Invoke Codex via the `codex:codex-rescue` subagent (Agent tool,
`subagent_type: "codex:codex-rescue"`), `--effort high`. If it still reports a
usage limit, either wait for reset or run an independent opus reviewer as a
stand-in and reconcile later (what we did for P2).

## Environment & commands (verified)

- No `.venv` in the worktree initially. CI installs only dependency groups:
  ```
  uv sync --locked --only-group dev --only-group test
  ```
- Run the gates exactly as CI does (heavy deps like cv2/insightface/osxphotos are
  NOT installed — tests are hermetic and must stay that way):
  ```
  uv run --no-sync pytest -q
  uv run --no-sync ruff check src/framedex tests
  uv run --no-sync ruff format --check src/framedex tests
  uv run --no-sync mypy src/framedex tests --ignore-missing-imports
  ```
- Running a module ad-hoc from source needs `PYTHONPATH=src` (the project isn't
  pip-installed in the CI env): `PYTHONPATH=src uv run --no-sync python -c "..."`.

## Conventions & gotchas (learned the hard way)

- **TDD is mandatory** (superpowers `test-driven-development`): failing test first,
  watch it fail *for the right reason*, then implement. If a RED fails on a setup
  error (e.g. thin mock metadata → KeyError), fix the test until it fails on the
  real assertion.
- Any new CLI flag goes in **BOTH** parsers (`index_videos.main` **and**
  `photos_indexer.main`) and threads through `ProcessOptions`
  (`pipeline.py:488`). `--frame-sampling` is the reference pattern
  (`index_videos.py` ~1304, `photos_indexer.py` ~252).
- Tunables are named module constants with a one-line comment, not flags, unless a
  real user decision exists.
- Tests are hermetic: mock subprocess/cv2/insightface; pure functions over numpy
  arrays / tmp sqlite. `mypy` flags `module.imported_name` access in tests — patch
  via string form `monkeypatch.setattr("framedex.mod.subprocess.run", ...)` or
  import the name directly.
- Docs stay in sync in the SAME PR: README.md, SKILL.md, docs/*.md, CHANGELOG.md.
- Phase 1 gave us `pipeline.atomic_write_text(path, text)` — use it for ALL new
  file writes (stub sidecars, any index). The sidecar is the resume marker; write
  everything else for a unit before it.
- Face writes only happen when detection actually ran (Phase 1 fix) — not relevant
  to P3 unless you touch the face path.
- `claude -p` transport is locked to `--permission-mode dontAsk --allowedTools
  Read` (path-scoped Read is NOT honored by the CLI — verified live). Don't touch.

## Verified code facts for Phase 3 (checked against source)

- `images.find_images(root, exclude) -> list[Path]` (sorted). Discovery skips
  dot-files and `_`-prefixed folders.
- Per-file loop for folder mode is in `index_videos.main` — discovery ends
  ~`index_videos.py:1336`, the processing loop starts ~`:1409`. Grouping pre-pass
  slots between them. Resume filter today: `[v for v in vids if not has_sidecar(v)]`
  (`index_videos.py` ~1329/1335).
- `ProcessOptions` (`pipeline.py:488`) — add `group_photos: bool = True` after the
  existing defaulted fields.
- `frame_sampling._signatures` computes the Laplacian sharpness at
  `frame_sampling.py:148-149`:
  `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); float(cv2.Laplacian(gray, cv2.CV_64F).var())`.
  Factor a shared `laplacian_sharpness(path: Path) -> float` and call it from both
  `_signatures` and the new grouping representative-pick (don't duplicate the math).
- `render_preview(image, out_dir)` (`images.py:202`) writes a **fixed** filename
  `preview.jpg` (`:248`); `_extract_raw_preview` writes fixed `raw_preview.jpg`
  (`:190`). Scoring N burst members needs a per-member temp subdir OR an
  output-name parameter — pick one; do NOT render into a shared dir (they'd
  collide).
- `images.RAW_EXTENSIONS` includes `.dng`; `RENDERABLE_EXTENSIONS` includes
  jpg/jpeg/png/tif/tiff/heic/heif/webp (`images.py:41-62`). Pairing must use
  **`.jpg`/`.jpeg` only**, not all renderables.
- `master_index.py` builds records + counts; `fdx-query` `matches()` filters.
- exiftool is an external binary (not a py dep), already called via
  `subprocess.run(["exiftool", "-json", ...])` in `pipeline.get_gps` and
  `images.get_image_metadata`. Batch with one call over an argfile.
- `pipeline.sidecar_path(media)` appends the FULL extension
  (`clip.MOV.description.md`); `SIDECAR_SUFFIX = ".description.md"`.

---

## Phase 3 — implementation plan (burst + RAW/JPEG grouping)

Goal: one moment = one vision call = one primary sidecar. Folder mode only
(`fdx --media images|all`). Reduces per-archive vision cost; never increases it.
Traces to PRD §"Phase 3". One PR, stacked on P2. TDD throughout; grouping is pure
logic over `(path, timestamp, make, model)` tuples → fully hermetic.

### Task 1 — `laplacian_sharpness` shared helper
Factor the cv2 Laplacian-variance math out of `frame_sampling._signatures` into
`laplacian_sharpness(path: Path) -> float` (new tiny helper; cv2 lazy-imported).
`_signatures` calls it. Test: monkeypatch cv2 or use a real tiny array via a
mocked `cv2.imread` — assert a sharp array scores higher than a blurred one
(hermetic; keep cv2 mocked as elsewhere).

### Task 2 — grouping pre-pass (new module `src/framedex/grouping.py`)
Pure logic, no cv2/exiftool inside the tested functions — pass in the metadata.
- **Metadata input**: one batched `exiftool -json -n -DateTimeOriginal
  -SubSecDateTimeOriginal -SubSecTimeOriginal -Make -Model -@ <argfile>` call over
  the whole image list (argfile dodges ARG_MAX). Wrap the subprocess in a thin
  `read_group_metadata(paths) -> dict[path, GroupMeta]`; the *pure* grouping
  functions take the already-parsed metadata so tests never shell out.
- **Timestamp resolution**: prefer `SubSecDateTimeOriginal`; else
  `DateTimeOriginal` + `SubSecTimeOriginal` fractional; else whole seconds. Files
  with no usable `DateTimeOriginal` never join a group (indexed individually).
- **Pairing** (`pair_raw_jpeg`): same directory + same stem (case-insensitive),
  one file in `RAW_EXTENSIONS`, the sibling extension in **`{.jpg, .jpeg}` only**.
  RAW is the primary (edit target); the JPEG is the vision/preview source when
  present. Pairs collapse to their primary before burst detection.
- **Burst** (`detect_bursts`): within one directory, same camera (`Make`+`Model`),
  sorted by timestamp, chain successive gaps `<= BURST_GAP_SEC = 2.0`; a chain of
  `>= BURST_MIN_SIZE = 3` is a burst. (Module constants.)
- Group id: `b_<sha1-of-sorted-member-paths[:8]>`.
- Tests (hermetic golden logic): pairing matrix (stem case, RAW-only, JPEG-only,
  RAW+two-JPEGs → still one pair + extra ungrouped), burst chaining at exactly
  2.0s edges, min-size boundary (2 vs 3), missing dates never group, mixed cameras
  interleaved don't merge, group id stable & order-independent.

### Task 3 — representative pick
Deterministic, local, explainable. Render each member's preview into a
**per-member temp dir** (or via a new `render_preview(..., out_name=...)`
param — decide + do ONE), score with `laplacian_sharpness`, highest wins, tie →
earliest timestamp. **No multi-image "ask the model to pick" call** (would scale
vision cost with shooting style). Test: given member→score, the highest is chosen,
ties break to earliest.

### Task 4 — sidecar semantics + group-aware resume
- **Primary**: full pipeline as today, plus a `group:` frontmatter block
  (`kind: burst|raw_jpeg`, `id`, `primary: true`, `members: [...]`, `sharpness`).
- **Members**: **stub sidecar** — no vision, no faces. Own file/EXIF/GPS fields +
  the group block (`primary: false`, `primary_file: <name>`, own `sharpness`) +
  **copies the primary's assessment fields** (`rating`, `technical`, `lighting`,
  `scene_type`, `keywords`, …) so `fdx-query` keeps working unmodified. Body:
  `See <primary>.description.md (burst primary).`
- **Write order & resume (critical)**: group completeness is defined by the
  **primary sidecar alone**. Write members' stubs first, primary last. On resume,
  skip a group only when its *primary* has a sidecar; if the primary is missing,
  **reprocess the whole group and rewrite all member stubs** — per-file
  `has_sidecar` on members must NOT filter them out of an incomplete group (a
  crash between stubs and primary must not leave stubs carrying a dead run's
  assessment forever). Stubs are cheap (no vision/faces) and atomic (P1 helper).
- Tests: stub frontmatter golden; primary-last write order; incomplete-group
  (primary sidecar removed) rewrites all stubs; a complete group is skipped.

### Task 5 — master_index / query
- `master_index` rating counts + cull-pile listings count **primaries only**
  (stubs recognizable by `group.primary: false`); totals gain a
  `grouped: N files in M groups` line.
- `fdx-query` gains `--primary-only` (default OFF — matching a burst member by
  keyword should still find it).
- Tests: counts exclude stubs; `--primary-only` filters stubs.

### Task 6 — flag + wiring
- `--no-group` in BOTH parsers, threaded as `ProcessOptions.group_photos: bool =
  True`. `fdx-photos` parser accepts it as a documented **no-op** for now (Apple
  Photos burst integration via osxphotos `burst`/`burst_selected` is a follow-up,
  out of scope).
- Grouping runs after `find_images`, before the per-file loop, folder mode only.

### Task 7 — XMP interaction (Phase 2)
- A pair → one primary → one `<stem>.xmp` (the P2 collision rule becomes moot for
  newly-indexed pairs). Burst members' stubs carry the copied rating, so they get
  XMP too, plus keyword `burst-alternate` in `dc:subject`; the primary gets
  `burst-pick`. (Only relevant once both phases are merged; add a light test if
  fdx-xmp reads the group block, else defer.)

### Task 8 — docs
README pipeline + cost sections (vision calls = `#groups + #ungrouped <= #files`),
SKILL.md, `docs/tuning.md` (the group constants), CHANGELOG. Known-limitation:
EXIF-gap burst inference can merge two rapid distinct moments; `--no-group` opts
out.

### Non-goals (PRD)
Cross-directory grouping; perceptual-similarity grouping (dedupe territory);
auto-culling non-picks; multi-image vision calls; Apple Photos burst integration.

### Open decisions (defaults chosen — from the PRD; change only with reason)
1. Burst members **inherit the primary's rating** (optimistic) rather than
   auto-review/cull.
2. Stubs **copy** assessment fields (drift-on-reindex accepted; cheap to
   regenerate) vs pointer-only.
3. Gap/min-size stay **constants** (`BURST_GAP_SEC=2.0`, `BURST_MIN_SIZE=3`),
   `--no-group` is the only flag.

### Definition of done (Phase 3)
A real burst-heavy folder indexes with measurably fewer vision calls (target
40-60% on burst-heavy dirs); kill -9 mid-group → re-run completes with correct
stubs and one primary; `fdx-query` finds both picks and alternates; `--no-group`
restores per-file behavior. Full gate green (pytest/ruff/mypy). Codex-reviewed
(plan + code). PR opened, CI green.
