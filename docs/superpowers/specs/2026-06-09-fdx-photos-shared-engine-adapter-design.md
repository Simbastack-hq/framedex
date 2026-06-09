# Design: Rework `fdx-photos` into a shared-engine source adapter

- **Date:** 2026-06-09
- **Status:** Approved (design), pending implementation plan
- **Author:** NJ (with Claude)
- **Repo:** Simbastack-hq/framedex
- **Builds on:** [2026-06-08 fdx images photo-indexing design](2026-06-08-fdx-images-photo-indexing-design.md) (PR #17)

## Summary

`fdx-photos` (the Apple Photos indexer) is today a ~530-line command that
**re-implements** the whisper / vision-backend / face-DB setup that `fdx`
already has, and indexes **videos only**. This rework turns it into a **thin
source adapter on the same engine** `fdx` uses, and adds
**`--media images|videos|all`** so it can also index Apple Photos **stills** —
with Photos-side metadata (albums, Apple's named persons, keywords, canonical
date, GPS) threaded into the same sidecar schema.

The smell being fixed is **not** "two commands" — it is "two commands with
**duplicated engines**". `fdx-photos` stays a distinct command (it needs ~6
source-only flags meaningless to a folder scan: `--library`, `--output`,
`--download`, `--album/--person/--keyword`, `--since/--until`, `--uuid`). It
just stops carrying its own copy of the engine.

Mental model: **framedex indexes media; the *source* is an adapter.** Two
sources today — a folder tree (`fdx`) and an Apple Photos library
(`fdx-photos`). The per-asset pipeline (probe/EXIF → geocode → vision → faces →
sidecar) and the `--media` selector are **shared**. A source only changes *how
assets are enumerated* and *where sidecars are written*.

## Goals

- Kill the engine duplication: one place for backend wiring, face-DB setup,
  whisper loading, cost announcement, and per-result reporting. Both `fdx` and
  `fdx-photos` consume it.
- `fdx-photos --media images|videos|all` indexes Apple Photos stills as well as
  videos, routing each asset to the right per-asset pipeline.
- Apple Photos stills get **both** the camera/EXIF block (from the real
  HEIC/JPEG/RAW on disk) **and** the Photos-side metadata (persons/albums/
  keywords/uuid/canonical date).
- DX consistency: both commands speak `--media`, write the same sidecar schema,
  and are queried by the same `fdx-query` / `fdx-master` / `fdx-summary`.
- Phase 1 is a **pure refactor** — the video sidecar output and `fdx` behavior
  stay byte-identical (golden-test enforced).

## Non-goals (YAGNI)

- A fully shared `run_index()` loop. Enumeration + per-asset override
  construction legitimately differ between a folder walk and a Photos query;
  forcing them into one loop is the over-abstraction we are declining. We share
  **setup + reporting helpers**, not the loop.
- Renaming `index_videos.py` (it backs the `fdx` entry point).
- Changing the search/query model, the vision prompt, or the sidecar schema
  beyond what PR #17 already established.
- An interactive confirmation prompt or a hard `--yes` gate for large runs (see
  Scale guardrail — we chose warn-and-proceed).

## Decisions (resolved in brainstorm)

1. **Scale guardrail: warn loudly, then proceed.** Runs are incremental,
   resumable, and mostly local; the user can watch sidecars appear and Ctrl-C /
   re-run any time. Above a threshold on the *post-resume/filter* todo count,
   and only when the run is **unfiltered**, print a loud heads-up (count + est
   API cost when `--backend api` + "Ctrl-C anytime; re-run resumes"). No hard
   gate, no prompt, no `--yes`. `--max-files` and any narrowing filter suppress
   the warning (they signal a deliberately scoped run).
2. **`fdx-photos --media` default = `all`.** Matches the folder `fdx` default
   and the "one mental model" goal. The warn-loudly guardrail is the backstop
   for a bare whole-library run.
3. **`[photos]` extra = osxphotos only.** Mirrors the folder side: `[video]`
   pulls whisperx/torch, `[images]` pulls Pillow/pillow-heif, `[photos]` pulls
   the source adapter (osxphotos). `fdx-photos` video → `[photos,video]`;
   stills → `[photos,images]`; everything → `[all]`. Lazy-import errors guide
   the user to the missing extra. Cleanest long-term; pre-1.0 churn acceptable.
4. **One PR, two commits.** Commit 1 = behavior-preserving dedupe (Phase 1);
   commit 2 = the `--media` feature (Phase 2). Reviewer can confirm the refactor
   is isolated.

## Architecture

### Module layout (after)

```
src/framedex/
  pipeline.py      # unchanged role — narrow media-agnostic primitives:
                   #   GPS+geocode, vision backends (cli/api/local),
                   #   parse_vision_response, dataclasses, Nominatim, serializer
  runner.py        # NEW — shared CLI run-orchestration (no heavy imports):
                   #   resolve_vision_model, announce_cost, wire_vision_backend,
                   #   setup_face_db, RunTally, record_result
  index_videos.py  # `fdx` entry + video-only steps (frames, audio, whisperx,
                   #   .video-context.md biasing) + setup_whisper (stays here)
  images.py        # still-only steps: render_preview, EXIF, process_one_image
  photos.py        # Apple Photos osxphotos adapter (enumerate_assets, projection,
                   #   materialize, override builders, mirror path)
  photos_indexer.py# `fdx-photos` entry — thin adapter over the shared engine
```

### Why `runner.py` (not `pipeline.py`, not `index_videos.py`)

- `pipeline.py` is deliberately narrow (PR #17 refinement #9: pure primitives
  only — no argparse, no printing, no CLI orchestration). The new helpers take
  an argparse `Namespace` and print — that is orchestration, not a primitive.
- `index_videos.py` is the `fdx` entry and already large; the helpers are
  media-agnostic, not video-specific, so importing CLI orchestration *from* the
  video entry into the photos entry is the wrong dependency direction.
- `runner.py` is a focused home both entries import. No heavy imports
  (whisperx/torch/Pillow/osxphotos) — it imports only `pipeline`, `face_db`,
  and (lazily, in the api branch) `anthropic`. No import cycle: `runner` →
  `pipeline`/`face_db`; `index_videos` → `runner`; `photos_indexer` →
  `runner` + `index_videos` + `photos` + `images`.

### `runner.py` surface

Small composable helpers so **each `main()` keeps its exact control flow** (no
forced reordering ⇒ no behavior change):

```python
def resolve_vision_model(args) -> tuple[str, float]:
    """Validate args.vision_model against VISION_MODELS; return
    (model_id, cost_per_call) for the chosen backend. sys.exit on unknown model."""

def announce_cost(backend: str, model_id: str, cost_per_call: float, n_todo: int) -> None:
    """Print the 'vision: ... / estimated cost' block (identical text to today)."""

def wire_vision_backend(args) -> Any:
    """Check cli/api/local availability and return the anthropic client (api) or
    None (cli/local). sys.exit with the existing actionable messages on failure.
    Prints the existing 'Vision: ...' notices, incl. the ANTHROPIC_API_KEY scrub
    note for cli."""

def setup_face_db(args) -> sqlite3.Connection | None:
    """init_face_app + open_db + stats print, or None when --no-faces / unavailable.
    Identical text to today."""

@dataclass
class RunTally:
    processed: int = 0
    errors: int = 0
    skipped_short: int = 0
    skipped_too_long: int = 0
    skipped_no_preview: int = 0
    actual_cost: float = 0.0

def record_result(result: ProcessResult, tally: RunTally, *, backend: str,
                  max_duration_min: int) -> None:
    """Apply one ProcessResult to the tally and print the per-file outcome line
    (skip reasons + the '-> sidecar (...)' success line). The caller prints the
    '[i/n] <header>' line (it differs per adapter: rel-path vs filename+uuid)."""
```

Notes:
- `record_result` knows every skip reason (`short`, `too_long`, `no_preview`,
  `vision_error`) so both loops report identically. `vision_error` increments
  `errors` (no sidecar written ⇒ retried next run). Photos-only `skipped_missing`
  (iCloud, decided *before* `process_one_*` in `materialize`) is **not** a
  `ProcessResult` reason and stays counted in the photos loop.
- `wire_vision_backend` / `setup_face_db` reference `check_claude_cli`,
  `check_local_endpoint`, `resolve_anthropic_key`, `face_db.*` so tests patch
  them at `runner.*`.

### Phase 1 — dedupe (byte-identical behavior)

1. Add `runner.py` with the six symbols above, lifting the **exact** code
   (prints included) from the two `main()`s.
2. `index_videos.main()` calls the helpers in its current order:
   resolve → announce → dry-run return → wire → face-db → `setup_whisper` (only
   if `need_video`) → loop (`record_result` per file).
3. `photos_indexer.main()` drops its inline whisper/diarize/backend/face-db code
   and instead: dry-run return (as today, earlier) → resolve → announce → wire →
   face-db → `setup_whisper` (only if the todo set contains a video) → loop
   (`record_result` per file). It imports `setup_whisper` from `index_videos`
   (deleting its private copy).
4. No change to `pipeline.py`, `images.py`, `write_sidecar`, or the sidecar
   bytes. The golden video-sidecar test and the image tests stay green.

**Behavior-preserving caveat (intended, minimal).** The byte-identical claim is
scoped to what matters: the **per-file processing output, the sidecar bytes, the
vision/cost-announce + backend + face lines, the result-reporting lines, the
run summary, and exit codes** — all verbatim (Codex code-review confirmed no
tally/exit drift). Three *intended* output changes fall out of removing the
duplicated copies, all on startup/diagnostic or failure paths:

1. `fdx-photos`' backend-wiring and face-DB **failure** messages now match
   `fdx`'s (more complete) shared text — failure paths only.
2. `fdx-photos` **video** runs now print `fdx`'s shared **whisper-setup banner**
   (via the reused `setup_whisper`: e.g. the canonical-fixes "loaded N rules
   from <path>" / "none" lines) instead of the old inline copy's terser
   variant. This is the point of the dedupe — one whisper setup — not a
   regression; the per-clip output and sidecar bytes are unchanged.
3. One existing test (`test_image_only_run_never_loads_whisper`) monkeypatches
   `index_videos.check_claude_cli`; since the cli check now lives in
   `runner.wire_vision_backend`, that patch target updates to
   `runner.check_claude_cli`. Its claim (an image-only `fdx` run never imports
   whisperx) is preserved.

### Phase 2 — `fdx-photos --media images|videos|all`

**`photos.py`:**
- `PhotosAsset` gains `media_type: str` (`"image"` | `"video"`), set in
  `_project` from `PhotoInfo.ismovie` (`"video" if photo.ismovie else "image"`).
- New `enumerate_assets(library, *, media="all", albums=..., persons=...,
  keywords=..., since=..., until=..., uuids=...)` — same filtering/sorting as
  today but `db.photos(images=(media in {"all","images"}), movies=(media in
  {"all","videos"}))`. `enumerate_videos(...)` becomes a thin wrapper
  `enumerate_assets(..., media="videos")` so any external caller / the existing
  contract is preserved.

**`images.py`:**
- `process_one_image` gains a `metadata_override: dict[str, Any] | None = None`
  kwarg (symmetric with `process_one_video`). After
  `metadata = get_image_metadata(image)`, `if metadata_override:
  metadata.update(metadata_override)`. This threads Photos' canonical
  `creation_time` over exiftool's. Folder `fdx` never passes it (no behavior
  change).

**`photos_indexer.py`:**
- Add `--media images|videos|all` (default `all`).
- Enumerate via `enumerate_assets(library, media=args.media, ...)`.
- Resume / iCloud-missing / `--max-files` logic unchanged, operating on the
  combined asset list.
- `need_video = any(a.media_type == "video" for a in todo)`; call
  `setup_whisper` only then (an images-only run never loads whisperx).
- Per asset, route by `asset.media_type`:
  - **video** → `process_one_video(video_path, library, opts, ctx,
    sidecar_path_override=mirror, parent_folder_override=..., metadata_override=
    to_metadata_override(asset), gps_override=to_gps_override(asset),
    place_override=None, extra_frontmatter=to_extra_frontmatter(asset),
    omit_path=downloaded, proper_nouns=[])` — exactly today's call.
  - **image** → `images.process_one_image(image_path, library, opts, ctx,
    sidecar_path_override=mirror, parent_folder_override=...,
    metadata_override=to_metadata_override(asset),
    gps_override=to_gps_override(asset), place_override=None,
    extra_frontmatter=to_extra_frontmatter(asset), omit_path=downloaded)`.
    Pixels come from `asset.path_original` (real HEIC/JPEG/RAW in the bundle);
    `get_image_metadata` (exiftool) + `render_preview` (Pillow/pillow-heif/RAW
    embedded preview) work on it directly.
- `record_result` handles both; the photos loop keeps its `[i/n] filename
  (uuid=...)` header and the `skipped_missing` (iCloud) counter.

**Scale warning (photos only):**
```python
def is_large_unfiltered_run(n_todo: int, has_filter: bool, threshold: int = 1000) -> bool:
    return n_todo > threshold and not has_filter
```
`has_filter = bool(album or person or keyword or since or until or uuid or
max_files)`. When true, print a loud multi-line warning (count, est API cost if
`--backend api`, "mostly local / Ctrl-C anytime / re-run resumes") and continue.
Pure predicate ⇒ unit-tested; the print stays in `main`.

## Sidecar schema

No new schema. Apple Photos stills produce the **same image sidecar** PR #17
defined (`media_type: image`, `camera:` block, `dimensions`, `scene_type`, …)
**plus** the existing Photos extras already emitted for videos
(`photos_uuid`, `photos_persons`, `photos_albums`, `photos_keywords`, `file`/
`original_filename` override, optional `photos_edited`) via
`to_extra_frontmatter`. Apple Photos videos are unchanged. `fdx-query` /
`fdx-master` / `fdx-summary` already read both shapes (query treats absent
`media_type` as `video`).

## Install extras (after)

| Extra | Pulls | For |
|---|---|---|
| `framedex[images]` | Pillow, pillow-heif | folder stills |
| `framedex[video]` | whisperx (+torch) | video |
| `framedex[photos]` | osxphotos (darwin) | Apple Photos **source** |
| `framedex[all]` | everything | |

`fdx-photos` video → `pip install framedex[photos,video]`; stills →
`framedex[photos,images]`; mixed/`--media all` → `framedex[all]`. The
`photos_indexer` whisperx-missing error is removed (whisper now loads via
`setup_whisper`, which already emits the correct `[video]`-extra message); the
osxphotos-missing error keeps pointing at `[photos]`. README + `--help`
install hints updated to match.

## Testing

CI installs **only** dev+test groups (`pyyaml`+`requests`) — every test must
import-safely without osxphotos/whisperx/Pillow. Patterns: stub `osxphotos` at
`sys.modules` (as `test_photos.py` does), patch `subprocess.run` for exiftool,
use **string-target monkeypatch** (`monkeypatch.setattr("framedex.x.y", fake)`),
annotate every helper (strict mypy).

New / updated tests:
- `runner.py`: `setup_face_db(--no-faces) → None`; `record_result` tallies each
  skip reason + the success line; `wire_vision_backend` cli/api/local branches
  (patch `runner.check_claude_cli` / `check_local_endpoint` /
  `resolve_anthropic_key`); `resolve_vision_model` happy + unknown-model exit.
- `photos.py`: `enumerate_assets` flips `images=`/`movies=` per `media`
  (capture kwargs via a stub `PhotosDB`); `_project` sets `media_type` from
  `ismovie`; `enumerate_videos` wrapper still returns movies-only.
- `images.py`: `process_one_image(metadata_override=...)` threads `creation_time`
  into the frontmatter.
- `photos_indexer.py` (new `tests/test_photos_indexer.py`, osxphotos stubbed):
  routing by `media_type` (image→`process_one_image`, video→`process_one_video`);
  images-only run never calls `setup_whisper`; `is_large_unfiltered_run`
  truth table; `--media` arg plumbs into `enumerate_assets`.
- Unchanged-and-green: the golden video-sidecar test
  (`test_video_sidecar_body_is_stable`) and all `test_images.py`.

Gates (CI's exact env): `ruff check`, `ruff format --check`,
`mypy src/framedex tests --ignore-missing-imports`, `pytest`, `uv lock --check`.

## Risks & mitigations

- **Refactor drift** changing video output → golden test + moving code verbatim.
- **Monkeypatch target moves** (`check_claude_cli`) → update the one test;
  assertion preserved.
- **HEIC stills need pillow-heif** → `render_preview` already raises an
  actionable `[images]`-extra error; surfaced only when an actual still is
  reached (lazy).
- **Whole-library run cost** (esp. `--backend api`) → warn-loudly with est cost;
  `--max-files` / filters as the scoped path; incremental + resumable.
- **`[photos]` no longer bundles whisperx** → existing `[photos]`-only installs
  that index video now get a clear `[video]`-extra error instead of a working
  run; documented in README + acceptable pre-1.0.

## Plan review triage (Codex gpt-5.5, high effort)

Verdict: **GO-WITH-CHANGES**. Findings folded into the contract above:

**Folded in:**
1. **`photos_indexer` import-safety (blocker).** The module imports `framedex.photos`
   (which raises without `osxphotos`) at import time. Move the guarded
   `import photos as photos_mod` *inside* `main()` so the module — and its pure
   helpers (`is_large_unfiltered_run`, `missing_media_extras`) — import without
   `osxphotos`. Tests then exercise helpers directly and stub `osxphotos` only
   for `main()` paths.
2. **`announce_cost` byte-identity (blocker).** It must take `local_base_url`
   (the `local` line prints `... @ {local_base_url}`). Add stdout snapshot tests
   asserting the `cli`/`api`/`local` announce lines match today's text verbatim.
3. **Date filters + undated assets (should-fix).** `enumerate_assets` excludes
   undated assets when `since` or `until` is set (today they silently pass a
   date filter). Tested.
4. **`media_type` contract.** Type it `Literal["image","video"]`; derive via
   `getattr(photo, "ismovie", False)`; unit-test `_project` for movie/still/
   missing-attr. Routing needs per-asset kind because `--media all` returns a
   mixed set.
5. **Extras preflight (contains the install trap).** With default `--media all`
   and `[photos] = osxphotos`-only, a bare `fdx-photos` would enumerate then
   fail lazily. Add a pure `missing_media_extras(need_video, need_image, *,
   has_whisperx, has_pillow) -> list[str]`; in `main()`, after the todo set is
   known, `sys.exit` once with `pip install -e '.[photos,<missing>]'` if extras
   are absent (`importlib.util.find_spec`, no heavy import). Folder `fdx` keeps
   its lazy errors (Phase-1 untouched).
6. **`runner.py` stays orchestration-only.** Guard test asserts importing
   `runner` pulls in no `whisperx`/`PIL`/`osxphotos`. Precise typing
   (`argparse.Namespace`, `sqlite3.Connection`, `Literal`).
7. **Cut `RunTally.skipped_short`.** Short skips are printed inline but never
   summarized today; the counter would be dead state. `RunTally` =
   `{processed, errors, skipped_too_long, skipped_no_preview, actual_cost}`.
8. **Claim accuracy (doc).** `query.py` + `master_index.py` are media-neutral
   (PR #17); `trip_summary.py` prose remains video-flavored (unchanged from
   PR #17) — out of scope here. Stills use `path_original` (the **unedited**
   original), consistent with the video path; `photos_edited: true` flags
   adjusted assets. An `--edited-pixels` mode is a future option, not v1.

**Deferred (pre-existing, out of scope — tracked as follow-ups):**
- **Face-DB idempotency on `--download`.** `face_db.write_faces` dedups on the
  media path; a materialized asset's temp path changes per run, so
  `--force --download` can duplicate face rows for one Photos UUID. Pre-existing
  for Photos videos; fix is a stable `photos:<uuid>` source id. Documented; not
  in this PR.
- **`fdx-query --person/--keyword` don't search `photos_persons`/
  `photos_keywords`.** Pre-existing for Photos videos. Photos-side metadata is in
  the sidecar frontmatter (greppable, visible to an LLM reading `_INDEX.md`); the
  structured query filters don't cover it yet. Follow-up.
- **`master_index` groups path-less (`--download`) records under "(unknown)".**
  No crash; pre-existing for Photos videos. Follow-up.
- **Image H1/prompt filename.** `process_one_image` uses the on-disk name for the
  sidecar H1 + prompt context while `file:` carries the nice original — this
  exactly matches `process_one_video`'s existing behavior; changing only the
  image side would diverge the two. Left consistent.

## Code review triage (Codex gpt-5.5, high effort)

Verdict: **GO-WITH-CHANGES** (no tally/exit drift; ruff + mypy-strict pass).
Folded in:
- **Byte-identity claim narrowed** to the processing/sidecar/cost/summary output
  (see the Phase-1 caveat above); the `fdx-photos` whisper-setup banner and
  failure messages now match `fdx`'s shared text by design.
- **Extras preflight checks `pillow-heif`, not just `PIL`.** `[images]` ships
  both; an Apple Photos library is HEIC-heavy, so requiring both avoids a
  false-pass that would fail mid-run on the first HEIC.
- **Subprocess import-safety test** added: a clean interpreter (no `osxphotos`
  stub) proves `import framedex.photos_indexer` succeeds and doesn't pull
  `osxphotos` at import time.
- **Stale "videos" wording** in the parser description + `photos.py` docstring
  updated to "media/assets".

## Rollout

Single PR off `main`, two commits (refactor → feature), CI green, stop at "PR
open". No merge without explicit instruction.
