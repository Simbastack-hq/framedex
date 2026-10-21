# fdx-develop: the model is the editor, a RAW engine does the pixels — design

- **Status:** v0 implemented (opt-in, experimental); Codex plan review applied (see the end)
- **Date:** 2026-10-21
- **Scope:** one new optional command, `fdx-develop`, that turns a folder of
  RAWs into edited web/export copies. A vision model looks at each photo,
  chooses edit parameters, the external RAW engine renders them, the model
  looks at the render and adjusts (a fixed, small number of rounds). The
  adjustments are kept in plain-text sidecars the user can re-open in the
  engine's own editor. No slider UI.

## Why this, why now

`fdx share` (in flight) turns a folder into a hosted gallery. Most folders a
photographer would share are unedited RAWs, and editing is the bottleneck. The
bet: vision models are good enough to act as the editor today, and get better
with every model release; the durable part is the loop and the plumbing
(look, choose parameters, render, look again), not a hand-tuned preset.

A blind A/B test on 10 real hand-edited RAWs (the maintainer's own Lightroom
edits vs the model + RapidRAW) gates whether this gets wired into
`fdx share --edit`. The test harness and results live outside this repo (the
photos are private); this PR ships the reusable module.

Not in scope: wiring into `fdx share` (follow-up after both PRs land), AI
subject/sky masks (the engine needs extra model downloads), retouching,
cropping/straightening decisions, batch "sync settings" across a burst, any
GUI.

## Architecture

```
RAW ──► engine (neutral render) ──► crop (from the user's Lightroom XMP, if any)
                                        │
                                        ▼
           ┌──── vision model: "here is the photo + context (+ the user's own
           │      before/after examples); return slider values as JSON"
           │                            │
           │                            ▼
           │      engine renders the values ──► crop ──► preview + stats
           └──────────────────────────────────────┘   (≤ N rounds, model may stop early)
                                        │
                                        ▼
              <out>/<name>.jpg   +   <out>/<name>.develop.json   (+ <raw>.rrdata)
```

- `framedex/develop_engine.py` — the engine adapter. Pure functions map the
  model's engine-neutral edit (a small JSON vocabulary: exposure, contrast,
  highlights, shadows, whites, blacks, WB temperature/tint, clarity, dehaze,
  texture, vibrance, saturation, 8-band HSL, 3-way color grading, a luma tone
  curve, vignette, up to 3 radial/graduated masks) onto RapidRAW's adjustments
  JSON, clamp every value to the engine's slider ranges, and convert mask
  geometry from fractions of the cropped frame to engine pixel coordinates
  (orientation-aware). Plus: the engine command line, Lightroom crop parsing
  from `.xmp` sidecars / embedded XMP, and crop math. Pixel work (PIL) is
  lazily imported so the module's pure core imports on a bare install.
- `framedex/develop.py` — the CLI and the loop: discovery, prompts, backend
  calls, parsing, sidecars, idempotency.

### Engine: RapidRAW, called as a separate program

RapidRAW (github.com/CyberTimon/RapidRAW, AGPL-3.0) has had a headless CLI
since 1.6: `rapidraw export <raw> --output <file> --adjustments <json>`. It is
GPU-accelerated (a 45 MP Z8 NEF renders in ~3–4 s on an M1 Max), and its
adjustments JSON is defined by its source (tested against 1.6.6). framedex
never vendors, links, modifies or imports it: it runs the user's
independently installed, unmodified binary through its documented CLI, like
exiftool and ffmpeg, and exchanges plain JSON. That keeps framedex's MIT code
independent of the AGPL program; redistributing or modifying RapidRAW is a
separate question the user's own install answers. framedex always passes
explicit `--adjustments`, so a neighbouring `.rrdata` can never leak into a
render. Located via `--engine`, then `$FDX_RAPIDRAW`, then
`rapidraw` on PATH, then `/Applications/RapidRAW.app/Contents/MacOS/RapidRAW`.
Renders are serialized (one engine process at a time).

The adapter boundary is the engine-neutral edit JSON, so a second engine
(`rawtherapee-cli` + generated `.pp3`, or darktable-cli + XMP) is a new
mapping function, not a new loop.

### The loop

Rounds are **stateless**: every call carries the neutral render, the latest
render, the latest values and the model's own stated intent. No multi-turn
conversation state, so the same loop runs on every backend framedex supports
(principle 2): `api` (Anthropic), `local` (any OpenAI-compatible server, e.g.
LM Studio), and `cli` (Claude Code). The model returns one JSON object; the
parser merges it onto a neutral template, so a missing key means "neutral",
never a crash. Bad JSON ends that photo's loop with the best render so far.

Each round the model gets a measured summary of the render (mean/median luma,
% of pixels near clipping, % near black, mean R/G/B, a 16-bin histogram) so it
can reason about clipping it can't see in a downscaled preview.

### Style from the user's own edits

`--examples DIR`: a folder of up to 3 pairs, `<name>.<raw>` + `<name>.jpg`
(the user's finished export of that RAW). framedex renders each RAW neutral
(cropped with the export's embedded Lightroom crop if present) and shows the
model before/after pairs: "edit like this photographer". Typically 2–3 hero
shots from the same shoot the user already edited. Examples are rendered once
per run and reused for every photo; the model is told never to copy a crop or
a black-and-white conversion unless the photo calls for it.

### Context

If a framedex `.description.md` sidecar exists for the RAW, its scene
description, place, time of day and lighting go into the prompt. Otherwise
EXIF (camera, focal length, aperture, shutter, ISO, capture time) only.

### Outputs and sidecars (principles 1 and 3)

- Originals are never modified, moved or renamed. Renders go to a temp file
  inside the output folder and are renamed atomically.
- `<out>/<name>.jpg` — the edited copy, long edge `--size` (default 2048),
  sRGB, quality 90.
- `<out>/<name>.develop.json` — plain text: status (`done`/`failed`), source
  fingerprint (name, size, mtime), engine + version, model, backend, every
  round's values / stats / cost, the final engine adjustments, the crop.
  Written last. A re-run skips a photo only if the record says `done`, the
  JPEG exists and the fingerprint matches; failed photos stay retryable;
  `--force` redoes done ones. An output JPEG framedex has no record for is
  never overwritten. Same-stem RAWs in one folder keep their extension in
  the output name. Subfolders are mirrored under `--out`.
  (Superseded by the diff review: see the last table.)
- `<raw>.rrdata` next to the original — RapidRAW's own sidecar, so the user
  can open the RAW in RapidRAW and keep editing from the AI's values, with
  the crop in RapidRAW's own `crop` field (verified pixel-identical to
  framedex's crop for unrotated crops; rotated crops are not carried).
  Created with `O_EXCL`, only if nothing (file or symlink) exists there, so
  the user's own edits are never overwritten; `--no-rrdata` skips it.
- Crop: if `<raw>.xmp` (or `<stem>.xmp`) carries a Lightroom crop
  (`crs:CropLeft/Top/Right/Bottom/Angle`), it is applied to every render, so
  the model edits the final framing. framedex never invents a crop.

### Cost (principle 4)

Fixed per photo: ≤ `--rounds` (default 3, max 4) vision calls, each with ≤ 2
photos at 1600 px, the calibration sheet, and up to 3 example pairs at
1024 px; output is capped at 16k tokens per call and SDK retries at 2. This is
framedex's one deliberate exception to "one call per file": the bound is the
round count. Measured on Claude Opus 5.5 at effort `high` on 10 real RAWs: $0.19–0.28
per photo ($0.23 average) with two example pairs, calibration and 3 rounds;
≈ $0.30 with 4 rounds. The run prints
an estimate up front and the actual at the end from API usage; the CLI
backend reports cost as unknown. `--max-files` caps a run. Local: $0.

## CLI

```
fdx-develop <folder> [--out DIR] [--engine PATH] [--backend api|local|cli]
            [--model ID] [--rounds N] [--examples DIR] [--size PX]
            [--max-files N] [--force] [--no-rrdata] [--dry-run]
```

`--dry-run` lists what would be processed and the cost estimate, renders
nothing, calls nothing.

## Testing

Hermetic (repo convention): no engine, no network, no real photos.
- Mapping: clamping, defaults, curve sanitizing, mask geometry for every
  orientation, graduated-mask direction.
- Parsing: JSON in prose, partial JSON merged onto the template, garbage.
- Lightroom crop: XMP attribute and element forms, absent crop.
- Crop math (PIL, skipped if Pillow is absent): synthetic gradients with known
  rectangles, orientation round-trip, angled crop size.
- Loop: the renderer and backend are injected, so orchestration tests use a
  fake backend and a fake renderer returning opaque bytes (no Pillow): early
  stop, the round cap, done-too-early, bad JSON first and later, backend
  errors, unknown cost, resume fingerprint, `.rrdata` never overwritten (incl.
  symlinks), originals byte-identical after a run, dry-run writes nothing.
  Pixel tests (crop math, orientation round trip, angled crop size, stats)
  `importorskip` Pillow and run wherever it is installed.

## Known limitations

- No AI subject/sky masks (RapidRAW's need ONNX model downloads); masks are
  radial/graduated only.
- No crop or straighten decisions; Lightroom crops are honored, nothing else.
- RapidRAW's lens-profile database doesn't cover every lens, so distortion
  and vignetting correction can differ from Lightroom's.
- Quality depends on the vision model; local models are supported but
  untested for quality.

## Codex plan review (2026-10-21) and what changed

| Finding | Resolution |
|---|---|
| Output collisions could overwrite foreign files | Same-stem RAWs keep their extension; subfolders mirrored; a JPEG without a framedex record is never overwritten |
| Sidecar presence insufficient for resume | Record has status + source fingerprint, written last; failed = retryable |
| Neutral render needs an explicit contract | Always explicit `--adjustments` (neutral JSON for r0), tested against 1.6.6 |
| Clamping isn't validation | Type-preserving merge onto the template (wrong types → neutral), bounded masks/curve points, absolute-values contract stated in the prompt, last valid render kept |
| Crop persistence incomplete | Crop carried into `.rrdata` in RapidRAW's own field for unrotated crops (verified identical); rotated crops documented as not carried |
| Call count doesn't bound spend | Output tokens capped, retries bounded, usage-based actual cost, unknown when unavailable; exception documented |
| Backend interface vague | One `Backend.complete(system, images, text)` contract, own max_tokens (no 800-token cap), local/cli tested by envelope |
| CI can't import Pillow | Renderer injected; loop tests need no Pillow; pixel tests skip without it |
| "Subprocess so MIT" too categorical | Scoped rationale above and in docs/develop.md |

## Codex diff review (2026-10-21) and what changed

| Finding | Resolution |
|---|---|
| Any record let framedex overwrite an existing JPEG | `owns_output`: only a `pending` record for this RAW, or a `done`/`partial` record whose `output_sha256` matches the file; a user's replacement is skipped |
| Collision names could still clash (`A.NEF` vs `A.nef.DNG`) | Colliding outputs use the full file name (`A.NEF.jpg`), unique per folder |
| Interrupt between JPEG and record orphaned the JPEG | `pending` record written before publishing, finalized after |
| `.rrdata` could be left truncated | Temp file + `os.link` (atomic, never replaces) |
| Masks ignored crop straightening | `FrameMap` maps frame fractions through the inverse crop rotation + orientation; ellipse `rotation` set; tested against `apply_crop` pixels |
| First-render failure aborted the batch | Failed record per photo; `main` has a per-photo exception boundary |
| A later-round failure was recorded `done` | Status `partial`: best earlier render ships, photo retried next run, exit code 1 |
| A failed call was reported as free | Known spend summed; any failed or unpriced call marks cost incomplete (`+ unknown`) |
