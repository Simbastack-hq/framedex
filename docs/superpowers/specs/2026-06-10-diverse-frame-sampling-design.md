# Diverse frame sampling (issue #5) — design

Date: 2026-06-10
Status: approved by NJ (diverse as default, pending real-clip validation)
Replaces the literal proposal in issue #5 (ffmpeg `select='gt(scene,0.4)'`).

## Goal

The vision step sends 5 frames per clip. Today they are evenly spaced, which is
content-blind: near-duplicate frames on static shots, missed distinct moments on
clips where things change. Replace even spacing with a selection that maximizes
the visual information in those same 5 frames, at unchanged vision cost.

## Why not the issue #5 proposal (measured, not guessed)

Benchmarked on ffmpeg 8.0.1 / Apple Silicon with synthetic and real clips:

- ffmpeg's scene score is `clip(min(mafd, |mafd - prev_mafd|)/100, 0, 1)` over
  adjacent frames. The `min()` deliberately suppresses *sustained* motion: a
  steady pan/orbit/walk scores ~0.01–0.07 and never crosses 0.3/0.4. framedex's
  corpus is mostly single-take unedited footage, so the threshold path would
  almost always fall back to even spacing — after paying a mandatory full-clip
  decode (~19s wall per minute of 4K HEVC; ~9 min for a 30-min clip).
- When the score does fire on home footage it ranks *abruptness*, not content: a
  0.1s white flash scores 1.000 twice (onset + offset) and outranks real cuts.
  Garbage frames feed the rating prompt and can flip keep → review/cull.
- Independent field report (HN): the author of a similar tool tested ffmpeg's
  detector and preferred histogram-difference analysis.

## Design: pick the 5 most mutually different, sharpest frames

Reframe from "detect scene cuts" to "select N maximally informative frames".
Cut detection asks "did adjacent frames change abruptly?"; we ask "which 5
moments differ most from each other?" — which also covers gradual single-take
drift (pan from beach to street: no cut anywhere, but the endpoints are
mutually distant) and subsumes edited clips (mutually distant frames land in
different shots).

### Pipeline (per clip, inside `extract_frames`)

1. **Candidate pool.** `M = clamp(round(duration / 2), 20, 96)` thumbnails at
   160px width, grabbed by *independent ffmpeg fast-seeks* at evenly spaced
   timestamps (the same `-ss TS -i clip -vframes 1` pattern the full-res
   extraction already uses). Measured ~0.27s per seek, flat in file size and
   position → ~6–26s for any clip length. No full decode, no `-skip_frame
   nokey`, no PTS parsing: timestamps are known by construction. Thumbnails are
   written to the existing flat tmp dir with a `cand_` prefix (the cleanup glob
   at the call site handles them; no subdirectories).
2. **Signatures** (cv2 + numpy, both already base deps; ~10ms at M=96):
   - 2D **H-S histogram** (Hue × Saturation, V channel dropped), normalized;
     pairwise distance via `cv2.compareHist(..., HISTCMP_BHATTACHARYYA)`.
     V is excluded because phone auto-exposure breathing alone produces
     V-inclusive distances (0.09–0.39) at or above genuine content change
     (~0.37 for pan endpoints) — brightness must not count as "different".
   - **Sharpness**: variance of Laplacian on the grayscale thumbnail
     (validated to still discriminate blur at 160px).
   - **Brightness gates**: drop candidates with mean V < `GATE_V_DARK` (20) or
     > `GATE_V_BLOWN` (245) (lens cap / pointed-at-the-sun; a blown frame has
     maximal distance to everything and would otherwise always win a slot).
3. **Degenerate-pool guard.** If gates leave fewer than `2 × num_frames`
   candidates (night clips straddle the dark gate), ungate and fall back to the
   exact legacy evenly-spaced timestamps.
4. **Static guard.** If the max pairwise distance is below `STATIC_GUARD`
   (initial 0.05 *on the H-S metric*, to be calibrated on real clips), the clip
   is visually static → legacy evenly-spaced timestamps.
5. **Selection.** Seed with the **medoid** (argmin of summed distances — resists
   outlier-chasing), then **greedy farthest-point**: repeatedly add the
   candidate maximizing min-distance-to-selected, until `num_frames`.
   Deterministic tie-breaking (lowest index wins). Sort chronologically.
6. **Sharpness swap.** For each pick, consider ±1 temporal neighbors whose
   distance to the pick is < `NEAR_DUP_D` (initial 0.1 on the H-S metric,
   calibrated with `STATIC_GUARD`); take the sharpest of that near-duplicate
   group. Avoids motion-blurred picks without ever crossing into different
   content.
7. **Full-res extraction** — unchanged: 5 fast-seek extractions at the selected
   timestamps, same quality flags, same 1920px cap.

Short-clip bypass: clips under `SHORT_CLIP_EVEN_CUTOFF` (20s) skip the pool
entirely (the pool costs more than it can return; <3s clips keep the existing
3-frame clamp). Static clips, gated-out pools, and `--frame-sampling even` all
produce byte-identical legacy behavior.

### Module layout

- New `src/framedex/frame_sampling.py`: pure functions over numpy arrays
  (`hs_histogram`, `pairwise_distances`, `select_diverse`, `sharpness_swap`,
  `candidate_timestamps`, gates). No subprocess calls, no I/O in the core
  functions → hermetic unit tests. All constants module-level, named, with
  one-line comments. cv2 stays out of `index_videos.py` (convention: cv2
  currently appears only in `face_db.py`).
- `index_videos.extract_frames(video, out_dir, num_frames=5, *, duration,
  sampling)` becomes a thin orchestrator returning **`list[tuple[Path,
  float]]`** (path, timestamp). Duration is passed in from the call site
  (drops the redundant second ffprobe).

### Required refactor (fixes a live bug)

`process_one_video` currently *re-derives* frame timestamps from the
even-spacing formula at `index_videos.py:903-905` instead of using what
`extract_frames` actually produced. Today, if any frame write fails, the
surviving frames' timestamps shift and `faces.db` `frame_time` silently
desyncs. The `(path, timestamp)` return kills this: the call site uses the
returned timestamps for `face_db.detect_faces_in_frames` and the sidecar.
Regression test: sidecar/faces `frame_time` equals what extraction reported,
including with a simulated failed frame.

### Prompt changes

`_build_vision_prompt` gains a `timestamps` parameter. The hard-coded
"(evenly sampled across the clip)" (line ~605) becomes "sampled at MM:SS,
MM:SS, …" for all backends. This also grounds the existing
`notable_timestamp` YAML output field.

### CLI surface

One flag: `--frame-sampling {diverse,even}`, default `diverse`, added to BOTH
parsers (`index_videos.main`, `photos_indexer.main`) and threaded as one new
`ProcessOptions` field. No `--scene-threshold`, no `--frame-candidates`: pool
size and all thresholds are documented module constants (issue #5's
tunability criterion explicitly allows a documented constant). `even` is the
escape hatch and pins legacy behavior.

### Tests

- Hermetic unit tests on synthetic numpy arrays: medoid seeding, farthest-point
  picks the constructed-distinct frames, determinism, chronological output,
  static guard, brightness gates, degenerate-pool fallback, sharpness swap
  stays within near-dup group.
- Golden test: `even` mode reproduces current timestamp formula exactly.
- Mocked-extraction test: `extract_frames` returns aligned (path, timestamp)
  pairs; call-site regression for faces `frame_time`.
- No test shells out to ffmpeg (existing suite convention).

### Docs

- README: pipeline step 4 wording; remove "evenly-spaced, not scene-detected"
  from Known limitations; one-paragraph explanation (sample small thumbnails
  across the clip, keep the 5 most mutually different and sharpest; unchanging
  clips keep the old even spacing).
- SKILL.md: update the known-limitations/roadmap line that currently promises
  the ffmpeg `select` mechanism.
- docs/tuning.md: note the constants and the `even` escape hatch.
- CHANGELOG entry.

### Validation before release

A/B `diverse` vs `even` on ~10 real clips from NJ's Downloads (drone, phone,
long single-take): eyeball 5-frame sets for diversity and absence of garbage
picks, confirm rating-relevant frames aren't degraded, calibrate
`STATIC_GUARD` / `NEAR_DUP_D`, measure real per-clip overhead. `diverse`
stays default only if this passes.

## Declined alternatives

- **ffmpeg select/scdet scene scoring** (issue #5 as written): wrong signal for
  single-take footage; flash false positives; linear-in-duration decode cost.
  See measurements above.
- **PySceneDetect**: its useful per-frame metric is ~30 lines of cv2 we write
  anyway; StatsManager forbids frame_skip (forces full decode); pre-1.0 API
  churn; headless variant required to avoid an opencv conflict.
- **TransNetV2 / shot-boundary ML**: detects cuts that single-take footage
  doesn't have; dormant since 2023.
- **CLIP/SigLIP diversity**: ~2 recall points over histograms in published
  comparisons; 40–350MB model in the base path; semantics are already supplied
  downstream by the vision model + transcript.
- **Edited-clip classifier hybrid**: rarely-executing heuristic (verified
  failure on heavily-edited montages); the simplicity-respecting version of it
  is this design.

## Cost summary

| | today | this design |
|---|---|---|
| ffmpeg invocations | 1 probe + 5 seeks | 1 probe + M seeks + 5 seeks |
| typical 1-min clip | ~1.4s | ~8–10s |
| 30-min 4K worst case | ~1.4s | ~26s |
| vision call | 5 frames | 5 frames (unchanged) |
| new dependencies | — | none |
