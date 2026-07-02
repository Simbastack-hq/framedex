# Phase 2 implementation plan — `fdx-xmp` (Lightroom XMP export)

Traces to [the PRD](../specs/2026-07-02-trust-xmp-burst-prd.md) Phase 2. One PR,
stacked on Phase 1 (`feature/phase1-trust-hardening`). New standalone command;
**nothing in the indexing pipeline changes.** Pure stdlib (no lxml). TDD,
golden-file byte comparisons.

## Shape

New module `src/framedex/xmp_export.py`, console script
`fdx-xmp = framedex.xmp_export:main` (added to `[project.scripts]`). Reads
`.description.md` sidecars under a root, writes `.xmp` sidecars next to the
originals. XMP is a **regenerable view** — delete every `.xmp`, re-run, get them
back. Originals and foreign `.xmp` files are never touched.

```
fdx-xmp /Volumes/SSD-2024                 # proprietary-RAW files with sidecars
fdx-xmp /Volumes/SSD-2024 --dry-run       # print would-write list, write nothing
```

**Scope decision (Codex plan review): `--all-files` is CUT from Phase 2.** The
99% value is proprietary-RAW → Lightroom Classic. `--all-files` would pull in
JPEG/HEIC/TIFF/DNG semantics, other-editor (darktable/digiKam) naming, and the
stem-collision matrix for marginal reach. The export set is always
`XMP_SIDECAR_EXTENSIONS`; re-adding `--all-files` later is a small follow-up.

## Constants (module-level, one-line comments)

- `XMP_SIDECAR_EXTENSIONS = images.RAW_EXTENSIONS - {".dng"}` — Lightroom reads
  `.xmp` *sidecars* only for proprietary RAW; DNG/JPEG/HEIC/TIFF embed metadata
  and ignore sidecars, so `.dng` is out of the default set.
- `RATING_MAP = {"keep": 3, "review": 2, "cull": 1}` — model `keep` is
  conservative archive-triage, not a pick; 3★ leaves headroom for the
  photographer's own 4/5★ selects. `cull` also gets `xmp:Label = "Red"`.
- `CREATOR_TOOL = f"framedex {framedex.__version__}"` — provenance stamp only;
  overwrite safety is manifest-hash-based, not CreatorTool-based.
- `MANIFEST_NAME = "_XMP_MANIFEST.json"`.

## Tasks (each: failing test first)

### Task 1 — sidecar reader (`read_sidecar_for_xmp`)
Returns `(frontmatter: dict, scene: str)`. Parses YAML frontmatter (reuse the
same split as `query.parse_sidecar`) **plus** the body's `**Scene:**` sentence
(one line, via regex — not the whole body). Returns None on parse failure.
- Test: a real sidecar → correct rating/keywords/scene_type + the Scene sentence
  only (not Subjects/Composition lines).

### Task 2 — XMP payload (`build_xmp(frontmatter, scene) -> str`)
Stdlib RDF/XML via `xml.sax.saxutils.escape`. `xmp:Rating` from `RATING_MAP`;
**an unknown/missing rating is not guessed** — the record is warned+skipped
(real sidecars always carry keep/review/cull; a missing one means a malformed
sidecar). `xmp:Label="Red"` only for `cull`; `dc:subject` rdf:Bag = `keywords`
+ `scene_type` (deduped, non-empty, `unclear`/`other` scene_type dropped);
`dc:description` x-default alt = the one Scene sentence (omit the block if empty);
`xmp:CreatorTool = CREATOR_TOOL`.
- Tests: golden byte-comparison per rating (keep/review/cull); keywords escaped
  (`&`, `<`); empty keywords → no `dc:subject`; empty scene → no `dc:description`;
  `scene_type: unclear` not added to the bag; non-ASCII scene round-trips as
  UTF-8 bytes.

### Task 3 — file targeting (pre-pass, not streaming)
For each sidecar, the original is `sidecar.parent / name-without(".description.md")`
(the sidecar sits next to it). **Skip + warn + count** when: `media_type ==
"video"`; `photos_uuid` present (Photos mirror sidecar); the original's extension
∉ `XMP_SIDECAR_EXTENSIONS`; or **the derived original file does not exist**
(moved/deleted → don't write an orphan `.xmp`). XMP target =
`original.with_suffix(".xmp")` (`DSC_1.RAF` → `DSC_1.xmp`; `IMG.2024.RAF` →
`IMG.2024.xmp`).
- Build the full {target_xmp → source} map first (pre-pass), THEN write. A
  same-run collision (two RAW originals sharing a stem, e.g. `DSC_1.CR2` +
  `DSC_1.NEF` → `DSC_1.xmp`) is resolved deterministically (first by sorted
  original path), the loser skipped+warned — no streaming clobber.
- Tests: video skipped; photos_uuid skipped; `.dng` skipped; JPEG skipped;
  missing-original skipped+warned; `IMG.2024.RAF` → `IMG.2024.xmp`; same-stem
  RAW/RAW collision resolves to one deterministic write.

### Task 4 — overwrite safety (manifest)
`_XMP_MANIFEST.json` at the scan root (written atomically via
`pipeline.atomic_write_text`), mapping the **root-relative** xmp path → SHA-1 of
the **exact UTF-8 payload bytes** we wrote (`payload.encode("utf-8")`; on re-run
compare against the existing file's `read_bytes()` — `atomic_write_text` writes
the payload verbatim, no added newline). Decision per target, in order:
1. No existing `.xmp` → **write**, record hash.
2. Existing `.xmp` whose current bytes SHA-1 == the manifest entry → unchanged
   since we wrote it → **overwrite** (atomic), update hash.
3. Anything else (foreign, or ours-but-since-edited, or no manifest entry) →
   **skip**, count, report as a conflict.
A missing/deleted manifest degrades safe: every existing `.xmp` is case 3.
- Tests: fresh write records hash; identical re-run is a no-op-equivalent
  (rewrites case 2, manifest stable); hand-edited `.xmp` (bytes ≠ manifest) →
  skip; missing manifest + existing `.xmp` → skip; changed sidecar → new payload,
  case 2 overwrite, manifest updated.

### Task 5 — `main()` / CLI
argparse: `root`, `--dry-run` (no `--all-files` — cut). Walk sidecars, apply
Tasks 1–4, write via `atomic_write_text`, update the manifest once at the end
(atomic). `--dry-run` prints the would-write list and writes nothing (no file,
no manifest). Fail-loud summary with per-category counts:
`wrote N, up-to-date K, conflicts C, skipped (video V, photos P, non-raw R,
missing-original O, malformed X)`.
- Tests: `--dry-run` writes nothing; end-to-end over a tmp tree with a RAW
  sidecar → `.xmp` exists with the right Rating; conflict + missing-original
  counts surfaced.

## Non-goals (PRD)
Writing into originals (never); embedded-XMP for JPEG/DNG (never — same reason);
IPTC location structs; person names (arrive with fdx-faces, same `dc:subject`
bag); a Lightroom plugin; any merge into a foreign `.xmp`.

## Verify before done
`ruff check` + `ruff format --check` + `mypy` + full `pytest`. README "Getting
ratings into Lightroom" section (with the "Read Metadata from Files" step),
SKILL.md verb, CHANGELOG.
