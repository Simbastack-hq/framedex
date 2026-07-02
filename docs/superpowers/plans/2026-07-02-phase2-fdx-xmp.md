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
fdx-xmp /Volumes/SSD-2024                 # RAW files with sidecars (default)
fdx-xmp /Volumes/SSD-2024 --all-files     # every image format
fdx-xmp /Volumes/SSD-2024 --dry-run       # print would-write list, write nothing
```

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
Stdlib RDF/XML via `xml.sax.saxutils.escape`. `xmp:Rating` from `RATING_MAP`
(default 2 for an unknown/missing rating — matches the sidecar default `review`);
`xmp:Label="Red"` only for `cull`; `dc:subject` rdf:Bag = `keywords` +
`scene_type` (deduped, non-empty, `unclear`/`other` scene_type dropped);
`dc:description` x-default alt = the one Scene sentence (omit the block if empty);
`xmp:CreatorTool = CREATOR_TOOL`.
- Tests: golden byte-comparison per rating (keep/review/cull); keywords escaped
  (`&`, `<`); empty keywords → no `dc:subject`; empty scene → no `dc:description`;
  `scene_type: unclear` not added to the bag.

### Task 3 — file targeting
For each sidecar, the original is `sidecar.parent / name-without(".description.md")`
(the sidecar sits next to it). Skip when: `media_type == "video"`;
`photos_uuid` present (Photos-managed mirror sidecar); the original's extension is
not in the active set (`XMP_SIDECAR_EXTENSIONS`, or all image exts with
`--all-files`). XMP target = `original.with_suffix(".xmp")` (`DSC_1.RAF` →
`DSC_1.xmp`).
- **Stem collision** (both `DSC_1.RAF` and `DSC_1.JPG` have sidecars → same
  `DSC_1.xmp`): the RAW wins (it's the edit target); print a one-line notice,
  skip the JPEG's write.
- Tests: video skipped; photos_uuid skipped; `.dng` skipped by default but
  included with `--all-files`; JPEG skipped by default; RAW-wins on collision.

### Task 4 — overwrite safety (manifest)
`_XMP_MANIFEST.json` at the scan root (written atomically via
`pipeline.atomic_write_text`), mapping the xmp path → SHA-1 of the payload we
wrote. Decision per target, in order:
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
argparse: `root`, `--all-files`, `--dry-run`. Walk sidecars, apply Tasks 1–4,
write via `atomic_write_text`, update the manifest once at the end (atomic).
`--dry-run` prints the would-write list and writes nothing (no file, no manifest).
Summary line: `wrote N, skipped M (conflicts), K up-to-date`.
- Tests: `--dry-run` writes nothing; end-to-end over a tmp tree with a RAW
  sidecar → `.xmp` exists with the right Rating; conflict count surfaced.

## Non-goals (PRD)
Writing into originals (never); embedded-XMP for JPEG/DNG (never — same reason);
IPTC location structs; person names (arrive with fdx-faces, same `dc:subject`
bag); a Lightroom plugin; any merge into a foreign `.xmp`.

## Verify before done
`ruff check` + `ruff format --check` + `mypy` + full `pytest`. README "Getting
ratings into Lightroom" section (with the "Read Metadata from Files" step),
SKILL.md verb, CHANGELOG.
