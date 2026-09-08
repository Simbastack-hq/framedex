# fdx-mcp: the archive as MCP tools — design + implementation plan

- **Status:** design, Codex plan review pending
- **Date:** 2026-09-09
- **Scope:** one new optional command, `fdx-mcp`, that exposes an indexed
  archive to any MCP host (Claude Code, Claude Desktop, LM Studio, anything
  that speaks MCP over stdio) as a handful of tools: query, read, overview,
  contact sheet, and one write (`set_user_rating`). Plus the three small core
  changes it needs: a `user_rating` override that survives re-indexing, an
  `effective_rating` used by every existing tool, and a `fdx-query` refactor so
  the CLI and the MCP tool share one filter implementation.

## Why this, why now

framedex already answers "find the sunset shot" and "what should I cull" when
it is used as a Claude Code skill: Claude runs `fdx-query` and reads
`_INDEX.md`. That surface is invisible to every other host. An MCP server
makes the same index reachable from Claude Desktop, from LM Studio with a
local model (principle 2: every AI step works with a local model), and from
any agent, without framedex owning a chat loop, a prompt, or a model choice.
The host does the reasoning; framedex hands it facts and pictures.

The one thing a text index cannot do is let the model *look*. A contact sheet
tool renders up to 20 thumbnails into one JPEG, numbered, so a host can rank
candidates against each other in a single image. That is one image per
question, not one per file: cost is bounded by what the user asks, not by the
archive (principle 4). The indexing pipeline stays untouched.

The one write, `set_user_rating`, records the human's decision next to the
model's, never over it (principle 6), in the sidecar (principle 3), atomically
(principle 1), and `fdx-xmp` carries it into Lightroom.

Not in scope: an in-repo chat UI or agent loop, semantic/embedding search,
`fdx-faces`, per-photo "critique" calls, anything that edits pixels, any
transport other than stdio, a read-only mode (the only write is one
frontmatter key in a sidecar; originals are never touched).

## Principles check

| Principle | How fdx-mcp honours it |
|---|---|
| 1 Originals sacred | Media files are only *read* (to render thumbnails). The single write is `user_rating`/`user_note`/`user_rated_at` in a sidecar's frontmatter via `atomic_write_text`; the body is preserved byte-for-byte. |
| 2 LLM-agnostic | The MCP host supplies the model. LM Studio with a local vision model gets the same tools as Claude Desktop. framedex makes no model call. |
| 3 Plain text | User picks land in the sidecar next to the original and are exported by `fdx-xmp`. Nothing is stored anywhere else. |
| 4 Predictable cost | Zero model calls inside framedex. A contact sheet is one image, capped at `SHEET_MAX_IMAGES = 20` thumbnails and ~1600 px wide. |
| 5 Earn every dependency | The official `mcp` SDK (v2) and Pillow live in a new optional extra `[mcp]`; the base install and every existing command are unchanged. No custom JSON-RPC. |
| 6 Professionals | `user_rating` is the photographer's call; it wins in `fdx-query`, `fdx-master`, `fdx-xmp`, and it survives `--force` re-indexing (like named face clusters do). |

## Tools (v1)

All paths are absolute or relative to a configured root; every path argument
must resolve under one of the roots the server was started with, or the call
fails (`ToolError`). Results are compact JSON; the host decides how to talk.

| Tool | Args | Returns |
|---|---|---|
| `list_roots` | – | the roots this server can see |
| `query_media` | `root?`, `rating?` (csv of keep/review/cull, matched against the *effective* rating), `media?` (image/video), `keywords?` (all must match), `place_contains?`, `person?`, `time_of_day?`, `lighting?`, `has_speech?`, `primary_only?`, `limit=50` (max 500) | `{matches: [{path, media_type, rating, user_rating, effective_rating, place, keywords, scene, creation_time}], total, truncated}` |
| `read_sidecar` | `path` | the full `.description.md` text |
| `archive_overview` | `root?` | `_INDEX.md` (truncated at 64 KB with a note), or a one-line instruction to run `fdx-master` |
| `contact_sheet` | `paths` (≤20), `columns=4` | one JPEG (numbered grid) + a legend `n. path — effective rating — scene` |
| `set_user_rating` | `path`, `rating` (keep/review/cull, or "" to clear), `note=""` | the file's `{rating, user_rating, user_note, user_rated_at}` after the write |

`query_media` builds the exact `fdx-query` argument list and parses it with
`fdx-query`'s own parser, so the two can never drift. `root` may be omitted
when the server has exactly one root.

Contact-sheet thumbnails: stills go through `images.render_preview` (RAW via
the embedded JPEG, HEIC via pillow-heif), videos through one mid-clip `ffmpeg`
frame. Each thumbnail is fitted into a `SHEET_THUMB_PX = 384` square and
numbered top-left; `columns` caps at 6. More than 20 paths is an error that
tells the host to narrow the query, never a silent truncation.

## Schema change

Three optional frontmatter keys, written only by `set_user_rating`:

```yaml
user_rating: keep        # keep | review | cull — the human's decision
user_note: "eyes closed but it's the only frame of the cub"
user_rated_at: 2026-09-09T02:41:10
```

`parsing.effective_rating(rec)` = `user_rating` when it is a valid value, else
`rating`. Used by `fdx-query --rating`, `fdx-master` (rating counts, cull
pile), and `fdx-xmp` (stars/label). The model's `rating` is never modified.
`pipeline.serialize_sidecar` carries these keys over from an existing sidecar
when it rewrites one (`--force`, regrouping), so a re-index cannot erase a
human decision.

## Module layout

- `src/framedex/mcp_tools.py` (new): everything testable without the SDK.
  `resolve_under_roots`, `query_media`, `read_sidecar`, `archive_overview`,
  `sheet_layout` (pure geometry), `build_contact_sheet` (Pillow compositing),
  `video_frame` (ffmpeg), `set_user_rating`, constants
  (`SHEET_MAX_IMAGES`, `SHEET_THUMB_PX`, `SHEET_MAX_COLUMNS`,
  `OVERVIEW_MAX_BYTES`, `QUERY_DEFAULT_LIMIT`, `QUERY_MAX_LIMIT`,
  `USER_RATING_VALUES`). Raises `ValueError` for caller mistakes; the server
  layer maps those to `ToolError`. No `mcp` import.
- `src/framedex/mcp_server.py` (new): `MCPServer("framedex")`, one thin
  `@mcp.tool()` wrapper per tool (docstrings are what the host's model reads),
  `main()` parses roots and runs stdio. `mcp`/Pillow are imported lazily with
  the same actionable "install the extra" message the video stack uses.
- `src/framedex/query.py`: `build_parser(exit_on_error=True)` and
  `run_query(root, args) -> list[dict]` factored out of `main()`; behavior of
  the CLI unchanged.
- `src/framedex/parsing.py`: `effective_rating`.
- `src/framedex/pipeline.py`: `serialize_sidecar` preserves the user keys.
- `src/framedex/master_index.py`, `query.py`, `xmp_export.py`: use
  `effective_rating`.
- `pyproject.toml`: extra `mcp = ["mcp>=2.0", "Pillow>=10.0.0"]`, added to
  `all`; script `fdx-mcp = "framedex.mcp_server:main"`; `uv.lock` updated.
- Docs: README "Talk to your archive (fdx-mcp)"; `docs/mcp.md` (host setup
  for Claude Code / Claude Desktop / LM Studio, what leaves the machine,
  cost); SKILL.md; CHANGELOG.

## Data flow

```
MCP host (Claude Desktop / LM Studio / Claude Code)
   | stdio JSON-RPC
fdx-mcp (mcp_server.py)  -- thin wrappers, ToolError mapping
   |
mcp_tools.py  -- roots guard, fdx-query parser + run_query, sidecar read/write,
   |             contact sheet (render_preview / ffmpeg + Pillow)
sidecars (.description.md)  <-- the only thing written (user_* keys)
originals                   <-- read only, for thumbnails
```

## Error handling

- Path outside every root → `ToolError("<path> is outside the configured roots: ...")`.
- Media not indexed (no sidecar) → `ToolError` naming the `fdx` command to run.
- Bad `rating` → rejected by the tool's `Literal` schema before the call; the
  tools layer re-validates (`ValueError`).
- `contact_sheet` with >20 paths → `ToolError` ("narrow the query"); a member
  that cannot be rendered (RAW without preview, ffmpeg failure) is drawn as a
  labelled grey cell, listed in the legend as "(no preview)", never dropped
  silently.
- `archive_overview` without `_INDEX.md` → the instruction string, not an
  error (the host can still query).
- The server never catches broad exceptions: an unexpected failure surfaces
  as the SDK's generic tool error with the message.

## Testing (hermetic; CI installs no `mcp`, no Pillow, no ffmpeg)

- `tests/test_mcp_tools.py`: roots guard (inside, outside, `..` traversal,
  symlink resolved, missing file); `query_media` returns effective ratings
  and honours `limit`/`truncated`, and goes through the real parser (a bad
  filter value raises, not exits); `read_sidecar` on unindexed media; overview
  present / absent / truncated; `sheet_layout` geometry (cells, rows,
  numbering, column cap); `video_frame` builds the ffprobe/ffmpeg argv with a
  mocked `subprocess.run`; `set_user_rating` writes the three keys, preserves
  the body bytes, clears on "", rejects bad values, uses the atomic writer, is
  idempotent. One Pillow compositing test guarded by
  `pytest.importorskip("PIL")` (runs locally, skips on CI).
- `tests/test_pipeline.py` (or `test_images.py`): `serialize_sidecar`
  preserves `user_*` across a rewrite and never invents them.
- `tests/test_parsing.py`: `effective_rating` matrix; `test_query.py`,
  `test_master_index.py`, `test_xmp_export.py`: the override wins.
- `tests/test_mcp_server.py`: guarded by `pytest.importorskip("mcp")`: the six
  tools are registered with the expected names and a `set_user_rating` call
  through the in-memory client round-trips; without `mcp`, `main()` exits
  with the install hint (import mocked).

## Implementation plan (TDD, one commit per task)

1. **`parsing.effective_rating`** + use in `query.matches` (`--rating`),
   `master_index` (counts, cull pile), `xmp_export` (`build_xmp` reads the
   effective rating; `run` validates it). Tests first in each.
2. **`serialize_sidecar` preserves `user_rating`/`user_note`/`user_rated_at`**
   from an existing sidecar (keys placed before `indexed_at`). Test: rewrite
   keeps them; absent stays absent; explicit new values win.
3. **`query.build_parser` / `run_query` refactor**; `main()` becomes
   parse → run → print. Existing CLI tests must pass unchanged; add one test
   that `run_query` returns records with resolved absolute paths.
4. **`mcp_tools`: roots + query + read + overview.** `resolve_under_roots`,
   `query_media(roots, ...)` (argv built from kwargs; `--limit` handled by
   `run_query`; `total`/`truncated` computed from an unlimited pass),
   `read_sidecar`, `archive_overview`.
5. **`mcp_tools.set_user_rating`**: frontmatter split/merge, `USER_RATING_VALUES`,
   `atomic_write_text`, returns the four fields.
6. **`mcp_tools` contact sheet**: `sheet_layout(n, columns)` → list of
   `(index, x, y)` + canvas size; `render_thumb(path, tmp_dir)` (still via
   `images.render_preview`, video via `video_frame`); `build_contact_sheet(
   entries, columns) -> tuple[bytes, list[str]]` with Pillow imported lazily.
7. **`mcp_server.py` + packaging**: tool wrappers, `main()`, extra, entry
   point, `uv lock`. Smoke test through the SDK's in-memory client (local
   only). Manual check: `fdx-mcp <root>` from `claude mcp add` and from LM
   Studio's `mcp.json`, one contact-sheet call each.
8. **Docs**: README section, `docs/mcp.md`, SKILL.md (new verbs: "show me",
   "mark as keep"), CHANGELOG.

## Open decisions (defaults chosen; flag if you disagree)

1. `user_rating` reuses the keep/review/cull vocabulary rather than 1-5 stars:
   one vocabulary across the tool, and the photographer's own 4/5★ picks stay
   Lightroom's business (the 3★ ceiling from `fdx-xmp` is unchanged; a
   `user_rating: keep` maps to 3★ like a model `keep`).
2. `query_media` exposes the common `fdx-query` filters only (no
   `--people-count`, `--face-count`, `--focus/--stability/--exposure`,
   `--dominant-color`, durations). They are five lines each to add if a host
   asks for them.
3. Videos on a contact sheet are one mid-clip frame, not the five indexed
   frames (one image per call, and the sidecar's description already covers
   the clip). `notable_timestamp`, when present, is preferred over the midpoint.
4. No read-only flag: the only write is a sidecar frontmatter key, and hosts
   that want read-only can leave `set_user_rating` unapproved.

## Revisions after Codex plan review (2026-09-09, ledger REVIEW.md Round 4)

The architecture stands (MCP over stdio, six tools, one write). These
decisions replace the corresponding text above:

- **Discovery.** `query_media` gains `folder` (a subpath under the scan root;
  only sidecars below it) and `offset` (with `limit`, max 500) for
  deterministic paging. `root` stays the *scan* root, never a subfolder (the
  sidecar `path` field is relative to it). `fdx-query` gains the same
  `--folder` and `--offset` so the CLI and the tool share one implementation.
- **No argv bridge.** `query.py` exposes a `Filters` dataclass (one field per
  CLI filter, with defaults) and `run_query(root, filters) -> QueryResult`
  (`records`, `total`, `skipped_malformed`). The CLI parser fills a `Filters`;
  `query_media` builds one directly from validated arguments (`rating` values
  must be keep/review/cull). One scan; count, then slice.
- **Containment on the resolved target, every time.** `_guard(path, roots)`
  resolves symlinks and checks containment before every read or write: the
  sidecar, the media file, `_INDEX.md`, each contact-sheet member. A
  frontmatter `path` that resolves outside its root is dropped from results
  (counted in `skipped_malformed`). Relative paths are rejected when more than
  one root is configured. Tests cover derived escapes, not just the helper.
- **Stable reference.** Results carry `sidecar` (always) and `path` (null for
  Photos-managed assets). `read_sidecar` and `set_user_rating` accept either a
  media path or a sidecar path.
- **`set_user_rating` semantics.** Reads and writes bytes (`atomic_write_bytes`)
  so the body is preserved byte-for-byte (CRLF included); malformed existing
  YAML is an error, nothing is written; unchanged rating+note is a no-op
  (timestamp untouched); `note=""` removes `user_note`; `rating=""` removes all
  three keys. A process-local lock per sidecar serialises concurrent tool
  calls (the SDK runs sync tools on worker threads). Running `fdx --force` on
  a folder while rating it in chat is documented as unsupported.
- **Surviving Phase 3.** `process_group` no longer deletes the primary's old
  sidecar; it rewrites it with `group.incomplete: true` (content kept), and the
  resume check treats that as not done. `serialize_sidecar` carries
  `user_rating` / `user_note` / `user_rated_at` over from the existing file.
  `atomic_write_*` create their temp file exclusively via `mkstemp` (unique
  name; a planted symlink is never followed).
- **Effective rating everywhere it is shown.** `fdx-query --with-description`
  prints `keep (user)` when overridden; JSON records gain `effective_rating`;
  `fdx-master` marks `(user)` in the cull list and adds "User ratings: N
  files"; `fdx-xmp` adds the keyword `user-rated` (stars unchanged: a human
  `keep` is still 3★). An invalid `user_rating` value warns on stderr and is
  ignored.
- **Contact sheet.** No `columns` argument: fixed 4-column geometry, 384 px
  cells, at most 20 images (≈1600 px wide). Zero paths is an error; a member
  that fails to render becomes a labelled grey cell with the reason; if
  nothing renders the call fails. `ffprobe`/`ffmpeg` run with `-nostdin`, a
  60 s timeout, a per-call temp directory, and validated finite durations;
  `notable_timestamp` (when finite and inside the clip) beats the midpoint.
- **Read-only mode.** `fdx-mcp --read-only` omits `set_user_rating`; read
  tools declare `readOnlyHint`.
- **Overview.** The text is prefixed with "Snapshot generated <generated_at>;
  run `fdx-master <root>` to refresh", and the docs say what `fdx-xmp`
  exports (proprietary RAW only, a separate command, conflicts skipped).
- **Errors.** Expected failures (missing `[mcp]`/`[images]` extra, ffmpeg or
  exiftool absent, permission denied, undecodable image) map to `ToolError`
  with the fix in the message; the server keeps running.
- **Cost / privacy wording.** "One bounded image per successful call; zero
  model calls inside framedex. The host decides what its model receives:
  thumbnails, paths, GPS, names, notes, transcripts."
- **Packaging / CI.** `mcp>=2,<3`; docs say `[mcp,images]` plus exiftool and
  ffmpeg. A second CI job installs the `mcp` extra and runs the integration
  tests for real (in-memory transport, generated pixels, mocked subprocesses);
  the missing-extra tests stay unconditional.
- **`scene`** comes from `parsing.scene_sentence(body)` (the `**Scene:**` line,
  moved out of `xmp_export`).
