# fdx-mcp: the archive as MCP tools

`fdx-mcp` serves one or more indexed roots to any MCP client over stdio.
The host (Claude Code, Claude Desktop, LM Studio, your own agent) supplies
the model; framedex makes no model call of its own. It hands the host facts
from the sidecars and, on request, one picture.

```bash
uv pip install -e '.[mcp]'        # MCP SDK + Pillow; add [images] for HEIC thumbnails
fdx-mcp /Volumes/SSD-2024         # one or more roots that fdx has indexed
fdx-mcp /Volumes/SSD-2024 /Volumes/SSD-2023 --read-only
```

`exiftool` (RAW previews) and `ffmpeg` (video frames) must be on PATH for
contact sheets; the other tools need neither.

## Host setup

The command must be the `fdx-mcp` on PATH of the environment you installed
into (`which fdx-mcp`); hosts do not source your shell profile.

**Claude Code**

```bash
claude mcp add framedex -- /path/to/.venv/bin/fdx-mcp /Volumes/SSD-2024
```

**Claude Desktop** (`claude_desktop_config.json`) and **LM Studio**
(`mcp.json`, Program → Integrations) use the same shape:

```json
{
  "mcpServers": {
    "framedex": {
      "command": "/path/to/.venv/bin/fdx-mcp",
      "args": ["/Volumes/SSD-2024"]
    }
  }
}
```

With LM Studio and a local vision model, nothing leaves the machine.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `list_roots` | – | the roots this server can see |
| `query_media` | `root?` (an exact value from `list_roots`; required with several roots), `folder?` (subpath of the root), `rating?` (comma list of keep/review/cull, matched against the effective rating), `media?` (image/video), `keywords?` (all must match, case-insensitive), `place_contains?`, `person?` (a stored face `cluster_id`), `time_of_day?` / `lighting?` (comma lists of the stored values), `has_speech?`, `primary_only?`, `offset=0`, `limit=50` (max 500). Filters AND together; results are ordered by sidecar path | `{matches: [{path (absolute media path, or null when the original is not on disk), sidecar_path (always usable with read_sidecar / set_user_rating), media_type, rating (the indexer's), user_rating, effective_rating, place, keywords, scene, creation_time, group_alternate}], total, offset, limit, has_more, skipped_malformed, invalid_user_ratings}` |
| `read_sidecar` | `path` (a media file or its sidecar) | the full `.description.md` text |
| `archive_overview` | `root?` | `_INDEX.md`, prefixed as the snapshot it is (regenerate with `fdx-master`), cut at 64 KB |
| `contact_sheet` | `paths` (1-20 media files on disk) | one JPEG: a numbered 4-column grid of 384 px cells (about 1600 px wide), plus a legend `n. <absolute path> — effective_rating=keep (user) — scene=...` in input order. A file that cannot be rendered keeps a numbered grey cell and the legend says why; if nothing renders the call fails. Comparing the sheet needs a vision-capable model on the host |
| `set_user_rating` | `path` (media or sidecar path), `rating` (keep/review/cull, or "" to clear), `note=""` | the sidecar's `rating` (the indexer's, unchanged), `user_rating`, `user_note`, `user_rated_at`, and whether anything changed. `note` replaces the existing note; omitting it removes the note; `rating=""` removes all three keys. Absent with `--read-only` |

`query_media` exposes a subset of `fdx-query`'s metadata filters through the
same implementation (`query.Filters`, `query.run_query`), so a filter means
the same thing in both; it is not semantic search. `folder` is a
subpath of the scan root; `root` is always the directory the sidecars were
written from (their `path` fields are relative to it).

Stills are rendered through the same path the indexer uses (RAW via the
embedded JPEG preview, HEIC via pillow-heif); a clip contributes one frame at
its `notable_timestamp` when the sidecar has one, else its midpoint.

## The one write

`set_user_rating` writes three frontmatter keys into the sidecar and nothing
else:

```yaml
user_rating: keep        # the person's decision
user_note: only frame of the cub
user_rated_at: 2026-09-09T02:41:10
```

The model's `rating` is never modified; the body (description, transcript)
is preserved byte-for-byte; the write is atomic. The same rating and note
again is a no-op (the timestamp stays); an empty `rating` removes all three
keys. A user rating wins wherever ratings are read: `fdx-query --rating` and
`--with-description` (`keep (user)`), `fdx-master` (counts, a "User ratings"
line, `(user)` in the cull list), and `fdx-xmp` (stars from the user rating,
plus the keyword `user-rated`; a user `keep` is still 3★, the 4/5★ picks stay
yours in Lightroom). Re-indexing (`--force`, regrouping) carries the keys
over. Do not run `fdx --force` on a folder while rating it in chat: the two
processes do not lock against each other.

`fdx-xmp` is a separate step, and exports proprietary RAW only (`.dng`, JPEG,
HEIC embed their metadata in the file, which framedex never modifies); a
`.xmp` it did not write is skipped as a conflict.

## Safety

- Every path argument, and every path derived from a sidecar (`path` fields,
  symlinked sidecars, `_INDEX.md`, each contact-sheet member), is resolved
  (symlinks, `..`) and must lie under a configured root; otherwise the call
  fails. With several roots, relative paths are refused as ambiguous.
- Media files are only ever read. The only write is the sidecar key set
  above; `--read-only` removes it.
- Caller mistakes come back as tool errors with the fix in the message; the
  server keeps running.

## Cost and privacy

fdx-mcp makes no model calls. Host inference is where cost lives: each tool
result is tokens, and each contact sheet is one image per successful call (at
most 20 thumbnails, about 1600 px wide) that a vision-capable model must
process. Tool results can include thumbnails, file paths, GPS coordinates and
place names, face cluster ids, keywords, notes, and (via `read_sidecar`)
transcripts; the host may forward all of that to its model provider. Keeping
it local requires a host configured for local inference that does not forward
results externally (LM Studio with a local vision model, for example). Local
processing is not local inference.

## Limits

- Videos on a contact sheet are one frame, not the five indexed frames.
- No semantic search: `query_media` is the keyword/metadata filter set of
  `fdx-query`. Ask the host to read `archive_overview` first.
- `archive_overview` is a snapshot of the last `fdx-master` run; ratings
  made in chat show up there after the next run.
- stdio only.
