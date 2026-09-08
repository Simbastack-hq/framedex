#!/usr/bin/env python3
"""
framedex.mcp_server — `fdx-mcp`: the archive as MCP tools over stdio.

A thin wrapper: every tool is a function in `mcp_tools`; this module only
registers them with the MCP SDK, maps caller mistakes to ToolError, and
parses the command line. The host (Claude Code, Claude Desktop, LM Studio,
any MCP client) supplies the model; framedex makes no model call.

Usage:
    fdx-mcp /Volumes/SSD-2024 [/Volumes/SSD-2023 ...] [--read-only]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Literal

from framedex.mcp_tools import (
    QUERY_DEFAULT_LIMIT,
    SHEET_MAX_IMAGES,
    Roots,
    archive_overview,
    build_contact_sheet,
    query_media,
    read_sidecar,
    set_user_rating,
)

INSTRUCTIONS = (
    "framedex indexes a photo/video archive into plain-text sidecars, one per "
    "file, with the indexer's rating (keep/review/cull), keywords, place, EXIF, "
    "and a description. Workflow: start with list_roots. Use archive_overview "
    "to see trip folders, counts, and top keywords (if it is missing, go "
    "straight to query_media). Search a trip with query_media(folder=..., "
    "rating='keep,review'): this filters stored metadata, not semantic "
    "similarity. Compare candidates by passing 1-"
    f"{SHEET_MAX_IMAGES} media paths to contact_sheet (batch larger sets). "
    "Read a sidecar for the full description or transcript. When the person "
    "decides, record it with set_user_rating (absent in read-only mode). In "
    "results, `path` is the media file (may be null when the original is not "
    "on disk) and `sidecar_path` is the description file; both read_sidecar "
    "and set_user_rating accept either."
)


def build_server(roots: Roots, *, read_only: bool) -> Any:
    """Construct the MCP server with the tools registered. Imports the SDK
    lazily so the error names the extra to install."""
    try:
        from mcp.server.mcpserver import Image, MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as e:
        raise RuntimeError(
            "fdx-mcp needs the 'mcp' extra: uv pip install -e '.[mcp]'  "
            f"(import failed: {e})"
        ) from e

    read = ToolAnnotations(read_only_hint=True)
    write = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=True
    )
    mcp = MCPServer("framedex", instructions=INSTRUCTIONS)

    def tool_error(e: Exception) -> ToolError:
        return ToolError(str(e))

    def list_roots_tool() -> list[str]:
        """Archive roots this server can see. Paths passed to the other tools
        must lie under one of them."""
        return [str(r) for r in roots.roots]

    def query_media_tool(
        root: str | None = None,
        folder: str | None = None,
        rating: str | None = None,
        media: Literal["image", "video"] | None = None,
        keywords: list[str] | None = None,
        place_contains: str | None = None,
        person: str | None = None,
        time_of_day: str | None = None,
        lighting: str | None = None,
        has_speech: bool = False,
        primary_only: bool = False,
        offset: int = 0,
        limit: int = QUERY_DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Find indexed media by stored metadata (a subset of fdx-query's
        filters; not semantic search). Filters AND together. `root`: an exact
        value from list_roots, required when several roots are configured.
        `folder`: a subfolder of the root (a trip or a shoot). `rating`: a
        comma list of keep/review/cull, matched against the effective rating
        (a person's user_rating overrides the indexer's rating). `keywords`:
        every one must match, case-insensitively. `person`: a face
        cluster_id as stored in sidecars. `lighting` / `time_of_day`: comma
        lists of the exact stored values (see a sidecar for the vocabulary).
        `has_speech=false` applies no speech filter. `primary_only=true`
        hides burst/pair alternates. Results are ordered by sidecar path, not
        by quality. Page with `offset`/`limit` (limit at most 500); `total` is
        the match count and `has_more` says whether another page exists. Each
        match carries `path` (absolute media path, or null when the original
        is not on disk), `sidecar_path` (always usable with read_sidecar and
        set_user_rating), `rating` (the indexer's), `user_rating`,
        `effective_rating`, `place`, `keywords`, and the one-line `scene`."""
        try:
            return query_media(
                roots,
                root=root,
                folder=folder,
                rating=rating,
                media=media,
                keywords=keywords,
                place_contains=place_contains,
                person=person,
                time_of_day=time_of_day,
                lighting=lighting,
                has_speech=has_speech,
                primary_only=primary_only,
                offset=offset,
                limit=limit,
            )
        except ValueError as e:
            raise tool_error(e) from e

    def read_sidecar_tool(path: str) -> str:
        """The full plain-text sidecar (`.description.md`) of one media file:
        frontmatter (EXIF, GPS, ratings, keywords, faces) and the description
        or transcript. `path` is the media file or its `sidecar_path` (use the
        latter when `path` is null)."""
        try:
            return read_sidecar(roots, path)
        except ValueError as e:
            raise tool_error(e) from e

    def archive_overview_tool(root: str | None = None) -> str:
        """The drive-level `_INDEX.md` (counts, top keywords and places, the
        cull list, per-folder summary), as a snapshot from the last
        `fdx-master` run."""
        try:
            return archive_overview(roots, root)
        except ValueError as e:
            raise tool_error(e) from e

    def contact_sheet_tool(paths: list[str]) -> list[Any]:
        """Render 1-20 media files (stills, or one frame of a clip) into one
        numbered grid image plus a legend (`n. <absolute path> —
        effective_rating=... — scene=...`), so candidates can be compared in a
        single look. Numbers follow input order; use the full path from the
        legend when recording a pick. Needs media files on disk (not a null
        `path`); split larger selections into batches. A file that cannot be
        rendered keeps a numbered grey cell and says why in the legend."""
        try:
            jpeg, legend = build_contact_sheet(roots, paths)
        except (ValueError, RuntimeError) as e:
            raise tool_error(e) from e
        return [Image(data=jpeg, format="jpeg"), "\n".join(legend)]

    def set_user_rating_tool(
        path: str, rating: Literal["keep", "review", "cull", ""], note: str = ""
    ) -> dict[str, Any]:
        """Set the person's rating for one media or sidecar path:
        `user_rating` keep/review/cull, plus an optional `note`. The indexer's
        stored `rating` stays unchanged; the person's rating wins in
        fdx-query, fdx-master and fdx-xmp (Lightroom). `note` replaces the
        existing user note; omitting it removes that note. `rating=""`
        removes the user rating, note, and timestamp. Never edits the media
        file."""
        try:
            return set_user_rating(roots, path, rating, note)
        except ValueError as e:
            raise tool_error(e) from e

    # add_tool, not @mcp.tool: the SDK is an optional extra, so under the base
    # (extras-free) mypy run `mcp` is Any and a decorator would be "untyped".
    mcp.add_tool(list_roots_tool, name="list_roots", annotations=read)
    mcp.add_tool(query_media_tool, name="query_media", annotations=read)
    mcp.add_tool(read_sidecar_tool, name="read_sidecar", annotations=read)
    mcp.add_tool(archive_overview_tool, name="archive_overview", annotations=read)
    mcp.add_tool(
        contact_sheet_tool,
        name="contact_sheet",
        annotations=read,
        structured_output=False,
    )
    if not read_only:
        mcp.add_tool(set_user_rating_tool, name="set_user_rating", annotations=write)
    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fdx-mcp",
        description="Serve an indexed framedex archive to an MCP host over stdio.",
    )
    parser.add_argument("roots", nargs="+", help="Archive root(s) the server may read")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Do not offer set_user_rating (no writes at all).",
    )
    args = parser.parse_args()
    try:
        roots = Roots.from_args(args.roots)
    except ValueError as e:
        sys.exit(str(e))
    try:
        server = build_server(roots, read_only=args.read_only)
    except RuntimeError as e:
        sys.exit(str(e))
    mode = " (read-only)" if args.read_only else ""
    print(
        f"fdx-mcp: serving {', '.join(str(r) for r in roots.roots)} over stdio{mode}",
        file=sys.stderr,
    )
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
