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
    "framedex indexes a photo/video archive into plain-text sidecars. "
    "Start with list_roots, then archive_overview or query_media (filter by "
    "rating, folder, keywords, place). Every match has a path; read_sidecar "
    "returns its full description, and contact_sheet renders up to "
    f"{SHEET_MAX_IMAGES} paths as one numbered image so you can compare them. "
    "set_user_rating records the person's decision (keep/review/cull) next to "
    "the model's; it never edits the media file."
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

    @mcp.tool(name="list_roots", annotations=read)
    def list_roots_tool() -> list[str]:
        """Archive roots this server can see. Paths passed to the other tools
        must lie under one of them."""
        return [str(r) for r in roots.roots]

    @mcp.tool(name="query_media", annotations=read)
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
        """Find indexed media. `rating` is a comma list of keep/review/cull and
        matches the effective rating (a person's rating overrides the
        model's). `folder` restricts to a subfolder of the root (a trip or a
        shoot). `keywords` must all match. Page with `offset`/`limit` (limit
        at most 500); `total` and `truncated` say whether more exist. Each
        match carries `path`, `sidecar`, ratings, `place`, `keywords`, and the
        one-line `scene`."""
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

    @mcp.tool(name="read_sidecar", annotations=read)
    def read_sidecar_tool(path: str) -> str:
        """The full plain-text sidecar (`.description.md`) of one media file:
        frontmatter (EXIF, GPS, ratings, keywords, faces) and the description
        or transcript. `path` is the media file or its sidecar."""
        try:
            return read_sidecar(roots, path)
        except ValueError as e:
            raise tool_error(e) from e

    @mcp.tool(name="archive_overview", annotations=read)
    def archive_overview_tool(root: str | None = None) -> str:
        """The drive-level `_INDEX.md` (counts, top keywords and places, the
        cull list, per-folder summary), as a snapshot from the last
        `fdx-master` run."""
        try:
            return archive_overview(roots, root)
        except ValueError as e:
            raise tool_error(e) from e

    @mcp.tool(name="contact_sheet", annotations=read, structured_output=False)
    def contact_sheet_tool(paths: list[str]) -> list[Any]:
        """Render 1-20 media files (stills, or one frame of a clip) into one
        numbered grid image plus a legend (`n. path — rating — scene`), so
        candidates can be compared in a single look. Narrow the query and
        call again for more."""
        try:
            jpeg, legend = build_contact_sheet(roots, paths)
        except (ValueError, RuntimeError) as e:
            raise tool_error(e) from e
        return [Image(data=jpeg, format="jpeg"), "\n".join(legend)]

    if not read_only:

        @mcp.tool(name="set_user_rating", annotations=write)
        def set_user_rating_tool(
            path: str, rating: Literal["keep", "review", "cull", ""], note: str = ""
        ) -> dict[str, Any]:
            """Record the person's decision for one file in its sidecar:
            `user_rating` keep/review/cull (an empty string clears it) and an
            optional `note`. The model's own rating stays; the person's wins
            in fdx-query, fdx-master and fdx-xmp (Lightroom). Never edits the
            media file."""
            try:
                return set_user_rating(roots, path, rating, note)
            except ValueError as e:
                raise tool_error(e) from e

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
