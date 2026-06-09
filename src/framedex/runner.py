#!/usr/bin/env python3
"""
framedex.runner — shared CLI run-orchestration for the `fdx` / `fdx-photos`
entry points.

This is the media-agnostic glue both entries need but that is not a pure
pipeline primitive: vision-model resolution, the cost/backend announce block,
backend availability checks + client construction, face-DB setup, and the
per-file result tally + reporting.

Kept deliberately free of heavy imports (whisperx / torch / Pillow / osxphotos)
and of the media pipelines themselves, so it stays a leaf dependency:
`runner -> pipeline / face_db` only. `index_videos` and `photos_indexer` import
*from* here; this module imports neither of them. `anthropic` is imported lazily
inside the api branch so a cli/local run never touches it.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framedex import face_db
from framedex.pipeline import (
    COST_PER_CALL_USD_CLI,
    COST_PER_CALL_USD_LOCAL,
    VISION_MODELS,
    ProcessResult,
    check_claude_cli,
    check_local_endpoint,
    resolve_anthropic_key,
)

# ---------------------------------------------------------------------------
# Vision model + backend
# ---------------------------------------------------------------------------


def resolve_vision_model(args: argparse.Namespace) -> tuple[str, float]:
    """Validate ``args.vision_model`` and return ``(model_id, cost_per_call)``
    for the selected backend. Exits with a clear message on an unknown model."""
    if args.vision_model not in VISION_MODELS:
        known = ", ".join(sorted(VISION_MODELS))
        sys.exit(f"--vision-model must be one of: {known}")
    model_cfg = VISION_MODELS[args.vision_model]
    if args.backend == "api":
        return str(model_cfg["api"]), float(model_cfg["cost_per_call_api"])
    if args.backend == "cli":
        return str(model_cfg["cli"]), COST_PER_CALL_USD_CLI
    # local: send the configured model name, or let the server pick.
    return (args.local_model or "(loaded model in LM Studio)"), COST_PER_CALL_USD_LOCAL


def announce_cost(
    backend: str,
    model_id: str,
    cost_per_call: float,
    n_todo: int,
    local_base_url: str,
) -> None:
    """Print the vision-backend + estimated-cost block. Byte-identical to the
    text both entries printed inline before the dedupe."""
    est_cost = n_todo * cost_per_call
    if backend == "api":
        print(f"  vision: api / {model_id}")
        print(f"  estimated Anthropic API cost: ~${est_cost:.2f}")
    elif backend == "cli":
        print(f"  vision: cli (Max) / {model_id}")
        print("  marginal cost: $0 (Max subscription)")
    else:
        print(f"  vision: local / {model_id} @ {local_base_url}")
        print("  marginal cost: $0 (fully local)")
    print()


def wire_vision_backend(args: argparse.Namespace) -> Any:
    """Check the selected backend is usable and return the anthropic client
    (api) or ``None`` (cli/local). Exits with an actionable message on failure
    and prints the same ``Vision: ...`` notices both entries used."""
    if args.backend == "api":
        api_key = resolve_anthropic_key()
        if not api_key:
            sys.exit(
                "--backend api requires ANTHROPIC_API_KEY env or "
                "~/.claude/credentials/anthropic-key.txt"
            )
        import anthropic as _anthropic

        client = _anthropic.Anthropic(api_key=api_key)
        print("Vision: direct Anthropic API\n")
        return client
    if args.backend == "cli":
        if not check_claude_cli():
            sys.exit(
                "--backend cli requires the `claude` CLI on PATH. "
                "Install Claude Code or pass --backend api with ANTHROPIC_API_KEY."
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "NOTE: ANTHROPIC_API_KEY is set in your environment. The script will\n"
                "      explicitly remove it from each `claude` subprocess so calls go\n"
                "      against your Max subscription instead of API billing.\n"
            )
        else:
            print("Vision: claude CLI -> Max subscription\n")
        return None
    # local
    ok, info = check_local_endpoint(args.local_base_url)
    if not ok:
        sys.exit(
            f"--backend local: cannot reach {args.local_base_url} ({info}). "
            "Start LM Studio and load a vision-capable model first."
        )
    print(f"Vision: local LM Studio at {args.local_base_url} ({info})\n")
    return None


# ---------------------------------------------------------------------------
# Face DB
# ---------------------------------------------------------------------------


def setup_face_db(args: argparse.Namespace) -> sqlite3.Connection | None:
    """Initialize insightface + open the face DB, or return ``None`` when
    ``--no-faces`` is set or detection is unavailable. Byte-identical prints."""
    if args.no_faces:
        return None
    ok, info = face_db.init_face_app()
    if not ok:
        print(f"Face detection unavailable: {info}")
        print("Continuing without face detection. Pass --no-faces to silence.")
        return None
    conn = face_db.open_db(Path(args.face_db))
    stats = face_db.db_stats(conn)
    print(f"Face detection: insightface ({info})")
    print(
        f"Face DB: {args.face_db} (currently {stats['faces']} faces, "
        f"{stats['clusters']} clusters, {stats['named_clusters']} named)\n"
    )
    return conn


# ---------------------------------------------------------------------------
# Per-file result tally + reporting
# ---------------------------------------------------------------------------


@dataclass
class RunTally:
    """Mutable counters accumulated across a run.

    Only the counters the end-of-run summary actually prints live here. Short
    skips are reported inline per-file but never summarized, so they get no
    counter (matching today's behavior).
    """

    processed: int = 0
    errors: int = 0
    skipped_too_long: int = 0
    skipped_no_preview: int = 0
    actual_cost: float = 0.0


def record_result(
    result: ProcessResult,
    tally: RunTally,
    *,
    backend: str,
    max_duration_min: int,
) -> None:
    """Apply one ``ProcessResult`` to ``tally`` and print the per-file outcome
    line. The caller prints the ``[i/n] <header>`` line first (it differs per
    adapter: a relative path for ``fdx`` vs ``filename (uuid=...)`` for
    ``fdx-photos``). Knows every skip reason so both loops report identically.
    """
    reason = result.skipped_reason
    if reason == "short":
        print("  skipped (duration < 0.5s)")
        return
    if reason == "too_long":
        print(f"  skipped (duration > --max-duration {max_duration_min} min)")
        tally.skipped_too_long += 1
        return
    if reason == "no_preview":
        print("  skipped (RAW without an embedded preview to read)")
        tally.skipped_no_preview += 1
        return
    if reason == "vision_error":
        # No sidecar written — count as an error so the run exits non-zero and
        # the file is retried on the next pass.
        tally.errors += 1
        return

    assert result.sidecar is not None
    tally.actual_cost += result.cost
    tally.processed += 1
    faces_note = (
        f", {len(result.detected_faces)} faces" if result.detected_faces else ""
    )
    rating_note = f", rated {result.rating}"
    if backend == "api":
        print(
            f"  -> {result.sidecar.name}  "
            f"(cost ~${tally.actual_cost:.2f}{rating_note}{faces_note})"
        )
    else:
        print(f"  -> {result.sidecar.name}  ({rating_note}{faces_note})")
