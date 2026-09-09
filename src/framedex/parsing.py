"""Pure helpers for parsing and normalizing model and tool output.

Stdlib-only by design: this module imports nothing heavy (no whisperx,
torch, requests), so it can be unit-tested without installing the full
runtime stack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, TypeGuard


def coerce_people_count(value: Any, face_count: int) -> int:
    """Normalize people_count to an int. Models occasionally return strings
    like 'many', 'lots', '~15'. Coerce defensively: clean int → use; fuzzy
    word → fall back to face_count (real ground truth from insightface);
    junk → 0."""
    if isinstance(value, bool):
        return 0  # avoid True→1 surprise
    if isinstance(value, int):
        return max(0, min(99, value))
    if isinstance(value, float):
        return max(0, min(99, int(value)))
    if isinstance(value, str):
        s = value.strip().lower().lstrip("~≈≥<>=")
        # Direct numeric parse
        try:
            return max(0, min(99, int(float(s))))
        except (ValueError, TypeError):
            pass
        # Word-style fuzzy counts
        if s in {
            "many",
            "lots",
            "lots of people",
            "a lot",
            "crowd",
            "crowded",
            "numerous",
            "several",
            "group",
        }:
            # Use face_count as a real lower bound; min 10 since "many" implies >10
            return max(10, min(99, face_count))
        if s in {"few", "couple", "pair"}:
            return 2
        if s in {"some", "a few"}:
            return 3
        if s in {"none", "no one", "empty", "no people", "no one in frame", ""}:
            return 0
    return 0


def is_usable_path(value: Any) -> TypeGuard[str]:
    """True if a sidecar `path` field is a usable, non-empty string.

    A record whose `path` is missing, blank, or a non-string is not directly
    usable for output that pipes to a media player. Callers warn and skip such
    records rather than crash on `Path()` or emit the `.description.md` sidecar
    path as if it were the media path (issue #14)."""
    return isinstance(value, str) and bool(value.strip())


def is_permission_denied(text: str) -> bool:
    """True if a Claude CLI response is a permission-denied message rather
    than a real description.

    The CLI exits 0 with text like "I need permission to read..." when a
    tool use is blocked; treating that as a description writes useless
    sidecars. The length guard avoids flagging a long, legitimate
    description that merely mentions the word "permission".
    """
    telltales = (
        "i need permission",
        "i don't have permission",
        "i do not have permission",
        "permission to read",
        "please grant",
        "i'm not able to read",
        "i am not able to read",
        "i cannot access",
        "request access",
    )
    lower = text.lower()
    return any(t in lower for t in telltales) and len(text) < 600


def pick_diar_auth_kwarg(params: Iterable[str]) -> str:
    """Choose the auth keyword for whisperx's DiarizationPipeline.

    Newer whisperx uses ``token``; older releases use ``use_auth_token``.
    Defaults to ``token`` when the signature exposes neither (e.g. auth is
    accepted via ``**kwargs``).
    """
    names = set(params)
    if "token" in names:
        return "token"
    if "use_auth_token" in names:
        return "use_auth_token"
    return "token"


def is_group_stub(rec: dict[str, Any]) -> bool:
    """A burst / RAW+JPEG member whose assessment was copied from its group
    primary (`group.primary: false`). Stubs are real sidecars but not
    assessments: drive stats and the cull pile count primaries only."""
    group = rec.get("group")
    return isinstance(group, dict) and group.get("primary") is False


# ---------------------------------------------------------------------------
# Ratings: the model's verdict vs the human's decision
# ---------------------------------------------------------------------------

RATING_VALUES = ("keep", "review", "cull")

_SCENE_RE = re.compile(r"\*\*Scene:\*\*\s*(.+)")


def is_user_rated(rec: dict[str, Any]) -> bool:
    """True when the sidecar carries a valid `user_rating` (written by
    fdx-mcp's set_user_rating: the photographer's own decision)."""
    return rec.get("user_rating") in RATING_VALUES


def effective_rating(rec: dict[str, Any]) -> Any:
    """The rating that counts everywhere ratings are read: a valid
    `user_rating` (the human's decision) wins over the model's `rating`. An
    invalid override is ignored (callers report it); the model value passes
    through unchanged, whatever it is."""
    if is_user_rated(rec):
        return rec["user_rating"]
    return rec.get("rating")


def scene_sentence(body: str) -> str:
    """The `**Scene:**` sentence of a sidecar body ("" when absent): the one-line
    summary fdx-xmp exports as the caption and fdx-mcp returns per match."""
    m = _SCENE_RE.search(body)
    return m.group(1).strip() if m else ""
