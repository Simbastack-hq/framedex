"""Pure helpers for parsing and normalizing model and tool output.

Stdlib-only by design: this module imports nothing heavy (no whisperx,
torch, requests), so it can be unit-tested without installing the full
runtime stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


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
