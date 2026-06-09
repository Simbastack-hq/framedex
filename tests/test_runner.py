"""Tests for framedex.runner — the shared CLI run-orchestration.

These pin the user-visible output that the Phase-1 dedupe must keep
byte-identical (the vision/cost announce lines and the per-file outcome
lines), plus the tally bookkeeping and backend/face-db branch behavior.

CI installs only pyyaml+requests, so nothing here may import anthropic /
insightface / whisperx; the heavy bits are patched or asserted-absent.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from framedex import pipeline, runner

# src/ dir so a clean-interpreter subprocess can import framedex (CI runs with
# pythonpath=src via pytest config, not an installed package).
_SRC_DIR = str(Path(runner.__file__).resolve().parent.parent)


def _ns(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        vision_model="haiku",
        backend="cli",
        local_model=None,
        local_base_url="http://localhost:1234/v1",
        no_faces=False,
        face_db="/tmp/faces.db",
    )
    base.update(kw)
    return argparse.Namespace(**base)


# --- resolve_vision_model --------------------------------------------------


def test_resolve_vision_model_cli() -> None:
    model_id, cost = runner.resolve_vision_model(
        _ns(backend="cli", vision_model="haiku")
    )
    assert model_id == "claude-haiku-4-5"
    assert cost == 0.0


def test_resolve_vision_model_api_uses_full_id_and_cost() -> None:
    model_id, cost = runner.resolve_vision_model(
        _ns(backend="api", vision_model="sonnet")
    )
    assert model_id == "claude-sonnet-4-6-20251001"
    assert cost == pytest.approx(0.008)


def test_resolve_vision_model_local_uses_local_model_name() -> None:
    model_id, cost = runner.resolve_vision_model(
        _ns(backend="local", local_model="google/gemma-4-26b")
    )
    assert model_id == "google/gemma-4-26b"
    assert cost == 0.0


def test_resolve_vision_model_local_default_label() -> None:
    model_id, _ = runner.resolve_vision_model(_ns(backend="local", local_model=None))
    assert model_id == "(loaded model in LM Studio)"


def test_resolve_vision_model_unknown_exits() -> None:
    with pytest.raises(SystemExit):
        runner.resolve_vision_model(_ns(vision_model="gpt9"))


# --- announce_cost (byte-identical snapshots) ------------------------------


def test_announce_cost_cli(capsys: pytest.CaptureFixture[str]) -> None:
    runner.announce_cost("cli", "claude-haiku-4-5", 0.0, 10, "http://x/v1")
    out = capsys.readouterr().out
    assert out == (
        "  vision: cli (Max) / claude-haiku-4-5\n"
        "  marginal cost: $0 (Max subscription)\n"
        "\n"
    )


def test_announce_cost_api(capsys: pytest.CaptureFixture[str]) -> None:
    runner.announce_cost("api", "claude-haiku-4-5-20251001", 0.002, 50, "http://x/v1")
    out = capsys.readouterr().out
    assert out == (
        "  vision: api / claude-haiku-4-5-20251001\n"
        "  estimated Anthropic API cost: ~$0.10\n"
        "\n"
    )


def test_announce_cost_local_includes_base_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner.announce_cost("local", "gemma", 0.0, 3, "http://localhost:1234/v1")
    out = capsys.readouterr().out
    assert out == (
        "  vision: local / gemma @ http://localhost:1234/v1\n"
        "  marginal cost: $0 (fully local)\n"
        "\n"
    )


# --- wire_vision_backend ---------------------------------------------------


def test_wire_cli_ok_returns_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("framedex.runner.check_claude_cli", lambda: True)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert runner.wire_vision_backend(_ns(backend="cli")) is None
    assert "Vision: claude CLI -> Max subscription" in capsys.readouterr().out


def test_wire_cli_missing_cli_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("framedex.runner.check_claude_cli", lambda: False)
    with pytest.raises(SystemExit):
        runner.wire_vision_backend(_ns(backend="cli"))


def test_wire_cli_scrubs_api_key_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("framedex.runner.check_claude_cli", lambda: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    runner.wire_vision_backend(_ns(backend="cli"))
    assert "ANTHROPIC_API_KEY is set" in capsys.readouterr().out


def test_wire_api_missing_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("framedex.runner.resolve_anthropic_key", lambda: None)
    with pytest.raises(SystemExit):
        runner.wire_vision_backend(_ns(backend="api"))


def test_wire_local_unreachable_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "framedex.runner.check_local_endpoint", lambda url: (False, "refused")
    )
    with pytest.raises(SystemExit):
        runner.wire_vision_backend(_ns(backend="local"))


def test_wire_local_ok_returns_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "framedex.runner.check_local_endpoint", lambda url: (True, "loaded: gemma")
    )
    assert runner.wire_vision_backend(_ns(backend="local")) is None
    assert "Vision: local LM Studio" in capsys.readouterr().out


# --- setup_face_db ---------------------------------------------------------


def test_setup_face_db_no_faces_returns_none_without_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Any:
        raise AssertionError("init_face_app must not run with --no-faces")

    monkeypatch.setattr("framedex.face_db.init_face_app", _boom)
    assert runner.setup_face_db(_ns(no_faces=True)) is None


def test_setup_face_db_unavailable_returns_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "framedex.face_db.init_face_app", lambda: (False, "no onnxruntime")
    )
    assert runner.setup_face_db(_ns(no_faces=False)) is None
    out = capsys.readouterr().out
    assert "Face detection unavailable: no onnxruntime" in out
    assert "Continuing without face detection" in out


def test_setup_face_db_available_opens_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "framedex.face_db.init_face_app", lambda: (True, "CPUExecutionProvider")
    )
    monkeypatch.setattr("framedex.face_db.open_db", lambda p: sentinel)
    monkeypatch.setattr(
        "framedex.face_db.db_stats",
        lambda c: {"faces": 5, "clusters": 2, "named_clusters": 1},
    )
    conn = runner.setup_face_db(_ns(no_faces=False, face_db="/tmp/f.db"))
    assert conn is sentinel
    out = capsys.readouterr().out
    assert "Face detection: insightface (CPUExecutionProvider)" in out
    assert "currently 5 faces, 2 clusters, 1 named" in out


# --- RunTally + record_result ----------------------------------------------


def _result(**kw: Any) -> pipeline.ProcessResult:
    base: dict[str, Any] = dict(sidecar=None, skipped_reason=None)
    base.update(kw)
    return pipeline.ProcessResult(**base)


def test_record_result_short_prints_no_counter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tally = runner.RunTally()
    runner.record_result(
        _result(skipped_reason="short"), tally, backend="cli", max_duration_min=30
    )
    assert capsys.readouterr().out == "  skipped (duration < 0.5s)\n"
    # short is never summarized — every counter stays zero
    assert tally == runner.RunTally()


def test_record_result_too_long(capsys: pytest.CaptureFixture[str]) -> None:
    tally = runner.RunTally()
    runner.record_result(
        _result(skipped_reason="too_long"), tally, backend="cli", max_duration_min=30
    )
    assert "skipped (duration > --max-duration 30 min)" in capsys.readouterr().out
    assert tally.skipped_too_long == 1


def test_record_result_no_preview(capsys: pytest.CaptureFixture[str]) -> None:
    tally = runner.RunTally()
    runner.record_result(
        _result(skipped_reason="no_preview"), tally, backend="cli", max_duration_min=30
    )
    assert "RAW without an embedded preview" in capsys.readouterr().out
    assert tally.skipped_no_preview == 1


def test_record_result_vision_error_counts_as_error() -> None:
    tally = runner.RunTally()
    runner.record_result(
        _result(skipped_reason="vision_error"),
        tally,
        backend="cli",
        max_duration_min=30,
    )
    assert tally.errors == 1
    assert tally.processed == 0


def test_record_result_success_cli_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tally = runner.RunTally()
    sidecar = tmp_path / "clip.mov.description.md"
    runner.record_result(
        _result(sidecar=sidecar, cost=0.0, rating="keep"),
        tally,
        backend="cli",
        max_duration_min=30,
    )
    assert capsys.readouterr().out == "  -> clip.mov.description.md  (, rated keep)\n"
    assert tally.processed == 1


def test_record_result_success_api_line_shows_running_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tally = runner.RunTally(actual_cost=0.10)
    sidecar = tmp_path / "p.jpg.description.md"
    runner.record_result(
        _result(sidecar=sidecar, cost=0.002, rating="keep"),
        tally,
        backend="api",
        max_duration_min=30,
    )
    out = capsys.readouterr().out
    assert out == "  -> p.jpg.description.md  (cost ~$0.10, rated keep)\n"
    assert tally.actual_cost == pytest.approx(0.102)


def test_record_result_success_with_faces_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tally = runner.RunTally()
    sidecar = tmp_path / "p.jpg.description.md"

    class _Face:
        pass

    runner.record_result(
        _result(sidecar=sidecar, rating="keep", detected_faces=[_Face(), _Face()]),
        tally,
        backend="cli",
        max_duration_min=30,
    )
    assert (
        capsys.readouterr().out
        == "  -> p.jpg.description.md  (, rated keep, 2 faces)\n"
    )


# --- orchestration-only guard ----------------------------------------------


def test_runner_imports_no_heavy_modules() -> None:
    """runner must stay a leaf: importing it pulls in no whisperx/torch/Pillow/
    osxphotos, so an fdx-photos images-only or a folder image-only run never
    drags the video stack in via the shared orchestration layer."""
    code = (
        "import sys, framedex.runner;"
        "heavy = [m for m in ('whisperx','torch','PIL','osxphotos') if m in sys.modules];"
        "print('LEAK:'+','.join(heavy) if heavy else 'CLEAN')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _SRC_DIR},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CLEAN", proc.stdout
