"""Hermetic tests for the fdx-develop loop: a fake backend and a fake renderer
stand in for the model and the RAW engine. No Pillow, no network, no photos."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import requests

from framedex import develop as D
from framedex import develop_engine as eng


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, replies: list[str], cost: float | None = 0.01) -> None:
        self.replies = list(replies)
        self.cost = cost
        self.calls: list[tuple[list[D.Img], str]] = []

    def complete(self, system: str, images: list[D.Img], text: str) -> D.Reply:
        self.calls.append((images, text))
        if not self.replies:
            raise RuntimeError("out of replies")
        return D.Reply(self.replies.pop(0), self.cost)


class FakeRenderer:
    def __init__(self) -> None:
        self.tags: list[str] = []
        self.adjustments: list[dict[str, Any]] = []

    def render(
        self,
        raw: Path,
        adjustments: dict[str, Any],
        tag: str,
        crop: eng.Crop,
        orientation: int,
    ) -> D.Rendered:
        self.tags.append(tag)
        self.adjustments.append(adjustments)
        return D.Rendered(
            preview=f"preview-{tag}".encode(),
            stats=f"stats-{tag}",
            full_size=(1000, 500),
            deliverable=f"jpeg-{tag}".encode(),
        )

    def calibration_sheet(
        self, raw: Path, full_size: tuple[int, int], crop: eng.Crop, orientation: int
    ) -> bytes:
        return b"sheet"

    def example_after(self, jpeg: Path) -> bytes:
        return b"after-" + jpeg.name.encode()


def edit(done: bool = False, exposure: float = 0.5, intent: str = "glow") -> str:
    return "Here:\n" + json.dumps(
        {"done": done, "intent": intent, "basic": {"exposure": exposure}}
    )


@pytest.fixture(autouse=True)
def no_exiftool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        D, "get_image_metadata", lambda raw: {"camera": {"model": "Z 8"}}
    )


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    p = tmp_path / "shoot" / "_DSC0001.NEF"
    p.parent.mkdir()
    p.write_bytes(b"RAW-BYTES-NEVER-TOUCH")
    return p


def run(
    raw: Path, backend: FakeBackend, renderer: FakeRenderer | None = None, **kw: Any
) -> D.PhotoResult:
    out = raw.parent / "developed"
    opts: dict[str, Any] = {
        "rounds": 3,
        "examples": [],
        "calibration": True,
        "write_engine_sidecar": True,
        "engine_info": {"name": "rapidraw", "version": "test"},
        "log": lambda m: None,
    }
    opts.update(kw)
    return D.develop_photo(
        raw,
        jpg=out / "_DSC0001.jpg",
        record=out / "_DSC0001.develop.json",
        backend=backend,
        renderer=renderer or FakeRenderer(),
        **opts,
    )


def record(raw: Path) -> dict[str, Any]:
    rec: dict[str, Any] = json.loads(
        (raw.parent / "developed" / "_DSC0001.develop.json").read_text()
    )
    return rec


def test_model_can_stop_early_and_the_accepted_render_ships(raw: Path) -> None:
    backend = FakeBackend([edit(exposure=0.3), edit(exposure=0.8), edit(done=True)])
    renderer = FakeRenderer()
    res = run(raw, backend, renderer)
    assert res.status == "done"
    assert renderer.tags == ["r0", "r1", "r2"]
    assert (raw.parent / "developed" / "_DSC0001.jpg").read_bytes() == b"jpeg-r2"
    rec = record(raw)
    assert rec["status"] == "done" and rec["final_round"] == 2
    assert rec["adjustments"]["exposure"] == 0.8
    assert rec["rounds"][-1]["accepted_previous"] is True
    assert res.cost_usd == pytest.approx(0.03)


def test_done_on_round_one_is_ignored(raw: Path) -> None:
    backend = FakeBackend([edit(done=True, exposure=1.0), edit(done=True)])
    renderer = FakeRenderer()
    run(raw, backend, renderer)
    assert renderer.tags == ["r0", "r1"]  # nothing to accept yet: r1 rendered
    assert record(raw)["final_round"] == 1


def test_model_can_accept_its_first_render(raw: Path) -> None:
    backend = FakeBackend([edit(exposure=0.7), edit(done=True)])
    renderer = FakeRenderer()
    res = run(raw, backend, renderer)
    assert renderer.tags == ["r0", "r1"] and res.status == "done"
    assert record(raw)["rounds"][-1]["accepted_previous"] is True


def test_round_cap(raw: Path) -> None:
    backend = FakeBackend([edit()] * 5)
    renderer = FakeRenderer()
    run(raw, backend, renderer, rounds=2)
    assert renderer.tags == ["r0", "r1", "r2"]
    assert len(backend.calls) == 2


def test_each_round_is_stateless_and_carries_the_latest_render(raw: Path) -> None:
    backend = FakeBackend([edit(exposure=0.3), edit(exposure=0.6), edit(done=True)])
    run(raw, backend)
    first_imgs, _ = backend.calls[0]
    second_imgs, second_text = backend.calls[1]
    assert [i.jpeg for i in first_imgs] == [b"sheet", b"preview-r0"]
    assert second_imgs[-1].jpeg == b"preview-r1"
    assert '"exposure": 0.3' in second_text and "stats-r1" in second_text
    assert "If it already hits the target, set done=true" in second_text
    assert "your intent" in second_text  # no examples: own intent is the target


def test_bad_first_reply_fails_and_stays_retryable(raw: Path) -> None:
    res = run(raw, FakeBackend(["I can't edit this."]))
    assert res.status == "failed"
    assert not (raw.parent / "developed" / "_DSC0001.jpg").exists()
    assert record(raw)["status"] == "failed"
    jpg = raw.parent / "developed" / "_DSC0001.jpg"
    assert not D.is_complete(
        raw, jpg, raw.parent / "developed" / "_DSC0001.develop.json"
    )


def test_bad_later_reply_keeps_the_last_good_render(raw: Path) -> None:
    res = run(raw, FakeBackend([edit(exposure=0.4), "garbage"]))
    assert res.status == "partial"  # ships round 1, stays retryable
    assert record(raw)["final_round"] == 1
    assert "bad reply" in record(raw)["rounds"][-1]["error"]


def test_backend_error_keeps_the_last_good_render(raw: Path) -> None:
    res = run(raw, FakeBackend([edit()]))  # second call raises
    assert res.status == "partial"
    assert res.cost_complete is False  # the failed call may have been billed
    assert record(raw)["rounds"][-1]["error"].startswith("backend:")


def test_unknown_cost_propagates(raw: Path) -> None:
    res = run(raw, FakeBackend([edit(), edit(), edit(done=True)], cost=None))
    assert res.cost_complete is False
    assert record(raw)["cost_complete"] is False


def test_original_is_untouched_and_rrdata_written_once(raw: Path) -> None:
    before = (raw.read_bytes(), raw.stat().st_mtime_ns)
    run(raw, FakeBackend([edit(exposure=0.3), edit(exposure=0.9), edit(done=True)]))
    assert (raw.read_bytes(), raw.stat().st_mtime_ns) == before
    rr = eng.rrdata_path(raw)
    assert json.loads(rr.read_text())["adjustments"]["exposure"] == 0.9
    rr.write_text("USER EDIT")
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    assert rr.read_text() == "USER EDIT"


def test_no_rrdata_flag(raw: Path) -> None:
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]), write_engine_sidecar=False)
    assert not eng.rrdata_path(raw).exists()


def test_write_rrdata_refuses_symlinks(tmp_path: Path) -> None:
    raw = tmp_path / "a.NEF"
    raw.write_bytes(b"x")
    os.symlink(tmp_path / "elsewhere", eng.rrdata_path(raw))
    assert D.write_rrdata(raw, {}) is False
    assert not (tmp_path / "elsewhere").exists()


def test_examples_are_shown_before_the_photo(raw: Path) -> None:
    ex = [D.Example("e1", b"b1", b"a1"), D.Example("e2", b"b2", b"a2")]
    backend = FakeBackend([edit(), edit(), edit(done=True)])
    run(raw, backend, examples=ex, calibration=False)
    imgs, text = backend.calls[0]
    assert [i.jpeg for i in imgs] == [b"b1", b"a1", b"b2", b"a2", b"preview-r0"]
    assert [i.cache_after for i in imgs] == [False, False, False, True, False]
    assert "photographer's own edits" in text


def test_lightroom_crop_sidecar_is_honored(raw: Path) -> None:
    raw.with_suffix(".xmp").write_text(
        'crs:HasCrop="True" crs:CropLeft="0.1" crs:CropRight="0.6" crs:CropTop="0.2" crs:CropBottom="0.8"'
    )
    assert D.read_crop(raw) == eng.Crop(0.1, 0.2, 0.6, 0.8, 0.0)
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    rr = json.loads(eng.rrdata_path(raw).read_text())["adjustments"]
    assert rr["crop"] == pytest.approx(
        {"x": 100, "y": 100, "width": 500, "height": 300}
    )
    assert record(raw)["crop"]["left"] == 0.1


def test_angled_crop_not_carried_into_rrdata() -> None:
    adj = D.rrdata_adjustments({}, (1000, 500), eng.Crop(0.1, 0.1, 0.9, 0.9, 1.0), 1)
    assert "crop" not in adj


def test_framedex_sidecar_context(raw: Path) -> None:
    sidecar = raw.with_name(raw.name + ".description.md")
    sidecar.write_text(
        "---\nlocation:\n  place: Maasai Mara\nlighting: golden_hour\ntime_of_day: unclear\n---\n\n"
        "## Description\n\nA lion at dusk.\n"
    )
    ctx, orientation = D.photo_context(raw)
    assert "Place: Maasai Mara" in ctx
    assert "Lighting: golden_hour" in ctx
    assert "Time of day" not in ctx
    assert "A lion at dusk." in ctx
    assert "Z 8" in ctx and orientation == 1


def test_is_complete_requires_matching_source(raw: Path) -> None:
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    out = raw.parent / "developed"
    jpg, rec = out / "_DSC0001.jpg", out / "_DSC0001.develop.json"
    assert D.is_complete(raw, jpg, rec)
    raw.write_bytes(b"RAW-BYTES-CHANGED!!!!!")
    assert not D.is_complete(raw, jpg, rec)


def test_plan_outputs_resolves_every_clash(tmp_path: Path) -> None:
    root, out = tmp_path, tmp_path / "developed"
    d = root / "day1"
    raws = [d / "A.ARW", d / "A.NEF", d / "A.NEF.DNG", d / "B.NEF", root / "A.NEF"]
    plan = D.plan_outputs(raws, root, out)
    names = {
        r.relative_to(root).as_posix(): plan[r][0].relative_to(out).as_posix()
        for r in raws
    }
    assert names == {
        "day1/A.ARW": "day1/A.ARW.jpg",
        "day1/A.NEF": "day1/A.NEF.jpg",
        "day1/A.NEF.DNG": "day1/A.NEF.DNG.jpg",
        "day1/B.NEF": "day1/B.jpg",
        "A.NEF": "A.jpg",  # different folder: no clash
    }
    assert plan[d / "B.NEF"][1] == out / "day1" / "B.develop.json"
    assert len({p[0] for p in plan.values()}) == len(raws)


def test_discover_skips_hidden_output_and_non_raw(tmp_path: Path) -> None:
    for rel in (
        "a.NEF",
        "b.cr3",
        "c.jpg",
        ".hidden/d.NEF",
        "developed/e.NEF",
        "sub/f.arw",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    found = D.discover(tmp_path, tmp_path / "developed")
    assert [p.relative_to(tmp_path).as_posix() for p in found] == [
        "a.NEF",
        "b.cr3",
        "sub/f.arw",
    ]


def test_main_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.NEF").write_bytes(b"x")
    (tmp_path / "a.ARW").write_bytes(b"x")
    assert D.main([str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 RAWs, 0 already developed, 2 to do" in out
    assert "a.NEF.jpg" in out and "a.ARW.jpg" in out  # same stem: full names
    assert not (tmp_path / "developed").exists()


def test_main_rejects_bad_rounds(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        D.main([str(tmp_path), "--rounds", "9"])


def test_local_backend_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": '{"done": false}'}}]}

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> Resp:
        seen.update(url=url, payload=json)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    reply = D.LocalBackend("http://localhost:1234/v1/", None).complete(
        "SYS", [D.Img("photo", b"\xff")], "go"
    )
    assert reply.text == '{"done": false}' and reply.cost_usd == 0.0
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    msgs = seen["payload"]["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["content"][1]["type"] == "image_url"
    assert seen["payload"]["max_tokens"] == D.MAX_OUTPUT_TOKENS


def test_api_cost_known_and_unknown_models() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert D._api_cost("claude-opus-5-5", usage) == pytest.approx(6.0)
    assert D._api_cost("some-other-model", usage) is None


def test_system_prompt_embeds_the_neutral_template() -> None:
    s = D.system_prompt()
    assert json.dumps(eng.default_edit()) in s
    assert "ABSOLUTE" in s


class BrokenRenderer(FakeRenderer):
    def render(
        self,
        raw: Path,
        adjustments: dict[str, Any],
        tag: str,
        crop: eng.Crop,
        orientation: int,
    ) -> D.Rendered:
        raise RuntimeError("rapidraw failed on _DSC0001.NEF (exit 1): unsupported")


def test_engine_failure_fails_the_photo_not_the_run(raw: Path) -> None:
    backend = FakeBackend([edit()])
    res = run(raw, backend, BrokenRenderer())
    assert res.status == "failed" and "unsupported" in res.message
    assert backend.calls == []
    assert record(raw)["status"] == "failed"


def _paths(raw: Path) -> tuple[Path, Path]:
    out = raw.parent / "developed"
    return out / "_DSC0001.jpg", out / "_DSC0001.develop.json"


def test_owns_output_rules(raw: Path) -> None:
    jpg, rec = _paths(raw)
    assert D.owns_output(raw, jpg, rec)  # nothing there yet
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    assert D.owns_output(raw, jpg, rec)  # our digest
    jpg.write_bytes(b"the user's own export")
    assert not D.owns_output(raw, jpg, rec)  # replaced by the user
    assert not D.is_complete(raw, jpg, rec)


def test_failed_record_does_not_claim_a_users_jpeg(raw: Path) -> None:
    jpg, rec = _paths(raw)
    run(raw, FakeBackend(["nope"]))
    jpg.write_bytes(b"user export")
    assert not D.owns_output(raw, jpg, rec)


def test_record_for_another_raw_does_not_claim(raw: Path, tmp_path: Path) -> None:
    jpg, rec = _paths(raw)
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    other = raw.with_name("_DSC0002.NEF")
    other.write_bytes(b"x")
    assert not D.owns_output(other, jpg, rec)


def test_interrupt_between_publish_and_finalize_is_recoverable(
    raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jpg, rec = _paths(raw)
    real = D.write_rrdata

    def crash(*a: Any, **k: Any) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(D, "write_rrdata", crash)
    with pytest.raises(KeyboardInterrupt):
        run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    assert jpg.exists() and record(raw)["status"] == "pending"
    assert D.owns_output(raw, jpg, rec) and not D.is_complete(raw, jpg, rec)
    monkeypatch.setattr(D, "write_rrdata", real)
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    assert D.is_complete(raw, jpg, rec)


def test_rrdata_publish_leaves_no_temp_file(raw: Path) -> None:
    assert D.write_rrdata(raw, {"exposure": 1})
    assert sorted(p.name for p in raw.parent.iterdir()) == [
        "_DSC0001.NEF",
        "_DSC0001.NEF.rrdata",
    ]
    assert not D.write_rrdata(raw, {"exposure": 2})
    assert json.loads(eng.rrdata_path(raw).read_text())["adjustments"] == {
        "exposure": 1
    }


def test_main_isolates_a_crashing_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for n in ("a.NEF", "b.NEF"):
        (tmp_path / n).write_bytes(b"x")
    seen: list[str] = []

    def fake_develop(raw: Path, **kw: Any) -> D.PhotoResult:
        seen.append(raw.name)
        if raw.name == "a.NEF":
            raise ValueError("boom")
        return D.PhotoResult(raw, "done", cost_usd=0.2)

    monkeypatch.setattr(eng, "find_engine", lambda explicit=None: Path("/bin/rr"))
    monkeypatch.setattr(D, "make_backend", lambda args: FakeBackend([]))
    monkeypatch.setattr(D, "develop_photo", fake_develop)
    assert D.main([str(tmp_path)]) == 1
    assert seen == ["a.NEF", "b.NEF"]
    out = capsys.readouterr().out
    assert "done 1, partial 0, failed 1" in out and "$0.20 + unknown" in out


def test_user_replacement_after_interrupt_is_not_overwritten(
    raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jpg, rec = _paths(raw)

    def crash(*a: Any, **k: Any) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(D, "write_rrdata", crash)
    with pytest.raises(KeyboardInterrupt):
        run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    assert record(raw)["status"] == "pending"
    jpg.write_bytes(b"user's own export")
    assert not D.owns_output(raw, jpg, rec)


def test_failed_redo_keeps_ownership_of_the_previous_output(raw: Path) -> None:
    jpg, rec = _paths(raw)
    run(raw, FakeBackend([edit(), edit(), edit(done=True)]))
    before = jpg.read_bytes()
    res = run(raw, FakeBackend([edit()]), BrokenRenderer())
    assert res.status == "failed"
    assert jpg.read_bytes() == before
    assert D.owns_output(raw, jpg, rec)  # still ours: a later --force may replace it
    assert not D.is_complete(raw, jpg, rec)  # and it gets retried


def _shoot(tmp_path: Path, names: Sequence[str]) -> Path:
    root = tmp_path / "shoot"
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return root


def test_files_from_develops_only_the_listed_subset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _shoot(tmp_path, ["a.NEF", "b.NEF", "sub/c.NEF", "d.NEF"])
    listing = tmp_path / "picks.txt"
    listing.write_text(f"# picks\n\nsub/c.NEF\n  b.NEF  \n{root / 'sub' / 'c.NEF'}\n")
    assert D.main([str(root), "--files-from", str(listing), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "4 RAWs, 2 listed, 0 already developed, 2 to do" in out
    planned = [
        line.split(" -> ")[0].strip() for line in out.splitlines() if " -> " in line
    ]
    assert planned == ["b.NEF", "sub/c.NEF"]  # discovery order, deduplicated


def test_files_from_keeps_full_run_output_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _shoot(tmp_path, ["A.NEF", "A.ARW", "B.NEF"])
    listing = tmp_path / "picks.txt"
    listing.write_text("A.NEF\nB.NEF\n")
    D.main([str(root), "--files-from", str(listing), "--dry-run"])
    subset = capsys.readouterr().out
    D.main([str(root), "--dry-run"])
    full = capsys.readouterr().out

    def targets(out: str) -> dict[str, str]:
        pairs = (
            line.strip().split(" -> ") for line in out.splitlines() if " -> " in line
        )
        return {src: dst for src, dst in pairs}

    assert targets(subset)["A.NEF"].endswith("developed/A.NEF.jpg")  # clash with A.ARW
    assert targets(subset) == {
        k: v for k, v in targets(full).items() if k in ("A.NEF", "B.NEF")
    }


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        ("../outside.NEF", "outside"),
        ("missing.NEF", "no such file"),
        ("notes.jpg", "not a RAW"),
        ("developed/old.NEF", "not a RAW"),
    ],
)
def test_files_from_rejects_bad_lines(tmp_path: Path, entry: str, why: str) -> None:
    root = _shoot(tmp_path, ["a.NEF", "notes.jpg", "developed/old.NEF"])
    (tmp_path / "outside.NEF").write_bytes(b"x")
    listing = tmp_path / "picks.txt"
    listing.write_text(f"a.NEF\n# fine so far\n{entry}\n")
    with pytest.raises(SystemExit) as exc:
        D.main([str(root), "--files-from", str(listing), "--dry-run"])
    msg = str(exc.value.code)
    assert f"line 3: {entry}" in msg and why in msg
    assert not (root / "developed" / "a.jpg").exists()


def test_files_from_symlink_aliases_stay_distinct(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _shoot(tmp_path, ["a.NEF", "notes.txt"])
    os.symlink(root / "a.NEF", root / "z.NEF")
    os.symlink(root / "a.NEF", root / "alias.jpg")
    listing = tmp_path / "picks.txt"
    listing.write_text("a.NEF\n")
    D.main([str(root), "--files-from", str(listing), "--dry-run"])
    planned = [
        ln.split(" -> ")[0].strip()
        for ln in capsys.readouterr().out.splitlines()
        if " -> " in ln
    ]
    assert planned == ["a.NEF"]  # not its alias z.NEF
    listing.write_text("alias.jpg\n")  # a non-RAW name pointing at a RAW
    with pytest.raises(SystemExit, match=r"line 1: alias\.jpg: not a RAW"):
        D.main([str(root), "--files-from", str(listing), "--dry-run"])


def test_files_from_case_variant_matches_only_the_same_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _shoot(tmp_path, ["A.NEF"])
    listing = tmp_path / "picks.txt"
    listing.write_text("a.nef\n")
    if not (
        root / "a.nef"
    ).exists():  # case-sensitive volume: a different (missing) file
        with pytest.raises(SystemExit, match="no such file"):
            D.main([str(root), "--files-from", str(listing), "--dry-run"])
        return
    D.main([str(root), "--files-from", str(listing), "--dry-run"])
    out = capsys.readouterr().out
    assert "A.NEF -> " in out  # the discovered name, not the listed casing


def test_files_from_empty_name_is_an_error_not_a_full_run(tmp_path: Path) -> None:
    root = _shoot(tmp_path, ["a.NEF"])
    with pytest.raises(SystemExit, match="needs a file name"):
        D.main([str(root), "--files-from", "", "--dry-run"])
