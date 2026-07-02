"""Tests for framedex.face_db — the persistent face DB.

Hermetic: sqlite in a tmpdir, hand-built DetectedFace objects, no insightface
(so this runs on CI without the heavy model stack). These pin the idempotency
guarantees: re-running the pipeline over the same media must never accumulate
duplicate face rows or inflate cluster member counts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from framedex import face_db


def _face(cluster_id: str, score: float = 0.9) -> face_db.DetectedFace:
    return face_db.DetectedFace(
        cluster_id=cluster_id,
        frame_time_seconds=0.0,
        bbox=[0, 0, 10, 10],
        detection_score=score,
        embedding=[0.1] * face_db.EMBEDDING_DIM,
    )


def _face_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0])


def _member_count(conn: sqlite3.Connection, cluster_id: str) -> int | None:
    row = conn.execute(
        "SELECT member_count FROM clusters WHERE cluster_id = ?", (cluster_id,)
    ).fetchone()
    return None if row is None else int(row[0])


def test_rewrite_same_file_is_idempotent(tmp_path: Path) -> None:
    conn = face_db.open_db(tmp_path / "f.db")
    v = Path("/media/clip.mov")
    s = Path("/media/clip.mov.description.md")
    faces = [_face("c1"), _face("c1"), _face("c2")]

    face_db.write_faces(conn, v, s, faces)
    face_db.write_faces(conn, v, s, faces)  # exact re-run

    assert _face_count(conn) == 3
    assert _member_count(conn, "c1") == 2  # not 4
    assert _member_count(conn, "c2") == 1


def test_member_count_never_inflates_across_reruns(tmp_path: Path) -> None:
    conn = face_db.open_db(tmp_path / "f.db")
    v = Path("/m/a.mov")
    s = Path("/m/a.mov.description.md")
    for _ in range(3):
        face_db.write_faces(conn, v, s, [_face("c1"), _face("c1")])
    assert _member_count(conn, "c1") == 2  # not 6


def test_dedup_keys_on_sidecar_path_when_video_path_changes(tmp_path: Path) -> None:
    """fdx-photos --download materializes each run into a fresh mkdtemp, so
    video_path differs run-to-run but the mirror sidecar_path is stable. Dedup
    must key on the sidecar too, or downloaded assets accumulate forever."""
    conn = face_db.open_db(tmp_path / "f.db")
    s = Path("/mirror/2024-08/IMG_ABCD1234.description.md")
    face_db.write_faces(conn, Path("/tmp/run1/IMG.jpg"), s, [_face("c1")])
    face_db.write_faces(conn, Path("/tmp/run2/IMG.jpg"), s, [_face("c1")])
    assert _face_count(conn) == 1


def test_dedup_keys_on_video_path_when_sidecar_changes(tmp_path: Path) -> None:
    conn = face_db.open_db(tmp_path / "f.db")
    v = Path("/media/clip.mov")
    face_db.write_faces(conn, v, Path("/a/old.description.md"), [_face("c1")])
    face_db.write_faces(conn, v, Path("/a/new.description.md"), [_face("c1")])
    assert _face_count(conn) == 1


def test_orphan_unnamed_cluster_is_reaped(tmp_path: Path) -> None:
    conn = face_db.open_db(tmp_path / "f.db")
    v = Path("/m/a.mov")
    s = Path("/m/a.mov.description.md")
    face_db.write_faces(conn, v, s, [_face("c1")])
    # Re-run detects a different face → c1 loses its last member and, being
    # unnamed, is reaped rather than left as a stale zero-member cluster.
    face_db.write_faces(conn, v, s, [_face("c2")])
    assert _member_count(conn, "c1") is None
    assert _member_count(conn, "c2") == 1


def test_named_cluster_with_zero_members_is_retained(tmp_path: Path) -> None:
    """A user label is real work — a named cluster must survive a re-index that
    strips its last face (only unnamed orphans are reaped)."""
    conn = face_db.open_db(tmp_path / "f.db")
    v = Path("/m/a.mov")
    s = Path("/m/a.mov.description.md")
    face_db.write_faces(conn, v, s, [_face("c1")])
    conn.execute("UPDATE clusters SET person_name = 'Mom' WHERE cluster_id = 'c1'")
    conn.commit()

    face_db.write_faces(conn, v, s, [_face("c2")])
    assert _member_count(conn, "c1") == 0  # retained, recomputed to 0


def test_sidecar_path_index_exists(tmp_path: Path) -> None:
    """The OR-keyed DELETE needs an index on sidecar_path to stay O(log n)."""
    conn = face_db.open_db(tmp_path / "f.db")
    names = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert "idx_faces_sidecar" in names
