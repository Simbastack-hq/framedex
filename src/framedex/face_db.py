"""
face_db.py — Face detection + embedding + persistent DB for framedex.

Uses insightface (RetinaFace detection + ArcFace 512-dim embedding via the
buffalo_l model package). All local, no network calls after first-time model
download. Embeddings persist to ~/.framedex/faces.db (SQLite).

Sidecars store only cluster IDs (small text refs); the actual 512-dim vectors
live in this DB so sidecars stay readable.

Pipeline integration:
    1. Frames already extracted for vision pass → reuse them
    2. For each frame: detect faces, generate embeddings
    3. INSERT each face row into faces.db with a temporary cluster_id
    4. Sidecar gets list of {cluster_id, frame_time, bbox, detection_quality}
    5. Later, fdx-faces clusters embeddings and replaces temp IDs with names
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Lazy import — insightface is heavy
_face_app = None
_face_app_init_attempted = False
_init_error: str | None = None


DB_PATH_DEFAULT = Path.home() / ".framedex" / "faces.db"
EMBEDDING_DIM = 512
# Min detection score we trust at all; below this we drop the face.
MIN_DETECTION_SCORE = 0.55
# Threshold to label a detection "high" vs "low" quality in the sidecar.
HIGH_QUALITY_DETECTION = 0.75


@dataclass
class DetectedFace:
    cluster_id: str  # temporary; replaced post-clustering
    frame_time_seconds: float
    bbox: list[int]  # [x, y, w, h]
    detection_score: float
    embedding: list[float] = field(default_factory=list)  # 512 floats
    source_frame_index: int = 0

    @property
    def detection_quality(self) -> str:
        return "high" if self.detection_score >= HIGH_QUALITY_DETECTION else "low"

    def to_sidecar_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "frame_time": round(self.frame_time_seconds, 2),
            "bbox": self.bbox,
            "detection_quality": self.detection_quality,
        }


# ---------------------------------------------------------------------------
# Insightface lifecycle
# ---------------------------------------------------------------------------


def init_face_app(model_pack: str = "buffalo_l") -> tuple[bool, str]:
    """Lazy-initialize the insightface FaceAnalysis app. Returns (ok, info_or_error)."""
    global _face_app, _face_app_init_attempted, _init_error
    if _face_app is not None:
        return True, "already loaded"  # type: ignore[unreachable]
    if _face_app_init_attempted and _init_error:
        return False, _init_error
    _face_app_init_attempted = True
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        _init_error = f"insightface not installed: {e}"
        return False, _init_error
    try:
        # CPUExecutionProvider works everywhere; CoreML is faster on M-series
        # but requires extra setup. Default to CPU for portability.
        providers = ["CPUExecutionProvider"]
        app = FaceAnalysis(name=model_pack, providers=providers)
        # det_size: detection input resolution. 640 is the standard sweet spot.
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
        return True, f"loaded {model_pack} on CPU"
    except Exception as e:
        _init_error = f"insightface init failed: {e}"
        return False, _init_error


def detect_faces_in_frame(
    frame_path: Path,
    frame_time_seconds: float,
    frame_index: int,
) -> list[DetectedFace]:
    """Run insightface on one JPEG. Returns list of DetectedFace (possibly empty)."""
    global _face_app
    if _face_app is None:
        ok, _ = init_face_app()
        if not ok:
            return []

    # cv2 is pulled in by insightface
    import cv2

    img = cv2.imread(str(frame_path))
    if img is None:
        return []

    try:
        faces = _face_app.get(img)  # type: ignore[union-attr]
    except Exception:
        return []

    out: list[DetectedFace] = []
    for f in faces:
        score = float(getattr(f, "det_score", 0))
        if score < MIN_DETECTION_SCORE:
            continue
        bbox_raw = getattr(f, "bbox", None)
        if bbox_raw is None:
            continue
        # bbox is [x1, y1, x2, y2] floats — convert to [x, y, w, h] ints
        x1, y1, x2, y2 = [round(v) for v in bbox_raw]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        emb = getattr(f, "normed_embedding", None)
        if emb is None or len(emb) != EMBEDDING_DIM:
            continue
        # Temp cluster ID is a hash of the embedding's first 8 bytes — gives a
        # short stable ref per face *for this run*. Real clustering happens
        # later via fdx-faces.
        emb_list = [float(v) for v in emb.tolist()]
        cid = (
            "tmp_"
            + hashlib.sha1(
                ",".join(f"{v:.4f}" for v in emb_list[:8]).encode()
            ).hexdigest()[:8]
        )
        out.append(
            DetectedFace(
                cluster_id=cid,
                frame_time_seconds=frame_time_seconds,
                bbox=bbox,
                detection_score=score,
                embedding=emb_list,
                source_frame_index=frame_index,
            )
        )
    return out


def detect_faces_in_frames(
    frame_paths: list[Path], frame_timestamps: list[float]
) -> list[DetectedFace]:
    """Run face detection on a batch of frames; collect all DetectedFaces."""
    all_faces: list[DetectedFace] = []
    for i, (fp, ts) in enumerate(zip(frame_paths, frame_timestamps, strict=True)):
        all_faces.extend(detect_faces_in_frame(fp, ts, i))
    return all_faces


# ---------------------------------------------------------------------------
# SQLite face DB
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS faces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id      TEXT NOT NULL,          -- temp like 'tmp_a3f7' until labeled
    person_name     TEXT,                    -- NULL until fdx-faces labels it
    video_path      TEXT NOT NULL,
    sidecar_path    TEXT NOT NULL,
    frame_time      REAL NOT NULL,
    bbox_x          INTEGER NOT NULL,
    bbox_y          INTEGER NOT NULL,
    bbox_w          INTEGER NOT NULL,
    bbox_h          INTEGER NOT NULL,
    det_score       REAL NOT NULL,
    embedding       BLOB NOT NULL,           -- 512 float32 packed
    inserted_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faces_video ON faces(video_path);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_name);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      TEXT PRIMARY KEY,
    person_name     TEXT,                    -- NULL until user labels
    member_count    INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    notes           TEXT
);
"""


def open_db(db_path: Path = DB_PATH_DEFAULT) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def write_faces(
    conn: sqlite3.Connection,
    video_path: Path,
    sidecar_path: Path,
    faces: list[DetectedFace],
) -> None:
    """Insert detected faces for a clip into the DB. Removes any prior entries
    for this video first so re-running with --force replaces cleanly."""
    import struct

    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.cursor()
    cur.execute("DELETE FROM faces WHERE video_path = ?", (str(video_path),))
    for f in faces:
        emb_blob = struct.pack(f"{EMBEDDING_DIM}f", *f.embedding)
        cur.execute(
            "INSERT INTO faces (cluster_id, video_path, sidecar_path, frame_time, "
            "bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding, inserted_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                f.cluster_id,
                str(video_path),
                str(sidecar_path),
                f.frame_time_seconds,
                f.bbox[0],
                f.bbox[1],
                f.bbox[2],
                f.bbox[3],
                f.detection_score,
                emb_blob,
                now,
            ),
        )
        # Upsert cluster row
        cur.execute(
            "INSERT INTO clusters (cluster_id, member_count, created_at, last_seen_at) "
            "VALUES (?, 1, ?, ?) "
            "ON CONFLICT(cluster_id) DO UPDATE SET "
            "member_count = member_count + 1, last_seen_at = excluded.last_seen_at",
            (f.cluster_id, now, now),
        )
    conn.commit()


def db_stats(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM faces")
    n_faces = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM clusters")
    n_clusters = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM clusters WHERE person_name IS NOT NULL")
    n_named = cur.fetchone()[0]
    return {"faces": n_faces, "clusters": n_clusters, "named_clusters": n_named}


if __name__ == "__main__":
    # Smoke test: init the model + DB
    print("face_db smoke test")
    ok, info = init_face_app()
    print(f"  insightface init: {ok} ({info})")
    conn = open_db()
    print(f"  DB: {DB_PATH_DEFAULT}")
    print(f"  stats: {db_stats(conn)}")
    conn.close()
