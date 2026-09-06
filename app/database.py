from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .fingerprint import metadata_hash, sha256_file
from .service import PipelineDossier

DEFAULT_DB_PATH = Path("output/evidence_vault.db")


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize the local evidence database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                metadata_hash TEXT NOT NULL,
                liveness_score REAL,
                is_live INTEGER,
                confidence REAL,
                distance REAL,
                threshold REAL,
                detector_model TEXT,
                social_platform TEXT,
                social_author TEXT,
                social_post_url TEXT,
                social_image_url TEXT,
                social_caption TEXT,
                artifact_path TEXT,
                annotated_path TEXT,
                blockchain_network TEXT,
                blockchain_tx_hash TEXT,
                blockchain_record_id INTEGER
            )
            """
        )
        conn.commit()


def insert_record(
    dossier: PipelineDossier,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Persist a verified pipeline dossier into the local database."""
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    liv = dossier.liveness
    post = dossier.social_post
    metrics = dossier.match_metrics
    hashes = dossier.evidence_hashes
    proof = dossier.blockchain_proof

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO evidence_records (
                created_at, status, artifact_hash, metadata_hash,
                liveness_score, is_live, confidence, distance, threshold,
                detector_model, social_platform, social_author, social_post_url,
                social_image_url, social_caption, artifact_path, annotated_path,
                blockchain_network, blockchain_tx_hash, blockchain_record_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                dossier.status,
                hashes.get("artifact_sha256", ""),
                hashes.get("metadata_sha256", ""),
                liv.get("score", 0.0),
                1 if liv.get("is_live", False) else 0,
                metrics.get("confidence", 0.0),
                metrics.get("distance", 0.0),
                metrics.get("threshold", 0.5),
                metrics.get("detector_model", "hog"),
                post.get("platform", "Web"),
                post.get("author"),
                post.get("post_url"),
                post.get("image_url"),
                post.get("caption"),
                metrics.get("artifact_path"),
                metrics.get("annotated_artifact"),
                proof.get("network"),
                proof.get("transaction_hash"),
                proof.get("record_id"),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_record(record_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Fetch a single record by primary key."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM evidence_records WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None


def list_records(limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve the most recent evidence records."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM evidence_records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def verify_offline_record(record_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Perform independent offline cryptographic verification of a stored record."""
    record = get_record(record_id, db_path)
    if not record:
        raise KeyError(f"Record #{record_id} not found in database")

    artifact_file = Path(record["artifact_path"] or "")
    if not artifact_file.is_file():
        # Fallback to standard output naming
        fallback = db_path.parent / "candidate_1.jpg"
        if fallback.is_file():
            artifact_file = fallback
        else:
            return {
                "record_id": record_id,
                "verified": False,
                "reason": f"Artifact file missing: {artifact_file}",
            }

    current_hash = sha256_file(artifact_file)
    matches_stored = current_hash == record["artifact_hash"]

    return {
        "record_id": record_id,
        "verified": matches_stored,
        "status": "VERIFIED" if matches_stored else "TAMPERED",
        "stored_hash": record["artifact_hash"],
        "recomputed_hash": current_hash,
        "social_platform": record["social_platform"],
        "confidence": record["confidence"],
    }


def find_offline_candidates(db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Fetch stored evidence records with valid image artifacts to use as offline search candidates."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM evidence_records
            WHERE artifact_path IS NOT NULL AND artifact_path != ''
            ORDER BY id DESC LIMIT 100
            """
        ).fetchall()
        valid = []
        for r in rows:
            d = dict(r)
            if d.get("artifact_path") and Path(d["artifact_path"]).is_file():
                valid.append(d)
        return valid

