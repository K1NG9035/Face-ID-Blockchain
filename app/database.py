from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class StoredRecord:
    record_id: int
    artifact_hash: str
    metadata_hash: str
    platform: str
    author: str | None
    source_url: str
    post_url: str | None
    confidence: float
    liveness_score: float
    tx_hash: str | None
    created_at: str
    raw_metadata: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        try:
            data["raw_metadata"] = json.loads(self.raw_metadata)
        except Exception:
            pass
        return data


class MatchDatabase:
    """High-performance local SQLite database for offline caching, indexing, and fast lookups."""

    def __init__(self, db_path: Path = Path("records.db")) -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS match_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    on_chain_record_id INTEGER,
                    artifact_hash TEXT NOT NULL UNIQUE,
                    metadata_hash TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    author TEXT,
                    source_url TEXT NOT NULL,
                    post_url TEXT,
                    confidence REAL NOT NULL,
                    liveness_score REAL NOT NULL,
                    tx_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    raw_metadata TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_hash ON match_records(artifact_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_platform ON match_records(platform)")
            conn.commit()

    def store_record(
        self,
        artifact_hash: str,
        metadata_hash: str,
        platform: str,
        source_url: str,
        confidence: float,
        liveness_score: float,
        raw_metadata: dict[str, Any],
        author: str | None = None,
        post_url: str | None = None,
        on_chain_record_id: int | None = None,
        tx_hash: str | None = None,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO match_records (
                    on_chain_record_id, artifact_hash, metadata_hash,
                    platform, author, source_url, post_url,
                    confidence, liveness_score, tx_hash, raw_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_hash) DO UPDATE SET
                    metadata_hash = excluded.metadata_hash,
                    on_chain_record_id = COALESCE(excluded.on_chain_record_id, match_records.on_chain_record_id),
                    tx_hash = COALESCE(excluded.tx_hash, match_records.tx_hash),
                    raw_metadata = excluded.raw_metadata
                """,
                (
                    on_chain_record_id,
                    artifact_hash,
                    metadata_hash,
                    platform,
                    author,
                    source_url,
                    post_url,
                    confidence,
                    liveness_score,
                    tx_hash,
                    json.dumps(raw_metadata),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_by_artifact_hash(self, artifact_hash: str) -> StoredRecord | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM match_records WHERE artifact_hash = ?", (artifact_hash,)
            ).fetchone()
            if not row:
                return None
            return StoredRecord(
                record_id=row["id"],
                artifact_hash=row["artifact_hash"],
                metadata_hash=row["metadata_hash"],
                platform=row["platform"],
                author=row["author"],
                source_url=row["source_url"],
                post_url=row["post_url"],
                confidence=row["confidence"],
                liveness_score=row["liveness_score"],
                tx_hash=row["tx_hash"],
                created_at=row["created_at"],
                raw_metadata=row["raw_metadata"],
            )

    def list_records(self, limit: int = 50) -> list[StoredRecord]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM match_records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                StoredRecord(
                    record_id=r["id"],
                    artifact_hash=r["artifact_hash"],
                    metadata_hash=r["metadata_hash"],
                    platform=r["platform"],
                    author=r["author"],
                    source_url=r["source_url"],
                    post_url=r["post_url"],
                    confidence=r["confidence"],
                    liveness_score=r["liveness_score"],
                    tx_hash=r["tx_hash"],
                    created_at=r["created_at"],
                    raw_metadata=r["raw_metadata"],
                )
                for r in rows
            ]

    def count(self) -> int:
        with self._get_connection() as conn:
            res = conn.execute("SELECT COUNT(*) FROM match_records").fetchone()
            return res[0] if res else 0
