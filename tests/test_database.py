from pathlib import Path
import pytest

from app.database import MatchDatabase, StoredRecord


def test_database_store_and_retrieve(tmp_path: Path):
    db_file = tmp_path / "test_records.db"
    db = MatchDatabase(db_file)
    assert db.count() == 0

    record_id = db.store_record(
        artifact_hash="hash_art_1",
        metadata_hash="hash_meta_1",
        platform="Twitter/X",
        author="@Sydney",
        source_url="https://pbs.twimg.com/media/1.jpg",
        post_url="https://x.com/Sydney/status/1",
        confidence=98.5,
        liveness_score=85.0,
        raw_metadata={"note": "test"},
        on_chain_record_id=1,
        tx_hash="0x123abc",
    )
    assert record_id > 0
    assert db.count() == 1

    stored = db.get_by_artifact_hash("hash_art_1")
    assert stored is not None
    assert stored.platform == "Twitter/X"
    assert stored.author == "@Sydney"
    assert stored.confidence == 98.5
    assert stored.tx_hash == "0x123abc"


def test_database_upsert_on_conflict(tmp_path: Path):
    db_file = tmp_path / "test_records2.db"
    db = MatchDatabase(db_file)

    db.store_record(
        artifact_hash="duplicate_hash",
        metadata_hash="meta_v1",
        platform="Web",
        source_url="https://example.com/1.jpg",
        confidence=80.0,
        liveness_score=70.0,
        raw_metadata={"v": 1},
    )

    # Store again with same artifact_hash -> updates metadata
    db.store_record(
        artifact_hash="duplicate_hash",
        metadata_hash="meta_v2",
        platform="Web",
        source_url="https://example.com/1.jpg",
        confidence=80.0,
        liveness_score=70.0,
        raw_metadata={"v": 2},
        tx_hash="0xupdated",
    )

    assert db.count() == 1
    updated = db.get_by_artifact_hash("duplicate_hash")
    assert updated is not None
    assert updated.metadata_hash == "meta_v2"
    assert updated.tx_hash == "0xupdated"
