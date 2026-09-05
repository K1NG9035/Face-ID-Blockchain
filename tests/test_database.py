from pathlib import Path
import pytest

from app.database import (
    get_record,
    init_db,
    insert_record,
    list_records,
    verify_offline_record,
)
from app.service import PipelineDossier


def test_database_insert_and_get(tmp_path: Path):
    db_file = tmp_path / "test_vault.db"
    init_db(db_file)

    dossier = PipelineDossier(
        status="VERIFIED_MATCH",
        liveness={"is_live": True, "score": 95.0, "reasons": []},
        social_post={
            "platform": "Twitter/X",
            "author": "@Alice",
            "post_url": "https://x.com/Alice/status/1",
            "image_url": "https://pbs.twimg.com/1.jpg",
            "caption": "Alice in Goa",
        },
        match_metrics={
            "distance": 0.18,
            "threshold": 0.5,
            "confidence": 89.2,
            "matched": True,
            "detector_model": "hog",
            "artifact_path": str(tmp_path / "cand.jpg"),
        },
        evidence_hashes={
            "artifact_sha256": "abc12345",
            "metadata_sha256": "def67890",
        },
        blockchain_proof={
            "status": "anchored",
            "network": "Ethereum Sepolia",
            "record_id": 1,
            "transaction_hash": "0x123",
        },
    )

    rec_id = insert_record(dossier, db_path=db_file)
    assert rec_id == 1

    fetched = get_record(rec_id, db_path=db_file)
    assert fetched is not None
    assert fetched["social_platform"] == "Twitter/X"
    assert fetched["social_author"] == "@Alice"
    assert fetched["artifact_hash"] == "abc12345"

    records = list_records(limit=10, db_path=db_file)
    assert len(records) == 1
    assert records[0]["id"] == 1


def test_verify_offline_record_tampered(tmp_path: Path):
    db_file = tmp_path / "test_vault.db"
    art_file = tmp_path / "art.jpg"
    art_file.write_bytes(b"initial_bytes")

    dossier = PipelineDossier(
        status="VERIFIED_MATCH",
        liveness={"is_live": True, "score": 90.0, "reasons": []},
        social_post={"platform": "Web", "image_url": "http://example.com/art.jpg"},
        match_metrics={
            "distance": 0.2,
            "threshold": 0.5,
            "confidence": 88.0,
            "matched": True,
            "artifact_path": str(art_file),
        },
        evidence_hashes={
            "artifact_sha256": "dummy_hash",
            "metadata_sha256": "meta_hash",
        },
        blockchain_proof={},
    )

    rec_id = insert_record(dossier, db_path=db_file)

    # Hashes don't match initial_bytes -> should report TAMPERED
    res = verify_offline_record(rec_id, db_path=db_file)
    assert res["status"] == "TAMPERED"
    assert res["verified"] is False
