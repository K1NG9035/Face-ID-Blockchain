from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from app.server import app
from app.service import PipelineDossier

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "VeriFace" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_records" in data


def test_records_list():
    response = client.get("/api/records")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_scan_api_mocked():
    mock_dossier = PipelineDossier(
        status="VERIFIED_MATCH",
        liveness={"is_live": True, "score": 90.0, "reasons": []},
        social_post={"platform": "Twitter/X", "author": "@Star", "post_url": "https://x.com/1"},
        match_metrics={"distance": 0.2, "confidence": 88.0, "matched": True},
        evidence_hashes={"artifact_sha256": "h1", "metadata_sha256": "h2"},
        blockchain_proof={"status": "skipped"},
    )

    with patch("app.server.run_pipeline_service", return_value=mock_dossier):
        response = client.post(
            "/api/scan",
            json={
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "threshold": 0.5,
                "skip_blockchain": True,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "VERIFIED_MATCH"
        assert result["social_post"]["platform"] == "Twitter/X"
