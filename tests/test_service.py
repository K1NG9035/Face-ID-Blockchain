import base64
import io
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pytest

from app.face import FaceMatch
from app.service import PipelineDossier, run_pipeline_service
from app.web_search import Candidate, MockSearchProvider


def _create_dummy_image(tmp_path: Path, filename: str = "test.jpg") -> Path:
    img = Image.new("RGB", (100, 100), color=(120, 150, 180))
    p = tmp_path / filename
    img.save(p)
    return p


def test_run_pipeline_service_with_path(tmp_path: Path):
    input_img = _create_dummy_image(tmp_path, "input.jpg")
    cand_img = _create_dummy_image(tmp_path, "cand.jpg")
    out_dir = tmp_path / "output"

    provider = MockSearchProvider([
        Candidate(cand_img.as_uri(), page_url="https://x.com/Star/status/123")
    ])

    with patch("app.pipeline.encode_single_face") as mock_ref, \
         patch("app.pipeline.find_best_matching_face") as mock_cand:
        mock_ref.return_value = [0.1] * 128
        mock_cand.return_value = (FaceMatch(0.2, 0.5, 88.0), (10, 40, 40, 10))

        dossier = run_pipeline_service(
            image_input=input_img,
            output_dir=out_dir,
            search_provider=provider,
            skip_blockchain=True,
        )

        assert isinstance(dossier, PipelineDossier)
        assert dossier.status == "VERIFIED_MATCH"
        assert dossier.social_post["platform"] == "Twitter/X"
        assert dossier.social_post["author"] == "@Star"
        assert dossier.match_metrics["confidence"] == 88.0
        assert dossier.match_metrics["distance"] == 0.2
        assert dossier.evidence_hashes["artifact_sha256"] != ""


def test_run_pipeline_service_with_base64(tmp_path: Path):
    cand_img = _create_dummy_image(tmp_path, "cand2.jpg")
    out_dir = tmp_path / "output_b64"

    # Generate base64 data URL
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), color=(100, 100, 100)).save(buf, format="JPEG")
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    provider = MockSearchProvider([
        Candidate(cand_img.as_uri(), page_url="https://reddit.com/r/technology/comments/1/ai_advancement")
    ])

    with patch("app.pipeline.encode_single_face") as mock_ref, \
         patch("app.pipeline.find_best_matching_face") as mock_cand:
        mock_ref.return_value = [0.1] * 128
        mock_cand.return_value = (FaceMatch(0.15, 0.5, 91.0), (5, 35, 35, 5))

        dossier = run_pipeline_service(
            image_input=b64_str,
            output_dir=out_dir,
            search_provider=provider,
            skip_blockchain=True,
        )

        assert dossier.status == "VERIFIED_MATCH"
        assert dossier.social_post["platform"] == "Reddit"
        assert dossier.social_post["author"] == "r/technology"
