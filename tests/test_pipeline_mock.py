from pathlib import Path
from unittest.mock import patch
import pytest

from app.face import FaceMatch
from app.pipeline import find_match
from app.web_search import Candidate, MockSearchProvider, LocalDirectorySearch


def test_mock_search_provider_returns_candidates():
    candidates = [Candidate("https://example.com/1.jpg"), Candidate("https://example.com/2.jpg")]
    provider = MockSearchProvider(candidates)
    assert provider.search(Path("ref.jpg")) == candidates


def test_local_directory_search(tmp_path: Path):
    (tmp_path / "img1.jpg").write_bytes(b"image1")
    (tmp_path / "img2.png").write_bytes(b"image2")
    (tmp_path / "ignore.txt").write_bytes(b"text")

    provider = LocalDirectorySearch(tmp_path)
    results = provider.search(Path("ref.jpg"))
    assert len(results) == 2
    urls = [c.url for c in results]
    assert any("img1.jpg" in u for u in urls)
    assert any("img2.png" in u for u in urls)


def test_find_match_with_mock(tmp_path: Path):
    ref_img = tmp_path / "ref.jpg"
    ref_img.write_bytes(b"ref_data")

    cand_img = tmp_path / "candidate.jpg"
    cand_img.write_bytes(b"cand_data")

    output_dir = tmp_path / "output"
    provider = MockSearchProvider([Candidate(cand_img.as_uri())])

    with patch("app.pipeline.encode_single_face") as mock_ref, \
         patch("app.pipeline.find_best_matching_face") as mock_cand:
        mock_ref.return_value = [0.1] * 128
        mock_cand.return_value = (FaceMatch(0.25, 0.5, 85.0), (10, 40, 40, 10))

        verified = find_match(ref_img, provider, threshold=0.5, output_dir=output_dir, annotate=False)
        assert verified.match.matched
        assert verified.match.distance == 0.25
        assert verified.metadata["confidence"] == 85.0
        assert verified.artifact_path.exists()
