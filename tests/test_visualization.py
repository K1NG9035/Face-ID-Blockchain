from pathlib import Path
from PIL import Image
import pytest

from app.visualization import annotate_face_match


def test_annotate_face_match(tmp_path: Path):
    src_img = tmp_path / "test_input.jpg"
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    img.save(src_img)

    out_img = tmp_path / "test_annotated.jpg"
    # face location: (top, right, bottom, left)
    face_loc = (30, 120, 120, 30)
    result_path = annotate_face_match(src_img, face_loc, distance=0.35, confidence=82.5, output_path=out_img)

    assert result_path.exists()
    with Image.open(result_path) as annotated:
        assert annotated.size == (200, 200)
