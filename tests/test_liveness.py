from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from app.liveness import LivenessResult, evaluate_liveness


def test_evaluate_liveness_natural_image(tmp_path: Path):
    # Create an image with natural gradient/texture
    img = Image.new("RGB", (200, 200))
    pixels = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    img_path = tmp_path / "natural.jpg"
    img.save(img_path)

    result = evaluate_liveness(img_path)
    assert isinstance(result, LivenessResult)
    assert result.is_live is True
    assert result.score >= 50.0
    assert "sharpness_variance" in result.details


def test_evaluate_liveness_flat_color(tmp_path: Path):
    # Create a completely flat solid image (0 variance -> print attack / blank)
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img_path = tmp_path / "flat.jpg"
    img.save(img_path)

    result = evaluate_liveness(img_path)
    assert result.is_live is False
    assert result.score < 50.0
    assert any("Low sharpness" in r or "Unnatural" in r for r in result.reasons)


def test_evaluate_liveness_bytes_input():
    # Pass raw image bytes directly
    img = Image.new("RGB", (100, 100), color=(80, 140, 200))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    result = evaluate_liveness(buf.getvalue())
    assert isinstance(result, LivenessResult)


def test_evaluate_liveness_with_face_location(tmp_path: Path):
    # Image with natural background and flat inner region
    img = Image.new("RGB", (300, 300))
    pixels = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    # Paint center box flat gray (simulating spoofed screen in center)
    pixels[100:200, 100:200] = [128, 128, 128]
    img = Image.fromarray(pixels)
    img_path = tmp_path / "spoof_crop.jpg"
    img.save(img_path)

    # When evaluating whole frame: outer random pixels raise sharpness
    res_whole = evaluate_liveness(img_path)

    # When evaluating specifically on the face location (100, 200, 200, 100)
    res_cropped = evaluate_liveness(img_path, face_location=(100, 200, 200, 100))
    # Cropped region is predominantly flat, so its sharpness/score must be lower than whole frame
    assert res_cropped.details["sharpness_variance"] < res_whole.details["sharpness_variance"]

