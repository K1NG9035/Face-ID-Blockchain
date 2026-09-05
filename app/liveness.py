from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
from PIL import Image
import numpy as np


@dataclass(frozen=True)
class LivenessResult:
    is_live: bool
    score: float
    reasons: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_liveness(
    image_source: Path | str | bytes | Image.Image,
    sharpness_threshold: float = 20.0,
    face_location: tuple[int, int, int, int] | None = None,
) -> LivenessResult:
    """Passive liveness and anti-spoofing analysis.
    If face_location (top, right, bottom, left) is supplied, evaluates the cropped
    facial region (with margin) rather than background elements.
    """
    if isinstance(image_source, (Path, str)):
        image = Image.open(image_source)
    elif isinstance(image_source, bytes):
        import io
        image = Image.open(io.BytesIO(image_source))
    else:
        image = image_source

    image = image.convert("RGB")

    # If face location is supplied, crop to facial bounding box + 15% margin
    if face_location is not None:
        top, right, bottom, left = face_location
        img_w, img_h = image.size
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        margin_x = int(box_w * 0.15)
        margin_y = int(box_h * 0.15)
        crop_box = (
            max(0, left - margin_x),
            max(0, top - margin_y),
            min(img_w, right + margin_x),
            min(img_h, bottom + margin_y),
        )
        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
            image = image.crop(crop_box)

    rgb = np.asarray(image, dtype=np.float64)
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    h, w = gray.shape
    if h < 20 or w < 20:
        return LivenessResult(
            is_live=False,
            score=0.0,
            reasons=["Resolution too low for liveness analysis"],
            details={},
        )

    # 1. Sharpness / Texture Variance (finite differences)
    gy, gx = np.gradient(gray)
    sharpness_var = float(np.var(gx) + np.var(gy))

    # 2. 2D FFT Moiré / Screen Grid Detection
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    cy, cx = h // 2, w // 2
    r_min = max(5, min(h, w) // 8)
    y, x = np.ogrid[:h, :w]
    high_pfmask = ((x - cx) ** 2 + (y - cy) ** 2) > (r_min ** 2)
    ehigh = float(np.mean(magnitude[high_pfmask]))
    etotal = float(np.mean(magnitude) + 1e-6)
    high_freq_ratio = ehigh / etotal

    # 3. Color Variance (detects desaturated paper prints)
    color_std = float(np.std(rgb))

    reasons: list[str] = []
    score = 100.0

    if sharpness_var < sharpness_threshold:
        reasons.append("Low sharpness / blurry texture (potential flat print attack)")
        penalty = (1.0 - (sharpness_var / max(1.0, sharpness_threshold))) * 40.0
        score -= penalty

    if high_freq_ratio > 0.85:
        reasons.append("Moire pattern detected (potential LCD/OLED screen attack)")
        score -= 35.0

    if color_std < 15.0:
        reasons.append("Unnatural color range (grayscale or poor color print)")
        score -= 25.0


    score = max(0.0, min(100.0, round(score, 1)))
    if not reasons:
        reasons.append("Natural facial texture and color distribution verified")

    is_live = score >= 50.0

    return LivenessResult(
        is_live=is_live,
        score=score,
        reasons=reasons,
        details={
            "sharpness_variance": round(sharpness_var, 2),
            "high_freq_ratio": round(high_freq_ratio, 3),
            "color_std": round(color_std, 2),
        },
    )
