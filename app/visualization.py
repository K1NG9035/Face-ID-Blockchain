from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def annotate_face_match(
    image_path: Path,
    face_location: tuple[int, int, int, int],
    distance: float,
    confidence: float,
    output_path: Path,
    label: str = "MATCH",
) -> Path:
    "Draw a high-contrast bounding box and match badge around a verified face."
    top, right, bottom, left = face_location
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    # Colors
    box_color = (0, 230, 118, 255)  # Neon green
    accent_color = (0, 200, 83, 255)
    overlay_fill = (0, 230, 118, 40)   # Subtle face tint

    # Tint the detected face area lightly
    draw.rectangle([(left, top), (right, bottom)], fill=overlay_fill)

    # Main bounding box
    line_width = max(3, int(min(image.size) * 0.005))
    draw.rectangle([(left, top), (right, bottom)], outline=box_color, width=line_width)

    # Corner brackets for a modern CV aesthetic
    corner_len = max(12, int(min(right - left, bottom - top) * 0.2))
    corner_w = line_width + 2
    # Top-left
    draw.line([(left, top), (left + corner_len, top)], fill=accent_color, width=corner_w)
    draw.line([(left, top), (left, top + corner_len)], fill=accent_color, width=corner_w)
    # Top-right
    draw.line([(right, top), (right - corner_len, top)], fill=accent_color, width=corner_w)
    draw.line([(right, top), (right, top + corner_len)], fill=accent_color, width=corner_w)
    # Bottom-left
    draw.line([(left, bottom), (left + corner_len, bottom)], fill=accent_color, width=corner_w)
    draw.line([(left, bottom), (left - corner_len, bottom)], fill=accent_color, width=corner_w)
    # Bottom-right
    draw.line([(right, bottom), (right - corner_len, bottom)], fill=accent_color, width=corner_w)
    draw.line([(right, bottom), (right, bottom - corner_len)], fill=accent_color, width=corner_w)

    # Badge text
    badge_text = f"{label} | Conf: {confidence:.1f}% | Dist: {distance:.3f}"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), badge_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad = 6
    badge_top = max(0, top - text_h - pad * 2 - 4)
    badge_bottom = badge_top + text_h + pad * 2
    badge_right = min(image.width, left + text_w + pad * 2)

    # Draw dark background badge with subtle green outline
    draw.rectangle([(left, badge_top), (badge_right, badge_bottom)], fill=(15, 23, 42, 230), outline=box_color, width=1)
    draw.text((left + pad, badge_top + pad), badge_text, fill=(255, 255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return output_path
