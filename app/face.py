from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FaceMatch:
    distance: float
    threshold: float

    @property
    def matched(self) -> bool:
        return self.distance <= self.threshold


def _library() -> Any:
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError("Install face-recognition to use live face processing") from exc
    return face_recognition


def encode_single_face(path: Path) -> Any:
    face_recognition = _library()
    image = face_recognition.load_image_file(path)
    locations = face_recognition.face_locations(image, model="hog")
    if len(locations) != 1:
        raise ValueError(f"Expected exactly one face, found {len(locations)}")
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    if len(encodings) != 1:
        raise ValueError("Unable to generate exactly one face encoding")
    return encodings[0]


def compare_encoding(reference: Any, candidate: Any, threshold: float) -> FaceMatch:
    face_recognition = _library()
    distances = face_recognition.face_distance([reference], candidate)
    if len(distances) != 1:
        raise ValueError("Face comparison returned an unexpected result")
    return FaceMatch(float(distances[0]), threshold)
