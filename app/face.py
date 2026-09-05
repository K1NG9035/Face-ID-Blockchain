from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")


@dataclass(frozen=True)
class FaceMatch:
    distance: float
    threshold: float
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.confidence == 0.0 and self.threshold > 0:
            object.__setattr__(self, "confidence", calculate_confidence(self.distance, self.threshold))

    @property
    def matched(self) -> bool:
        return self.distance <= self.threshold


def calculate_confidence(distance: float, threshold: float = 0.5) -> float:
    if threshold <= 0:
        return 0.0
    if distance <= threshold:
        confidence = 100.0 - (distance / threshold) * 30.0
    else:
        excess = (distance - threshold) / max(0.01, 1.0 - threshold)
        confidence = max(0.0, 70.0 - excess * 70.0)
    return round(confidence, 2)


def _library() -> Any:
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError("Install face-recognition to use live face processing") from exc
    return face_recognition


def _prominent_face(locations: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(locations) < 2:
        return locations
    areas = [(bottom - top) * (right - left) for top, right, bottom, left in locations]
    largest = max(areas)
    prominent = [location for location, area in zip(locations, areas) if area >= largest * 0.2]
    return prominent


def encode_single_face(path: Path, upsample_times: int = 0, model: str = "hog") -> Any:
    if upsample_times < 0:
        raise ValueError("upsample_times must be non-negative")
    if model not in {"hog", "cnn"}:
        raise ValueError("model must be 'hog' or 'cnn'")
    face_recognition = _library()
    image = face_recognition.load_image_file(path)
    locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample_times, model=model)
    locations = _prominent_face(locations)
    if len(locations) != 1:
        raise ValueError(f"Expected exactly one face, found {len(locations)}")
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    if len(encodings) != 1:
        raise ValueError("Unable to generate exactly one face encoding")
    return encodings[0]


def encode_all_faces(
    path: Path,
    upsample_times: int = 0,
    model: str = "hog",
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    if upsample_times < 0:
        raise ValueError("upsample_times must be non-negative")
    if model not in {"hog", "cnn"}:
        raise ValueError("model must be 'hog' or 'cnn'")
    face_recognition = _library()
    image = face_recognition.load_image_file(path)
    locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample_times, model=model)
    if not locations:
        return []
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    return list(zip(encodings, locations))


def compare_encoding(reference: Any, candidate: Any, threshold: float) -> FaceMatch:
    face_recognition = _library()
    distances = face_recognition.face_distance([reference], candidate)
    if len(distances) != 1:
        raise ValueError("Face comparison returned an unexpected result")
    dist = float(distances[0])
    return FaceMatch(dist, threshold, calculate_confidence(dist, threshold))


def find_best_matching_face(
    reference: Any,
    candidate_path: Path,
    threshold: float,
    upsample_times: int = 0,
    model: str = "hog",
) -> tuple[FaceMatch, tuple[int, int, int, int]] | None:
    faces = encode_all_faces(candidate_path, upsample_times=upsample_times, model=model)
    if not faces:
        return None
    face_recognition = _library()
    best_match: FaceMatch | None = None
    best_location: tuple[int, int, int, int] | None = None

    candidate_encodings = [enc for enc, _ in faces]
    distances = face_recognition.face_distance(candidate_encodings, reference)

    for dist, (_, location) in zip(distances, faces):
        dist_float = float(dist)
        if dist_float <= threshold:
            if best_match is None or dist_float < best_match.distance:
                best_match = FaceMatch(dist_float, threshold, calculate_confidence(dist_float, threshold))
                best_location = location

    if best_match is not None and best_location is not None:
        return best_match, best_location
    return None
