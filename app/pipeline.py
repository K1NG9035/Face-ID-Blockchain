from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .face import FaceMatch, encode_single_face, find_best_matching_face
from .fingerprint import metadata_hash, sha256_file
from .visualization import annotate_face_match
from .web_search import Candidate, download_candidate


class SearchProvider(Protocol):
    def search(self, image_path: Path) -> list[Candidate]: ...


@dataclass(frozen=True)
class VerifiedCandidate:
    candidate: Candidate
    artifact_path: Path
    match: FaceMatch
    artifact_hash: str
    metadata: dict[str, Any]
    metadata_hash: str
    annotated_path: Path | None = None
    face_location: tuple[int, int, int, int] | None = None
    reference_encoding: list[float] | None = None


def find_match(
    image_path: Path,
    search_provider: SearchProvider,
    threshold: float,
    output_dir: Path,
    upsample_times: int = 0,
    detector_model: str = "hog",
    annotate: bool = True,
) -> VerifiedCandidate:
    reference = encode_single_face(image_path, upsample_times=upsample_times, model=detector_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(search_provider.search(image_path), start=1):
        artifact_path = output_dir / f"candidate_{index}.jpg"
        try:
            download_candidate(candidate, artifact_path)
            best = find_best_matching_face(
                reference,
                artifact_path,
                threshold=threshold,
                upsample_times=upsample_times,
                model=detector_model,
            )
        except (OSError, ValueError, RuntimeError):
            artifact_path.unlink(missing_ok=True)
            continue

        if best is None:
            artifact_path.unlink(missing_ok=True)
            continue

        match, location = best
        annotated_path: Path | None = None
        if annotate:
            try:
                annotated_path = output_dir / f"candidate_{index}_annotated.jpg"
                annotate_face_match(
                    artifact_path,
                    location,
                    match.distance,
                    match.confidence,
                    annotated_path,
                )
            except Exception:
                annotated_path = None

        metadata = {
            "source_url": candidate.url,
            "page_url": candidate.page_url,
            "face_distance": round(match.distance, 4),
            "threshold": threshold,
            "confidence": round(match.confidence, 2),
            "cosine_similarity": round(getattr(match, "cosine_similarity", 0.0), 4),
            "detector_model": detector_model,
            "face_location": {
                "top": location[0],
                "right": location[1],
                "bottom": location[2],
                "left": location[3],
            },
        }
        return VerifiedCandidate(
            candidate=candidate,
            artifact_path=artifact_path,
            match=match,
            artifact_hash=sha256_file(artifact_path),
            metadata=metadata,
            metadata_hash=metadata_hash(metadata),
            annotated_path=annotated_path,
            face_location=location,
            reference_encoding=[round(float(x), 5) for x in reference] if hasattr(reference, "__iter__") else None,
        )
    raise LookupError("No web candidate passed local face verification")
