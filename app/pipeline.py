from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .face import FaceMatch, compare_encoding, encode_single_face
from .fingerprint import metadata_hash, sha256_file
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


def find_match(
    image_path: Path,
    search_provider: SearchProvider,
    threshold: float,
    output_dir: Path,
) -> VerifiedCandidate:
    reference = encode_single_face(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(search_provider.search(image_path), start=1):
        artifact_path = output_dir / f"candidate_{index}.jpg"
        try:
            download_candidate(candidate, artifact_path)
            candidate_encoding = encode_single_face(artifact_path)
            match = compare_encoding(reference, candidate_encoding, threshold)
        except (OSError, ValueError, RuntimeError):
            artifact_path.unlink(missing_ok=True)
            continue
        if not match.matched:
            artifact_path.unlink(missing_ok=True)
            continue
        metadata = {"source_url": candidate.url, "page_url": candidate.page_url, "face_distance": match.distance, "threshold": threshold}
        return VerifiedCandidate(candidate, artifact_path, match, sha256_file(artifact_path), metadata, metadata_hash(metadata))
    raise LookupError("No web candidate passed local face verification")
