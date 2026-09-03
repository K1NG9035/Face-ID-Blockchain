from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Candidate:
    url: str
    page_url: str | None = None


def unique_candidates(candidates: list[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        parsed = urlparse(candidate.url)
        if parsed.scheme not in {"http", "https"} or candidate.url in seen:
            continue
        seen.add(candidate.url)
        result.append(candidate)
    return result


class GoogleVisionSearch:
    def search(self, image_path: Path) -> list[Candidate]:
        try:
            from google.cloud import vision
        except ImportError as exc:
            raise RuntimeError("Install google-cloud-vision to use live web discovery") from exc
        client = vision.ImageAnnotatorClient()
        response = client.web_detection(image=vision.Image(content=image_path.read_bytes()))
        if response.error.message:
            raise RuntimeError(response.error.message)
        detection = response.web_detection
        candidates = [Candidate(image.url) for image in detection.full_matching_images]
        candidates += [Candidate(image.url) for image in detection.visually_similar_images]
        candidates += [Candidate(page.url) for page in detection.pages_with_matching_images]
        return unique_candidates(candidates)


def download_candidate(candidate: Candidate, destination: Path, timeout: int = 15) -> Path:
    request = Request(candidate.url, headers={"User-Agent": "FaceWebChain/1.0"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError("Candidate URL did not return an image")
        destination.write_bytes(response.read())
    return destination
