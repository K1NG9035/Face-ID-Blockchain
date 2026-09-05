from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import shutil
from urllib.request import Request, url2pathname, urlopen


@dataclass(frozen=True)
class Candidate:
    url: str
    page_url: str | None = None


def unique_candidates(candidates: list[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        parsed = urlparse(candidate.url)
        if parsed.scheme not in {"http", "https", "file"} or candidate.url in seen:
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
        try:
            client = vision.ImageAnnotatorClient()
        except Exception as exc:
            raise RuntimeError("Google Vision credentials are required for live web discovery") from exc
        response = client.web_detection(image=vision.Image(content=image_path.read_bytes()))
        if response.error.message:
            raise RuntimeError(response.error.message)
        detection = response.web_detection
        candidates = [Candidate(image.url) for image in detection.full_matching_images]
        candidates += [Candidate(image.url) for image in detection.visually_similar_images]
        candidates += [Candidate(page.url) for page in detection.pages_with_matching_images]
        return unique_candidates(candidates)


class MockSearchProvider:
    """Mock search provider returning pre-configured candidates for testing/development."""

    def __init__(self, candidates: list[Candidate] | None = None) -> None:
        self._candidates = candidates or []

    def search(self, image_path: Path) -> list[Candidate]:
        return list(self._candidates)


class LocalDirectorySearch:
    """Offline search provider scanning a local directory for candidate images."""

    def __init__(self, candidate_dir: Path) -> None:
        self.candidate_dir = candidate_dir

    def search(self, image_path: Path) -> list[Candidate]:
        if not self.candidate_dir.is_dir():
            raise FileNotFoundError(f"Mock candidate directory not found: {self.candidate_dir}")
        supported = {".jpg", ".jpeg", ".png", ".webp"}
        candidates: list[Candidate] = []
        for file in sorted(self.candidate_dir.iterdir()):
            if file.is_file() and file.suffix.lower() in supported:
                candidates.append(Candidate(file.resolve().as_uri(), page_url=f"local://{file.name}"))
        return candidates


def download_candidate(candidate: Candidate, destination: Path, timeout: int = 15) -> Path:
    if candidate.url.startswith("file://"):
        parsed = urlparse(candidate.url)
        path_str = url2pathname(parsed.path)
        if len(path_str) > 2 and path_str[0] == "\\" and path_str[2] == ":":
            path_str = path_str[1:]
        shutil.copyfile(path_str, destination)
        return destination

    request = Request(candidate.url, headers={"User-Agent": "FaceWebChain/1.0"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError("Candidate URL did not return an image")
        destination.write_bytes(response.read())
    return destination
