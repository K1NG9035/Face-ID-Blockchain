from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
import numpy as np

from .face import calculate_confidence, encode_all_faces, encode_single_face

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    distance: float
    matched: bool
    face_location: tuple[int, int, int, int]


class FaceClassifier:
    """Trained Face Recognition Classifier storing 128-d metric embeddings and prototypes."""

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.embeddings: np.ndarray = embeddings if embeddings is not None else np.empty((0, 128))
        self.labels: list[str] = labels if labels is not None else []
        self.threshold: float = threshold
        self.centroids: dict[str, np.ndarray] = self._compute_centroids()

    def _compute_centroids(self) -> dict[str, np.ndarray]:
        centroids: dict[str, np.ndarray] = {}
        unique_labels = sorted(set(self.labels))
        for lbl in unique_labels:
            indices = [i for i, name in enumerate(self.labels) if name == lbl]
            class_vecs = self.embeddings[indices]
            centroid = np.mean(class_vecs, axis=0)
            norm = np.linalg.norm(centroid)
            centroids[lbl] = centroid / (norm if norm > 0 else 1.0)
        return centroids

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.labels))

    def train_from_directory(
        self,
        dataset_dir: Path,
        model: str = "hog",
        upsample_times: int = 0,
    ) -> int:
        """Scan directory for labeled face images.
        Supports two structures:
        1. Subdirectories per class: dataset_dir/<label>/<image>.jpg
        2. Direct image files: dataset_dir/<label>_<index>.jpg or single identity folder.
        """
        all_embeddings: list[np.ndarray] = []
        all_labels: list[str] = []

        subdirs = [p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if subdirs:
            for subdir in subdirs:
                label = subdir.name
                for file_path in subdir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                        try:
                            enc = encode_single_face(file_path, upsample_times=upsample_times, model=model)
                            all_embeddings.append(np.asarray(enc, dtype=np.float64))
                            all_labels.append(label)
                        except Exception:
                            continue
        else:
            label = dataset_dir.name
            for file_path in dataset_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    try:
                        enc = encode_single_face(file_path, upsample_times=upsample_times, model=model)
                        all_embeddings.append(np.asarray(enc, dtype=np.float64))
                        all_labels.append(label)
                    except Exception:
                        continue

        if not all_embeddings:
            raise ValueError(f"No valid faces could be encoded from {dataset_dir}")

        self.embeddings = np.vstack(all_embeddings)
        self.labels = all_labels
        self.centroids = self._compute_centroids()
        return len(all_labels)

    def predict(
        self,
        image_path: Path,
        threshold: float | None = None,
        model: str = "hog",
        upsample_times: int = 0,
    ) -> list[ClassificationResult]:
        """Detect and classify all faces in an image against trained identities."""
        if len(self.embeddings) == 0:
            raise RuntimeError("Model has not been trained yet.")

        thresh = threshold if threshold is not None else self.threshold
        faces = encode_all_faces(image_path, upsample_times=upsample_times, model=model)
        results: list[ClassificationResult] = []

        for enc, location in faces:
            enc_arr = np.asarray(enc, dtype=np.float64)
            dists = np.linalg.norm(self.embeddings - enc_arr, axis=1)
            min_idx = int(np.argmin(dists))
            best_dist = float(dists[min_idx])
            best_label = self.labels[min_idx]

            matched = best_dist <= thresh
            confidence = calculate_confidence(best_dist, thresh)
            pred_label = best_label if matched else "Unknown"

            results.append(
                ClassificationResult(
                    label=pred_label,
                    confidence=confidence,
                    distance=best_dist,
                    matched=matched,
                    face_location=location,
                )
            )
        return results


    def save(self, output_path: Path) -> Path:
        """Serialize trained model to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embeddings": self.embeddings,
            "labels": self.labels,
            "threshold": self.threshold,
            "centroids": self.centroids,
        }
        with output_path.open("wb") as stream:
            pickle.dump(payload, stream)
        return output_path

    @classmethod
    def load(cls, model_path: Path) -> FaceClassifier:
        """Load trained model from file."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        with model_path.open("rb") as stream:
            payload = pickle.load(stream)
        classifier = cls(
            embeddings=payload["embeddings"],
            labels=payload["labels"],
            threshold=payload.get("threshold", 0.5),
        )
        classifier.centroids = payload.get("centroids", classifier._compute_centroids())
        return classifier


def train_face_classifier(
    dataset_dir: Path,
    output_model_path: Path,
    model: str = "hog",
    upsample_times: int = 0,
    threshold: float = 0.5,
) -> FaceClassifier:
    """Convenience helper to train and persist a face recognition model."""
    classifier = FaceClassifier(threshold=threshold)
    classifier.train_from_directory(dataset_dir, model=model, upsample_times=upsample_times)
    classifier.save(output_model_path)
    return classifier
