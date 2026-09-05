from pathlib import Path
import numpy as np
import pytest

from app.classifier import FaceClassifier


def test_classifier_init_and_classes():
    emb1 = np.ones(128, dtype=np.float64)
    emb2 = np.zeros(128, dtype=np.float64)
    labels = ["Alice", "Bob"]
    classifier = FaceClassifier(embeddings=np.vstack([emb1, emb2]), labels=labels, threshold=0.5)

    assert classifier.classes == ["Alice", "Bob"]
    assert "Alice" in classifier.centroids
    assert "Bob" in classifier.centroids


def test_classifier_save_and_load(tmp_path: Path):
    emb1 = np.ones(128, dtype=np.float64)
    labels = ["Alice"]
    classifier = FaceClassifier(embeddings=emb1.reshape(1, -1), labels=labels, threshold=0.45)

    model_file = tmp_path / "model.pkl"
    classifier.save(model_file)
    assert model_file.exists()

    loaded = FaceClassifier.load(model_file)
    assert loaded.classes == ["Alice"]
    assert loaded.threshold == 0.45
    np.testing.assert_allclose(loaded.embeddings, classifier.embeddings)


def test_predict_without_training_raises():
    classifier = FaceClassifier()
    with pytest.raises(RuntimeError, match="not been trained"):
        classifier.predict(Path("dummy.jpg"))


def test_predict_with_mock_faces():
    from unittest.mock import patch
    emb1 = np.ones(128, dtype=np.float64)
    labels = ["Alice"]
    classifier = FaceClassifier(embeddings=emb1.reshape(1, -1), labels=labels, threshold=0.5)

    with patch("app.classifier.encode_all_faces") as mock_encode:
        # Mock finding two faces: Alice (identical embedding) and stranger (zero embedding)
        mock_encode.return_value = [
            (np.ones(128, dtype=np.float64), (10, 50, 50, 10)),
            (np.zeros(128, dtype=np.float64), (60, 90, 90, 60)),
        ]
        results = classifier.predict(Path("mock.jpg"))
        assert len(results) == 2
        assert results[0].label == "Alice"
        assert results[0].matched is True
        assert results[0].distance == 0.0
        assert results[0].confidence == 100.0

        assert results[1].label == "Unknown"
        assert results[1].matched is False
