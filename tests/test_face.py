from pathlib import Path
import pytest

from app.face import FaceMatch, _prominent_face, calculate_confidence, encode_all_faces, encode_single_face


def test_threshold_is_inclusive():
    match = FaceMatch(0.5, 0.5)
    assert match.matched
    assert match.confidence > 0.0

    not_matched = FaceMatch(0.51, 0.5)
    assert not not_matched.matched
    assert not_matched.confidence < match.confidence


def test_calculate_confidence_scale():
    assert calculate_confidence(0.0, 0.5) == 100.0
    assert calculate_confidence(0.5, 0.5) == 70.0
    assert calculate_confidence(0.75, 0.5) == 35.0
    assert calculate_confidence(1.0, 0.5) == 0.0
    assert calculate_confidence(0.1, 0.0) == 0.0


def test_prominent_face_filtering():
    # Only 1 face -> returned directly
    single = [(10, 50, 50, 10)]
    assert _prominent_face(single) == single

    # Two faces: one large (area 1600), one tiny (area 25)
    large = (0, 40, 40, 0)       # area = 40 * 40 = 1600
    tiny = (50, 55, 55, 50)      # area = 5 * 5 = 25 (< 20% of 1600 = 320)
    filtered = _prominent_face([large, tiny])
    assert filtered == [large]


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="non-negative"):
        encode_single_face(Path("nonexistent.jpg"), upsample_times=-1)

    with pytest.raises(ValueError, match="model must be"):
        encode_single_face(Path("nonexistent.jpg"), model="invalid")

    with pytest.raises(ValueError, match="non-negative"):
        encode_all_faces(Path("nonexistent.jpg"), upsample_times=-1)

    with pytest.raises(ValueError, match="model must be"):
        encode_all_faces(Path("nonexistent.jpg"), model="invalid")


def test_cosine_similarity_calculation():
    from app.face import compute_cosine_similarity
    import numpy as np

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0)

    v3 = np.array([0.0, 1.0, 0.0])
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0)

    # FaceMatch approximate cosine similarity property
    match = FaceMatch(distance=0.0, threshold=0.5)
    assert match.cosine_similarity == pytest.approx(1.0)

    # At distance 0.5: cos_sim ~ 1 - (0.25)/2 = 0.875
    match_mid = FaceMatch(distance=0.5, threshold=0.5)
    assert match_mid.cosine_similarity == pytest.approx(0.875, abs=1e-3)

