from app.face import FaceMatch


def test_threshold_is_inclusive():
    assert FaceMatch(0.5, 0.5).matched
    assert not FaceMatch(0.51, 0.5).matched
