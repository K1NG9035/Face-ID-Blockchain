from app.fingerprint import canonical_json, metadata_hash, sha256_bytes


def test_canonical_metadata_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert metadata_hash({"a": 1, "b": 2}) == metadata_hash({"b": 2, "a": 1})


def test_sha256_is_stable():
    assert sha256_bytes(b"FaceWebChain") == "2d0962e23ab1af1adc8e2a0925d087caa21250252c2e6c336fc050e579050bcd"
