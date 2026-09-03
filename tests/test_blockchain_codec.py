import pytest

from app.blockchain import as_bytes32, from_bytes32


def test_bytes32_round_trip():
    value = "0x" + "ab" * 32
    assert from_bytes32(as_bytes32(value)) == value


def test_bytes32_rejects_wrong_length():
    with pytest.raises(ValueError):
        as_bytes32("00")