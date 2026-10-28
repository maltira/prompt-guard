import pytest

from data.golden_hash import GoldenChangedError, read_hash, register, sha256_bytes, verify_file


def test_register_creates_then_keeps(tmp_path):
    f = tmp_path / "golden.sha256"
    assert register("aaa", f) == "created"
    assert register("aaa", f) == "unchanged"
    assert read_hash(f) == "aaa"


def test_changed_hash_is_rejected_without_flag(tmp_path):
    f = tmp_path / "golden.sha256"
    register("aaa", f)
    with pytest.raises(GoldenChangedError):
        register("bbb", f)
    assert read_hash(f) == "aaa"
    assert register("bbb", f, update=True) == "updated"
    assert read_hash(f) == "bbb"


def test_verify_file(tmp_path):
    golden, f = tmp_path / "golden.jsonl", tmp_path / "golden.sha256"
    golden.write_bytes(b"abc\n")
    with pytest.raises(GoldenChangedError):
        verify_file(golden, f)  # хэш ещё не записан
    register(sha256_bytes(b"abc\n"), f)
    assert verify_file(golden, f) == sha256_bytes(b"abc\n")
    golden.write_bytes(b"abd\n")
    with pytest.raises(GoldenChangedError):
        verify_file(golden, f)
