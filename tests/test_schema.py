import pytest

from data.schema import Record, read_jsonl, write_jsonl


def rec(**kw):
    base = dict(id="a1", text="hello", label="attack", lang="en", category="x", source="own")
    return Record(**{**base, **kw})


def test_roundtrip(tmp_path):
    items = [rec(), rec(id="a2", text="привет", label="benign", lang="ru")]
    write_jsonl(tmp_path / "d.jsonl", items)
    assert read_jsonl(tmp_path / "d.jsonl") == items


@pytest.mark.parametrize("kw", [{"label": "spam"}, {"lang": "de"}, {"text": "  "},
                                {"category": "benign_borderline"}])
def test_invalid(kw):
    with pytest.raises(ValueError):
        rec(**kw)


def test_duplicate_ids(tmp_path):
    write_jsonl(tmp_path / "d.jsonl", [rec(), rec()])
    with pytest.raises(ValueError):
        read_jsonl(tmp_path / "d.jsonl")
