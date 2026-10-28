import json

import numpy as np

from data.schema import Record, write_jsonl
from detectors.embeddings_cache import embed_split


class FakeEncoder:
    model_name = "org/fake-model"
    prefix = "query: "

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def make_split(tmp_path, texts):
    recs = [Record(id=f"r{i}", text=t, label="benign", lang="en", category="benign", source="t")
            for i, t in enumerate(texts)]
    write_jsonl(tmp_path / "val.jsonl", recs)


def test_cache_hit_and_order(tmp_path):
    make_split(tmp_path, ["a", "bb", "ccc"])
    enc, cache = FakeEncoder(), tmp_path / "cache"
    first = embed_split(enc, "val", tmp_path, cache)
    second = embed_split(enc, "val", tmp_path, cache)
    assert enc.calls == 1
    assert first[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert np.array_equal(first, second)
    meta = json.loads((cache / "fake-model__val.json").read_text(encoding="utf-8"))
    assert meta["n"] == 3 and meta["dim"] == 2


def test_cache_invalidated_when_split_changes(tmp_path):
    make_split(tmp_path, ["a", "bb"])
    enc, cache = FakeEncoder(), tmp_path / "cache"
    embed_split(enc, "val", tmp_path, cache)
    make_split(tmp_path, ["a", "bb", "dddd"])
    vecs = embed_split(enc, "val", tmp_path, cache)
    assert enc.calls == 2 and len(vecs) == 3


def test_cache_invalidated_when_prefix_changes(tmp_path):
    make_split(tmp_path, ["a"])
    enc, cache = FakeEncoder(), tmp_path / "cache"
    embed_split(enc, "val", tmp_path, cache)
    enc.prefix = "passage: "
    embed_split(enc, "val", tmp_path, cache)
    assert enc.calls == 2


def test_cached_embeddings_does_not_compute(tmp_path):
    from detectors.embeddings_cache import cached_embeddings
    make_split(tmp_path, ["a", "bb"])
    enc, cache = FakeEncoder(), tmp_path / "cache"
    assert cached_embeddings(enc, "val", tmp_path, cache) is None
    assert enc.calls == 0
    embed_split(enc, "val", tmp_path, cache)
    assert cached_embeddings(enc, "val", tmp_path, cache) is not None
    assert enc.calls == 1
