import hashlib
import json

import pytest

from detectors.implementations.lora import best_threshold, load_precomputed


def test_best_threshold_prefers_separating_value():
    y = [True] * 5 + [False] * 5
    probs = [0.9, 0.8, 0.85, 0.7, 0.95, 0.1, 0.2, 0.15, 0.3, 0.05]
    th, row = best_threshold(y, probs)
    assert 0.3 < th <= 0.7 and row["recall"] == 1.0 and row["fpr"] == 0.0


def test_best_threshold_handles_no_positive_predictions():
    th, row = best_threshold([True, False], [0.01, 0.02])
    assert 0.05 <= th <= 0.95


def _write_split(path, texts):
    path.write_text("\n".join(json.dumps({"id": str(i), "text": t, "label": "benign", "lang": "en", "category": "benign",
                                         "source": "x"}) for i, t in enumerate(texts)) + "\n", encoding="utf-8")


def test_load_precomputed_maps_texts_to_probs(tmp_path):
    splits, adapter = tmp_path / "splits", tmp_path / "adapter"
    splits.mkdir(), adapter.mkdir()
    _write_split(splits / "val.jsonl", ["a", "b"])
    sha = hashlib.sha256((splits / "val.jsonl").read_bytes()).hexdigest()
    (adapter / "probs_val.json").write_text(json.dumps({"split": "val", "split_sha256": sha, "n": 2, "probs": [0.1, 0.9]}))
    assert load_precomputed(adapter, splits) == {"a": 0.1, "b": 0.9}


def test_load_precomputed_rejects_changed_split(tmp_path):
    splits, adapter = tmp_path / "splits", tmp_path / "adapter"
    splits.mkdir(), adapter.mkdir()
    _write_split(splits / "val.jsonl", ["a"])
    (adapter / "probs_val.json").write_text(json.dumps({"split": "val", "split_sha256": "0" * 64, "n": 1, "probs": [0.1]}))
    with pytest.raises(SystemExit):
        load_precomputed(adapter, splits)
