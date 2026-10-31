"""подход 2: xlm-roberta-base с LoRA-надстройками (дообучение и вероятности на val и golden считает Colab)

адаптеры и probs_val.json, probs_golden.json лежат в results/lora-xlmr/adapter (из notebooks/train_lora.ipynb)
для новых текстов нужна база xlm-roberta-base в кэше huggingface
порог подбирается на val: uv run python -m detectors.implementations.lora --tune
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.schema import read_jsonl
from detectors.base import Detector
from eval.metrics import scores

BASE_MODEL = "FacebookAI/xlm-roberta-base"
MAX_LEN = 384
RESULTS_DIR = Path("results/lora-xlmr")
ADAPTER_DIR = RESULTS_DIR / "adapter"
SPLITS_DIR = Path("data/splits")
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))


def best_threshold(y_true: list[bool], probs: list[float]) -> tuple[float, dict]:
    """порог с наибольшим F1 на val (тот же критерий, что у эмбеддингов); неопределённый F1 считается нулём"""
    best, best_row = None, None
    for th in THRESHOLDS:
        s = scores(y_true, [p >= th for p in probs])
        if best is None or (s["f1"] or 0.0) > (best_row["f1"] or 0.0):
            best, best_row = th, {k: s[k] for k in ("recall", "precision", "f1", "fpr")}
    return best, best_row


def load_precomputed(adapter_dir: Path = ADAPTER_DIR, splits_dir: Path = SPLITS_DIR) -> dict[str, float]:
    """вероятности атаки, посчитанные в Colab: текст -> вероятность; выборка сверяется по хэшу файла"""
    probs: dict[str, float] = {}
    for split in ("val", "golden"):
        path = adapter_dir / f"probs_{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        src = splits_dir / f"{split}.jsonl"
        if hashlib.sha256(src.read_bytes()).hexdigest() != data["split_sha256"]:
            raise SystemExit(f"{path} посчитан на другой версии {src.name}: переобучите или пересчитайте в Colab")
        probs.update(zip((r.text for r in read_jsonl(src)), data["probs"]))
    return probs


class LoraDetector(Detector):
    name = "lora-xlmr"
    cost_per_1000 = 0.0  # модель работает локально
    latency_note = "вероятности из Colab: в задержку не входит работа модели"
    results_dir = RESULTS_DIR  # подклассы (prompt tuning) меняют каталог и ноутбук
    notebook = "notebooks/train_lora.ipynb"
    base_model = BASE_MODEL  # подклассы с другой базой меняют это поле

    def __init__(self, adapter_dir: Path | None = None):
        adapter_dir = adapter_dir or self.results_dir / "adapter"
        if not (adapter_dir / "adapter_config.json").exists():
            raise SystemExit(f"нет адаптеров в {adapter_dir}: обучите {self.notebook} и распакуйте архив из Colab туда")
        self.adapter_dir = adapter_dir
        self.precomputed = load_precomputed(adapter_dir)
        self._tok = self._model = None
        cfg = self.results_dir / "threshold.json"
        self.threshold = json.loads(cfg.read_text(encoding="utf-8"))["threshold"] if cfg.exists() else 0.5

    def _load_model(self) -> None:
        """модель с адаптером нужна только для текстов вне val и golden"""
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.base_model)  # токенизатор у базы и адаптера один
        try:
            # модель уже в кэше: грузим без обращения к сети
            base = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=2, local_files_only=True)
        except OSError:
            base = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=2)
        self._model = PeftModel.from_pretrained(base, self.adapter_dir).eval()

    @torch.no_grad()
    def _probs(self, texts: list[str]) -> list[float]:
        missing = [t for t in texts if t not in self.precomputed]
        if missing and self._model is None:
            self._load_model()
        computed = {}
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            enc = self._tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            computed.update(zip(batch, torch.softmax(self._model(**enc).logits, dim=-1)[:, 1].tolist()))
        return [self.precomputed[t] if t in self.precomputed else computed[t] for t in texts]

    def predict(self, text: str) -> tuple[str, float]:
        p = self._probs([text])[0]
        return ("attack" if p >= self.threshold else "benign"), p


def tune(detector_cls: type[LoraDetector] = LoraDetector) -> dict:
    """подбирает порог на val и сохраняет его вместе с таблицей"""
    det = detector_cls()
    val = read_jsonl(SPLITS_DIR / "val.jsonl")
    probs = det._probs([r.text for r in val])
    y = [r.label == "attack" for r in val]
    th, row = best_threshold(y, probs)
    det.results_dir.mkdir(parents=True, exist_ok=True)
    (det.results_dir / "threshold.json").write_text(json.dumps({"threshold": th, "criterion": "максимум F1 на val", "val": row},
                                                              ensure_ascii=False, indent=2), encoding="utf-8")
    return {"threshold": th, **row}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="подобрать порог на val")
    if parser.parse_args().tune:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tune().items()})
