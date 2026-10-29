"""подход 1: эмбеддинги текста и классификатор поверх них (логистическая регрессия, kNN, MLP)

обучение: uv run python -m detectors.implementations.embeddings --kind knn
классификатор обучается на train, настройки и порог подбираются на val, golden не используется
"""
import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from data.schema import read_jsonl
from detectors.base import Detector
from detectors.embeddings_cache import SPLITS_DIR, Encoder, cached_embeddings, embed_split
from eval.metrics import scores

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
RESULTS_DIR = Path("results")
SEED = 13
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))

# вид классификатора: (конструктор, сетка настроек, которые перебираются на val)
KINDS = {
    "logreg": (lambda **p: LogisticRegression(max_iter=2000, **p),
               {"C": [0.1, 1.0, 10.0, 100.0]}),
    "knn": (lambda **p: KNeighborsClassifier(metric="cosine", algorithm="brute", **p),
            {"n_neighbors": [5, 15, 50], "weights": ["uniform", "distance"]}),
    "mlp": (lambda **p: MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED, **p),
            {"hidden_layer_sizes": [(128,), (256, 128)], "alpha": [1e-4, 1e-2]}),
}


def detector_name(model_name: str, kind: str = "logreg") -> str:
    base = "embeddings-" + model_name.split("/")[-1].replace("multilingual-", "")
    return base if kind == "logreg" else f"{base}-{kind}"


class EmbeddingDetector(Detector):
    cost_per_1000 = 0.0  # модель работает локально

    def __init__(self, model_name: str = DEFAULT_MODEL, kind: str = "logreg", precomputed: bool = False,
                 encoder: Encoder | None = None):
        """precomputed - брать эмбеддинги val и golden из кэша (посчитаны вне ноутбука, например в Colab).

        тогда задержка в run_eval не включает кодирование текста моделью
        """
        self.model_name = model_name
        self.name = detector_name(model_name, kind)
        path = RESULTS_DIR / self.name / "classifier.joblib"
        if not path.exists():
            raise SystemExit(f"нет обученного классификатора {path}, сначала запустите обучение")
        artifact = joblib.load(path)
        self.clf, self.threshold = artifact["clf"], artifact["threshold"]
        self.encoder = encoder or Encoder(model_name)  # шлюз передаёт общий кодировщик на все классификаторы модели
        self._lookup = {}
        self.latency_note = None
        if precomputed:
            for split in ("val", "golden"):
                vecs = cached_embeddings(self.encoder, split)
                if vecs is None:
                    raise SystemExit(f"нет кэша эмбеддингов {split} для {model_name} в results/cache/embeddings/")
                texts = [r.text for r in read_jsonl(SPLITS_DIR / f"{split}.jsonl")]
                self._lookup.update(zip(texts, vecs))
            self.latency_note = "эмбеддинги из кэша: в задержку не входит кодирование моделью"

    def _vector(self, text: str):
        vec = self._lookup.get(text)
        return vec[None, :] if vec is not None else self.encoder.encode([text])

    def _label(self, score: float) -> str:
        return "attack" if score >= self.threshold else "benign"

    def predict(self, text: str) -> tuple[str, float]:
        score = float(self.clf.predict_proba(self._vector(text))[0, 1])
        return self._label(score), score

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        probs = self.clf.predict_proba(self.encoder.encode(texts))[:, 1]
        return [(self._label(float(p)), float(p)) for p in probs]


def train(model_name: str = DEFAULT_MODEL, kind: str = "logreg", cache_only: bool = False) -> dict:
    """cache_only - не считать эмбеддинги, а брать только готовые из кэша (защита от долгого счёта на ноутбуке)"""
    make, grid = KINDS[kind]
    encoder = Encoder(model_name)
    if cache_only:
        x_train, x_val = (cached_embeddings(encoder, s) for s in ("train", "val"))
        if x_train is None or x_val is None:
            raise SystemExit("нет кэша эмбеддингов train/val, положите файлы в results/cache/embeddings/")
    else:
        x_train, x_val = embed_split(encoder, "train"), embed_split(encoder, "val")
    y_train = np.array([r.label == "attack" for r in read_jsonl(SPLITS_DIR / "train.jsonl")])
    y_val = [r.label == "attack" for r in read_jsonl(SPLITS_DIR / "val.jsonl")]

    rows, best, best_clf = [], None, None
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        clf = make(**params).fit(x_train, y_train)
        probs = clf.predict_proba(x_val)[:, 1]
        for th in THRESHOLDS:
            s = scores(y_val, [bool(p >= th) for p in probs])
            row = {"params": params, "threshold": th,
                   **{k: s[k] for k in ("recall", "precision", "f1", "fpr")}}
            rows.append(row)
            # f1 неопределён, если детектор не назвал ни одной атаки: такая точка не лучшая
            if best is None or (row["f1"] or 0.0) > (best["f1"] or 0.0):
                best, best_clf = row, clf

    out = RESULTS_DIR / detector_name(model_name, kind)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": best_clf, "threshold": best["threshold"], "params": best["params"],
                 "kind": kind, "model": model_name}, out / "classifier.joblib")
    (out / "tuning.json").write_text(
        json.dumps({"model": model_name, "kind": kind, "criterion": "максимум F1 на val",
                    "best": best, "grid": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--kind", default="logreg", choices=[*KINDS, "all"])
    parser.add_argument("--cache-only", action="store_true", help="не считать эмбеддинги, только из кэша")
    args = parser.parse_args()
    for kind in (KINDS if args.kind == "all" else [args.kind]):
        best = train(args.model, kind, args.cache_only)
        print(f"{kind}: лучшая точка на val:",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})
