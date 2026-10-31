"""классический бейзлайн: TF-IDF по словам и символьным n-граммам + логистическая регрессия, без нейросетей

обучение: uv run python -m detectors.implementations.tfidf
классификатор обучается на train, параметр C и порог подбираются на val, golden не используется
"""
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from data.schema import read_jsonl
from detectors.base import Detector
from eval.metrics import scores

NAME = "tfidf-logreg"
RESULTS_DIR = Path("results") / NAME
SPLITS_DIR = Path("data/splits")
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def make_pipeline(c: float) -> Pipeline:
    """слова и биграммы ловят шаблонные фразы, символьные n-граммы устойчивее к опечаткам, смешению языков и разметке"""
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000)),
    ])
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(C=c, max_iter=2000))])


class TfidfDetector(Detector):
    name = NAME
    cost_per_1000 = 0.0  # модель работает локально
    latency_note = None

    def __init__(self):
        path = RESULTS_DIR / "classifier.joblib"
        if not path.exists():
            raise SystemExit(f"нет обученного классификатора {path}, сначала запустите обучение: uv run python -m detectors.implementations.tfidf")
        artifact = joblib.load(path)
        self.pipe, self.threshold = artifact["pipe"], artifact["threshold"]

    def _label(self, score: float) -> str:
        return "attack" if score >= self.threshold else "benign"

    def predict(self, text: str) -> tuple[str, float]:
        score = float(self.pipe.predict_proba([text])[0, 1])
        return self._label(score), score

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [(self._label(float(p)), float(p)) for p in self.pipe.predict_proba(texts)[:, 1]]


def train() -> dict:
    train_recs = read_jsonl(SPLITS_DIR / "train.jsonl")
    val_recs = read_jsonl(SPLITS_DIR / "val.jsonl")
    x_train, y_train = [r.text for r in train_recs], [r.label == "attack" for r in train_recs]
    x_val, y_val = [r.text for r in val_recs], [r.label == "attack" for r in val_recs]

    rows, best, best_pipe = [], None, None
    for c in C_GRID:
        pipe = make_pipeline(c).fit(x_train, y_train)
        probs = pipe.predict_proba(x_val)[:, 1]
        for th in THRESHOLDS:
            s = scores(y_val, [bool(p >= th) for p in probs])
            row = {"C": c, "threshold": th, **{k: s[k] for k in ("recall", "precision", "f1", "fpr")}}
            rows.append(row)
            # f1 неопределён, если детектор не назвал ни одной атаки: такая точка не лучшая
            if best is None or (row["f1"] or 0.0) > (best["f1"] or 0.0):
                best, best_pipe = row, pipe

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": best_pipe, "threshold": best["threshold"], "params": {"C": best["C"]}}, RESULTS_DIR / "classifier.joblib")
    (RESULTS_DIR / "tuning.json").write_text(
        json.dumps({"model": "tfidf (слова 1-2 + символы 2-5) + логрегрессия", "criterion": "максимум F1 на val",
                    "best": best, "grid": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


if __name__ == "__main__":
    best = train()
    print("лучшая точка на val:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})
