"""подход 5: каскад - быстрая первая ступень, неуверенные тексты отдаются судье

p < lo - безопасно, p > hi - атака, иначе решает судья
"""
import json
import time
from pathlib import Path

from detectors.base import Detector

THRESHOLDS_DIR = Path("results/cascade")  # пороги лежат в results/cascade/<имя первой ступени>/thresholds.json


class CascadeDetector(Detector):
    name = "cascade"

    def __init__(self, first: Detector, judge: Detector, lo: float, hi: float):
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"нужно 0 <= lo <= hi <= 1, получено lo={lo}, hi={hi}")
        self.first, self.judge, self.lo, self.hi = first, judge, lo, hi
        self.n_requests = self.n_judged = 0
        self.last_latency_ms = None
        self.latency_note = "задержка первой ступени замерена вживую, задержка судьи взята из исходных запросов"

    def predict(self, text: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        _, p = self.first.predict(text)
        latency = (time.perf_counter() - t0) * 1000
        self.n_requests += 1
        if p < self.lo:
            self.last_latency_ms, result = latency, ("benign", p)
        elif p > self.hi:
            self.last_latency_ms, result = latency, ("attack", p)
        else:
            self.n_judged += 1
            result = self.judge.predict(text)
            self.last_latency_ms = latency + (self.judge.last_latency_ms or 0.0)
        return result

    @property
    def judged_share(self) -> float | None:
        return self.n_judged / self.n_requests if self.n_requests else None

    @property
    def cost_per_1000(self) -> float | None:
        """цена на 1000 запросов каскада: цена судьи на 1000 запросов, умноженная на долю запросов к нему"""
        judge_cost = getattr(self.judge, "cost_per_1000", None)
        if judge_cost is None or self.judged_share is None:
            return None
        return judge_cost * self.judged_share


def load_thresholds(first_stage: str) -> dict:
    path = THRESHOLDS_DIR / first_stage / "thresholds.json"
    if not path.exists():
        raise SystemExit(f"нет {path}: сначала подберите пороги на val (python -m eval.cascade_sweep --first-stage {first_stage})")
    return json.loads(path.read_text(encoding="utf-8"))
