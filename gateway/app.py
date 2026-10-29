"""шлюз между письмом и ИИ-ассистентом: проверка письма выбранным детектором

запуск: uv run uvicorn gateway.app:app --port 8000
POST /check принимает тему, тело (текст или HTML) и имя детектора, POST /compare прогоняет письмо через все готовые
детекторы по очереди, GET /detectors показывает готовые детекторы, GET /metrics отдаёт счётчики по журналу решений
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gateway.parse import compose_text, parse_mail
from gateway.registry import Registry

DEFAULT_LOG = Path("results/gateway/decisions.jsonl")
PREVIEW_CHARS = 120


class CheckRequest(BaseModel):
    subject: str = ""
    body: str
    detector: str | None = None  # без значения берётся лучший из готовых


class CompareRequest(BaseModel):
    subject: str = ""
    body: str


class Journal:
    """журнал решений в JSONL"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("mode", "check") == "check"]  # запуски сравнения в счётчики не входят
    latencies = [r["latency_ms"] for r in rows]
    by_detector: dict[str, dict] = {}
    by_stage: dict[str, int] = {}
    for r in rows:
        d = by_detector.setdefault(r["detector"], {"checked": 0, "blocked": 0})
        d["checked"] += 1
        d["blocked"] += r["action"] == "block"
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    return {"checked": len(rows), "blocked": sum(r["action"] == "block" for r in rows),
            "allowed": sum(r["action"] == "allow" for r in rows),
            "hidden_content": sum(r["hidden_found"] for r in rows),
            "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)} if latencies else None,
            "by_detector": by_detector, "by_stage": by_stage}


def create_app(registry: Registry | None = None, log_path: Path | None = None) -> FastAPI:
    registry = registry or Registry()
    journal = Journal(log_path or Path(os.environ.get("GATEWAY_LOG", DEFAULT_LOG)))
    predict_lock = threading.Lock()  # модели не рассчитаны на параллельные вызовы, нагрузка демонстрационная
    app = FastAPI(title="prompt-guard gateway")
    # страница демо открывается с другого адреса
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok", "default_detector": registry.default()}

    @app.get("/detectors")
    def detectors():
        available, unavailable = [], []
        for name in registry.names():
            problem = registry.problem(name)
            (unavailable.append({"name": name, "reason": problem}) if problem else available.append(name))
        return {"default": registry.default(), "available": available, "unavailable": unavailable}

    @app.post("/check")
    def check(req: CheckRequest):
        name = req.detector or registry.default()
        if name is None:
            raise HTTPException(503, "нет ни одного готового детектора")
        if name not in registry.names():
            raise HTTPException(404, f"неизвестный детектор: {name}")
        problem = registry.problem(name)
        if problem:
            raise HTTPException(503, f"детектор {name} не готов: {problem}")
        parsed = parse_mail(req.subject, req.body)
        text = compose_text(req.subject, req.body)
        started = time.perf_counter()
        with predict_lock:
            detector = registry.get(name)
            label, score = detector.predict(text)
            stage = getattr(detector, "last_stage", None) or detector.name
        latency_ms = (time.perf_counter() - started) * 1000
        action = "block" if label == "attack" else "allow"
        journal.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": "check", "detector": name, "label": label,
                        "score": round(score, 4), "action": action, "stage": stage, "latency_ms": round(latency_ms, 1),
                        "hidden_found": parsed.hidden_found, "preview": text[:PREVIEW_CHARS]})
        return {"label": label, "score": score, "action": action, "detector": name, "stage": stage,
                "latency_ms": latency_ms, "hidden_found": parsed.hidden_found, "hidden_text": parsed.hidden_text}

    @app.post("/compare")
    def compare(req: CompareRequest):
        parsed = parse_mail(req.subject, req.body)
        text = compose_text(req.subject, req.body)
        results, unavailable = [], []
        with predict_lock:
            # порядок реестра: судья идёт раньше каскадов, и они берут его ответ из общего кэша без второго запроса к API
            for name in registry.names():
                problem = registry.problem(name)
                if problem:
                    unavailable.append({"detector": name, "reason": problem})
                    continue
                started = time.perf_counter()
                try:
                    detector = registry.get(name)
                    label, score = detector.predict(text)
                except Exception as e:  # сбой одного детектора не должен ломать сравнение
                    results.append({"detector": name, "error": f"{type(e).__name__}: {e}"})
                    continue
                latency_ms = (time.perf_counter() - started) * 1000
                stage = getattr(detector, "last_stage", None) or detector.name
                results.append({"detector": name, "label": label, "score": score, "stage": stage, "latency_ms": latency_ms,
                                "action": "block" if label == "attack" else "allow"})
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in results:
            if "error" not in r:
                journal.append({"ts": ts, "mode": "compare", "detector": r["detector"], "label": r["label"],
                                "score": round(r["score"], 4), "action": r["action"], "stage": r["stage"],
                                "latency_ms": round(r["latency_ms"], 1), "hidden_found": parsed.hidden_found,
                                "preview": text[:PREVIEW_CHARS]})
        answered = [r for r in results if "error" not in r]
        return {"results": results, "unavailable": unavailable, "hidden_found": parsed.hidden_found,
                "hidden_text": parsed.hidden_text,
                "summary": {"attack": sum(r["label"] == "attack" for r in answered),
                            "benign": sum(r["label"] == "benign" for r in answered), "total": len(answered),
                            "errors": len(results) - len(answered)}}

    @app.get("/metrics")
    def metrics():
        return summarize(journal.rows())

    return app


app = create_app()
