"""реестр детекторов шлюза: какие из них готовы работать на новых письмах и ленивая загрузка

готовность проверяется по наличию файлов, модели при этом не загружаются; детектор строится при первом обращении
"""
import os
import threading
from functools import partial
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import try_to_load_from_cache

from detectors.base import Detector
from detectors.embeddings_cache import Encoder
from detectors.implementations.cascade import THRESHOLDS_DIR, CascadeDetector, load_thresholds
from detectors.implementations.embeddings import KINDS, EmbeddingDetector, detector_name
from detectors.implementations.judge import JudgeDetector
from detectors.llm_cache import CachedLLM
from detectors.implementations.keyword import KeywordDetector
from detectors.implementations.lora import LoraDetector
from detectors.implementations.lora_mdeberta import LoraMdebertaDetector
from detectors.implementations.tfidf import RESULTS_DIR as TFIDF_DIR, TfidfDetector

RESULTS_DIR = Path("results")
EMBEDDING_MODELS = ("intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large",
                    "BAAI/bge-m3")
LORA_DETECTORS = (LoraDetector, LoraMdebertaDetector)
CASCADE_FIRST_STAGES = tuple(detector_name(m, k) for m in EMBEDDING_MODELS for k in KINDS) + (
    "tfidf-logreg", "lora-xlmr", "lora-mdeberta")
# порядок предпочтения детектора по умолчанию: лучший из готовых
DEFAULT_ORDER = ("cascade-lora-mdeberta", "cascade-lora-xlmr", "cascade-embeddings-e5-large",
                 "cascade-embeddings-e5-base-mlp", "lora-mdeberta", "lora-xlmr", "embeddings-e5-base-mlp",
                 "embeddings-e5-large", "tfidf-logreg", "keyword")


class MemoEncoder(Encoder):
    """кодировщик с памятью: письмо кодируется один раз, хотя у модели три классификатора"""
    MEMO_LIMIT = 256

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._memo: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if missing:
            if len(self._memo) + len(missing) > self.MEMO_LIMIT:
                self._memo.clear()
            self._memo.update(zip(missing, super().encode(missing, batch_size)))
        return np.stack([self._memo[t] for t in texts])


def model_cached(repo_id: str) -> bool:
    """есть ли веса модели в кэше huggingface"""
    return any(isinstance(try_to_load_from_cache(repo_id, f), str) for f in ("model.safetensors", "pytorch_model.bin"))


def _embedding_problem(model: str, kind: str) -> str | None:
    if not (RESULTS_DIR / detector_name(model, kind) / "classifier.joblib").exists():
        return "нет обученного классификатора"
    return None if model_cached(model) else f"нет модели {model} в кэше"


def _lora_problem(cls: type[LoraDetector]) -> str | None:
    adapter = cls.results_dir / "adapter"
    if not (adapter / "adapter_config.json").exists() or not (adapter / "adapter_model.safetensors").exists():
        return "нет адаптера"
    return None if model_cached(cls.base_model) else f"нет базы {cls.base_model} в кэше"


def _tfidf_problem() -> str | None:
    return None if (TFIDF_DIR / "classifier.joblib").exists() else "нет обученного классификатора"


def _judge_problem() -> str | None:
    load_dotenv()
    if not (os.environ.get("ANTHROPIC_BASE_URL") and os.environ.get("ANTHROPIC_AUTH_TOKEN") and os.environ.get("LLM_MODEL")):
        return "не заданы ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN и LLM_MODEL в .env"
    return None


def _judge() -> JudgeDetector:
    """судья для шлюза: ответ дольше JUDGE_TIMEOUT_S секунд считается таймаутом"""
    return JudgeDetector(llm=CachedLLM(timeout_s=float(os.environ.get("JUDGE_TIMEOUT_S", 20))))


class Registry:
    def __init__(self):
        self._problems: dict[str, callable] = {"keyword": lambda: None, "judge-haiku": _judge_problem,
                                               "tfidf-logreg": _tfidf_problem}
        self._factories: dict[str, callable] = {"keyword": KeywordDetector, "judge-haiku": _judge,
                                                "tfidf-logreg": TfidfDetector}
        for cls in LORA_DETECTORS:
            self._problems[cls.name] = partial(_lora_problem, cls)
            self._factories[cls.name] = cls
        for model in EMBEDDING_MODELS:
            for kind in KINDS:
                name = detector_name(model, kind)
                self._problems[name] = partial(_embedding_problem, model, kind)
                self._factories[name] = partial(self._embedding, model, kind)
        for first in CASCADE_FIRST_STAGES:
            name = "cascade-" + first
            self._problems[name] = partial(self._cascade_problem, first)
            self._factories[name] = partial(self._build_cascade, first)
        self._instances: dict[str, Detector] = {}
        self._encoders: dict[str, MemoEncoder] = {}
        self._lock = threading.RLock()  # сборка каскада вызывает get для ступеней

    def _embedding(self, model: str, kind: str) -> EmbeddingDetector:
        encoder = self._encoders.setdefault(model, MemoEncoder(model))
        return EmbeddingDetector(model, kind, False, encoder)

    def _cascade_problem(self, first: str) -> str | None:
        if not (THRESHOLDS_DIR / first / "thresholds.json").exists():
            return "не подобраны пороги каскада"
        return self._problems[first]() or _judge_problem()

    def _build_cascade(self, first: str) -> CascadeDetector:
        cfg = load_thresholds(first)
        return CascadeDetector(first=self.get(first), judge=self.get("judge-haiku"), lo=cfg["lo"], hi=cfg["hi"])

    def names(self) -> list[str]:
        return list(self._factories)

    def problem(self, name: str) -> str | None:
        """почему детектор не готов, или None, если готов"""
        return self._problems[name]()

    def available(self) -> list[str]:
        return [n for n in self.names() if self.problem(n) is None]

    def default(self) -> str | None:
        ready = set(self.available())
        return next((n for n in DEFAULT_ORDER if n in ready), None)

    def get(self, name: str) -> Detector:
        with self._lock:
            if name not in self._instances:
                self._instances[name] = self._factories[name]()
            return self._instances[name]
