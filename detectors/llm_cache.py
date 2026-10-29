"""клиент LLM с дисковым кэшем ответов, расходом токенов и задержкой исходного запроса"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_CACHE_DIR = Path("results/cache")


class CachedLLM:
    def __init__(self, model: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 client=None, min_interval_s: float | None = None):
        load_dotenv()
        # минимальная пауза между реальными запросами к API (кэш не задерживается), из LLM_MIN_INTERVAL_S
        self.min_interval_s = float(os.environ.get("LLM_MIN_INTERVAL_S", 0)) if min_interval_s is None else min_interval_s
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0
        self.model = model or os.environ.get("LLM_MODEL", "")
        if not self.model:
            raise ValueError("не задана модель: аргумент model или LLM_MODEL в .env")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.calls = 0  # число реальных обращений к API
        self.last_usage = None  # токены последнего ответа (и из кэша тоже)
        self.last_latency_ms = None  # задержка исходного запроса, не чтения из кэша

    @property
    def client(self):
        if self._client is None:
            base_url = os.environ.get("ANTHROPIC_BASE_URL")
            key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            if not base_url or not key:
                raise RuntimeError("не заданы ANTHROPIC_BASE_URL и ANTHROPIC_AUTH_TOKEN в .env")
            # openai-клиент сам добавляет /chat/completions, поэтому нужен суффикс /v1
            self._client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=key)
        return self._client

    def _key(self, system: str, user: str, temperature: float | None, max_tokens: int) -> str:
        payload = json.dumps(
            [self.model, system, user, temperature, max_tokens], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _wait_slot(self) -> None:
        """выдерживает паузу между запросами из разных потоков"""
        if self.min_interval_s <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self.min_interval_s
        if wait:
            time.sleep(wait)

    def _remember(self, saved: dict) -> None:
        self.last_usage = saved.get("usage")
        self.last_latency_ms = saved.get("latency_ms")

    def complete(self, user: str, system: str = "", temperature: float | None = None,
                 max_tokens: int = 512) -> str:
        # none - не передавать temperature: haiku 5.5 принимает только значение по умолчанию
        path = self.cache_dir / f"{self._key(system, user, temperature, max_tokens)}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self._remember(saved)
            return saved["response"]
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        params = {} if temperature is None else {"temperature": temperature}
        self._wait_slot()
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, **params)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        saved = {
            "model": self.model, "system": system, "user": user,
            "temperature": temperature, "max_tokens": max_tokens, "response": text,
            "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                      "completion_tokens": getattr(usage, "completion_tokens", None)} if usage else None,
            "latency_ms": round(latency_ms, 1),
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
        }
        path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        self._remember(saved)
        return text
