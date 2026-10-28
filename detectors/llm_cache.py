"""клиент LLM через OpenAI-совместимый шлюз с дисковым кэшем ответов"""
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_CACHE_DIR = Path("results/cache")


class CachedLLM:
    def __init__(self, model: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 client=None):
        load_dotenv()
        self.model = model or os.environ.get("LLM_MODEL", "")
        if not self.model:
            raise ValueError("не задана модель: аргумент model или LLM_MODEL в .env")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.calls = 0  # число реальных обращений к API

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

    def complete(self, user: str, system: str = "", temperature: float | None = None,
                 max_tokens: int = 512) -> str:
        # none - не передавать temperature: haiku 5.5 на шлюзе принимает только значение по умолчанию
        path = self.cache_dir / f"{self._key(system, user, temperature, max_tokens)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        params = {} if temperature is None else {"temperature": temperature}
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, **params)
        self.calls += 1
        text = resp.choices[0].message.content or ""
        path.write_text(json.dumps(
            {"model": self.model, "system": system, "user": user,
             "temperature": temperature, "max_tokens": max_tokens, "response": text},
            ensure_ascii=False), encoding="utf-8")
        return text
