"""слабый локальный ассистент: небольшая открытая модель на процессоре, без обращений к API

генерация жадная, поэтому ответы воспроизводятся; результаты кэшируются на диск, как у клиента API
"""
import hashlib
import json
import threading
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CACHE_DIR = Path("results/cache/local")
MAX_NEW_TOKENS = 150  # пересказ короткий, длинный вывод на процессоре только замедлил бы ответ


def model_available(name: str) -> bool:
    """есть ли веса модели в кэше huggingface (скачивание во время запроса не запускается)"""
    return any(isinstance(try_to_load_from_cache(name, f), str) for f in ("model.safetensors", "pytorch_model.bin"))


class LocalLLM:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_name = model
        self.cache_dir = Path(cache_dir)
        self.max_new_tokens = max_new_tokens
        self.calls = 0  # число реальных генераций
        self._tok = self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.float32, local_files_only=True).eval()

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str:
        import torch

        if self._model is None:
            self._load()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        ids = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def _path(self, system: str, user: str, max_new_tokens: int) -> Path:
        key = hashlib.sha256(json.dumps([self.model_name, system, user, max_new_tokens], ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def complete(self, user: str, system: str = "", temperature: float | None = None, max_tokens: int = 512) -> str:
        """тот же интерфейс, что у клиента API; temperature не используется, генерация жадная"""
        new_tokens = min(max_tokens, self.max_new_tokens)
        path = self._path(system, user, new_tokens)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        with self._lock:
            reply = self._generate(system, user, new_tokens)
        self.calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model_name, "system": system, "user": user, "max_new_tokens": new_tokens,
                                    "response": reply}, ensure_ascii=False), encoding="utf-8")
        return reply
