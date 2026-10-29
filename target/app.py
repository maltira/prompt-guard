"""ассистент-мишень: POST /summarize пересказывает письмо

MODE=direct: письмо сразу уходит ассистенту; MODE=gateway: сначала проверка шлюзом, опасное письмо ассистент не получает
режим можно переопределить в запросе; если шлюз недоступен, письмо не пересылается (закрыто по умолчанию)
запуск: uv run uvicorn target.app:app --port 8001
"""
import os
import time
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detectors.llm_cache import CachedLLM
from target.assistant import DEFAULT_SECRET, MAX_TOKENS, PERSONAS, build_system_prompt, build_user_message, leaked
from target.gateway_client import GatewayClient, GatewayError
from target.local_llm import DEFAULT_MODEL, LocalLLM, model_available

Mode = Literal["direct", "gateway"]
Persona = Literal["default", "naive"]
Backend = Literal["api", "local"]


class SummarizeRequest(BaseModel):
    subject: str = ""
    body: str
    mode: Mode | None = None  # без значения берётся MODE из окружения
    detector: str | None = None  # детектор шлюза, только для режима gateway
    backend: Backend | None = None  # api: Haiku через API, local: слабая локальная модель; без значения берётся TARGET_BACKEND
    persona: Persona | None = None  # naive: намеренно уязвимая конфигурация для демонстрации; без значения берётся TARGET_PERSONA


def create_app(llm=None, gateway=None, secret: str | None = None, default_mode: Mode | None = None,
               default_persona: Persona | None = None, local_llm=None, default_backend: Backend | None = None,
               local_available=model_available) -> FastAPI:
    load_dotenv()
    secret = secret or os.environ.get("TARGET_SECRET", DEFAULT_SECRET)
    default_mode = default_mode or os.environ.get("MODE", "direct")
    if default_mode not in ("direct", "gateway"):
        raise ValueError(f"MODE должен быть direct или gateway, получено {default_mode!r}")
    default_persona = default_persona or os.environ.get("TARGET_PERSONA", "default")
    if default_persona not in PERSONAS:
        raise ValueError(f"TARGET_PERSONA должен быть одним из {PERSONAS}, получено {default_persona!r}")
    default_backend = default_backend or os.environ.get("TARGET_BACKEND", "api")
    if default_backend not in ("api", "local"):
        raise ValueError(f"TARGET_BACKEND должен быть api или local, получено {default_backend!r}")
    local_model = os.environ.get("TARGET_LOCAL_MODEL", DEFAULT_MODEL)
    gateway = gateway or GatewayClient(os.environ.get("GATEWAY_URL", "http://localhost:8000"))
    systems = {persona: build_system_prompt(secret, persona) for persona in PERSONAS}
    state = {"api": llm, "local": local_llm}  # клиенты создаются при первом обращении: API нужны ключи из .env, локальной модели веса
    app = FastAPI(title="prompt-guard target")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": default_mode, "persona": default_persona, "backend": default_backend,
                "local_model": local_model, "local_ready": state["local"] is not None or local_available(local_model)}

    @app.post("/summarize")
    def summarize(req: SummarizeRequest):
        mode = req.mode or default_mode
        persona = req.persona or default_persona
        backend = req.backend or default_backend
        started = time.perf_counter()
        verdict = None
        if mode == "gateway":
            try:
                verdict = gateway.check(req.subject, req.body, req.detector)
            except GatewayError as e:
                raise HTTPException(502, str(e))
            if verdict["action"] == "block":
                return {"mode": mode, "persona": persona, "backend": backend, "blocked": True, "reply": None, "leaked": False, "gateway": verdict,
                        "latency_ms": (time.perf_counter() - started) * 1000}
        if state[backend] is None:
            if backend == "local":
                if not local_available(local_model):
                    raise HTTPException(503, f"модель {local_model} не скачана: слабый ассистент недоступен")
                state["local"] = LocalLLM(local_model)
            else:
                state["api"] = CachedLLM()
        try:
            reply = state[backend].complete(build_user_message(req.subject, req.body), system=systems[persona], max_tokens=MAX_TOKENS)
        except Exception as e:
            raise HTTPException(502, f"ассистент недоступен: {type(e).__name__}: {e}")
        return {"mode": mode, "persona": persona, "backend": backend, "blocked": False, "reply": reply, "leaked": leaked(reply, secret), "gateway": verdict,
                "latency_ms": (time.perf_counter() - started) * 1000}

    return app


app = create_app()
