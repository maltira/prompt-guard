# общий образ для шлюза и мишени: код внутри, результаты и веса моделей монтируются снаружи
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

# обычный пользователь с uid 1000, чтобы файлы в смонтированном results/ принадлежали владельцу репозитория
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# зависимости отдельным слоем и с кэшем загрузок uv: при правке кода этот слой берётся из кэша, при смене uv.lock пересобирается
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# код копируется последним, сразу с нужным владельцем (отдельный chown раздул бы образ копией всех файлов)
COPY --chown=app:app . .
USER app

ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/app HF_HOME=/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
