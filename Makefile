.DEFAULT_GOAL := help
.PHONY: help install test gateway target build up down logs plots

help: ## список команд
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

install: ## поставить зависимости
	@uv sync

test: ## запустить тесты
	@uv run pytest -q

gateway: ## шлюз локально на порту 8000
	@uv run uvicorn gateway.app:app --port 8000

target: ## мишень локально на порту 8001
	@uv run uvicorn target.app:app --port 8001

build: ## собрать образ (слои из кэша)
	@docker compose build

up: ## поднять шлюз и мишень в Docker
	@docker compose up -d

down: ## остановить сервисы
	@docker compose down

logs: ## логи сервисов
	@docker compose logs -f
