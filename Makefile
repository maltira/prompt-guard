.DEFAULT_GOAL := help
.PHONY: help install test gateway build up down restart logs plots

help: ## список команд
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

install: ## поставить зависимости
	@uv sync

test: ## запустить тесты
	@uv run pytest -q

gateway: ## бэкенд (шлюз) локально на порту 8000
	@uv run uvicorn --app-dir services/backend gateway.app:app --port 8000

build: ## собрать образ (слои из кэша)
	@docker compose build

up: ## поднять сервисы в Docker (страница и бэкенд за nginx)
	@docker compose up -d

down: ## остановить сервисы
	@docker compose down

restart: ## пересобрать и перезапустить сервисы
	@docker compose up -d --build
	@docker compose restart nginx

logs: ## логи сервисов
	@docker compose logs -f
