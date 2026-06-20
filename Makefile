.PHONY: install lint format type-check run test dashboard docker-build clean

install:
	uv sync && uv sync --extra dev

lint:
	uv run ruff check src
	uv run ruff format --check src

format:
	uv run ruff check --fix src
	uv run ruff format src

type-check:
	uv run mypy src

check: format lint type-check

run:
	uv run python -m src.main

test:
	uv run pytest tests -v -m "not smoke"

dashboard:
	uv run streamlit run src/streamlit_app.py

docker-build:
	docker build -t energy-forecast:latest .

docker-build-streamlit:
	docker build -t demand-forecast-app:v1 .

acr-build:
	@echo "Usage: az acr build --registry <your-registry> --image demand-forecast-app:v1 ."
	@echo "Run these steps first:"
	@echo "  1. az login"
	@echo "  2. az group create --name energy-forecast-rg --location southeastasia"
	@echo "  3. az acr create --resource-group energy-forecast-rg --name <your-acr-name> --sku Basic --admin-enabled true"
	@echo '  4. az acr build --registry <your-acr-name> --image demand-forecast-app:v1 .'

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
