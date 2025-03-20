# Amazon Recommendation Engine Makefile

.PHONY: help setup data models api frontend all clean test lint

help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	pip install -r requirements.txt

data: ## Generate sample data
	python scripts/create_sample_data.py

models: ## Train models
	python scripts/simulate_training.py

api: ## Start API server
	python src/api/app.py

frontend: ## Start frontend server
	python src/frontend/serve.py

all: ## Start all services
	python scripts/start_services.py

clean: ## Clean generated files
	@echo "Cleaning generated files..."
	@rm -f models/*.pkl
	@rm -rf data/processed/*
	@echo "Done!"

test: ## Run tests
	pytest tests/

lint: ## Run linting
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	pylint src/ scripts/ tests/ 