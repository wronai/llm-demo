.PHONY: help install build up down stop restart logs clean test lint format typecheck check-env \
	model-install model-create model-run model-push model-pull model-list \
	venv venv-clean deps deps-update install-dev install-ci \
	open open-ui open-ollama open-docs \
	publish test-publish docs coverage

# Default target
help:
	@echo "\n🚀 WronAI - Available commands:\n"
	@echo "  Environment:"
	@echo "    make venv           Create Python virtual environment"
	@echo "    make deps           Install Python dependencies"
	@echo "    make deps-update    Update all Python dependencies"
	@echo "    make venv-clean     Remove virtual environment"
	@echo "    make install-dev    Install package in development mode"
	@echo "    make install-ci     Install package for CI environment"

	@echo "\n  Docker & Services:"
	@echo "    make build          Build Docker containers"
	@echo "    make up             Start all services in detached mode"
	@echo "    make down           Stop and remove all containers, networks, and volumes"
	@echo "    make stop           Stop all running containers"
	@echo "    make restart        Restart all services"
	@echo "    make logs           Follow container logs"
	@echo "    make logs-ollama    Follow Ollama container logs"
	@echo "    make logs-ui        Follow Streamlit UI logs"
	@echo "    make clean          Remove all containers, networks, and volumes"

	@echo "\n  Model Management:"
	@echo "    make model-install  Install model dependencies"
	@echo "    make model-create   Create Ollama model from Modelfile"
	@echo "    make model-run      Run the WronAI model"
	@echo "    make model-push     Push model to Ollama registry"
	@echo "    make model-pull     Pull model from Ollama registry"
	@echo "    make model-list     List available models"

	@echo "\n  Development:"
	@echo "    make test           Run tests with coverage"
	@echo "    make test-fast      Run tests quickly without coverage"
	@echo "    make lint           Run linters (flake8, black, isort)"
	@echo "    make format         Format code with black and isort"
	@echo "    make typecheck      Run static type checking with mypy"
	@echo "    make coverage       Generate and open coverage report"
	@echo "    make docs           Generate documentation"

	@echo "\n  Package Publishing:"
	@echo "    make build-pkg      Build source and wheel package"
	@echo "    make test-publish   Upload package to test PyPI"
	@echo "    make publish        Upload package to PyPI"

	@echo "\n  Shell Access:"
	@echo "    make shell-ollama   Open shell in Ollama container"
	@echo "    make shell-ui       Open shell in Streamlit UI container"

	@echo "\n  Open in Browser:"
	@echo "    make open           Open all services in browser"
	@echo "    make open-ui        Open Streamlit UI in browser"
	@echo "    make open-ollama    Open Ollama API in browser"
	@echo "    make open-docs      Open documentation in browser"
	@echo "    make open-coverage  Open coverage report in browser"

# Check if .env file exists
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Creating from .env.example..."; \
		cp -n .env.example .env || true; \
	fi

# Check if Docker is running
check-docker:
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker is not running. Please start Docker and try again."; \
		exit 1; \
	fi

# Virtual Environment
venv:
	@echo "Creating Python virtual environment..."
	python -m venv .venv
	@echo "\nTo activate virtual environment, run:"
	@echo "  source .venv/bin/activate  # Linux/Mac"
	@echo "  .venv\\Scripts\\activate   # Windows"

venv-clean:
	@echo "Removing virtual environment..."
	rm -rf .venv

# Dependencies
deps: venv
	@echo "Installing Python dependencies..."
	. .venv/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt -r model_requirements.txt

deps-update: venv
	@echo "Updating Python dependencies..."
	. .venv/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -U -r requirements.txt -r model_requirements.txt

# Install package in development mode
install-dev: deps
	@echo "Installing package in development mode..."
	. .venv/bin/activate && \
	pip install -e ".[dev]" && \
	if ! command -v pre-commit &> /dev/null; then \
		echo "Installing pre-commit..."; \
		pip install pre-commit; \
	fi && \
	pre-commit install

# Install for CI environment
install-ci:
	@echo "Installing for CI environment..."
	pip install --upgrade pip setuptools wheel && \
	pip install -e .[dev] && \
	pip install pytest-cov

# Build Docker containers
build: check-env
	@echo "Building Docker containers..."
	docker-compose build

# Start all services in detached mode
up: check-env
	@echo "Starting all services..."
	docker-compose up -d

# Stop and remove all containers, networks, and volumes
down:
	@echo "Stopping and removing all containers..."
	docker-compose down -v

# Stop and remove all containers, networks, and images
stop:
	@echo "Stopping and removing all containers, networks, and images..."
	docker-compose down --rmi all --volumes --remove-orphans
	@echo "Removing unused Docker resources..."
	docker system prune -a -f --volumes
	@echo "Removing all unused Docker networks..."
	docker network prune -f
	@echo "Removing all unused Docker volumes..."
	docker volume prune -f

# Restart all services
restart: stop up

# Follow container logs
logs:
	docker-compose logs -f

# Follow Ollama container logs
logs-ollama:
	docker-compose logs -f ollama

# Follow Streamlit UI logs
logs-ui:
	docker-compose logs -f streamlit-ui

# Alias for stop (for backward compatibility)
clean: stop

# Model Management
model-install:
	@echo "Installing model dependencies..."
	@if [ ! -f "model_requirements.txt" ]; then \
		echo "model_requirements.txt not found. Creating with default dependencies..."; \
		echo "torch>=2.0.0" > model_requirements.txt; \
		echo "transformers>=4.35.0" >> model_requirements.txt; \
		echo "peft>=0.7.0" >> model_requirements.txt; \
	fi
	. .venv/bin/activate && \
	pip install -r model_requirements.txt

model-create-simple:
	@echo "Creating Ollama model from simplified Modelfile..."
	ollama create wronai -f Modelfile.simple

model-create-full:
	@echo "Creating Ollama model from full Modelfile..."
	@if [ ! -f "my_custom_model.gguf" ]; then \
		echo "Error: my_custom_model.gguf not found. Please convert your model first."; \
		exit 1; \
	fi
	ollama create wronai -f Modelfile

# Alias for backward compatibility
model-create: model-create-simple

model-run:
	@echo "Running WronAI model..."
	ollama run wronai

model-push:
	@echo "Pushing model to Ollama registry..."
	ollama push wronai

model-pull:
	@echo "Pulling latest model..."
	ollama pull wronai

model-list:
	@echo "Available models:"
	ollama list

# Development
test: install-dev
	@echo "Running tests with coverage..."
	@mkdir -p tests  # Ensure tests directory exists
	. .venv/bin/activate && \
	pytest --cov=wronai --cov-report=term-missing -v tests/ || (echo "\n💡 No tests found. Consider adding tests to the tests/ directory." && exit 1)

test-fast: install-dev
	@echo "Running tests quickly..."
	. .venv/bin/activate && \
	pytest -v tests/

lint: install-dev
	@echo "Running linters..."
	. .venv/bin/activate && \
	echo "\n=== black ===\n" && \
	black --check --diff . && \
	echo "\n=== isort ===\n" && \
	isort --check-only --diff . && \
	echo "\n=== flake8 ===\n" && \
	flake8 wronai/ tests/ && \
	echo "\n✅ All checks passed!"

format: install-dev
	@echo "Formatting code..."
	. .venv/bin/activate && \
	black . && \
	isort .

typecheck: install-dev
	@echo "Running type checking..."
	. .venv/bin/activate && \
	mypy wronai/

coverage: test
	@echo "Generating coverage report..."
	. .venv/bin/activate && \
	coverage html && \
	python -m webbrowser -t "htmlcov/index.html"

docs: install-dev
	@echo "Generating documentation..."
	. .venv/bin/activate && \
	cd docs && make html

# Package building and publishing
build-pkg: clean
	@echo "Building source and wheel package..."
	. .venv/bin/activate && \
	python -m build

publish: build-pkg
	@echo "Uploading package to PyPI..."
	. .venv/bin/activate && \
	twine upload dist/*

test-publish: build-pkg
	@echo "Uploading package to TestPyPI..."
	. .venv/bin/activate && \
	twine upload --repository testpypi dist/*

# Open shell in Ollama container
shell-ollama:
	docker-compose exec ollama /bin/sh

# Open shell in Streamlit UI container
shell-ui:
	docker-compose exec streamlit-ui /bin/sh

# Open all services in browser
open: open-ui open-ollama

# Open Streamlit UI in browser
open-ui:
	@echo "Opening Streamlit UI..."
	@xdg-open http://localhost:8501 2>/dev/null || open http://localhost:8501 2>/dev/null || start http://localhost:8501 2>/dev/null || echo "Could not open the browser. Please open http://localhost:8501 manually"

# Open Ollama API in browser
open-ollama:
	@echo "Opening Ollama API..."
	@xdg-open http://localhost:11436 2>/dev/null || open http://localhost:11436 2>/dev/null || start http://localhost:11436 2>/dev/null || echo "Could not open the browser. Please open http://localhost:11436 manually"

# Open documentation in browser
open-docs:
	@echo "Opening documentation..."
	@xdg-open https://github.com/wronai/llm-demo#readme 2>/dev/null || open https://github.com/wronai/llm-demo#readme 2>/dev/null || start https://github.com/wronai/llm-demo#readme 2>/dev/null || echo "Could not open the browser. Please open https://github.com/wronai/llm-demo#readme manually"
