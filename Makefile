.PHONY: help install build up down stop restart logs clean test lint format check-env

# Default target
help:
	@echo "\nLLM Demo - Available commands:\n"
	@echo "  make install         Install Python dependencies"
	@echo "  make build           Build Docker containers"
	@echo "  make up              Start all services in detached mode"
	@echo "  make down            Stop and remove all containers, networks, and volumes"
	@echo "  make stop            Stop all running containers"
	@echo "  make restart         Restart all services"
	@echo "  make logs            Follow container logs"
	@echo "  make logs-ollama     Follow Ollama container logs"
	@echo "  make logs-ui         Follow Streamlit UI logs"
	@echo "  make clean           Remove all containers, networks, and volumes"
	@echo "  make test            Run tests"
	@echo "  make lint            Run linter"
	@echo "  make format          Format code"
	@echo "  make shell-ollama    Open shell in Ollama container"
	@echo "  make shell-ui        Open shell in Streamlit UI container"

# Check if .env file exists
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Please create one from .env.example"; \
		exit 1; \
	fi

# Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt

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

# Stop all running containers
stop:
	@echo "Stopping all containers..."
	docker-compose stop

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

# Remove all containers, networks, and volumes
clean: down
	@echo "Removing all unused containers, networks, and volumes..."
	docker system prune -f
	docker volume prune -f

# Run tests
test:
	@echo "Running tests..."
	# Add your test command here
	# Example: python -m pytest tests/

# Lint code
lint:
	@echo "Running linter..."
	# Add your lint command here
	# Example: pylint app/

# Format code
format:
	@echo "Formatting code..."
	# Add your format command here
	# Example: black app/

# Open shell in Ollama container
shell-ollama:
	docker-compose exec ollama /bin/sh

# Open shell in Streamlit UI container
shell-ui:
	docker-compose exec streamlit-ui /bin/sh
