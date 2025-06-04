---
license: apache-2.0
base_model:
- mistralai/Mistral-7B-Instruct-v0.3
pipeline_tag: translation
tags:
- llm
- devops
- development
- polish
- english
- python
- iac
---

# 🚀 WronAI - End-to-End LLM Toolkit

[![PyPI Version](https://img.shields.io/pypi/v/wronai.svg)](https://pypi.org/project/wronai/)
[![Python Version](https://img.shields.io/pypi/pyversions/wronai.svg)](https://python.org)
[![License](https://img.shields.io/pypi/l/wronai.svg)](https://github.com/wronai/llm-demo/blob/main/LICENSE)
[![Tests](https://github.com/wronai/llm-demo/actions/workflows/tests.yml/badge.svg)](https://github.com/wronai/llm-demo/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://wronai.readthedocs.io/)
[![Docker Pulls](https://img.shields.io/docker/pulls/wronai/wronai)](https://hub.docker.com/r/wronai/wronai)

> A comprehensive toolkit for creating, fine-tuning, and deploying large language models with support for both Polish and English.

## 🌟 Features

- **Ready-to-use WronAI package** - All functionality available through the `wronai` package
- **Model Management** - Easy installation and management of LLM models
- **Multiple Model Support** - Works with various models via Ollama
- **Optimizations** - 4-bit quantization, LoRA, FP16 support
- **CLI Tools** - Command-line interface for all operations
- **Production Ready** - Easy deployment with Docker
- **Web Interface** - User-friendly Streamlit-based web UI

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Install the package
pip install wronai

# Start Ollama (if not already running)
ollama serve &

# Pull the required model (e.g., mistral:7b-instruct)
ollama pull mistral:7b-instruct
```

### Basic Usage

#### Using Python Package

```python
from wronai import WronAI

# Initialize with default settings
wron = WronAI()

# Chat with the model
response = wron.chat("Explain quantum computing in simple terms")
print(response)
```

#### Command Line Interface

```bash
# Start interactive chat
wronai chat

# Run a single query
wronai query "Explain quantum computing in simple terms"
```

#### Web Interface

```bash
# Start the web UI
wronai web
```

## 🔧 Model Management

List available models:
```bash
ollama list
```

Pull a model (if not already available):
```bash
ollama pull mistral:7b-instruct
```

## 🐳 Docker Support

Run with Docker:
```bash
docker run -p 8501:8501 wronai/wronai web
```

## 🛠️ Development

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/wronai/llm-demo.git
cd llm-demo

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=wronai --cov-report=term-missing
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact us at [email protected]
