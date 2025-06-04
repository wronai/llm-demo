# WronAI

A Python package for AI model management and deployment.

## Installation

```bash
# Install with pip from local source
pip install -e .
```

## Usage

```bash
# Show version
wronai --version

# Show help
wronai --help

# Manage models
wronai model pull <model_name>
wronai model push <model_name>
```

## Development

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest
   ```

## License

Apache 2.0
