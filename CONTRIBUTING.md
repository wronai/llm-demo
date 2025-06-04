# Contributing to WronAI

Thank you for considering contributing to WronAI! We appreciate your time and effort in helping us improve this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Pull Request Process](#-pull-request-process)
- [Reporting Issues](#-reporting-issues)
- [Feature Requests](#-feature-requests)
- [License](#-license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/your-username/llm-demo.git
   cd llm-demo
   ```
3. **Set up** the development environment
   ```bash
   make venv
   make install-dev
   ```
4. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔄 Development Workflow

1. Make your changes following the code style guidelines
2. Run tests to ensure nothing is broken
   ```bash
   make test
   ```
3. Run linters and type checkers
   ```bash
   make lint
   make typecheck
   ```
4. Format your code
   ```bash
   make format
   ```
5. Commit your changes with a descriptive message
   ```bash
   git commit -m "feat: add new feature"
   ```
6. Push to your fork and submit a pull request

## 🎨 Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Keep functions small and focused on a single responsibility
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

## 🧪 Testing

- Write tests for all new features and bug fixes
- Run all tests before submitting a pull request
  ```bash
  make test
  ```
- Aim for good test coverage (80%+)
- Test edge cases and error conditions

## 📚 Documentation

- Update documentation for any new features or changes
- Keep docstrings up to date
- Add examples for new functionality
- Document any breaking changes

## 🔄 Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters
3. Increase the version number in `pyproject.toml` and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/)
4. You may merge the Pull Request once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you

## 🐛 Reporting Issues

When reporting issues, please include:

- A clear title and description
- Steps to reproduce the issue
- Expected vs. actual behavior
- Any relevant error messages or logs
- Version information (Python, package versions, etc.)

## 💡 Feature Requests

We welcome feature requests! Please open an issue to discuss your idea before implementing it to ensure it aligns with the project's goals.

## 📄 License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
