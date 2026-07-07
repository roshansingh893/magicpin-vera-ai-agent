# Contributing to Vera AI Agent

Thank you for your interest in contributing to Vera AI Agent! This document provides guidelines and instructions for contributing.

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip
- A Groq API key ([get one here](https://console.groq.com))

### Local Setup

```bash
cd vera-agent

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux

# Edit .env and add your GROQ_API_KEY
```

### Running the Server

```bash
uvicorn app.main:app --reload
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Development Guidelines

### Code Style

- Follow **PEP 8** conventions.
- Use **type hints** on all function signatures.
- Write **docstrings** for all public functions and classes.
- Keep modules focused — one responsibility per file.

### Architecture Rules

- **API routes** are thin handlers — no business logic in `app/api/`.
- **Business logic** lives in `app/services/`.
- **LLM integration** is isolated in `app/llm/`.
- **Prompts** are templates in `app/prompts/` — never hardcode prompt strings elsewhere.
- **Evaluation** is fully separated from the production API in `evaluation/`.

### Groq Free Tier

This project uses the Groq Free Tier with strict rate limits:

- **Never** make unnecessary API calls.
- **Always** check the response cache before calling the LLM.
- **Never** build brute-force prompt comparison loops.
- Use `--dry-run` to validate changes before running batch operations.

---

## Pull Request Process

1. **Fork** the repository and create a feature branch.
2. **Write tests** for any new functionality.
3. **Run the full test suite** before submitting: `pytest tests/ -v`
4. **Run the release verification**: `python scripts/verify_release.py --skip-api`
5. **Update documentation** if your changes affect the API or configuration.
6. **Submit a PR** with a clear description of what changed and why.

---

## Reporting Issues

When filing an issue, please include:

- Python version (`python --version`)
- Operating system
- Steps to reproduce the issue
- Expected vs. actual behavior
- Relevant log output

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
