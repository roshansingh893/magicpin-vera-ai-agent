# Vera AI Agent

An AI-powered merchant assistant for the [magicpin AI Challenge](../challenge-brief.md). Composes context-aware WhatsApp messages for merchants and their customers, modeled on magicpin's production assistant **Vera**.

---

## Architecture

```
vera-agent/
│
├── app/
│   ├── main.py              # FastAPI application factory + uvicorn entrypoint
│   ├── api/
│   │   └── routes.py         # All /v1 endpoint definitions
│   ├── core/
│   │   ├── config.py          # Settings loaded from .env via pydantic-settings
│   │   ├── logging.py         # Centralized logging configuration
│   │   └── exceptions.py      # Custom exceptions + global error handlers
│   ├── models/
│   │   ├── requests.py        # Pydantic request schemas (4-context framework)
│   │   └── responses.py       # Pydantic response schemas (ComposedMessage)
│   ├── services/
│   │   └── composer.py        # Message composition engine (Phase 2)
│   ├── prompts/               # Prompt templates (Phase 2)
│   └── llm/                   # LLM client abstractions (Phase 2)
│
├── tests/
│   └── test_endpoints.py      # pytest suite for all API endpoints
│
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Design Principles

- **Separation of Concerns** — API routes are thin handlers; business logic lives in `services/`; LLM integration is isolated in `llm/`.
- **Schema-first** — All request/response bodies are Pydantic v2 models with type hints and validation.
- **Config from environment** — No hardcoded secrets. Settings are loaded via `pydantic-settings` + `python-dotenv`.
- **Structured logging** — Every endpoint logs request in → validation → response out.
- **Fail-safe error handling** — Custom exception hierarchy with consistent JSON error responses; stack traces never leak to clients.

---

## Quick Start

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
cd vera-agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

### Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `GET` | `/v1/healthz` | Liveness probe | ✅ Implemented |
| `GET` | `/v1/metadata` | Bot name, version, description | ✅ Implemented |
| `POST` | `/v1/context` | Compose a message from the 4-context framework | ✅ Complete (Groq API integrated) |
| `POST` | `/v1/reply` | Handle merchant reply in multi-turn conversation | ✅ Validates input (logic in Phase 2) |
| `POST` | `/v1/tick` | Scheduled cadence check for proactive outreach | ✅ Validates input (logic in Phase 2) |

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Backend foundation — project structure, API, models, config, tests | ✅ Complete |
| **Phase 2** | Message composition — prompt engineering, LLM integration, structured output | ✅ Complete |
| **Phase 3** | Multi-turn conversation handling — auto-reply detection, intent routing | ✅ Complete |
| **Phase 4** | Evaluation — generate `submission.jsonl` for the 30 test pairs | 🔲 Next |
