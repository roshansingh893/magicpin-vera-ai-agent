# 🤖 Vera AI Agent

> AI-powered merchant engagement assistant for the [magicpin AI Challenge](../challenge-brief.md).  
> Composes context-aware WhatsApp messages for merchants and their customers using Groq's LLM API.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Project Overview

Vera is magicpin's intelligent merchant assistant that drives engagement through personalized WhatsApp messaging. This project implements a production-grade AI agent that:

- **Composes** context-aware messages using a 4-context framework (Category → Merchant → Trigger → Customer)
- **Handles** multi-turn conversations with intent detection and state management
- **Evaluates** message quality across 5 dimensions (Specificity, Merchant Fit, Category Fit, Trigger Relevance, Engagement)
- **Optimizes** for the Groq Free Tier with response caching, rate limiting, and smart retries

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  /v1/healthz  /v1/metadata  /v1/context  /v1/reply  /v1/tick│
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Service Layer                             │
│  Composer → Prompt Builder → Output Validator                │
│  Reply Handler → Intent Detector → State Machine             │
│  Tick Handler → Conversation Intelligence                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    LLM Layer (Groq)                          │
│  BaseLLMClient → GroqClient (OpenAI-compatible SDK)          │
│  Response Cache → Rate Limiter → Retry Strategy              │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

```
Client Request
    │
    ▼
[API Route] ── validates via Pydantic
    │
    ▼
[Prompt Builder] ── selects merchant vs. customer flow
    │               ── assembles system + user prompts
    ▼
[Groq LLM] ── sends chat completion request
    │           ── retries on transient errors
    ▼
[Output Validator] ── parses JSON response
    │                  ── validates ComposedMessage schema
    ▼
[API Response] ── returns structured JSON
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI 0.115+ | Async REST API with auto-generated docs |
| **Validation** | Pydantic v2 | Request/response schemas with type safety |
| **LLM** | Groq API (Llama 3.3 70B) | Message composition via OpenAI-compatible SDK |
| **Config** | pydantic-settings + python-dotenv | Environment-based configuration |
| **Testing** | pytest + pytest-asyncio + httpx | Async test client for FastAPI |
| **Deployment** | Docker + Render | Containerized production deployment |
| **Caching** | File-based JSON cache | Eliminates redundant LLM calls |

---

## 📁 Folder Structure

```
vera-agent/
│
├── app/                          # Production application
│   ├── main.py                   # FastAPI factory + uvicorn entrypoint
│   ├── api/
│   │   └── routes.py             # All /v1 endpoint definitions
│   ├── core/
│   │   ├── config.py             # Settings from .env via pydantic-settings
│   │   ├── logging.py            # Centralized logging (dev/prod formats)
│   │   └── exceptions.py         # Custom exceptions + global error handlers
│   ├── models/
│   │   ├── requests.py           # 4-context framework request schemas
│   │   └── responses.py          # ComposedMessage + response schemas
│   ├── services/
│   │   ├── composer.py           # Message composition orchestrator
│   │   ├── prompt_builder.py     # Prompt routing (merchant vs. customer)
│   │   ├── output_validator.py   # LLM output parsing + validation
│   │   ├── reply_handler.py      # Multi-turn reply processing
│   │   ├── intent_detector.py    # Merchant intent classification
│   │   ├── tick_handler.py       # Proactive follow-up scheduling
│   │   ├── conversation_manager.py # Conversation state persistence
│   │   ├── conversation_intelligence.py # Engagement analysis
│   │   └── state_machine.py      # Conversation stage transitions
│   ├── prompts/
│   │   ├── system_prompt.py      # Global system prompt
│   │   ├── merchant_prompt.py    # Merchant-facing prompt builder
│   │   ├── customer_prompt.py    # Customer-facing prompt builder
│   │   └── reply_prompt.py       # Multi-turn reply prompt builder
│   └── llm/
│       ├── base.py               # Abstract LLM client interface
│       └── groq_client.py        # Groq API client (OpenAI SDK)
│
├── cache/                        # Response caching (Phase 5)
│   ├── response_cache.py         # File-based LLM response cache
│   └── rate_limiter.py           # Sliding-window rate limiter
│
├── evaluation/                   # Offline evaluation framework
│   ├── batch_runner.py           # Batch execution with cache + resume
│   ├── dataset_loader.py         # Dataset loading + scenario building
│   ├── evaluator.py              # Output validation + scoring
│   ├── metrics.py                # 5-dimension quality scoring
│   ├── prompt_comparator.py      # A/B prompt comparison
│   └── report_generator.py       # Markdown report generation
│
├── scripts/
│   ├── generate_submission.py    # submission.jsonl generator
│   ├── evaluate_dataset.py       # Full dataset evaluation
│   ├── verify_release.py         # Production readiness checker
│   └── run_all_checks.py         # Master validation script
│
├── tests/                        # Test suite
│   ├── test_endpoints.py         # API endpoint tests
│   ├── test_evaluation.py        # Evaluation pipeline tests
│   ├── test_phase2.py            # Phase 2 composition tests
│   ├── test_phase3.py            # Phase 3 multi-turn tests
│   └── test_phase35.py           # Phase 3.5 intelligence tests
│
├── Dockerfile                    # Production container
├── docker-compose.yml            # Local container orchestration
├── render.yaml                   # Render.com deployment config
├── requirements.txt              # Python dependencies
├── submission.jsonl              # Challenge submission (25 entries)
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
└── README.md                     # This file
```

---

## ✨ Features

### Phase 1 — Backend Foundation
- FastAPI application with structured project layout
- Pydantic v2 request/response schemas for the 4-context framework
- Environment-based configuration with validation
- Structured logging (dev: human-readable, prod: JSON)
- Custom exception hierarchy with consistent error responses

### Phase 2 — Message Composition
- Two-flow prompt routing: merchant-facing and customer-facing
- Groq LLM integration via OpenAI-compatible SDK
- JSON output parsing with retry on malformed responses
- Output validation against the ComposedMessage schema
- Deterministic output (temperature=0.0)

### Phase 3 — Multi-Turn Conversations
- Intent detection: approval, rejection, question, edit, autoresponder
- Conversation state machine: initial → follow_up → merchant_replied → closed
- Deterministic replies (no LLM) for approval/rejection intents
- LLM-powered replies for questions and edit requests
- Proactive tick-based follow-up scheduling

### Phase 4 — Evaluation Pipeline
- Dataset loading from seed files (categories, merchants, triggers, customers)
- 5-dimension heuristic scoring (no LLM calls)
- Batch execution with rate limiting
- Markdown report generation
- submission.jsonl output

### Phase 5 — Release Engineering
- Response cache (file-based, SHA-256 keyed)
- Resume-able batch generation (`--resume`)
- Dry-run mode (`--dry-run`)
- Smart rate limiting (sliding window, 28 RPM default)
- Retry strategy (transient errors only, max 3 attempts)
- Docker + Render deployment
- Production verification script

---

## 💬 Conversation Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   INITIAL    │────▶│  FOLLOW_UP   │────▶│    CLOSED    │
│              │     │              │     │              │
│ Bot sends    │     │ Waiting for  │     │ Conversation │
│ first msg    │     │ merchant     │     │ complete     │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     merchant replies
                            │
                     ┌──────▼───────┐
                     │  MERCHANT    │
                     │  _REPLIED    │
                     │              │
                     │ Intent       │
                     │ detected:    │
                     │ • approve    │──▶ deterministic reply
                     │ • reject     │──▶ deterministic reply
                     │ • question   │──▶ LLM reply
                     │ • edit       │──▶ LLM reply
                     │ • autorespond│──▶ ignored
                     └──────────────┘
```

---

## 📊 Evaluation Pipeline

Messages are scored on 5 dimensions (0–10 each):

| Dimension | What It Measures |
|-----------|-----------------|
| **Specificity** | Concrete facts: numbers, dates, names, prices, percentages |
| **Merchant Fit** | Personalization: owner name, city, performance metrics, offers |
| **Category Fit** | Vocabulary match for the business vertical + voice taboo checks |
| **Trigger Relevance** | How clearly the message explains *why now* |
| **Engagement** | Compulsion levers: CTA quality, loss aversion, social proof, urgency |

All scoring is **heuristic** (no LLM calls) — fast and deterministic.

---

## 🚀 Deployment

### Local Development

```bash
cd vera-agent
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # then edit .env with your GROQ_API_KEY
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t vera-agent .
docker run -p 8000:8000 --env-file .env vera-agent
```

### Docker Compose

```bash
docker-compose up --build
```

### Render

Push to GitHub and connect your repo to [Render](https://render.com). The `render.yaml` configures everything automatically. Set `GROQ_API_KEY` as a secret environment variable in the Render dashboard.

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/healthz` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/v1/metadata` | Bot name, version, and description |
| `POST` | `/v1/context` | Compose a message from the 4-context framework |
| `POST` | `/v1/reply` | Handle a merchant reply in a multi-turn conversation |
| `POST` | `/v1/tick` | Scheduled cadence check for proactive follow-up |

### Interactive Docs

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 📝 Example Requests & Responses

### Health Check

```bash
curl http://localhost:8000/v1/healthz
```

```json
{"status": "ok"}
```

### Metadata

```bash
curl http://localhost:8000/v1/metadata
```

```json
{
  "name": "Vera AI Agent",
  "version": "1.0.0",
  "description": "AI-powered merchant assistant for the magicpin platform..."
}
```

### Compose Message

```bash
curl -X POST http://localhost:8000/v1/context \
  -H "Content-Type: application/json" \
  -d '{
    "category": {
      "slug": "dentists",
      "display_name": "Dentists",
      "voice": {"tone": "clinical-warm"}
    },
    "merchant": {
      "merchant_id": "m_001",
      "identity": {"name": "SmileCare Dental", "owner_first_name": "Dr. Meera", "city": "Delhi"},
      "subscription": {"status": "active", "plan": "Pro"}
    },
    "trigger": {
      "id": "t_001",
      "scope": "merchant",
      "kind": "research_digest",
      "source": "external",
      "merchant_id": "m_001"
    }
  }'
```

```json
{
  "message": "Message composed successfully.",
  "result": {
    "body": "Hi Meera, a recent dental research digest highlights...",
    "cta": "binary_yes_stop",
    "send_as": "vera",
    "suppression_key": "research:dentists:2026-W17",
    "rationale": "Uses trigger context to provide actionable insight..."
  }
}
```

---

## 🧪 Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific phase tests
pytest tests/test_endpoints.py -v
pytest tests/test_phase2.py -v
pytest tests/test_phase3.py -v

# Verify release readiness (without running server)
python scripts/verify_release.py --skip-api

# Dry run submission generation
python scripts/generate_submission.py --dry-run
```

---

## 🗺️ Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Backend foundation — project structure, API, models, config, tests | ✅ Complete |
| **Phase 2** | Message composition — prompt engineering, LLM integration, structured output | ✅ Complete |
| **Phase 3** | Multi-turn conversations — intent detection, state machine, reply handling | ✅ Complete |
| **Phase 3.5** | Conversation intelligence — engagement analysis, proactive follow-up | ✅ Complete |
| **Phase 4** | Evaluation pipeline — dataset loading, metrics, batch execution, reports | ✅ Complete |
| **Phase 5** | Release engineering — caching, deployment, verification, submission | ✅ Complete |

---

## 💡 Lessons Learned

1. **Groq Free Tier demands caching** — With strict RPM limits, a file-based response cache was essential. The cache-first approach cut API calls by 80%+ during iterative development.

2. **Deterministic output matters** — Setting `temperature=0.0` ensured reproducible results, which made evaluation meaningful and debugging tractable.

3. **Heuristic scoring is underrated** — The 5-dimension heuristic evaluator (no LLM calls) provided fast, deterministic quality signals that guided prompt iteration without burning API credits.

4. **Separation of concerns pays off** — Keeping prompts, LLM clients, services, and evaluation fully decoupled made it easy to iterate on each layer independently.

5. **Schema-first design prevents drift** — Pydantic models at every boundary (request → service → LLM → response) caught data issues early and made the API self-documenting.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
