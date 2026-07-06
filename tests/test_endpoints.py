"""Endpoint tests for the Vera AI Agent API.

Covers all v1 endpoints with status code and response shape assertions.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.responses import ComposedMessage


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client from a fresh app instance."""
    app = create_app()
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────
# GET /v1/healthz
# ──────────────────────────────────────────────────────────────────

class TestHealthz:
    """Health check endpoint tests."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/v1/healthz")
        assert response.status_code == 200

    def test_returns_status_ok(self, client: TestClient) -> None:
        data = client.get("/v1/healthz").json()
        assert data["status"] == "ok"


# ──────────────────────────────────────────────────────────────────
# GET /v1/metadata
# ──────────────────────────────────────────────────────────────────

class TestMetadata:
    """Bot metadata endpoint tests."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/v1/metadata")
        assert response.status_code == 200

    def test_contains_required_fields(self, client: TestClient) -> None:
        data = client.get("/v1/metadata").json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Vera AI Agent"
        assert data["version"] == "1.0.0"


# ──────────────────────────────────────────────────────────────────
# POST /v1/context
# ──────────────────────────────────────────────────────────────────

# Minimal valid payload matching the Pydantic models
VALID_CONTEXT_PAYLOAD = {
    "category": {
        "slug": "dentists",
    },
    "merchant": {
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "identity": {
            "name": "Dr. Meera's Dental Clinic",
        },
        "subscription": {
            "status": "active",
            "plan": "Pro",
        },
    },
    "trigger": {
        "id": "trg_001_research_digest_dentists",
        "scope": "merchant",
        "kind": "research_digest",
        "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi",
    },
}

MOCK_COMPOSED = ComposedMessage(
    body="Dr. Meera, JIDA's Oct issue has a finding for your practice.",
    cta="open_ended",
    send_as="vera",
    suppression_key="research:dentists:2026-W17",
    rationale="Research digest with clinical anchor.",
)


class TestComposeContext:
    """Context composition endpoint tests."""

    @patch("app.api.routes.compose", new_callable=AsyncMock, return_value=MOCK_COMPOSED)
    def test_valid_payload_returns_200(self, mock_compose: AsyncMock, client: TestClient) -> None:
        response = client.post("/v1/context", json=VALID_CONTEXT_PAYLOAD)
        assert response.status_code == 200

    @patch("app.api.routes.compose", new_callable=AsyncMock, return_value=MOCK_COMPOSED)
    def test_valid_payload_returns_composed_message(self, mock_compose: AsyncMock, client: TestClient) -> None:
        data = client.post("/v1/context", json=VALID_CONTEXT_PAYLOAD).json()
        assert data["message"] == "Message composed successfully."
        assert data["result"] is not None
        assert data["result"]["body"] == MOCK_COMPOSED.body
        assert data["result"]["cta"] == "open_ended"
        assert data["result"]["send_as"] == "vera"

    @patch("app.api.routes.compose", new_callable=AsyncMock, return_value=MOCK_COMPOSED)
    def test_with_customer_returns_200(self, mock_compose: AsyncMock, client: TestClient) -> None:
        payload = {
            **VALID_CONTEXT_PAYLOAD,
            "customer": {
                "customer_id": "c_001_priya_for_m001",
                "merchant_id": "m_001_drmeera_dentist_delhi",
                "identity": {"name": "Priya"},
            },
        }
        response = client.post("/v1/context", json=payload)
        assert response.status_code == 200

    def test_missing_required_fields_returns_422(self, client: TestClient) -> None:
        """Category slug is required — omitting it should fail validation."""
        bad_payload = {
            "category": {},
            "merchant": {
                "merchant_id": "m_001",
                "identity": {"name": "Test"},
                "subscription": {"status": "active"},
            },
            "trigger": {
                "id": "trg_001",
                "scope": "merchant",
                "kind": "test",
                "source": "external",
                "merchant_id": "m_001",
            },
        }
        response = client.post("/v1/context", json=bad_payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/context", json={})
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────
# POST /v1/reply
# ──────────────────────────────────────────────────────────────────

class TestReply:
    """Reply endpoint tests."""

    def test_valid_payload_returns_200(self, client: TestClient) -> None:
        payload = {
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "merchant_message": "Yes please",
        }
        response = client.post("/v1/reply", json=payload)
        assert response.status_code == 200

    def test_returns_placeholder_message(self, client: TestClient) -> None:
        payload = {
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "merchant_message": "Yes please",
        }
        data = client.post("/v1/reply", json=payload).json()
        assert data["message"] == "Reply endpoint."

    def test_missing_fields_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/reply", json={})
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────
# POST /v1/tick
# ──────────────────────────────────────────────────────────────────

class TestTick:
    """Tick endpoint tests."""

    def test_valid_payload_returns_200(self, client: TestClient) -> None:
        payload = {"timestamp": "2026-04-26T10:00:00Z"}
        response = client.post("/v1/tick", json=payload)
        assert response.status_code == 200

    def test_empty_payload_returns_200(self, client: TestClient) -> None:
        """All tick fields are optional."""
        response = client.post("/v1/tick", json={})
        assert response.status_code == 200

    def test_returns_placeholder_message(self, client: TestClient) -> None:
        data = client.post("/v1/tick", json={}).json()
        assert data["message"] == "Tick endpoint."
