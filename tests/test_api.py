"""
Testes de integração para a API REST (FastAPI).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Cliente de teste para a API FastAPI."""
    return TestClient(app)


class TestHealthEndpoint:
    """Testa o endpoint de health check."""

    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        data = response = client.get("/api/health").json()
        assert data["status"] == "healthy"

    def test_health_contains_version(self, client):
        data = client.get("/api/health").json()
        assert "version" in data


class TestUploadEndpoint:
    """Testa validação de upload de documentos."""

    def test_reject_unsupported_format(self, client):
        """Deve rejeitar arquivos que não são PDF ou TXT."""
        response = client.post(
            "/api/upload",
            files={"file": ("test.docx", b"fake content", "application/octet-stream")}
        )
        assert response.status_code == 400
        assert "Formato não suportado" in response.json()["detail"]

    def test_reject_image_as_document(self, client):
        """Deve rejeitar imagens no endpoint de documentos."""
        response = client.post(
            "/api/upload",
            files={"file": ("photo.jpg", b"fake image", "image/jpeg")}
        )
        assert response.status_code == 400


class TestChatEndpoint:
    """Testa o endpoint de chat."""

    def test_empty_question_rejected(self, client):
        """Deve rejeitar perguntas vazias."""
        response = client.post(
            "/api/chat",
            json={"question": "", "session_id": "test"}
        )
        assert response.status_code == 400

    def test_chat_without_body_rejected(self, client):
        """Deve rejeitar requisição sem body."""
        response = client.post("/api/chat")
        assert response.status_code == 422


class TestSessionsEndpoint:
    """Testa os endpoints de sessão."""

    def test_list_sessions(self, client):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        assert "sessions" in response.json()

    def test_get_empty_session_history(self, client):
        response = client.get("/api/sessions/nonexistent_session")
        assert response.status_code == 200
        assert response.json()["history"] == []


class TestDocumentsEndpoint:
    """Testa os endpoints de listagem de documentos."""

    def test_list_documents(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert "documents" in response.json()

    def test_delete_nonexistent_document(self, client):
        response = client.delete("/api/documents/inexistente.pdf")
        assert response.status_code == 404


class TestFrontend:
    """Testa se o frontend é servido corretamente."""

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_static_css_accessible(self, client):
        response = client.get("/style.css")
        assert response.status_code == 200
