import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import database


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with an isolated SQLite database."""

    test_db = tmp_path / "test_medical_imaging.db"

    # Replace the application's database path for this test.
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    with TestClient(app) as test_client:
        yield test_client