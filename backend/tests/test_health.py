def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_health_response_is_json(client):
    response = client.get("/api/health")

    assert response.headers["content-type"].startswith(
        "application/json"
    )