from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_setosa():
    response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
                }
            )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_species"] == "Setosa"
    assert "probabilities" in data

def test_predict_probabilities():
    response = client.post(
            "predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
                }
            )
    probabilities = response.json()["probabilities"]

    assert probabilities["Setosa"] == 1.0
    assert probabilities["Versicolor"] == 0.0
    assert probabilities["Virginica"] == 0.0

def test_reject_megative_measurement():
    response = client.post(
            "/predict",
            json={
                "sepal_length": -5,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
                }
            )

    assert response.status_code == 422

def test_reject_missing_measurement():
    response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4
                }
            )

    assert response.status_code == 422
