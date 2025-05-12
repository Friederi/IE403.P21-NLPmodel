from fastapi.testclient import TestClient
from inference.app import app
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

client = TestClient(app)

def test_predict_valid_input():
    res = client.post("/predict", json={"text": "You are amazing!"})
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert "confidence" in data

def test_predict_missing_input():
    res = client.post("/predict", json={})
    assert res.status_code == 422  

def test_predict_empty_string():
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 200
    assert res.json()["text"] == ""

def test_predict_invalid_json():
    res = client.post("/predict", data="not a json")
    assert res.status_code == 422

def test_predict_output_range():
    res = client.post("/predict", json={"text": "Random text"})
    data = res.json()
    assert 0 <= data["predicted_class"] <= 1
    assert 0.0 <= data["confidence"] <= 1.0