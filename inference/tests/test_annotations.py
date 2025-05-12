# %% [markdown]
import pytest
from inference.Annotation_Untils import get_emotion, get_sentiment, get_toxicity, get_hate_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# %%

# test_scalar_features.py
def test_get_emotion():
    result = get_emotion("I love this!")
    assert result in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def test_get_sentiment():
    result = get_sentiment("I hate this")
    assert result in ['negative', 'neutral', 'positive', 'UNKNOWN']

def test_get_toxicity():
    scores = get_toxicity("You are awful")
    assert isinstance(scores, dict)
    assert all(isinstance(v, float) for v in scores.values())

def test_get_hate_score():
    score = get_hate_score("I dislike you")
    assert isinstance(score, float)
# %%

# test_api.py
from fastapi.testclient import TestClient
from inference.app import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["message"] == "API is working"

def test_annotate():
    sample_input = {"text": "You are amazing!"}
    res = client.post("/predict", json=sample_input)
    assert res.status_code == 200
    json_data = res.json()
    for field in ["confidence", "predicted_class", "text"]:
        assert field in json_data