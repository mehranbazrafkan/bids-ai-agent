"""Tests for the Flask API endpoints."""

import pytest
from main import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_response(self, client):
        resp = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {
                "severity": "err",
                "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
                "message": "ReferencesAndLinks must be array",
                "field": "ReferencesAndLinks",
            },
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "response" in data
        assert "intent" in data
        assert data["intent"] in ("explain", "fix")

    def test_predict_missing_fields_returns_400(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_get_config_returns_llm_and_retrieval(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "llm" in data
        assert "retrieval" in data
        llm = data["llm"]
        assert "model_name" in llm
        assert "max_new_tokens" in llm
        assert "temperature" in llm
        assert "top_p" in llm
        assert "top_k" in llm
        assert "repetition_penalty" in llm
        assert "enable_thinking" in llm
        assert "load_in_4bit" in llm

    def test_update_config_max_new_tokens(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["max_new_tokens"]

        resp = client.put("/config", json={"max_new_tokens": 500})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["max_new_tokens"] == 500

        client.put("/config", json={"max_new_tokens": original})

    def test_update_config_temperature(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["temperature"]

        resp = client.put("/config", json={"temperature": 0.5})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["temperature"] == 0.5

        client.put("/config", json={"temperature": original})

    def test_update_config_enable_thinking(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["enable_thinking"]

        resp = client.put("/config", json={"enable_thinking": False})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["enable_thinking"] is False

        client.put("/config", json={"enable_thinking": original})

    def test_update_config_top_p(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["top_p"]

        resp = client.put("/config", json={"top_p": 0.9})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["top_p"] == 0.9

        client.put("/config", json={"top_p": original})

    def test_update_config_top_k(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["top_k"]

        resp = client.put("/config", json={"top_k": 50})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["top_k"] == 50

        client.put("/config", json={"top_k": original})

    def test_update_config_repetition_penalty(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["repetition_penalty"]

        resp = client.put("/config", json={"repetition_penalty": 1.1})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["repetition_penalty"] == 1.1

        client.put("/config", json={"repetition_penalty": original})

    def test_update_config_load_in_4bit(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["load_in_4bit"]

        resp = client.put("/config", json={"load_in_4bit": False})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["load_in_4bit"] is False

        client.put("/config", json={"load_in_4bit": original})

    def test_update_config_model_name(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["model_name"]

        resp = client.put("/config", json={"model_name": "Qwen/Qwen3-1.7B"})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["model_name"] == "Qwen/Qwen3-1.7B"

        client.put("/config", json={"model_name": original})


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------


class TestModels:
    def test_list_models_returns_list(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_models_have_required_fields(self, client):
        resp = client.get("/models")
        models = resp.get_json()["models"]
        for m in models:
            assert "name" in m
            assert "description" in m
            assert "size" in m

    def test_models_include_qwen(self, client):
        resp = client.get("/models")
        names = [m["name"] for m in resp.get_json()["models"]]
        assert any("Qwen" in n for n in names)
