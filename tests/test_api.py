"""Tests for the Flask API endpoints.

The ``/predict`` tests run against a stubbed agent so they exercise the
API contract (metadata truthfulness, error handling, single
classification) without loading Laya or the LLM. The one test that runs
the real model stack is marked ``slow`` and excluded by default
(see pytest.ini).
"""

import inspect

import pytest

import main
from main import app
from planner import PlannerResult


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
    main._agent = None  # never leak a stub or a real agent between tests


@pytest.fixture()
def stub_agent(monkeypatch):
    """Install a stub agent and count every ``run`` call on it."""

    class StubAgent:
        def __init__(self, result=None, error=None):
            self.result = result
            self.error = error
            self.calls = 0

        def run(self, user_input, context, history=None):
            self.calls += 1
            if self.error is not None:
                raise self.error
            return self.result

    def install(result=None, error=None):
        agent = StubAgent(result=result, error=error)
        monkeypatch.setattr(main, "_agent", agent)
        return agent

    return install


# ---------------------------------------------------------------------------
# /predict — response metadata (mocked agent)
# ---------------------------------------------------------------------------


class TestPredictMetadata:
    def test_explain_metadata(self, client, stub_agent):
        stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="laya",
            retrieval_found=True, retrieval_quality="rule_exact",
        ))
        resp = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "An explanation."
        assert data["intent"] == "explain"
        # No tool executes in this phase — the API must say so.
        assert data["tool_used"] is None
        assert data["execution_status"] is None
        assert data["classification_source"] == "laya"

    def test_action_request_metadata(self, client, stub_agent):
        stub_agent(result=PlannerResult(
            response="Execution is not available...", intent="fix",
            tool_used=None, execution_status="not_available",
            classification_source="laya",
        ))
        resp = client.post("/predict", json={
            "user_input": "Fix this error in my dataset.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"] == "fix"
        assert data["tool_used"] is None
        assert data["execution_status"] == "not_available"

    def test_classification_is_not_recomputed_by_the_api(self, client, stub_agent):
        """The intent in the response is the one the pipeline produced.

        ``/predict`` must not rerun Laya to build response metadata: the
        handler only forwards what ``Agent.run`` returned, and it is
        called exactly once.
        """
        agent = stub_agent(result=PlannerResult(
            response="r", intent="unclear", classification_source="fallback",
        ))
        resp = client.post("/predict", json={
            "user_input": "Can you help me with this?",
            "context": {},
        })
        assert resp.status_code == 200
        assert agent.calls == 1
        assert resp.get_json()["intent"] == "unclear"
        assert resp.get_json()["classification_source"] == "fallback"
        # The endpoint itself never invokes the classifier (names in the
        # function body; ``classification_source`` is just a response
        # field forwarded from the pipeline result).
        import ast

        fn = ast.parse(inspect.getsource(main.predict)).body[0]
        names = [n.id for n in ast.walk(fn) if isinstance(n, ast.Name)]
        names += [n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)]
        forbidden = {"laya", "classify", "classify_intent", "intent_classifier"}
        assert not any(
            n.lower() in forbidden or n.lower().startswith("laya")
            for n in names
        )

    def test_agent_failure_returns_500_with_reason(self, client, stub_agent):
        stub_agent(error=ValueError("llm exploded"))
        resp = client.post("/predict", json={
            "user_input": "Explain.", "context": {"rule_id": "X"},
        })
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data
        assert "ValueError" in data["error"]

    def test_predict_missing_fields_returns_400(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /predict — full local model stack (slow; excluded by default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPredictRealStack:
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
        assert data["intent"] in ("explain", "fix", "unclear")
        assert data["tool_used"] is None
        assert isinstance(data["response"], str) and data["response"]


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
