"""Tests for the Flask API endpoints.

The ``/predict`` tests run against a stubbed agent so they exercise the
API contract (metadata truthfulness, error handling, single
classification) without loading Laya or the LLM. The one test that runs
the real model stack is marked ``slow`` and excluded by default
(see pytest.ini).
"""

import inspect
import logging
import time

import agent_state
import model_downloads
import pytest

import main
from config import load_config
from main import app
from planner import PlannerResult


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    """Keep each test's ``PUT /config`` writes out of the real config.json."""
    path = tmp_path / "config.json"
    monkeypatch.setenv("BIDS_AGENT_CONFIG_FILE", str(path))
    return path


@pytest.fixture()
def client(config_path):
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
            # What the API handed over, so a test can assert on the
            # contract rather than on the stub's own assumptions.
            self.seen = {}

        def run(self, user_input, context, history=None,
                dataset_structure=None, file_content=None):
            self.calls += 1
            self.seen = {
                "user_input": user_input,
                "context": context,
                "history": history,
                "dataset_structure": dataset_structure,
                "file_content": file_content,
            }
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


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}

    def test_health_requests_are_not_logged(self, client, caplog):
        caplog.set_level(logging.INFO, logger="bids_agent")
        client.get("/health")
        assert not [
            record for record in caplog.records
            if record.name == "bids_agent" and "/health" in record.getMessage()
        ]

    def test_openai_discovery_probes_are_not_logged(self, client, caplog):
        """A /v1/models liveness probe is not our endpoint; stay quiet."""
        caplog.set_level(logging.INFO, logger="bids_agent")
        resp = client.get("/v1/models")
        assert resp.status_code == 404
        assert not [
            record for record in caplog.records
            if record.name == "bids_agent" and "/v1/models" in record.getMessage()
        ]


class TestStatus:
    def test_status_starts_idle(self, client):
        agent_state.finish()
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.get_json() == {"phase": "idle", "generating_for": None}

    def test_status_reports_the_generating_clock(self, client):
        """A waiting client anchors its clock on a duration, so it never
        has to trust its own clock to match the agent's."""
        agent_state.finish()
        agent_state.generating()
        try:
            body = client.get("/status").get_json()
        finally:
            agent_state.finish()
        assert body["phase"] == "generating"
        assert isinstance(body["generating_for"], float)

    def test_status_requests_are_not_logged(self, client, caplog):
        """The client polls several times a second while it waits."""
        caplog.set_level(logging.INFO, logger="bids_agent")
        client.get("/status")
        assert not [
            record for record in caplog.records
            if record.name == "bids_agent" and "/status" in record.getMessage()
        ]


class TestAgentState:
    """The phase and the load measurement behind ``/status`` and
    ``/predict``'s ``model_load_seconds``."""

    def test_a_load_is_measured_until_generation_begins(self):
        agent_state.finish()
        agent_state.loading()
        time.sleep(0.05)
        agent_state.generating()
        try:
            assert agent_state.load_seconds() > 0.0
        finally:
            agent_state.finish()
        assert agent_state.load_seconds() == 0.0

    def test_a_model_already_in_memory_has_no_load_time(self):
        agent_state.finish()
        agent_state.generating()
        try:
            assert agent_state.load_seconds() == 0.0
        finally:
            agent_state.finish()

    def test_a_late_load_call_does_not_stop_the_generation_clock(self):
        agent_state.finish()
        agent_state.loading()
        agent_state.generating()
        agent_state.loading()
        try:
            assert agent_state.snapshot()["phase"] == agent_state.GENERATING
        finally:
            agent_state.finish()


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
        assert set(data) == {
            "response", "intent", "execution_status", "classification_source",
            "model", "prompt_tokens", "generated_tokens", "generation_seconds",
            "model_load_seconds", "log",
        }
        assert data["response"] == "An explanation."
        assert data["intent"] == "explain"
        assert data["execution_status"] is None
        assert data["classification_source"] == "laya"
        # No model call means no load to measure.
        assert data["model_load_seconds"] == 0.0

    def test_the_model_load_is_reported_beside_the_generation_time(
        self, client, stub_agent,
    ):
        """A cold first answer waits for the checkpoint and then for the
        tokens; the client shows both numbers, so it asks for both."""
        stub = stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="guard",
        ))
        import prompt_log

        original = stub.run

        def run(**kwargs):
            prompt_log.begin("SYSTEM", "QUESTION")
            agent_state.loading()
            time.sleep(0.05)
            agent_state.generating()
            prompt_log.record_generation("An explanation.", model="stub/m")
            return original(**kwargs)

        stub.run = run
        resp = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model_load_seconds"] > 0.0
        # The record that opens under the answer carries it too.
        assert data["log"]["model_load_seconds"] == data["model_load_seconds"]

    def test_the_prompt_record_travels_in_the_response(self, client, monkeypatch):
        """The record of the exchange replaces the old ``prompt_log.txt``:
        the system prompt, the user's question, the output and how the
        answer was produced, per message, without touching disk. The LLM
        layer records during ``run``; the stub does the same."""
        import prompt_log

        class RecordingAgent:
            def run(self, user_input, context, history=None,
                    dataset_structure=None, file_content=None):
                prompt_log.begin("SYSTEM PROMPT", "Why is this wrong?")
                prompt_log.record_generation(
                    "An explanation.", model="stub/stub-7b-instruct",
                    prompt_tokens=120, generated_tokens=64, seconds=1.5,
                )
                return PlannerResult(
                    response="An explanation.", intent="explain",
                    classification_source="laya",
                )

        monkeypatch.setattr(main, "_agent", RecordingAgent())
        resp = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        record = data["log"]
        assert record["system_prompt"] == "SYSTEM PROMPT"
        assert record["user_prompt"] == "Why is this wrong?"
        assert record["output"] == "An explanation."
        assert data["model"] == "stub/stub-7b-instruct"
        assert data["prompt_tokens"] == 120
        assert data["generated_tokens"] == 64
        assert data["generation_seconds"] == 1.5

    def test_the_record_shows_the_answer_the_client_receives(
        self, client, monkeypatch,
    ):
        """The quality gate can repair the model's draft (or replace it
        with the grounded template). "Show full data" is the record of
        the answer the app shows, so its output is that final text, not
        the draft that was rewritten."""
        import prompt_log

        class RewritingAgent:
            def run(self, user_input, context, history=None,
                    dataset_structure=None, file_content=None):
                prompt_log.begin("SYSTEM PROMPT", "Why is this wrong?")
                prompt_log.record_generation("the draft", model="stub/m")
                return PlannerResult(
                    response="the answer the user reads",
                    intent="explain",
                    classification_source="laya",
                )

        monkeypatch.setattr(main, "_agent", RewritingAgent())
        data = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
        }).get_json()
        assert data["log"]["output"] == data["response"]
        assert data["log"]["output"] == "the answer the user reads"

    def test_no_log_when_the_model_was_never_called(self, client, stub_agent):
        """A stubbed agent that never reaches the model reports no record:
        without the reset, a request would show the previous request's
        prompt log."""
        import prompt_log

        stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="laya",
        ))
        prompt_log.reset()
        prompt_log.begin("SYSTEM PROMPT", "old question")
        prompt_log.record_generation("old answer", model="old/model")
        try:
            resp = client.post("/predict", json={
                "user_input": "Explain this to me.",
                "context": {"severity": "err", "rule_id": "X"},
            })
        finally:
            prompt_log.reset()
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["log"] is None
        assert data["generated_tokens"] is None

    def test_internal_diagnostics_are_not_exposed(self, client, stub_agent):
        """Diagnostics stay internal; the API response remains narrow."""
        stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="laya",
            diagnostics={"retrieval": {"found": True}},
        ))
        resp = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "diagnostics" not in data
        assert "tool_used" not in data

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
        assert data["execution_status"] == "not_available"
        assert "tool_used" not in data

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


class TestPredictEvidence:
    """The two dataset fields, from the wire to the agent.

    The agent never sees the disk — the desktop app walks it and sends
    what it found. Nothing here may be dropped at the door, and nothing
    that was absent may arrive as an empty string: an empty section is
    a header with nothing under it.
    """

    def test_the_fields_reach_the_agent(self, client, stub_agent):
        agent = stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="manual",
        ))
        resp = client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
            "history": [{"role": "user", "content": "Earlier?"}],
            "dataset_structure": "my_dataset/\n└── participants.tsv",
            "file_content": "TSV table: 3 columns, 12 data rows",
        })

        assert resp.status_code == 200
        assert agent.seen["history"] == [
            {"role": "user", "content": "Earlier?"},
        ]
        assert agent.seen["dataset_structure"].startswith("my_dataset/")
        assert "12 data rows" in agent.seen["file_content"]

    def test_absent_or_blank_becomes_none_not_empty(self, client, stub_agent):
        agent = stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="manual",
        ))
        client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
            "dataset_structure": "   ",
            "file_content": None,
        })

        assert agent.seen["dataset_structure"] is None
        assert agent.seen["file_content"] is None

    def test_a_runaway_field_is_truncated_not_forwarded(self, client, stub_agent):
        """A prompt is the wrong place to discover an unbounded field:
        the client compiles these down, and this is the backstop."""
        agent = stub_agent(result=PlannerResult(
            response="An explanation.", intent="explain",
            classification_source="manual",
        ))
        client.post("/predict", json={
            "user_input": "Explain this to me.",
            "context": {"severity": "err", "rule_id": "X"},
            "dataset_structure": "x" * (main.EVIDENCE_LIMIT + 500),
        })

        seen = agent.seen["dataset_structure"]
        assert len(seen) <= main.EVIDENCE_LIMIT + 60
        assert seen.endswith("(truncated by the agent)")


# ---------------------------------------------------------------------------
# /prompt — the request side, with no generation
# ---------------------------------------------------------------------------


class TestPromptEndpoint:
    """The dataset export's endpoint: prompt assembly, never an answer."""

    @pytest.fixture()
    def prompt_agent(self, monkeypatch):
        class StubAgent:
            def __init__(self, result=None, error=None):
                self.result = result
                self.error = error
                self.calls = 0
                self.seen = {}

            def build_prompt(self, user_input, context, history=None,
                             dataset_structure=None, file_content=None):
                self.calls += 1
                self.seen = {
                    "user_input": user_input,
                    "context": context,
                    "history": history,
                    "dataset_structure": dataset_structure,
                    "file_content": file_content,
                }
                if self.error is not None:
                    raise self.error
                return self.result

            def build_prompt_log(self, user_input, context, history=None,
                                 dataset_structure=None, file_content=None):
                record = self.build_prompt(
                    user_input, context, history=history,
                    dataset_structure=dataset_structure,
                    file_content=file_content,
                )
                return (
                    "[System prompt]\n"
                    "---------------\n"
                    f"{record.get('system_prompt', '')}\n\n"
                    "[User prompt]\n"
                    "-------------\n"
                    f"{record.get('user_prompt', user_input)}\n"
                )

        def install(result=None, error=None):
            agent = StubAgent(result=result, error=error)
            monkeypatch.setattr(main, "_agent", agent)
            return agent

        return install

    def test_the_request_record_is_returned_verbatim(self, client, prompt_agent):
        agent = prompt_agent(result={
            "system_prompt": "SYSTEM",
            "user_prompt": "Explain this error to me.",
            "intent": "explain",
            "classification_source": "manual",
            "prompt_available": True,
            "retrieval_found": True,
            "retrieval_quality": "strong",
        })
        resp = client.post("/prompt", json={
            "user_input": "Explain this error to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })

        assert resp.status_code == 200
        assert resp.get_json() == agent.result
        assert agent.calls == 1

    def test_evidence_reaches_the_agent(self, client, prompt_agent):
        agent = prompt_agent(result={"system_prompt": "S"})
        client.post("/prompt", json={
            "user_input": "Explain this error to me.",
            "context": {"severity": "err", "rule_id": "X"},
            "history": [{"role": "user", "content": "Earlier?"}],
            "dataset_structure": "my_dataset/\n└── participants.tsv",
            "file_content": "JSON keys: Name",
        })

        assert agent.seen["history"] == [
            {"role": "user", "content": "Earlier?"},
        ]
        assert agent.seen["dataset_structure"].startswith("my_dataset/")
        assert "JSON keys" in agent.seen["file_content"]

    def test_missing_fields_are_rejected(self, client, prompt_agent):
        prompt_agent(result={})
        resp = client.post("/prompt", json={"user_input": "hi"})
        assert resp.status_code == 400
        assert "context" in resp.get_json()["error"]

    def test_a_failure_is_reported_not_hidden(self, client, prompt_agent):
        prompt_agent(error=RuntimeError("no retriever"))
        resp = client.post("/prompt", json={
            "user_input": "Explain this error to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 500
        assert "no retriever" in resp.get_json()["error"]

    def test_the_answer_side_is_never_returned(self, client, prompt_agent):
        """There is no generation, so there is no answer — not even an
        empty one that a client could mistake for a failed model call."""
        prompt_agent(result={"system_prompt": "S", "user_prompt": "Q"})
        data = client.post("/prompt", json={
            "user_input": "Q",
            "context": {"severity": "err", "rule_id": "X"},
        }).get_json()
        assert "response" not in data

    def test_the_prompt_log_is_plain_text_without_an_answer(
        self, client, prompt_agent,
    ):
        """The export writes one readable file per finding, not JSON."""
        prompt_agent(result={
            "system_prompt": "SYSTEM",
            "user_prompt": "Explain this error to me.",
        })
        resp = client.post("/prompt/log", json={
            "user_input": "Explain this error to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })

        assert resp.status_code == 200
        assert resp.content_type.startswith("text/plain")
        text = resp.get_data(as_text=True)
        assert text == (
            "[System prompt]\n"
            "---------------\n"
            "SYSTEM\n\n"
            "[User prompt]\n"
            "-------------\n"
            "Explain this error to me.\n"
        )
        assert "[Agent output]" not in text

    def test_the_prompt_log_uses_the_same_evidence_as_prompt(
        self, client, prompt_agent,
    ):
        agent = prompt_agent(result={"system_prompt": "S"})
        client.post("/prompt/log", json={
            "user_input": "Explain this error to me.",
            "context": {"severity": "err", "rule_id": "X"},
            "dataset_structure": "my_dataset/\n participants.tsv",
            "file_content": "JSON keys: Name",
        })

        assert agent.calls == 1
        assert agent.seen["dataset_structure"].startswith("my_dataset/")
        assert agent.seen["file_content"] == "JSON keys: Name"

    def test_the_prompt_log_rejects_missing_fields(self, client, prompt_agent):
        prompt_agent(result={})
        resp = client.post("/prompt/log", json={"user_input": "hi"})
        assert resp.status_code == 400
        assert "context" in resp.get_json()["error"]

    def test_the_prompt_log_reports_a_failure(self, client, prompt_agent):
        prompt_agent(error=RuntimeError("no retriever"))
        resp = client.post("/prompt/log", json={
            "user_input": "Explain this error to me.",
            "context": {"severity": "err", "rule_id": "X"},
        })
        assert resp.status_code == 500
        assert "no retriever" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# /predict - full local model stack (slow; excluded by default)
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
        retrieval = data["retrieval"]
        assert "retriever_type" in retrieval
        assert "embedding_model" in retrieval
        assert "vector_store_dir" in retrieval
        assert "intent" in data
        assert data["intent"]["mode"] in ("auto", "explain", "fix")
        assert data["intent"]["classifier"] in ("llm", "nli", "laya")
        prompt = data["prompt"]
        assert "system_prompt" in prompt
        assert prompt["effective_system_prompt"]
        assert isinstance(prompt["include_dataset_structure"], bool)
        assert isinstance(prompt["include_file_content"], bool)

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
        llm = client.get("/config").get_json()["llm"]
        original, original_quant = llm["load_in_4bit"], llm["quantization"]

        resp = client.put("/config", json={"load_in_4bit": False})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["load_in_4bit"] is False

        client.put("/config", json={
            "load_in_4bit": original,
            "quantization": original_quant,
        })

    def test_update_config_is_persisted(self, client, config_path):
        original = client.get("/config").get_json()["llm"]["max_new_tokens"]

        client.put("/config", json={"max_new_tokens": 333})

        assert config_path.exists()
        saved = load_config(config_file=config_path)
        assert saved.llm.max_new_tokens == 333

        client.put("/config", json={"max_new_tokens": original})

    def test_update_config_accepts_the_nested_get_shape(self, client):
        original = client.get("/config").get_json()["llm"]["quantization"]
        other = "none" if original == "4bit" else "4bit"

        resp = client.put("/config", json={"llm": {"quantization": other}})

        assert resp.status_code == 200
        assert resp.get_json()["llm"]["quantization"] == other
        client.put("/config", json={"quantization": original})

    def test_changing_quantization_offloads_the_cached_model(
        self, client, monkeypatch
    ):
        from model_manager import manager as model_manager

        calls = []
        monkeypatch.setattr(
            model_manager, "offload_current", lambda: calls.append(1)
        )
        original = client.get("/config").get_json()["llm"]["quantization"]
        other = "none" if original == "4bit" else "4bit"

        client.put("/config", json={"quantization": other})

        assert calls, "a load-affecting change must drop the cached model"
        client.put("/config", json={"quantization": original})

    def test_generation_only_change_keeps_the_cached_model(
        self, client, monkeypatch
    ):
        from model_manager import manager as model_manager

        calls = []
        monkeypatch.setattr(
            model_manager, "offload_current", lambda: calls.append(1)
        )
        original = client.get("/config").get_json()["llm"]["temperature"]

        client.put("/config", json={"temperature": original})

        assert not calls, "sampling settings do not require a model reload"

    def test_update_config_quality_toggle(self, client):
        original = client.get("/config").get_json()["quality"]
        resp = client.put("/config", json={
            "quality": {"enabled": False, "repair_attempts": 0},
        })
        assert resp.status_code == 200
        data = resp.get_json()["quality"]
        assert data["enabled"] is False
        assert data["repair_attempts"] == 0
        assert data["fallback_enabled"] is True  # untouched default survives
        client.put("/config", json={"quality": original})

    def test_update_config_quality_attempts_are_clamped(self, client):
        original = client.get("/config").get_json()["quality"]
        resp = client.put("/config", json={"quality": {"repair_attempts": 99}})
        assert resp.status_code == 200
        assert resp.get_json()["quality"]["repair_attempts"] == 3
        client.put("/config", json={"quality": original})

    def test_update_config_prompt_and_evidence_switches(self, client):
        original = client.get("/config").get_json()["prompt"]

        resp = client.put("/config", json={
            "system_prompt": "  Custom instructions.  ",
            "include_dataset_structure": False,
            "include_file_content": False,
        })

        assert resp.status_code == 200
        prompt = resp.get_json()["prompt"]
        assert prompt["system_prompt"] == "Custom instructions."
        assert prompt["effective_system_prompt"] == "Custom instructions."
        assert prompt["include_dataset_structure"] is False
        assert prompt["include_file_content"] is False

        client.put("/config", json=original)

    def test_update_config_prompt_is_persisted(self, client, config_path):
        original = client.get("/config").get_json()["prompt"]

        client.put("/config", json={
            "system_prompt": "Persisted instructions.",
            "include_dataset_structure": False,
            "include_file_content": False,
        })

        saved = load_config(config_file=config_path)
        assert saved.prompt.system_prompt == "Persisted instructions."
        assert saved.prompt.include_dataset_structure is False
        assert saved.prompt.include_file_content is False

        client.put("/config", json=original)

    def test_empty_system_prompt_falls_back_to_the_builtin(self, client):
        from system_prompt import BIDS_SYSTEM_PROMPT

        original = client.get("/config").get_json()["prompt"]

        resp = client.put("/config", json={"system_prompt": ""})

        assert resp.status_code == 200
        prompt = resp.get_json()["prompt"]
        assert prompt["system_prompt"] == ""
        assert prompt["effective_system_prompt"] == BIDS_SYSTEM_PROMPT

        client.put("/config", json=original)

    def test_update_config_model_name(self, client):
        resp = client.get("/config")
        original = resp.get_json()["llm"]["model_name"]

        resp = client.put("/config", json={"model_name": "Qwen/Qwen3-1.7B"})
        assert resp.status_code == 200
        assert resp.get_json()["llm"]["model_name"] == "Qwen/Qwen3-1.7B"

        client.put("/config", json={"model_name": original})

    def test_update_config_retriever_backend(self, client):
        retrieval = client.get("/config").get_json()["retrieval"]
        original_type = retrieval["retriever_type"]
        original_model = retrieval["embedding_model"]

        resp = client.put("/config", json={
            "retriever_type": "embedding",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_device": "cpu",
            "embedding_pooling": "mean",
            "embedding_min_score": 0.42,
            "vector_store_dir": "./tmp/vectors",
        })
        assert resp.status_code == 200
        data = resp.get_json()["retrieval"]
        assert data["retriever_type"] == "embedding"
        assert data["embedding_device"] == "cpu"
        assert data["embedding_pooling"] == "mean"
        assert data["embedding_min_score"] == pytest.approx(0.42)
        assert data["vector_store_dir"] == "./tmp/vectors"

        client.put("/config", json={
            "retriever_type": original_type,
            "embedding_model": original_model,
            "embedding_device": "auto",
            "embedding_pooling": "cls",
            "embedding_min_score": 0.35,
            "vector_store_dir": "./knowledge-base/vector_store",
        })

    def test_update_config_intent_mode_and_classifier(self, client):
        original = client.get("/config").get_json()["intent"]

        resp = client.put("/config", json={
            "intent": {"mode": "fix", "classifier": "llm"},
        })
        assert resp.status_code == 200
        data = resp.get_json()["intent"]
        assert data["mode"] == "fix"
        assert data["classifier"] == "llm"

        client.put("/config", json={"intent": original})

    def test_update_config_rejects_unknown_intent_modes(self, client):
        original = client.get("/config").get_json()["intent"]
        resp = client.put("/config", json={
            "intent": {"mode": "autonomous", "classifier": "oracle"},
        })
        assert resp.status_code == 200
        data = resp.get_json()["intent"]
        assert data["mode"] == "auto"
        assert data["classifier"] == "llm"
        client.put("/config", json={"intent": original})


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

    def test_models_report_what_is_local(self, client):
        """The desktop app flags ready-to-use models from this field:
        a plain directory listing, no model and no network."""
        resp = client.get("/models")
        models = resp.get_json()["models"]
        for m in models:
            assert isinstance(m["downloaded"], bool)

    def test_models_report_the_cache_directory(self, client):
        resp = client.get("/models")
        models_dir = resp.get_json()["models_dir"]
        assert isinstance(models_dir, str) and models_dir


# ---------------------------------------------------------------------------
# /models/download, /models/download/status
# ---------------------------------------------------------------------------


class TestModelDownloads:
    def test_download_answers_immediately_with_the_status(self, client, monkeypatch):
        """The answer must not block on the fetch: the download runs in
        its own thread and the client polls for progress."""
        calls = {}

        def fake_start(name):
            calls["name"] = name
            return {"name": name, "status": "downloading",
                    "downloaded_bytes": 0, "total_bytes": None}

        monkeypatch.setattr(model_downloads, "start", fake_start)
        resp = client.post("/models/download", json={"name": "stub/stub-7b"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "downloading"
        assert calls["name"] == "stub/stub-7b"

    def test_download_requires_a_name(self, client):
        resp = client.post("/models/download", json={})
        assert resp.status_code == 400

    def test_download_status_reports_progress(self, client, monkeypatch):
        state = {"name": "stub/stub-7b", "status": "downloading",
                 "downloaded_bytes": 812_000, "total_bytes": 3_400_000,
                 "error": ""}
        monkeypatch.setattr(model_downloads, "status", lambda: state)
        resp = client.get("/models/download/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "downloading"
        assert data["downloaded_bytes"] == 812_000
        assert data["total_bytes"] == 3_400_000
