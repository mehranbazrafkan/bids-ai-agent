"""Tests for the planner's LCEL pipeline and the StepLogger.

The pipeline must be transparent orchestration: the deterministic
stages (retrieval, prompt rendering, response checks) behave exactly as
before, the model step is the only one that calls out, and the
observability handler reports step names and durations — never step
content.
"""

import logging
import os

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from laya_decision import INTENT_EXPLAIN, LayaDecision
from planner import Planner, PlannerResult
from observability import StepLogger
from retriever import Retriever

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "knowledge-base")

from config import RetrievalConfig

ISSUE_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array",
    "field": "ReferencesAndLinks",
}


@pytest.fixture(scope="module")
def kb_retriever():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


class SpyLLM:
    def __init__(self, response="STUB", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.response)


class SpyLaya:
    def __init__(self, decision=None):
        self.decision = decision or LayaDecision(INTENT_EXPLAIN, 0.9, "laya")
        self.calls = 0

    def classify_intent(self, user_input, context=""):
        self.calls += 1
        return self.decision


# ---------------------------------------------------------------------------
# Pipeline mechanics
# ---------------------------------------------------------------------------


class TestPlannerPipeline:
    def test_execute_runs_the_pipeline_and_reports(self, kb_retriever):
        llm = SpyLLM("ANSWER.")
        p = Planner(llm=llm, retriever=kb_retriever, laya=SpyLaya())
        result = p.execute(system_prompt="S", user_input="Explain.",
                           context=ISSUE_001)
        assert isinstance(result, PlannerResult)
        assert result.response == "ANSWER."
        assert result.intent == "explain"
        assert result.retrieval_found is True
        assert len(llm.calls) == 1
        # the model saw the assembled system message and the question
        assert llm.calls[0][0].content.startswith("System: S")
        assert llm.calls[0][-1].content == "Explain."

    def test_runnable_llm_is_used_directly(self, kb_retriever):
        """A real LangChain runnable (not a duck-typed spy) plugs straight
        into the pipeline — no wrapper needed."""
        calls = []

        def gen(prompt_value):
            calls.append(prompt_value)
            return AIMessage(content="RUNNABLE ANSWER")

        runnable = RunnableLambda(gen, name="fake-llm")
        p = Planner(llm=runnable, retriever=kb_retriever, laya=SpyLaya())
        result = p.execute(system_prompt="S", user_input="Explain.",
                           context=ISSUE_001)
        assert result.response == "RUNNABLE ANSWER"
        assert calls
        # the branch handed the runnable the rendered prompt value
        from langchain_core.prompt_values import ChatPromptValue
        assert isinstance(calls[0], ChatPromptValue)
        messages = calls[0].to_messages()
        assert messages[0].type == "system"
        assert messages[-1].content == "Explain."

    def test_model_error_propagates(self, kb_retriever):
        """A generation failure is not swallowed by the orchestration."""
        llm = SpyLLM(error=RuntimeError("gpu oom"))
        p = Planner(llm=llm, retriever=kb_retriever, laya=SpyLaya())
        with pytest.raises(RuntimeError, match="gpu oom"):
            p.execute(system_prompt="S", user_input="Explain.",
                      context=ISSUE_001)

    def test_planner_result_fields_are_stable(self):
        """The API contract: these are the field names /predict reads."""
        assert set(PlannerResult.model_fields) == {
            "response", "intent", "tool_used", "execution_status",
            "classification_source", "retrieval_found",
            "retrieval_quality", "diagnostics",
        }

    def test_repeated_requests_share_the_pipeline(self, kb_retriever):
        """The pipeline is built once and re-invoked per request."""
        llm = SpyLLM()
        p = Planner(llm=llm, retriever=kb_retriever, laya=SpyLaya())
        first = p._pipeline
        p.execute(system_prompt="S", user_input="One.", context=ISSUE_001)
        p.execute(system_prompt="S", user_input="Two.", context=ISSUE_001)
        assert p._pipeline is first
        assert [c[-1].content for c in llm.calls] == ["One.", "Two."]


# ---------------------------------------------------------------------------
# StepLogger
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture()
def captured_step_log():
    capture = _Capture()
    logger = logging.getLogger("bids_agent.pipeline.test")
    logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    try:
        yield capture
    finally:
        logger.removeHandler(capture)
        logger.setLevel(logging.NOTSET)


def test_step_logger_records_named_steps(captured_step_log):
    step_logger = StepLogger(log=logging.getLogger("bids_agent.pipeline.test"))
    chain = (
        RunnableLambda(lambda x: x + 1, name="step_one")
        | RunnableLambda(lambda x: x * 2, name="step_two")
    )
    chain.invoke(1, config={"callbacks": [step_logger]})
    joined = "\n".join(captured_step_log.messages)
    assert "pipeline step step_one" in joined
    assert "pipeline step step_two" in joined


def test_step_logger_records_failures(captured_step_log):
    step_logger = StepLogger(log=logging.getLogger("bids_agent.pipeline.test"))

    def boom(_):
        raise RuntimeError("step exploded")

    chain = RunnableLambda(boom, name="bad_step")
    with pytest.raises(RuntimeError, match="step exploded"):
        chain.invoke("x", config={"callbacks": [step_logger]})
    joined = "\n".join(captured_step_log.messages)
    assert "bad_step failed" in joined


def test_step_logger_never_logs_step_content(captured_step_log):
    """Inputs carry dataset content; only names and durations may appear."""
    step_logger = StepLogger(log=logging.getLogger("bids_agent.pipeline.test"))
    sentinel = "DATASET-CONTENT-SECRET"
    chain = RunnableLambda(lambda x: x, name="quiet_step")
    chain.invoke(sentinel, config={"callbacks": [step_logger]})
    joined = "\n".join(captured_step_log.messages)
    assert sentinel not in joined
    assert "quiet_step" in joined
