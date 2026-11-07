"""Planner tests — explanation-only pipeline with a mocked classifier and LLM.

These tests pin the behavior required of the explanation-only phase:

* Laya (mocked here) decides the intent exactly ONCE per request and the
  decision is carried through — never re-derived.
* No fixed keyword list routes requests: routing follows the decision the
  classifier returned, whatever the wording of the input.
* No tool is ever dispatched; ``tool_used`` stays ``None``.
* Action requests get an honest "execution not available" treatment with
  grounded evidence, ambiguous requests get a conservative clarification,
  and retrieval misses are declared as misses instead of being presented
  as documentation.
"""

import os

import pytest

import planner as planner_module
from config import RetrievalConfig
from laya_decision import INTENT_ACTION, INTENT_EXPLAIN, INTENT_UNCLEAR, LayaDecision
from planner import (
    CLARIFICATION_MESSAGE,
    EXECUTION_NOT_AVAILABLE,
    Planner,
    PlannerResult,
)
from retriever import Retriever
from tool_registry import ToolExecutionUnavailable, ToolRegistry

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "knowledge-base")

ISSUE_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  \u00b7  Use a value of the correct type.",
    "field": "ReferencesAndLinks",
}

ISSUE_002 = {
    "severity": "warn",
    "rule_id": "bidsmgr.todo_placeholder",
    "message": "field 'EthicsApprovals' contains a TODO placeholder",
    "field": "EthicsApprovals",
}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class SpyLaya:
    """Returns a canned decision and counts classify calls."""

    def __init__(self, decision=None, error=None):
        self.decision = decision or LayaDecision(INTENT_EXPLAIN, 0.9, "laya")
        self.error = error
        self.calls = 0

    def classify_intent(self, user_input, context=""):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.decision


class SpyLLM:
    """Records every prompt it receives and returns a canned response."""

    def __init__(self, response="STUB RESPONSE"):
        self.response = response
        # list of (system_prompt, user_prompt, history) — the history is
        # recorded because *where* it goes is the behaviour under test:
        # as chat turns between the system prompt and the question, and
        # as a section of the system prompt built from those same turns.
        self.calls = []

    def generate_response(self, system_prompt, user_prompt, history=None):
        self.calls.append((system_prompt, user_prompt, history))
        return self.response


class SpyRegistry(ToolRegistry):
    """Fails the test if anything tries to execute a tool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execute_calls = []

    def execute(self, name, **kwargs):
        self.execute_calls.append((name, kwargs))
        raise AssertionError(
            f"tool {name!r} was dispatched; no tool may run in this phase"
        )


@pytest.fixture(scope="module")
def kb_retriever():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


def make_planner(retriever, laya=None, llm=None, tools=None):
    laya = laya if laya is not None else SpyLaya()
    llm = llm if llm is not None else SpyLLM()
    tools = tools if tools is not None else SpyRegistry()
    p = Planner(llm=llm, retriever=retriever,
                tools=tools, laya=laya)
    return p, llm, laya, tools


# ---------------------------------------------------------------------------
# Explain requests
# ---------------------------------------------------------------------------


class TestExplainPath:
    def test_explain_produces_grounded_prompt_and_metadata(self, kb_retriever):
        llm = SpyLLM(response="A grounded explanation.")
        laya = SpyLaya(LayaDecision(INTENT_EXPLAIN, 0.92, "laya"))
        p, _, _, _ = make_planner(laya=laya, llm=llm, retriever=kb_retriever)

        result = p.execute(
            system_prompt="SYSTEM BOUNDARY",
            user_input="What does this error mean?",
            context=ISSUE_001,
        )

        assert isinstance(result, PlannerResult)
        assert result.intent == "explain"
        assert result.tool_used is None
        assert result.execution_status is None
        assert result.classification_source == "laya"
        assert result.response == "A grounded explanation."

        # Exactly one classification, one generation.
        assert laya.calls == 1
        assert len(llm.calls) == 1

        system_prompt, user_prompt, _history = llm.calls[0]
        # system instructions first, user's actual question forwarded
        assert system_prompt.startswith("System: SYSTEM BOUNDARY")
        assert user_prompt == "What does this error mean?"
        # issue data present
        assert "JSON_SCHEMA_VALIDATION_ERROR" in system_prompt
        assert "ReferencesAndLinks" in system_prompt
        # retrieved knowledge present as evidence
        assert "[Retrieved BIDS Knowledge" in system_prompt
        assert "=== Knowledge Item" in system_prompt
        # no action-unavailable banner on a pure explain request
        assert "Execution Not Available" not in system_prompt

        # retrieval outcome is reported truthfully
        assert result.retrieval_found is True
        assert result.retrieval_quality == "rule_exact"

    def test_retrieval_miss_is_declared_not_faked(self, kb_retriever):
        """A miss must not be rendered as a retrieved document."""
        llm = SpyLLM()
        p, _, _, _ = make_planner(llm=llm, retriever=kb_retriever)

        result = p.execute(
            system_prompt="SYSTEM BOUNDARY",
            user_input="Explain this.",
            context=ISSUE_002,  # bidsmgr rule — not in the knowledge base
        )

        assert result.retrieval_found is False
        assert result.retrieval_quality == "none"
        system_prompt = llm.calls[0][0]
        assert "No relevant knowledge-base record was found" in system_prompt
        assert "=== Knowledge Item" not in system_prompt

    def test_repeated_wording_does_not_change_routing(self, kb_retriever):
        """Routing follows the classifier's decision, not the wording.

        The input is saturated with action words, yet the (mocked)
        classifier says explain — the planner must obey the decision,
        which is only possible if no keyword rule short-circuits it.
        """
        laya = SpyLaya(LayaDecision(INTENT_EXPLAIN, 0.8, "laya"))
        p, llm, _, _ = make_planner(laya=laya, retriever=kb_retriever)
        result = p.execute(
            system_prompt="S",
            user_input="fix fix fix rename repair validate clean correct modify",
            context=ISSUE_001,
        )
        assert result.intent == "explain"
        assert result.execution_status is None

    def test_planner_has_no_keyword_routing_machinery(self):
        # The old ``_FIX_KEYWORDS`` six-word list is gone for good, and
        # the legacy keyword/zero-shot classifier is not wired in.
        assert not hasattr(planner_module, "_FIX_KEYWORDS")
        assert "intent_classifier" not in vars(planner_module)


# ---------------------------------------------------------------------------
# Action requests
# ---------------------------------------------------------------------------


class TestActionPath:
    def test_action_request_discloses_unavailable_execution(self, kb_retriever):
        llm = SpyLLM(response="The current version cannot modify your dataset. ...")
        laya = SpyLaya(LayaDecision(INTENT_ACTION, 0.88, "laya"))
        tools = SpyRegistry()
        p, _, _, _ = make_planner(laya=laya, llm=llm, tools=tools,
                                  retriever=kb_retriever)

        result = p.execute(
            system_prompt="SYSTEM BOUNDARY",
            user_input="Fix this error in my dataset.",
            context=ISSUE_001,
        )

        assert result.intent == "fix"
        assert result.tool_used is None
        assert result.execution_status == EXECUTION_NOT_AVAILABLE == "not_available"
        assert result.classification_source == "laya"

        # No tool was dispatched (SpyRegistry.execute would have failed the test).
        assert tools.execute_calls == []

        system_prompt = llm.calls[0][0]
        assert "[Action Requested — Execution Not Available]" in system_prompt
        # The request is still served with grounded evidence.
        assert "JSON_SCHEMA_VALIDATION_ERROR" in system_prompt
        assert "[Retrieved BIDS Knowledge" in system_prompt
        assert result.retrieval_found is True

    def test_mixed_explain_and_act_request_still_explains(self, kb_retriever):
        """'Explain ... and then repair it' is an action request that must
        still get the explanation evidence plus the honesty banner."""
        llm = SpyLLM(response="Explanation ... Note: I cannot apply the fix.")
        laya = SpyLaya(LayaDecision(INTENT_ACTION, 0.7, "laya"))
        p, _, _, _ = make_planner(laya=laya, llm=llm, retriever=kb_retriever)

        result = p.execute(
            system_prompt="SYSTEM BOUNDARY",
            user_input="Can you explain this error and then repair it?",
            context=ISSUE_001,
        )

        assert result.intent == "fix"
        assert result.execution_status == "not_available"
        system_prompt = llm.calls[0][0]
        assert "Execution Not Available" in system_prompt
        assert "=== Knowledge Item" in system_prompt  # explanation evidence

    def test_action_request_with_empty_context_still_honest(self, kb_retriever):
        laya = SpyLaya(LayaDecision(INTENT_ACTION, 0.9, "laya"))
        p, llm, _, _ = make_planner(laya=laya, retriever=kb_retriever)
        result = p.execute(system_prompt="S",
                           user_input="Rename this subject for me.",
                           context={})
        assert result.intent == "fix"
        assert result.tool_used is None
        assert result.execution_status == "not_available"


# ---------------------------------------------------------------------------
# Unclear / fallback
# ---------------------------------------------------------------------------


class TestUnclearPath:
    def test_unclear_without_context_asks_for_clarification(self, kb_retriever):
        llm = SpyLLM()
        laya = SpyLaya(LayaDecision(INTENT_UNCLEAR, 0.0, "fallback"))
        p, _, _, _ = make_planner(laya=laya, llm=llm, retriever=kb_retriever)

        result = p.execute(system_prompt="S",
                           user_input="Can you help me with this?",
                           context={})

        assert result.intent == "unclear"
        assert result.response == CLARIFICATION_MESSAGE
        # No ungrounded generation is spent on an empty request.
        assert llm.calls == []
        assert result.classification_source == "fallback"

    def test_unclear_with_issue_context_gets_safe_explanation(self, kb_retriever):
        llm = SpyLLM(response="Based on the issue ...")
        laya = SpyLaya(LayaDecision(INTENT_UNCLEAR, 0.2, "fallback"))
        p, _, _, _ = make_planner(laya=laya, llm=llm, retriever=kb_retriever)

        result = p.execute(system_prompt="S",
                           user_input="Can you help me with this?",
                           context=ISSUE_001)

        assert result.intent == "unclear"
        assert result.classification_source == "fallback"
        assert llm.calls, "issue context supports a safe explanation"
        system_prompt = llm.calls[0][0]
        assert "[Intent Uncertain]" in system_prompt
        # explanation-only behavior is preserved
        assert result.tool_used is None
        assert result.execution_status is None

    def test_classifier_exception_degrades_conservatively(self, kb_retriever):
        laya = SpyLaya(error=RuntimeError("laya checkpoint missing"))
        p, llm, _, _ = make_planner(laya=laya, retriever=kb_retriever)

        result = p.execute(system_prompt="S",
                           user_input="Explain this.",
                           context={})
        assert laya.calls == 1
        assert result.classification_source == "fallback"
        assert result.intent == "unclear"
        assert result.response == CLARIFICATION_MESSAGE  # no LLM call
        assert llm.calls == []


# ---------------------------------------------------------------------------
# Pipeline mechanics
# ---------------------------------------------------------------------------


class TestPipelineMechanics:
    def test_classification_happens_exactly_once(self, kb_retriever):
        for decision, ctx in [
            (LayaDecision(INTENT_EXPLAIN, 0.9, "laya"), ISSUE_001),
            (LayaDecision(INTENT_ACTION, 0.9, "laya"), ISSUE_001),
            (LayaDecision(INTENT_UNCLEAR, 0.1, "fallback"), {}),
        ]:
            laya = SpyLaya(decision)
            p, _, _, _ = make_planner(laya=laya, retriever=kb_retriever)
            p.execute(system_prompt="S", user_input="?", context=ctx)
            assert laya.calls == 1

    def test_str_context_is_handled(self, kb_retriever):
        p, llm, _, _ = make_planner(retriever=kb_retriever)
        result = p.execute(system_prompt="S",
                           user_input="What does this mean?",
                           context="rule_id:JSON_SCHEMA_VALIDATION_ERROR message:bad")
        assert result.intent == "explain"
        assert llm.calls

    def test_malformed_context_is_handled(self, kb_retriever):
        p, llm, _, _ = make_planner(retriever=kb_retriever)
        result = p.execute(system_prompt="S", user_input="Explain.",
                           context=["not", "a", "dict"])
        assert result.intent == "explain"
        assert llm.calls

    def test_history_travels_as_chat_turns_and_as_a_section(self, kb_retriever):
        """In both places, and the same turns both times.

        The chat turns are what keeps a re-asked question from reading
        as a fresh start ("give me the error", with the error in the
        context the whole time); the rendered section keeps what was
        said in view while the answer is composed, for models that
        weight system text above their own earlier messages. Duplicating
        the text is harmless — the two drifting apart would not be, so
        the assertion is that the section shows exactly the turns sent.
        """
        p, llm, _, _ = make_planner(retriever=kb_retriever)
        history = [{"role": "user", "content": "Earlier question?"},
                   {"role": "assistant", "content": "Earlier answer."}]
        p.execute(system_prompt="S", user_input="And now?",
                  context=ISSUE_001, history=history)
        system_prompt, _, sent = llm.calls[0]
        assert sent == history
        assert "[Conversation History" in system_prompt
        for turn in history:
            assert (f"{turn['role'].capitalize()}: {turn['content']}"
                    in system_prompt)

    def test_history_omitted_when_absent(self, kb_retriever):
        p, llm, _, _ = make_planner(retriever=kb_retriever)
        p.execute(system_prompt="S", user_input="Explain.", context=ISSUE_001)
        assert llm.calls[0][2] is None
        assert "[Conversation History" not in llm.calls[0][0]

    def test_dataset_evidence_gets_its_own_sections(self, kb_retriever):
        """The tree and the file digest are sections, not context.

        Both arrive as their own fields of ``POST /predict`` and leave
        as their own sections, so a model can tell "here is the file"
        from "here is the finding". Absent fields cost no lines: a
        finding with no dataset behind it must not print empty headers.
        """
        p, llm, _, _ = make_planner(retriever=kb_retriever)
        p.execute(
            system_prompt="S", user_input="Explain.", context=ISSUE_001,
            dataset_structure="my_dataset/\n├── participants.tsv",
            file_content="TSV table: 3 columns, 12 data rows",
        )
        system_prompt = llm.calls[0][0]
        assert "[Dataset Structure" in system_prompt
        assert "my_dataset/" in system_prompt
        assert "[File Content" in system_prompt
        assert "12 data rows" in system_prompt

        # …and the same request without them says nothing about them.
        p, llm, _, _ = make_planner(retriever=kb_retriever)
        p.execute(system_prompt="S", user_input="Explain.", context=ISSUE_001)
        bare = llm.calls[0][0]
        assert "[Dataset Structure" not in bare
        assert "[File Content" not in bare

    def test_registry_execute_raises_when_nothing_registered(self):
        # The kept-for-future abstraction never pretends to work.
        reg = ToolRegistry()
        with pytest.raises(ToolExecutionUnavailable):
            reg.execute("validate_dataset")
