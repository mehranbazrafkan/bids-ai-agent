"""Tests for the LLM intent router and the forced explain/fix modes."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage

from config import CONFIG, RetrievalConfig
from intent import (
    INTENT_EXPLAIN,
    INTENT_FIX,
    INTENT_UNCLEAR,
    IntentDecision,
    LLMIntentClassifier,
    parse_label,
    rule_intent,
)
from planner import Planner
from retriever import Retriever

KB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge-base",
)

ISSUE = {
    "severity": "error",
    "rule_id": "LICENSE_INVALID",
    "message": "License is invalid",
    "field": "License",
}


class FakeClassifierLLM:
    def __init__(self, response="EXPLAIN"):
        self.response = response
        self.calls = []

    def classify(self, system_prompt, user_prompt, max_new_tokens=8):
        self.calls.append((system_prompt, user_prompt, max_new_tokens))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SpyClassifier:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def classify_intent(self, user_input):
        self.calls += 1
        return self.decision


class SpyChatModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        return AIMessage(content="Grounded explanation.")


@pytest.fixture()
def retriever():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


@pytest.fixture(autouse=True)
def legacy_laya_off(monkeypatch):
    monkeypatch.setattr(CONFIG.laya, "classify", False)
    monkeypatch.setattr(CONFIG.intent, "mode", "auto")


def test_parse_label_accepts_wrapped_labels_and_rejects_ambiguity():
    assert parse_label("EXPLAIN") == INTENT_EXPLAIN
    assert parse_label("**FIX**") == INTENT_FIX
    assert parse_label("The request is UNCLEAR.") == INTENT_UNCLEAR
    assert parse_label("EXPLAIN or FIX") is None
    assert parse_label("I cannot decide") is None


def test_rule_guard_distinguishes_explanation_from_action():
    assert rule_intent("What does this error mean?").intent == INTENT_EXPLAIN
    assert rule_intent("How can I fix this?").intent == INTENT_EXPLAIN
    assert rule_intent("Is this a problem?").intent == INTENT_EXPLAIN
    assert rule_intent("Tell me the steps to rename a subject.").intent == INTENT_EXPLAIN
    assert rule_intent("Fix it for me.").intent == INTENT_FIX
    assert rule_intent("Can you fix this for me?").intent == INTENT_FIX
    assert rule_intent("Rename every subject folder to sub-XX.").intent == INTENT_FIX
    assert rule_intent("Validate my dataset and correct the errors.").intent == INTENT_FIX
    assert rule_intent("Explain this error and fix it.").intent == INTENT_FIX
    assert rule_intent("Do it.").intent == INTENT_FIX
    assert rule_intent("Can you help me fix this?").intent == INTENT_FIX
    assert rule_intent("I want to understand this error.").intent == INTENT_EXPLAIN
    assert rule_intent("Do not fix it.").intent == INTENT_EXPLAIN
    assert rule_intent("Hello").intent == INTENT_UNCLEAR
    assert rule_intent("Can you help me with this?").intent == INTENT_UNCLEAR
    assert rule_intent("The value seems off.") is None


def test_rule_guard_precedes_the_model():
    llm = FakeClassifierLLM("FIX")
    classifier = LLMIntentClassifier(llm)
    decision = classifier.classify_intent("How can I fix this?")
    assert decision.intent == INTENT_EXPLAIN
    assert decision.source == "guard"
    assert llm.calls == []


def test_rule_guard_matches_the_labelled_intent_set():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "data" / "intent_eval.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for item in items:
        decision = rule_intent(item["text"])
        assert decision is not None, item["text"]
        assert decision.intent == item["label"], (item["text"], decision)


def test_llm_classifier_returns_model_decision_and_uses_few_shot_prompt():
    llm = FakeClassifierLLM("FIX")
    classifier = LLMIntentClassifier(llm)
    decision = classifier.classify_intent("The value seems off.")
    assert decision.intent == INTENT_FIX
    assert decision.source == "llm"
    assert llm.calls
    prompt = llm.calls[0][1]
    assert "How can I fix this?" in prompt
    assert "Label: EXPLAIN" in prompt
    assert llm.calls[0][2] == 8


def test_llm_classifier_falls_back_on_bad_or_failed_output():
    classifier = LLMIntentClassifier(FakeClassifierLLM("EXPLAIN or FIX"))
    decision = classifier.classify_intent("The value seems off.")
    assert decision.intent == INTENT_UNCLEAR
    assert decision.source == "fallback"

    classifier = LLMIntentClassifier(FakeClassifierLLM(RuntimeError("model down")))
    decision = classifier.classify_intent("The value seems off.")
    assert decision.intent == INTENT_UNCLEAR
    assert decision.source == "fallback"


def test_llm_classifier_prompt_contains_only_the_current_request():
    llm = FakeClassifierLLM("FIX")
    classifier = LLMIntentClassifier(llm)
    classifier.classify_intent("Proceed with that.")

    prompt = llm.calls[0][1]
    assert "Proceed with that." in prompt
    assert "How should I fix the License?" not in prompt
    assert "Change it to a valid identifier." not in prompt


def test_forced_explain_mode_never_calls_classifier(retriever, monkeypatch):
    monkeypatch.setattr(CONFIG.intent, "mode", "explain")
    spy = SpyClassifier(IntentDecision(INTENT_FIX, 1.0, "llm"))
    p = Planner(
        llm=SpyChatModel(),
        retriever=retriever,
        intent_classifier=spy,
    )
    result = p.execute("S", "Fix this for me.", ISSUE)
    assert result.intent == INTENT_EXPLAIN
    assert result.classification_source == "configured"
    assert spy.calls == 0
    assert result.execution_status is None


def test_forced_fix_mode_never_calls_classifier(retriever, monkeypatch):
    monkeypatch.setattr(CONFIG.intent, "mode", "fix")
    spy = SpyClassifier(IntentDecision(INTENT_EXPLAIN, 1.0, "llm"))
    p = Planner(
        llm=SpyChatModel(),
        retriever=retriever,
        intent_classifier=spy,
    )
    result = p.execute("S", "What does this mean?", ISSUE)
    assert result.intent == INTENT_FIX
    assert result.classification_source == "configured"
    assert result.execution_status == "not_available"
    assert spy.calls == 0


def test_auto_mode_uses_injected_classifier_once(retriever):
    spy = SpyClassifier(IntentDecision(INTENT_FIX, 0.9, "llm"))
    p = Planner(
        llm=SpyChatModel(),
        retriever=retriever,
        intent_classifier=spy,
    )
    result = p.execute("S", "Please fix it.", ISSUE)
    assert result.intent == INTENT_FIX
    assert result.classification_source == "llm"
    assert result.execution_status == "not_available"
    assert spy.calls == 1
