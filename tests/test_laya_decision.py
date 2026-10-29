"""Tests for the Laya-based intent classifier (mocked model inference).

The response fixtures mirror the *installed* laya 0.3.x
``Agent.system_one`` / ``Agent.predict`` schema exactly (verified against
site-packages/laya/agent.py):::

    {"model": "laya-rl-agent",
     "answers": {"<qid>": {"type": "choice", "choice": "<criterion key>",
                           "probabilities": {...}, "confidence": 0.7,
                           "action": {"act_probability": ...}}},
     "usage": {...}}

The historical defect: the integration read ``answer_confidence`` (a key
the installed API never returns), so confidence was always 0.0 and every
request silently fell through to a fallback. These tests pin the real key
and the surrounding failure handling.
"""

import math

import pytest

from config import CONFIG
from laya_decision import (
    INTENT_ACTION,
    INTENT_EXPLAIN,
    INTENT_UNCLEAR,
    LayaDecisionEngine,
)


# ---------------------------------------------------------------------------
# Helpers — a fake Agent whose .predict returns canned real-schema payloads
# ---------------------------------------------------------------------------


class FakeLayaAgent:
    """Stands in for ``laya.load(...)``'s Agent object."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return self.response


def choice_response(choice, confidence, qid="intent"):
    """A real-schema choice answer."""
    probs = {choice: confidence}
    if choice != INTENT_EXPLAIN:
        probs[INTENT_EXPLAIN] = round(1.0 - confidence, 4)
    else:
        probs[INTENT_ACTION] = round(1.0 - confidence, 4)
    return {
        "model": "laya-rl-agent",
        "answers": {
            qid: {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": confidence,
                "action": {"act_probability": 0.42},
            }
        },
        "usage": {"input_tokens": 51, "output_tokens": 0},
    }


def make_engine(response=None, error=None, **kwargs):
    engine = LayaDecisionEngine(**kwargs)
    engine._agent = FakeLayaAgent(response=response, error=error)
    return engine


# ---------------------------------------------------------------------------
# Parsing the installed response schema
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_parses_explain_choice_from_real_schema(self):
        engine = make_engine(choice_response(INTENT_EXPLAIN, 0.83))
        decision = engine.classify_intent("What does this error mean?")
        assert decision.intent == INTENT_EXPLAIN
        assert decision.confidence == pytest.approx(0.83)
        assert decision.source == "laya"
        assert decision.is_model_decision

    def test_parses_fix_choice_from_real_schema(self):
        engine = make_engine(choice_response(INTENT_ACTION, 0.71))
        decision = engine.classify_intent("Fix this error in my dataset.")
        assert decision.intent == INTENT_ACTION
        assert decision.confidence == pytest.approx(0.71)
        assert decision.source == "laya"

    def test_state_carries_user_input(self):
        engine = make_engine(choice_response(INTENT_EXPLAIN, 0.9))
        engine.classify_intent("Why is this field invalid?", "Context: rule_id:X")
        state, questions = engine._agent.calls[0]
        assert "Why is this field invalid?" in state
        assert "rule_id:X" in state
        # The model is asked a *semantic* choice question whose criteria
        # are the intent labels themselves — not a keyword list. All three
        # intents are offered, so Laya classifies every one of them.
        criteria = questions["intent"]["criteria"]
        assert set(criteria) == {INTENT_EXPLAIN, INTENT_ACTION, INTENT_UNCLEAR}

    def test_unclear_criterion_is_offered_to_the_model(self):
        engine = make_engine(choice_response(INTENT_UNCLEAR, 0.6))
        engine.classify_intent("Hi there")
        _, questions = engine._agent.calls[0]
        unclear_text = questions["intent"]["criteria"][INTENT_UNCLEAR].lower()
        assert "vague" in unclear_text or "greeting" in unclear_text

    def test_legacy_answer_confidence_key_still_read(self):
        """Defensive: older/other builds spelled the key differently.

        With ``probabilities`` absent the gate falls back to the
        ``confidence`` / ``answer_confidence`` key so a response build
        without class probabilities can still produce a decision.
        """
        response = {
            "model": "laya-rl-agent",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": INTENT_EXPLAIN,
                    "answer_confidence": 0.9,
                }
            },
            "usage": {},
        }
        engine = make_engine(response)
        decision = engine.classify_intent("Explain this.")
        assert decision.intent == INTENT_EXPLAIN
        assert decision.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Acceptance gate (chosen-class probability)
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_default_threshold_comes_from_config(self):
        engine = make_engine()
        assert engine.confidence_threshold == CONFIG.laya.confidence_threshold

    def test_low_support_falls_back_to_unclear(self):
        # probabilities[choice] = 0.31 < 0.5 -> no accepted model decision.
        engine = make_engine(choice_response(INTENT_ACTION, 0.31))
        decision = engine.classify_intent("Fix it.")
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "fallback"
        assert decision.confidence == 0.0

    def test_class_probability_gates_not_entropy_confidence_key(self):
        # The gate reads probabilities[choice]; the (un)calibrated
        # ``confidence`` key must not be what decides acceptance.
        response = choice_response(INTENT_EXPLAIN, 0.6)
        response["answers"]["intent"]["confidence"] = 0.05
        engine = make_engine(response)
        decision = engine.classify_intent("Explain this.")
        assert decision.intent == INTENT_EXPLAIN
        assert decision.source == "laya"
        assert decision.confidence == pytest.approx(0.6)

    def test_unclear_choice_accepted_at_low_support(self):
        # ``unclear`` is the safe outcome: accepting it can never misroute,
        # so no threshold applies to the model's own unclear choice.
        response = choice_response(INTENT_UNCLEAR, 0.2)
        engine = make_engine(response)
        decision = engine.classify_intent("Hmm.")
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "laya"
        assert decision.confidence == pytest.approx(0.2)

    def test_explain_below_threshold_is_fallback_not_model_unclear(self):
        response = choice_response(INTENT_EXPLAIN, 0.44)
        engine = make_engine(response)
        decision = engine.classify_intent("What does this error mean?")
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "fallback"

    def test_threshold_is_configurable_per_engine(self):
        # 0.4 is rejected at the default 0.5 but accepted at 0.3.
        default_engine = make_engine(choice_response(INTENT_EXPLAIN, 0.4))
        strict = default_engine.classify_intent("Explain.")
        assert strict.intent == INTENT_UNCLEAR

        custom = make_engine(choice_response(INTENT_EXPLAIN, 0.4),
                             confidence_threshold=0.3)
        relaxed = custom.classify_intent("Explain.")
        assert relaxed.intent == INTENT_EXPLAIN
        assert relaxed.source == "laya"

    def test_threshold_boundary_is_inclusive(self):
        engine = make_engine(choice_response(INTENT_EXPLAIN, 0.5),
                             confidence_threshold=0.5)
        assert engine.classify_intent("Explain.").source == "laya"

    def test_env_var_overrides_threshold(self, monkeypatch):
        monkeypatch.setenv("BIDS_AGENT_LAYA_CONFIDENCE_THRESHOLD", "0.2")
        from config import load_config
        cfg = load_config(config_file="__no_such_config__.json")
        assert cfg.laya.confidence_threshold == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Malformed / missing / invalid outputs
# ---------------------------------------------------------------------------


class TestMalformedResponses:
    @pytest.mark.parametrize("response", [
        None,                              # no result at all
        [],                                # wrong container type
        "explain",                         # wrong type
        {"answers": None},                 # answers not a dict
        {"answers": {}},                   # no answer for our question id
        {"answers": {"intent": None}},     # answer is null
        {"answers": {"intent": "explain"}},  # answer not a dict
        {"answers": {"intent": {"type": "choice"}}},   # no choice
        {"answers": {"intent": {"choice": None}}},     # null choice
        {"answers": {"intent": {"choice": 42}}},       # non-string choice
        {"answers": {"other_question": {"choice": INTENT_EXPLAIN}}},
    ])
    def test_malformed_response_yields_unclear(self, response):
        engine = make_engine(response)
        decision = engine.classify_intent("Explain this.")
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "fallback"
        assert decision.confidence == 0.0

    @pytest.mark.parametrize("choice", ["maybe", "EXPLAIN", "action_request", ""])
    def test_invalid_label_yields_unclear(self, choice):
        """Only the criterion keys actually offered to the model are valid."""
        engine = make_engine(choice_response(choice, 0.99))
        decision = engine.classify_intent("Explain this.")
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "fallback"

    @pytest.mark.parametrize("confidence", ["high", None, [0.8]])
    def test_non_numeric_confidence_treated_as_no_confidence(self, confidence):
        """No usable class probability AND no usable confidence key => fallback."""
        response = {
            "model": "laya-rl-agent",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": INTENT_EXPLAIN,
                    "confidence": confidence,
                }
            },
            "usage": {},
        }
        engine = make_engine(response)
        decision = engine.classify_intent("Explain.")
        # unusable confidence -> below any sane threshold -> fallback
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "fallback"

    def test_non_numeric_class_probability_falls_back_to_confidence_key(self):
        response = choice_response(INTENT_EXPLAIN, 0.9)
        response["answers"]["intent"]["probabilities"] = {"explain": "high"}
        engine = make_engine(response)
        decision = engine.classify_intent("Explain.")
        assert decision.intent == INTENT_EXPLAIN
        assert decision.source == "laya"
        assert decision.confidence == pytest.approx(0.9)

    def test_nan_class_probability_falls_back_to_confidence_key(self):
        response = choice_response(INTENT_EXPLAIN, 0.9)
        response["answers"]["intent"]["probabilities"] = {"explain": math.nan}
        engine = make_engine(response)
        decision = engine.classify_intent("Explain.")
        assert decision.intent == INTENT_EXPLAIN
        assert decision.confidence == pytest.approx(0.9)

    def test_nan_confidence_treated_as_no_confidence(self):
        # probabilities absent entirely: the legacy confidence key is all
        # there is, and NaN there means "no usable confidence".
        response = {
            "model": "laya-rl-agent",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": INTENT_EXPLAIN,
                    "confidence": math.nan,
                }
            },
            "usage": {},
        }
        engine = make_engine(response)
        assert engine.classify_intent("Explain.").source == "fallback"

    def test_confidence_clamped_to_unit_interval(self):
        # Defensive: a >1 value must not crash threshold comparison.
        response = choice_response(INTENT_EXPLAIN, 0.9)
        response["answers"]["intent"]["probabilities"] = {"explain": 1.7}
        engine = make_engine(response)
        decision = engine.classify_intent("Explain.")
        assert decision.confidence <= 1.0
        assert decision.source == "laya"


# ---------------------------------------------------------------------------
# Model failures
# ---------------------------------------------------------------------------


class TestModelFailure:
    @pytest.mark.parametrize("error", [
        RuntimeError("checkpoint missing"),
        OSError("disk full"),
        ValueError("question options exceed head_max_len"),
    ])
    def test_exception_degrades_to_unclear_not_crash(self, error):
        engine = make_engine(error=error)
        decision = engine.classify_intent("Explain this to me.")
        assert decision.intent == INTENT_UNCLEAR
        assert decision.source == "fallback"
        assert decision.confidence == 0.0

    def test_fallback_is_not_labeled_as_a_model_decision(self):
        engine = make_engine(error=RuntimeError("boom"))
        decision = engine.classify_intent("Fix it.")
        assert not decision.is_model_decision


# ---------------------------------------------------------------------------
# Resource behaviour
# ---------------------------------------------------------------------------


class TestResources:
    def test_engine_construction_does_not_load_the_model(self):
        # The default engine must be cheap: the checkpoint loads lazily.
        engine = LayaDecisionEngine()
        assert engine._agent is None

    def test_classification_happens_once_per_call(self):
        engine = make_engine(choice_response(INTENT_EXPLAIN, 0.9))
        engine.classify_intent("Explain this.")
        engine.classify_intent("Explain this again.")
        assert len(engine._agent.calls) == 2  # one predict per classify
