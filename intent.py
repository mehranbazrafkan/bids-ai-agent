"""Intent routing for the BIDS agent.

Laya was fast but consistently over-selected ``fix`` and under-selected
``explain`` on real questions.  The default classifier here is therefore a
hybrid: a high-precision deterministic guard resolves unmistakable question
and imperative forms, and genuinely ambiguous requests go to the project's
instruction-tuned LLM with a compact few-shot prompt whose only job is to
return ``EXPLAIN``, ``FIX`` or ``UNCLEAR``.  The LLM reuses the already-loaded
model, so it adds no checkpoint and no VRAM.

The legacy engines remain available through ``intent.classifier`` for
comparison, but they are no longer the default:

* ``llm``  — this module (default).
* ``nli``  — the old zero-shot cross-encoder.
* ``laya`` — the old Laya decision engine.

Operators can also bypass classification entirely with
``intent.mode = "explain"`` or ``intent.mode = "fix"``.  Forced modes never
call a classifier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import prompt_log
from config import IntentConfig

logger = logging.getLogger(__name__)

INTENT_EXPLAIN = "explain"
INTENT_FIX = "fix"
INTENT_UNCLEAR = "unclear"

VALID_INTENTS = (INTENT_EXPLAIN, INTENT_FIX, INTENT_UNCLEAR)
VALID_MODES = ("auto", INTENT_EXPLAIN, INTENT_FIX)
VALID_CLASSIFIERS = ("llm", "nli", "laya")

_LABEL_TO_INTENT = {
    "EXPLAIN": INTENT_EXPLAIN,
    "FIX": INTENT_FIX,
    "UNCLEAR": INTENT_UNCLEAR,
}


@dataclass(frozen=True)
class IntentDecision:
    """One routing decision.

    ``source`` is ``"llm"`` / ``"nli"`` / ``"laya"`` for a model decision,
    ``"guard"`` when the deterministic explain/action guard resolved the
    request, ``"configured"`` for an operator-forced mode, ``"manual"`` when
    no classifier is configured, and ``"fallback"`` when neither a model nor
    a guard decision was available.
    """

    intent: str
    confidence: float = 0.0
    source: str = "llm"
    raw: str = ""

    @property
    def is_model_decision(self) -> bool:
        return self.source in ("llm", "nli", "laya")


# A small local model can confuse "How do I fix this?" (question) with
# "Fix this for me" (action). These patterns are intentionally high
# precision: they only fire on unmistakable grammatical forms, and anything
# else falls through to the model. This guard is what makes the
# explain/action distinction reliable even when the model has a bad day.
_QUESTION_START = re.compile(
    r"^\s*(?:what|why|how|when|where|which|who|is|are|does|do|did|"
    r"should|would|will|can|could)\b",
    re.I,
)
_DIRECT_REQUEST_START = re.compile(
    r"^\s*(?:(?:please|kindly)\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:go\s+ahead\s+and\s+)?"
    r"(?:help\s+me\s+(?:to\s+)?)?"
    r"(?:fix|repair|correct|rename|validate|update|change|clean|resolve|"
    r"modify|delete|remove|move|sort\s+out|apply|add|run|execute|make)\b",
    re.I,
)
_FOLLOW_UP_ACTION = re.compile(
    r"^\s*(?:(?:please|kindly)\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:do|perform|apply)\s+(?:it|that|this|so|them)\b",
    re.I,
)
_NEGATED_ACTION = re.compile(
    r"\b(?:do\s+not|don't|never)\s+"
    r"(?:fix|repair|correct|rename|validate|update|change|clean|resolve|"
    r"modify|delete|remove|move|sort\s+out|apply|add|run|execute)\b",
    re.I,
)
_FIX_OBJECT = re.compile(
    r"\b(?:fix|repair|correct|rename|validate|update|change|clean\s+up|"
    r"resolve|modify|delete|remove|move|sort\s+out|apply|add)\s+"
    r"(?:it|this|that|these|those|my|the|all|every|a|an)\b",
    re.I,
)
_INSTRUCTION = re.compile(
    r"\b(?:tell\s+me|show\s+me(?:\s+how)?|explain\w*|interpret|describe|"
    r"understand|summarize|summary|clarify|background|steps|"
    r"walk\s+me\s+through|"
    r"what\s+does\s+.*\s+mean|what\s+went\s+wrong|possible\s+causes|"
    r"recommended|why|how\s+(?:do|can|should|would)|"
    r"what\s+(?:is|are|does|happened)|which\s+bids\s+rule|"
    r"where\s+in\s+the\s+bids|is\s+this|are\s+these)\b",
    re.I,
)
_FIX_AND_EXPLAIN = re.compile(
    r"\b(?:explain|describe|tell\s+me\s+about)\b[^.!?]*\b(?:and|then)\b"
    r"[^.!?]*\bfix\b"
    r"|\bfix\b[^.!?]*\b(?:and|then)\b[^.!?]*\b(?:explain|describe)\b",
    re.I,
)
_GREETING_OR_VAGUE = re.compile(
    r"^\s*(?:hi(?:\s+there)?|hello|hey|thanks|thank\s+you|ok|okay|hmm+|"
    r"ping|any\s+ideas\??|what\s+do\s+you\s+think\??|"
    r"can\s+you\s+(?:help\s+me(?:\s+with\s+(?:this|that|it))?|"
    r"take\s+a\s+look)\??|something\s+is\s+wrong\s+with\s+my\s+data|"
    r"i'?m\s+not\s+sure\s+what\s+i'?m\s+looking\s+for\s+here|"
    r"this\s+probably\s+needs\s+attention\s+sometime|"
    r"tell\s+me\s+something\s+about\s+bids.*)\s*[.!?]*$",
    re.I,
)


def rule_intent(text: Any) -> Optional[IntentDecision]:
    """High-confidence deterministic routing for the explain/action split.

    Returns ``None`` whenever the request is not an unmistakable question,
    instruction request, imperative action, greeting or vague opener. Those
    requests go to the configured model classifier instead.
    """
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return IntentDecision(INTENT_UNCLEAR, 1.0, "guard", "empty request")
    if _GREETING_OR_VAGUE.match(normalized):
        return IntentDecision(INTENT_UNCLEAR, 1.0, "guard", "greeting/vague")
    if _NEGATED_ACTION.search(normalized):
        return IntentDecision(INTENT_EXPLAIN, 1.0, "guard", "negated action")
    if _FIX_AND_EXPLAIN.search(normalized):
        return IntentDecision(INTENT_FIX, 1.0, "guard", "explicit action")
    if _FOLLOW_UP_ACTION.match(normalized):
        return IntentDecision(INTENT_FIX, 1.0, "guard", "follow-up action")
    if _DIRECT_REQUEST_START.match(normalized):
        return IntentDecision(INTENT_FIX, 1.0, "guard", "direct action request")
    if _QUESTION_START.match(normalized):
        return IntentDecision(INTENT_EXPLAIN, 1.0, "guard", "question")
    if _INSTRUCTION.search(normalized):
        return IntentDecision(INTENT_EXPLAIN, 1.0, "guard", "explanation request")
    if _FIX_OBJECT.search(normalized):
        return IntentDecision(INTENT_FIX, 1.0, "guard", "action with object")
    return None


def normalize_mode(mode: Any) -> str:
    value = str(mode or "auto").strip().lower()
    return value if value in VALID_MODES else "auto"


def forced_decision(mode: str) -> IntentDecision:
    """Return the decision for ``intent.mode=explain|fix``."""
    return IntentDecision(normalize_mode(mode), 1.0, "configured", "")


SYSTEM_PROMPT = """\
You are the intent router for a BIDS dataset assistant.
The assistant can explain BIDS issues and describe fixes, but it cannot
execute changes on the user's files.

Classify the USER REQUEST into exactly one label:

EXPLAIN
  The user wants to understand, diagnose, interpret, or receive instructions.
  Questions such as "What does this error mean?", "Why did this fail?",
  "Is this a problem?", and "How do I fix this?" are EXPLAIN.

FIX
  The user asks the assistant to perform the repair or change itself, for
  example "Fix it for me", "Rename all subjects", or "Validate and correct my
  dataset".

UNCLEAR
  A greeting, thanks, vague text, or no actionable BIDS request.

Classify what the user asks the assistant to do, not what the validation
error itself says. Return exactly one word: EXPLAIN, FIX, or UNCLEAR.
"""

_FEW_SHOT = """\
Examples:
User: What does this error mean?
Label: EXPLAIN

User: Why is the License field invalid?
Label: EXPLAIN

User: How can I fix this?
Label: EXPLAIN

User: Can you fix this for me?
Label: FIX

User: Rename every subject folder to sub-XX.
Label: FIX

User: Validate my dataset and correct the errors.
Label: FIX

User: Hello
Label: UNCLEAR

User: Thanks, that helps.
Label: UNCLEAR
"""


def parse_label(raw: str) -> Optional[str]:
    """Parse a classifier answer into one intent label, or ``None``.

    A response that names more than one label is not a decision; it is
    deliberately rejected instead of taking the first word (a model saying
    "EXPLAIN or FIX" must not silently become one of them).  A single label
    may be wrapped in punctuation, backticks or a short explanation.
    """
    text = str(raw or "").strip().upper()
    if not text:
        return None
    labels = re.findall(r"\b(EXPLAIN|FIX|UNCLEAR)\b", text)
    unique = set(labels)
    if len(unique) != 1:
        return None
    return _LABEL_TO_INTENT[unique.pop()]


class LLMIntentClassifier:
    """Hybrid router: deterministic guard first, LLM for ambiguous requests."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def _user_prompt(self, user_input: str) -> str:
        parts = [
            _FEW_SHOT.rstrip(),
            "Now classify this request.",
            f"User: {str(user_input or '').strip()}",
            "Label:",
        ]
        return "\n\n".join(parts)

    def classify_intent(self, user_input: str) -> IntentDecision:
        # The intent decision is made from the user's request alone. Issue
        # metadata and conversation history are deliberately withheld: they
        # belong to answering, and context contamination changes routing.
        if not str(user_input or "").strip():
            return IntentDecision(INTENT_UNCLEAR, 0.0, "fallback", "")
        # The guard is high precision and resolves the forms a small local
        # model still confuses ("How can I fix this?" vs "Fix it for me").
        # Only ambiguous paraphrases pay for a model call.
        guarded = rule_intent(user_input)
        if guarded is not None:
            return guarded
        try:
            with prompt_log.suspended():
                raw = self.llm.classify(
                    SYSTEM_PROMPT,
                    self._user_prompt(user_input),
                    max_new_tokens=8,
                )
        except Exception:
            logger.exception("LLM intent classification failed; using unclear.")
            return IntentDecision(INTENT_UNCLEAR, 0.0, "fallback", "")
        label = parse_label(raw)
        if label is None:
            logger.warning("Unparseable intent label from LLM: %r", raw)
            return IntentDecision(INTENT_UNCLEAR, 0.0, "fallback", str(raw or ""))
        return IntentDecision(label, 0.9, "llm", str(raw or ""))


class NLIAdapter:
    """Adapter for the legacy zero-shot cross-encoder classifier."""

    def __init__(self, classifier: Optional[Any] = None) -> None:
        if classifier is None:
            from intent_classifier import IntentClassifier

            classifier = IntentClassifier()
        self.classifier = classifier

    def classify_intent(self, user_input: str) -> IntentDecision:
        # Issue context and history are intentionally not part of intent
        # classification: this adapter sees the user's request only.
        try:
            label = self.classifier.classify(user_input)
        except Exception:
            logger.exception("NLI intent classification failed; using unclear.")
            return IntentDecision(INTENT_UNCLEAR, 0.0, "fallback", "")
        if label == INTENT_FIX:
            return IntentDecision(INTENT_FIX, 0.5, "nli", str(label))
        if label == INTENT_EXPLAIN:
            return IntentDecision(INTENT_EXPLAIN, 0.5, "nli", str(label))
        return IntentDecision(INTENT_UNCLEAR, 0.0, "fallback", str(label or ""))


def build_intent_classifier(config: IntentConfig, llm: Any = None) -> Optional[Any]:
    """Build the auto-mode classifier selected by ``config.classifier``.

    Forced modes return ``None``: the planner reports ``configured`` and must
    not load or call any model.
    """
    mode = normalize_mode(config.mode)
    if mode != "auto":
        return None
    kind = str(config.classifier or "llm").strip().lower()
    if kind == "laya":
        from laya_decision import LayaDecisionEngine

        return LayaDecisionEngine()
    if kind == "nli":
        return NLIAdapter()
    if kind != "llm":
        logger.warning("Unknown intent.classifier %r; using the LLM.", kind)
    if llm is None:
        logger.warning(
            "intent.classifier='llm' needs the chat model; "
            "falling back to manual explanation routing."
        )
        return None
    return LLMIntentClassifier(llm)
