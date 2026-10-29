"""
Laya-based decision engine for the BIDS AI agent.

Laya is the PRIMARY intent classifier. The LLM is used only for output
generation (explanations). In the current (explanation-only) phase the
engine decides *intent* once per request:

  * ``explain``   — the user wants to understand an issue / get guidance.
  * ``fix``       — the user asks the agent to *perform* an operation
                    (repair, rename, validate-and-correct, ...). The agent
                    currently cannot execute tools, so this intent is served
                    with an explanation plus an explicit
                    "execution is not available" disclosure.
  * ``unclear``   — either Laya itself chose the explicit ``unclear``
                    criterion (vague / greeting / no concrete target), or
                    Laya could not produce a usable choice (weak support,
                    invalid label, malformed response, or Laya failed). The
                    planner handles this conservatively (clarification, or a
                    safe explanation when the surrounding issue context
                    clearly supports it).

The question offered to Laya is a three-way choice (``explain`` / ``fix`` /
``unclear``), so Laya is the classifier for *all three* intents. An
``explain`` or ``fix`` choice is accepted only when the model's own
probability for that class reaches ``confidence_threshold`` (default 0.5 =
the class is the model's clear preference); an ``unclear`` choice is always
accepted because it is the safe outcome. Anything else degrades to
``unclear`` with ``source="fallback"``.

The acceptance gate uses ``probabilities[choice]`` rather than the answer's
``confidence`` field: in the installed package ``confidence`` is the
entropy score ``1 - H(p)/log(k)`` (see ``laya.common.confidence_from_probs``)
and the package itself warns that its shipped temperatures are invalid,
i.e. "Treat confidence ... as uncalibrated". Gating on that entropy score at
0.5 would require p >= ~0.9 for a two-way split and made ~90% of real
requests fall back (measured 0.31-0.35 accuracy on the labelled eval set
against a 0.235 always-unclear baseline); the class-probability gate scores
~0.63 on the same set. ``probabilities`` is preferred, but if a response
build ever omits it, the ``confidence`` key is used as a defensive
fallback.

The installed Laya API (verified against laya v0.3.x ``Agent.system_one``,
exposed as ``Agent.predict``) returns::

    {
      "model": "laya-rl-agent",
      "answers": {
        "<qid>": {
          "type": "choice",
          "choice": "<criterion key>",
          "probabilities": {"<key>": 0.2, ...},
          "confidence": 0.7,        # entropy-derived, uncalibrated
          "action": {"act_probability": ...}
        }
      },
      "usage": {...},
    }

Note the key is ``confidence`` — NOT ``answer_confidence``. An earlier
version of this module read ``answer_confidence`` (always missing), so the
confidence was always 0.0 and *every* request silently fell through to a
keyword fallback in the planner. Parsing here is defensive: any malformed
shape is treated as an invalid prediction, never as a guess.

No fixed keyword list is used anywhere in this module. When Laya cannot
decide, the result is explicitly ``unclear`` with ``source="fallback"`` so
callers (and the API) can tell the decision did not come from the model.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Suppress Laya's RuntimeWarning (e.g. tokenizer parallelism)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# Intent labels. ``fix`` is kept as the action-request label to preserve the
# existing API contract (clients and docs already know "explain"/"fix");
# semantically it is "action_request".
INTENT_EXPLAIN = "explain"
INTENT_ACTION = "fix"
INTENT_UNCLEAR = "unclear"

# Every intent Laya may choose, including the explicit safe outcome.
_VALID_INTENTS = (INTENT_EXPLAIN, INTENT_ACTION, INTENT_UNCLEAR)


@dataclass(frozen=True)
class LayaDecision:
    """The outcome of one intent classification.

    ``intent`` is one of ``explain`` / ``fix`` / ``unclear``.
    ``source`` is ``"laya"`` when the model produced an accepted choice
    (its own criterion key, with ``explain``/``fix`` supported by at
    least ``confidence_threshold`` class probability), and ``"fallback"``
    when the conservative unclear result was used instead. ``confidence``
    is the value the gate used: the model's probability for the chosen
    class, or 0.0 for a fallback. Callers must not present a fallback as
    a model decision.
    """

    intent: str
    confidence: float
    source: str

    @property
    def is_model_decision(self) -> bool:
        return self.source == "laya"


def _answer_confidence(answer: Any) -> float:
    """Extract the confidence from a Laya choice answer.

    Reads ``confidence`` (the installed API) and, defensively, the legacy
    ``answer_confidence`` spelling. Anything missing or non-numeric is
    treated as 0.0 (no usable confidence), and the value is clamped to
    [0, 1] because it is used against a probability-style threshold.
    """
    if not isinstance(answer, dict):
        return 0.0
    raw = answer.get("confidence", answer.get("answer_confidence", 0.0))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN guard
        return 0.0
    return min(1.0, max(0.0, value))


def _choice_confidence(answer: Any, choice: str) -> float:
    """The model's own support for the class it chose.

    Prefers ``probabilities[choice]`` — the class probability the model
    assigned to its answer, which is what the acceptance gate thresholds.
    Falls back to the answer's ``confidence`` key (entropy-derived and
    package-acknowledged as uncalibrated) only when ``probabilities`` is
    missing or unusable, so older/other response builds still produce a
    bounded number. Missing, non-numeric and NaN values all mean 0.0.
    """
    if not isinstance(answer, dict):
        return 0.0
    probs = answer.get("probabilities")
    if isinstance(probs, dict):
        raw = probs.get(choice)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None  # None => no usable class probability, try the key
        else:
            if value == value:  # not NaN
                return min(1.0, max(0.0, value))
    return _answer_confidence(answer)


class LayaDecisionEngine:
    """Wraps Laya for agent decision-making.

    The Laya checkpoint is loaded lazily on first use and then reused for
    the lifetime of this engine — the same resource strategy the LLM uses
    (lazy load, single residency). Constructing the engine is cheap, which
    keeps ``Planner``/``Agent`` construction and ordinary unit tests free
    of model downloads; unit tests inject a mock engine instead.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        # Imported here (not at module import time) so importing this module
        # in a test environment never pulls in torch / laya.
        from config import CONFIG

        laya_cfg = CONFIG.laya
        self.model_name = model or laya_cfg.model
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else laya_cfg.confidence_threshold
        )
        self._agent: Any = None

    # ------------------------------------------------------------------
    # Lazy model access
    # ------------------------------------------------------------------

    def _ensure_agent(self) -> Any:
        if self._agent is None:
            import laya

            self._agent = laya.load(self.model_name)
        return self._agent

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, user_input: str, context: str = "") -> LayaDecision:
        """Classify the user's intent as explain / fix / unclear.

        Laya answers a three-way choice question and this method applies
        the acceptance gate: ``explain``/``fix`` only when the model's
        probability for that class reaches ``confidence_threshold``,
        ``unclear`` at any support level. Always returns a
        :class:`LayaDecision`; never raises. An accepted model choice is
        reported with ``source="laya"``; anything else yields the
        conservative ``unclear`` result with ``source="fallback"``.
        """
        state = f"{user_input}\nContext: {context}" if context else user_input

        questions: Dict[str, Dict[str, Any]] = {
            "intent": {
                "type": "choice",
                "instructions": (
                    "What does the user want to do with this issue? "
                    "Consider the complete wording of the request."
                ),
                "criteria": {
                    INTENT_EXPLAIN: (
                        "The user wants to understand, learn about, "
                        "interpret, diagnose, or get an explanation of the "
                        "issue, or wants *instructions* on how to deal with "
                        "it themselves. They are not asking the agent to "
                        "perform the operation."
                    ),
                    INTENT_ACTION: (
                        "The user asks the agent to perform an operation on "
                        "their data: fix, repair, correct, rename, validate "
                        "and correct, clean, or otherwise modify something. "
                        "A request that asks for an explanation *and* also "
                        "asks the agent to carry out the operation counts "
                        "as an action request."
                    ),
                    INTENT_UNCLEAR: (
                        "The request is vague, a greeting or social "
                        "message, lacks a concrete target, or otherwise "
                        "does not clearly ask for an explanation or for an "
                        "operation."
                    ),
                },
            }
        }

        try:
            result = self._ensure_agent().predict(state, questions)
            answers = result.get("answers") if isinstance(result, dict) else None
            answer = answers.get("intent") if isinstance(answers, dict) else None
            choice = answer.get("choice") if isinstance(answer, dict) else None

            if isinstance(choice, str) and choice.strip() in _VALID_INTENTS:
                choice = choice.strip()
            else:
                choice = None

            if choice is not None:
                # The gate: the model's probability for the class it chose.
                # ``unclear`` is accepted at any support level because it is
                # the safe outcome — accepting it can never misroute.
                confidence = _choice_confidence(answer, choice)
                if (
                    choice == INTENT_UNCLEAR
                    or confidence >= self.confidence_threshold
                ):
                    return LayaDecision(choice, confidence, "laya")
                logger.info(
                    "Laya intent choice %r lacks support (p=%.3f, "
                    "threshold=%.2f); using conservative 'unclear' fallback.",
                    choice, confidence, self.confidence_threshold,
                )
            else:
                logger.info(
                    "Laya returned no valid intent choice; using "
                    "conservative 'unclear' fallback."
                )
        except Exception:
            # A model failure must degrade to the conservative outcome, not
            # crash the request. The source field keeps this honest: this
            # decision did not come from Laya.
            logger.exception(
                "Laya intent classification failed; using 'unclear' fallback."
            )

        return LayaDecision(INTENT_UNCLEAR, 0.0, "fallback")

    # ------------------------------------------------------------------
    # Tool selection (reserved for a future tool-execution phase)
    # ------------------------------------------------------------------

    def select_tool(self, user_input: str, context: str = "") -> Optional[str]:
        """Decide which tool should handle the given issue.

        Kept for a future tool-execution phase. In the current
        explanation-only phase no tool is executed, so the planner does not
        call this. Returns the tool name, or None when no tool matches
        confidently.
        """
        state = f"{user_input}\nContext: {context}" if context else user_input

        questions: Dict[str, Dict[str, Any]] = {
            "tool": {
                "type": "choice",
                "instructions": "Which tool should be used to fix this issue?",
                "criteria": {
                    "validate": (
                        "The issue is about validating the dataset, checking "
                        "BIDS compliance, or verifying structure and metadata."
                    ),
                    "repair_metadata": (
                        "The issue is about repairing, correcting, or fixing "
                        "metadata fields, values, or JSON/TSV content."
                    ),
                    "rename_subject": (
                        "The issue is about renaming subjects, sessions, or "
                        "directories to match BIDS naming conventions."
                    ),
                },
            }
        }

        try:
            result = self._ensure_agent().predict(state, questions)
            answers = result.get("answers") if isinstance(result, dict) else None
            answer = answers.get("tool") if isinstance(answers, dict) else None
            choice = answer.get("choice") if isinstance(answer, dict) else None
            confidence = _answer_confidence(answer)

            valid = ("validate", "repair_metadata", "rename_subject")
            if (
                isinstance(choice, str)
                and choice.strip() in valid
                and confidence >= self.confidence_threshold
            ):
                return choice.strip()
        except Exception:
            logger.exception("Laya tool selection failed; returning None.")
        return None
