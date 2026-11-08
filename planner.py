"""
Planner — routes one request through Laya, retrieval, and the LLM.

This is the explanation-only phase of the agent:

* Intent handling happens exactly once per request. By default
  (``laya.classify`` is ``False`` for now) every request is routed as
  ``explain`` without consulting Laya, and the API reports
  ``classification_source: "manual"``. When ``laya.classify`` is
  enabled, Laya decides once per request (see
  :class:`laya_decision.LayaDecisionEngine`) and the resulting decision
  is carried through the pipeline and reported back to the API
  unchanged — nothing is re-classified later.
* ``explain`` requests produce a knowledge-grounded explanation built
  from the issue data, the user's question, and the retrieved records.
* ``fix`` (action-request) intents are served by an explanation that
  explicitly discloses that direct execution is not available in this
  version. No tool is invoked and nothing is simulated:
  ``tool_used`` is always ``None`` and ``execution_status`` is
  ``"not_available"``.
* ``unclear`` intents are handled conservatively: a clarification
  question when no issue context is present, or a safe explanation-only
  answer when the surrounding issue context clearly supports it.

The :class:`tool_registry.ToolRegistry` abstraction is kept for a future
tool-execution phase; the planner never dispatches tools in this phase.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import CONFIG
from laya_decision import (
    INTENT_EXPLAIN,
    INTENT_UNCLEAR,
    LayaDecision,
    LayaDecisionEngine,
)
from retriever import Retriever, RetrievalResult
from tool_registry import ToolRegistry
from utils import compact_json

logger = logging.getLogger(__name__)

# Truthful execution status reported for action requests in this phase.
EXECUTION_NOT_AVAILABLE = "not_available"

# Keys that mark a context as carrying a concrete BIDS issue. Used only to
# decide whether "unclear" can safely fall back to an explanation (context
# clearly supports that choice) — never as an intent classifier.
_ISSUE_KEYS = ("rule_id", "message", "severity", "field", "path", "suffix", "issues")

# ---------------------------------------------------------------------------
# Deterministic response checks (output validation stage).
#
# The capability boundary of this agent is absolute: it never executes
# tools and never modifies a dataset. These patterns flag a generated
# answer that *claims* otherwise. They are deliberately conservative —
# disclaimers such as "I cannot apply changes" or "I cannot modify your
# dataset" must never match — and they only *record* the finding in the
# diagnostics: the answer is never silently rewritten.
# ---------------------------------------------------------------------------
_EXECUTION_CLAIM_PATTERNS = (
    (re.compile(
        r"\b(?:i\s+(?:have\s+|already\s+)?|i'?ve\s+(?:already\s+)?)"
        r"(?:fixed|repaired|renamed|modified|updated|corrected|patched)\b",
        re.IGNORECASE,
    ), "claims to have changed the dataset"),
    (re.compile(r"\bi\s+(?:have\s+|already\s+)?ran\b", re.IGNORECASE),
     "claims to have run something"),
    (re.compile(
        r"\bsuccessfully\s+(?:fixed|repaired|renamed|corrected|updated)\b",
        re.IGNORECASE,
    ), "claims a successful fix"),
    (re.compile(
        r"\b(?:validation|check|repair|update)s?\s+"
        r"(?:completed|finished)\s+successfully\b",
        re.IGNORECASE,
    ), "claims a successful validation run"),
    (re.compile(
        r"\b(?:the|your|this)\s+(?:issue|error|problem)\s+"
        r"(?:has been|was|is now)\s+(?:fixed|resolved|corrected|updated)\b",
        re.IGNORECASE,
    ), "claims the issue was already fixed"),
    (re.compile(
        r"\bi\s+(?:have\s+)?(?:applied|made)\s+(?:the\s+)?"
        r"(?:fix|change|correction|update)s?\b",
        re.IGNORECASE,
    ), "claims to have applied a fix"),
    (re.compile(
        r"\b(?:the\s+)?(?:tool|script|validator)\s+"
        r"(?:ran|executed|completed)\b",
        re.IGNORECASE,
    ), "claims a tool/validator ran"),
)


def _response_checks(response: str) -> Dict[str, Any]:
    """Deterministic post-generation checks; flags only, never rewrites."""
    text = response or ""
    claims = [
        label
        for pattern, label in _EXECUTION_CLAIM_PATTERNS
        if pattern.search(text)
    ]
    return {
        "empty_response": not text.strip(),
        "execution_claims": claims,
    }


def _retrieval_diagnostics(
    retrieval: Optional[RetrievalResult],
) -> Dict[str, Any]:
    """Why each record was kept or rejected — internal, structured."""
    if retrieval is None:
        return {"performed": False}
    return {
        "performed": True,
        "found": retrieval.found,
        "quality": retrieval.quality,
        "evidence": [a.as_dict() for a in retrieval.assessments],
        "rejected": [a.as_dict() for a in retrieval.rejected],
    }


@dataclass
class PlannerResult:
    """Everything the API needs to report what actually happened."""

    response: str
    intent: str  # the intent actually used: "explain" | "fix" | "unclear"
    # No tools execute in the explanation-only phase, so this is always
    # None. The field is kept for client compatibility.
    tool_used: Optional[str] = None
    # "not_available" when the user requested an action, else None.
    execution_status: Optional[str] = None
    # "laya" when Laya made the decision, "fallback" when the conservative
    # unclear result was used (Laya unavailable / unconfident / invalid),
    # "manual" when classification is disabled (laya.classify is False)
    # and every request is routed as explain without consulting the model.
    classification_source: str = "laya"
    retrieval_found: Optional[bool] = None
    # "rule_exact" | "field_exact" | "lexical" | None (no retrieval done).
    retrieval_quality: Optional[str] = None
    # Structured, internal-only reasoning trace (never part of the
    # user-facing text): per-record retrieval assessments and rejection
    # reasons, which evidence sections were present, and the
    # deterministic checks run on the generated answer. Exposed by the
    # API under ``diagnostics`` so a reviewer can see *why* the answer
    # looks the way it does.
    diagnostics: Optional[Dict[str, Any]] = None


# Deterministic clarification for an ambiguous request with no issue
# context to stand on. No LLM call is spent on it: there is nothing
# grounded to generate yet.
CLARIFICATION_MESSAGE = (
    "I could not tell what you would like me to do. Please send a specific "
    "BIDS issue or error together with your question — for example "
    "'What does this error mean?' — and I will explain it."
)

# Appended to the quarantine block of records that were retrieved but
# ruled out by their own applicability selectors.
_REJECTION_INSTRUCTION = (
    "These records were retrieved by keyword overlap, but their own "
    "applicability conditions conflict with this issue's file. Do not use "
    "them to explain this issue; mention them, if at all, only to say why "
    "they do not apply."
)


def _retrieval_note(retrieval: RetrievalResult) -> str:
    """How the model must treat each relevance class of the evidence.

    Rendered right after the records themselves so the classification is
    acted on, not just displayed: background records may never become the
    issue's definition, and an absence of direct records must be stated
    rather than papered over with the closest matching document.
    """
    direct = [a for a in retrieval.assessments if a.relevance == "direct"]
    lines = ["[Retrieval Note]"]
    if direct:
        lines.append(
            "- Records marked \"Directly relevant\" document this issue's "
            "rule or the exact constraint the message refers to; use them "
            "to support statements about this issue."
        )
    else:
        lines.append(
            "- None of the records above is marked \"Directly relevant\": "
            "the knowledge base does not directly document this issue's "
            "rule. State that explicitly and explain only what the issue "
            "details and file content show — do not present a background "
            "record as the definition, cause, or rule behind this issue."
        )
    lines.append(
        "- Records marked \"Related background\" are context only (field "
        "definitions, general file rules, examples). Do not apply their "
        "file conditions (suffix, extension, selectors) to this issue, "
        "and do not merge their behavior into this issue's explanation."
    )
    if retrieval.quality == "lexical":
        lines.append(
            "- The best match only shares vocabulary with the issue: treat "
            "every record above as weak background at most."
        )
    return "\n".join(lines)


class Planner:
    def __init__(
        self,
        llm: Any,
        retriever: Retriever,
        tools: Optional[ToolRegistry] = None,
        laya: Optional[LayaDecisionEngine] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        # Kept for the future tool-execution phase; never dispatched here.
        self.tools = tools if tools is not None else ToolRegistry()
        # The Laya engine loads its checkpoint lazily, so default
        # construction is cheap; unit tests inject a mock instead.
        self.laya = laya if laya is not None else LayaDecisionEngine()

    # ------------------------------------------------------------------
    # Entry point — one classification per request, passed through
    # ------------------------------------------------------------------

    def execute(
        self,
        system_prompt: str,
        user_input: str,
        context: dict,
        history: Optional[List[Dict[str, str]]] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> PlannerResult:
        str_context = (
            compact_json(context) if isinstance(context, dict)
            else str(context or "")
        )
        if not CONFIG.laya.classify:
            # Explain-only phase (laya.classify is off): every request is
            # served as 'explain' and the model is never consulted. The
            # source keeps this honest — this decision did not come from
            # Laya. Flip laya.classify back on to restore classification.
            decision = LayaDecision(INTENT_EXPLAIN, 0.0, "manual")
        else:
            try:
                decision = self.laya.classify_intent(user_input, str_context)
            except Exception:
                # :class:`LayaDecisionEngine` already degrades internally, but
                # the planner must stay honest and robust for ANY classifier
                # implementation: a failure yields the conservative unclear
                # result, never a guessed routing.
                logger.exception("Intent classification failed; using 'unclear'.")
                decision = LayaDecision(INTENT_UNCLEAR, 0.0, "fallback")
        intent = decision.intent

        if intent == "fix":
            return self._action_unavailable(
                system_prompt, user_input, context, str_context,
                decision, history, dataset_structure, file_content,
            )

        if intent == "unclear" and not self._has_issue_context(context):
            # No issue to stand an explanation on: ask, cheaply and
            # deterministically, instead of generating an ungrounded answer.
            return PlannerResult(
                response=CLARIFICATION_MESSAGE,
                intent="unclear",
                classification_source=decision.source,
            )

        # "explain" — or "unclear" with issue context, where the context
        # clearly supports a safe explanation-only answer.
        return self._explain(
            system_prompt, user_input, context, str_context,
            decision, history, dataset_structure, file_content,
            unclear=(intent == "unclear"),
        )

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    def _explain(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Any,
        str_context: str,
        decision: LayaDecision,
        history: Optional[List[Dict[str, str]]],
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
        unclear: bool = False,
    ) -> PlannerResult:
        retrieval = self._retrieve(context, user_prompt)
        built = self.build_system_prompt(
            system_prompt=system_prompt,
            context=str_context,
            retrieval=retrieval,
            action_unavailable=False,
            unclear=unclear,
            history=history,
            dataset_structure=dataset_structure,
            file_content=file_content,
        )
        # …and as chat turns as well, which is what keeps a re-asked
        # question reading as a second question: see ``LLM._generate``.
        response = self.llm.generate_response(
            system_prompt=built, user_prompt=user_prompt, history=history,
        )
        return PlannerResult(
            response=response,
            intent="explain" if not unclear else "unclear",
            classification_source=decision.source,
            retrieval_found=retrieval.found if retrieval else None,
            retrieval_quality=retrieval.quality if retrieval else None,
            diagnostics=self._diagnostics(retrieval, response, history,
                                          dataset_structure, file_content),
        )

    def _action_unavailable(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Any,
        str_context: str,
        decision: LayaDecision,
        history: Optional[List[Dict[str, str]]],
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> PlannerResult:
        """Serve an action request with an honest, grounded explanation.

        No tool is selected, dispatched, or simulated. The response must
        still explain the issue and give safe manual guidance when the
        evidence supports it, and must state that execution is
        unavailable in the current version.
        """
        retrieval = self._retrieve(context, user_prompt)
        built = self.build_system_prompt(
            system_prompt=system_prompt,
            context=str_context,
            retrieval=retrieval,
            action_unavailable=True,
            unclear=False,
            history=history,
            dataset_structure=dataset_structure,
            file_content=file_content,
        )
        response = self.llm.generate_response(
            system_prompt=built, user_prompt=user_prompt, history=history,
        )
        return PlannerResult(
            response=response,
            intent="fix",
            tool_used=None,
            execution_status=EXECUTION_NOT_AVAILABLE,
            classification_source=decision.source,
            retrieval_found=retrieval.found if retrieval else None,
            retrieval_quality=retrieval.quality if retrieval else None,
            diagnostics=self._diagnostics(retrieval, response, history,
                                          dataset_structure, file_content),
        )

    # ------------------------------------------------------------------
    # Diagnostics (internal reasoning trace, never user-facing text)
    # ------------------------------------------------------------------

    @staticmethod
    def _diagnostics(
        retrieval: Optional[RetrievalResult],
        response: str,
        history: Optional[List[Dict[str, str]]],
        dataset_structure: Optional[str],
        file_content: Optional[str],
    ) -> Dict[str, Any]:
        """Structured record of *why* the answer looks the way it does.

        Contains record ids, relevance/applicability decisions and
        rejection reasons — never the dataset's own content (no file
        digests, no trees, no history), so it is safe to log or return.
        """
        return {
            "retrieval": _retrieval_diagnostics(retrieval),
            "evidence": {
                "history": bool(history),
                "dataset_structure": bool(dataset_structure),
                "file_content": bool(file_content),
            },
            "response_checks": _response_checks(response),
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _has_issue_context(context: Any) -> bool:
        if not isinstance(context, dict):
            return False
        return any(
            context.get(k) not in (None, "", [], {}) for k in _ISSUE_KEYS
        )

    def _retrieve(
        self, context: Any, user_prompt: str
    ) -> Optional[RetrievalResult]:
        """Retrieve issue knowledge explicitly; a miss stays a miss.

        Returns None when there is nothing sensible to retrieve (no
        context and an empty question), so callers can distinguish
        "not retrieved" from "retrieved and missed".
        """
        if isinstance(context, dict) and self._has_issue_context(context):
            return self.retriever.retrieve_issue_structured(
                context, user_question=user_prompt
            )
        question = (
            (str(context or "") + " " if not isinstance(context, dict) else "")
            + (user_prompt or "")
        ).strip()
        if not question:
            return None
        return self.retriever.retrieve_structured(question)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        system_prompt: str,
        context: str,
        retrieval: Optional[RetrievalResult],
        action_unavailable: bool = False,
        unclear: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> str:
        """Assemble the system half of the prompt.

        The system instructions come first and define the capability
        boundary; the issue, the dataset around it and the retrieved
        records are then presented as *data* the model may not treat as
        instructions.

        The conversation appears twice on purpose: as real chat turns
        (see ``LLM._generate``) *and* as a section here. The chat turns
        are what keeps a re-asked question from reading as a fresh
        start; the section keeps what was said in view while the answer
        is composed, for models that weight system text more heavily
        than their own earlier messages. Both render the same turns, so
        the prompt builder must not invent a second version of them.

        ``dataset_structure`` and ``file_content`` are pre-compiled by
        the client — a condensed tree and a file digest — and are left
        out entirely when absent, so a finding with no dataset behind
        it costs no lines here.
        """
        parts: List[str] = [f"System: {system_prompt}"]

        if action_unavailable:
            parts.append(
                "[Action Requested — Execution Not Available]\n"
                "The user asked the agent to perform an operation (fix, "
                "repair, rename, validate-and-correct, ...). The current "
                "version cannot execute tools and cannot modify, repair, "
                "validate, or rename anything in the user's dataset. "
                "1) State clearly and early that direct execution is not "
                "available in this version. 2) Do not claim that any tool "
                "ran, any file was read, or any change was applied. "
                "3) Still explain the issue as thoroughly as the evidence "
                "allows and give concrete manual steps the user can "
                "perform themselves, distinguishing what the retrieved "
                "documentation supports from what is only an inference. "
                "4) If the evidence is insufficient for a fix, say so."
            )

        if unclear:
            parts.append(
                "[Intent Uncertain]\n"
                "The classifier could not confidently determine whether the "
                "user wants an explanation or an action. Answer helpfully "
                "as an explanation-only assistant based on the provided "
                "context, and end with one short clarifying question if "
                "the user's goal is genuinely ambiguous."
            )

        parts.append(f"[Issue / Application Context]\n{context}")

        if dataset_structure:
            parts.append(
                "[Dataset Structure — condensed tree]\n"
                "How the dataset around this finding is laid out. A "
                "summary: folders and files that are not drawn are "
                "still there.\n"
                f"{dataset_structure}"
            )

        if file_content:
            parts.append(
                "[File Content — compiled digest]\n"
                "What is in the file the finding is about, compiled "
                "rather than copied whole.\n"
                f"{file_content}"
            )

        if retrieval is None:
            parts.append(
                "[Retrieved BIDS Knowledge]\n"
                "No retrieval was performed (no issue context provided)."
            )
        elif not retrieval.found:
            section = (
                "[Retrieved BIDS Knowledge]\n"
                "No relevant knowledge-base record was found for this "
                "issue. Do not invent documentation, rules, or citations. "
                "Explain only what the issue details themselves show, and "
                "say explicitly what could not be determined from the "
                "available knowledge."
            )
            if retrieval.rejection_text:
                section += (
                    "\n\n" + retrieval.rejection_text + "\n"
                    + _REJECTION_INSTRUCTION
                )
            parts.append(section)
        else:
            parts.append(
                "[Retrieved BIDS Knowledge]\n"
                f"{retrieval.text}\n\n{_retrieval_note(retrieval)}"
            )
            if retrieval.rejection_text:
                parts.append(
                    "[Retrieved Records — Not Applicable]\n"
                    f"{retrieval.rejection_text}\n{_REJECTION_INSTRUCTION}"
                )

        if history:
            # The same turns that travel as chat messages, rendered
            # here as well: see the docstring above.
            rendered = "\n\n".join(
                f"{str(turn.get('role') or '').capitalize()}: "
                f"{turn.get('content')}"
                for turn in history
            )
            parts.append(
                "[Conversation History — earlier turns of this same "
                "conversation]\n"
                "Repeated from the chat messages; it is one "
                "conversation, not two.\n"
                f"{rendered}"
            )

        return "\n\n".join(parts)
