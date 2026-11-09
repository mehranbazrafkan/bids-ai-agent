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

Mechanically, the explain/fix/unclear-with-context paths run as one
small LCEL pipeline (``_build_pipeline``): a ``bids_evidence`` step
retrieves and renders the system half of the prompt, the
``BIDS_PROMPT | LLM`` branch generates the answer, and a
``bids_result`` step assembles the :class:`PlannerResult`. The
deterministic parts — classification, the clarification shortcut,
retrieval, prompt rendering, the response checks — are ordinary
functions, not model steps: LangChain orchestrates here, it does not
decide.

The :class:`tool_registry.ToolRegistry` abstraction is kept for a future
tool-execution phase; the planner never dispatches tools in this phase.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel

from answer_contract import AnswerContract, build_contract
from answer_repair import (
    deterministic_repair,
    render_grounded_template,
    render_repair_instructions,
)
from config import CONFIG, QualityConfig
from kb_retriever import KBRetriever
from laya_decision import (
    INTENT_EXPLAIN,
    INTENT_UNCLEAR,
    LayaDecision,
    LayaDecisionEngine,
)
from observability import StepLogger
from output_gate import check as check_contract
from output_gate import response_checks as _gate_response_checks
from prompt_log import suspended as suspend_prompt_log
from prompts import BIDS_PROMPT, render_system_content, to_chat_history
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
# The patterns live in :mod:`output_gate` now; this function keeps the exact
# historical name and return shape (the diagnostics and the tests depend on
# both). It remains flag-only.
# ---------------------------------------------------------------------------


def _response_checks(response: str) -> Dict[str, Any]:
    """Deterministic post-generation checks; flags only, never rewrites."""
    return _gate_response_checks(response)


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


class PlannerResult(BaseModel):
    """Everything the API needs to report what actually happened."""

    response: str
    # the intent actually used: "explain" | "fix" | "unclear"
    intent: str
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


class Planner:
    def __init__(
        self,
        llm: Any,
        retriever: Retriever,
        tools: Optional[ToolRegistry] = None,
        laya: Optional[LayaDecisionEngine] = None,
    ):
        self.llm = llm
        # The planner speaks the LangChain retriever interface; a raw
        # ``Retriever`` is wrapped on the spot so either spelling works.
        self.retriever: KBRetriever = (
            retriever if isinstance(retriever, KBRetriever)
            else KBRetriever(core=retriever)
        )
        # Kept for the future tool-execution phase; never dispatched here.
        self.tools = tools if tools is not None else ToolRegistry()
        # The Laya engine loads its checkpoint lazily, so default
        # construction is cheap; unit tests inject a mock instead.
        self.laya = laya if laya is not None else LayaDecisionEngine()
        self._step_logger = StepLogger()
        self._pipeline: Runnable = self._build_pipeline()

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

        if intent == "unclear" and not self._has_issue_context(context):
            # No issue to stand an explanation on: ask, cheaply and
            # deterministically, instead of generating an ungrounded answer.
            return PlannerResult(
                response=CLARIFICATION_MESSAGE,
                intent="unclear",
                classification_source=decision.source,
            )

        # "fix", or "explain", or "unclear" with issue context (where the
        # context clearly supports a safe explanation-only answer): all
        # three are the same pipeline run, differing only in the decision
        # they carry.
        state: Dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_input": user_input,
            "context": context,
            "str_context": str_context,
            "history": history,
            "dataset_structure": dataset_structure,
            "file_content": file_content,
            "decision": decision,
            "action_unavailable": intent == "fix",
            "unclear": intent == "unclear",
        }
        return self._pipeline.invoke(
            state, config={"callbacks": [self._step_logger]}
        )

    # ------------------------------------------------------------------
    # The pipeline (LangChain orchestration; deterministic internals)
    # ------------------------------------------------------------------

    def _llm_runnable(self) -> Runnable:
        """The generation step as a runnable.

        A ``Runnable`` (the production :class:`local_chat_model.LocalChatModel`,
        or a mock) is used as-is; any other object is adapted through its
        ``invoke(messages)`` method so the same pipeline serves both
        without the tests needing to subclass LangChain's chat models.
        """
        llm = self.llm
        if isinstance(llm, Runnable):
            return llm

        def call(prompt_value: Any) -> AIMessage:
            messages: List[BaseMessage] = (
                prompt_value.to_messages()
                if isinstance(prompt_value, ChatPromptValue)
                else list(prompt_value)
            )
            result = llm.invoke(messages)
            return result if isinstance(result, AIMessage) else AIMessage(
                content=str(result)
            )

        return RunnableLambda(call, name="llm")

    def _build_pipeline(self) -> Runnable:
        """One request: gather evidence, generate, report what happened.

        * ``bids_evidence`` — retrieve for the issue and render the
          system half of the prompt (deterministic; no model);
        * ``BIDS_PROMPT | llm`` — the conversation shape: the assembled
          system message, the earlier turns as chat messages, the
          current question;
        * ``bids_result`` — run the deterministic response checks and
          assemble the :class:`PlannerResult`.
        """
        answer_branch = BIDS_PROMPT | self._llm_runnable()
        return (
            RunnableLambda(self._prepare, name="bids_evidence")
            | {"answer": answer_branch, "state": RunnablePassthrough()}
            | RunnableLambda(self._build_result, name="bids_result")
        )

    def _prepare(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evidence step: retrieval + system-content rendering.

        Returns the prepared state the prompt template reads
        (``system_content`` / ``history`` / ``user_input``) plus the
        pieces the result step needs (retrieval outcome, decision,
        which banners were requested, the dataset evidence fields).
        """
        retrieval = self._retrieve(state["context"], state["user_input"])
        contract = build_contract(
            state["context"], retrieval, state["file_content"]
        )
        system_content = render_system_content(
            system_prompt=state["system_prompt"],
            context=state["str_context"],
            retrieval=retrieval,
            action_unavailable=state["action_unavailable"],
            unclear=state["unclear"],
            history=state["history"],
            dataset_structure=state["dataset_structure"],
            file_content=state["file_content"],
            contract=contract,
        )
        return {
            "system_content": system_content,
            "user_input": state["user_input"],
            "history": to_chat_history(state["history"]),
            "retrieval": retrieval,
            "contract": contract,
            "decision": state["decision"],
            "action_unavailable": state["action_unavailable"],
            "unclear": state["unclear"],
            "dataset_structure": state["dataset_structure"],
            "file_content": state["file_content"],
            "str_context": state["str_context"],
            "raw_history": state["history"],
        }

    def _build_result(self, bundle: Dict[str, Any]) -> PlannerResult:
        """Result step: deterministic checks + honest metadata."""
        state = bundle["state"]
        answer = bundle["answer"]
        raw_response = (
            answer.content if isinstance(answer, AIMessage) else str(answer)
        )
        contract: Optional[AnswerContract] = state.get("contract")
        response, gate_report = self._apply_quality_gate(
            raw_response, state, contract
        )
        retrieval: Optional[RetrievalResult] = state["retrieval"]
        decision: LayaDecision = state["decision"]
        action_unavailable: bool = state["action_unavailable"]
        intent = (
            "fix" if action_unavailable
            else "unclear" if state["unclear"]
            else "explain"
        )
        return PlannerResult(
            response=response,
            intent=intent,
            tool_used=None,
            execution_status=EXECUTION_NOT_AVAILABLE if action_unavailable else None,
            classification_source=decision.source,
            retrieval_found=retrieval.found if retrieval else None,
            retrieval_quality=retrieval.quality if retrieval else None,
            diagnostics=self._diagnostics(
                retrieval, response, state["history"],
                state["dataset_structure"], state["file_content"],
                gate_report,
            ),
        )

    # ------------------------------------------------------------------
    # Quality gate: contract validation, repair, deterministic fallback
    # ------------------------------------------------------------------

    def _apply_quality_gate(
        self,
        response: str,
        state: Dict[str, Any],
        contract: Optional[AnswerContract],
    ) -> tuple[str, Dict[str, Any]]:
        """Validate the answer against its contract and repair it if needed.

        Tiers, cheapest first: deterministic repair, one model repair, the
        grounded template. The report is diagnostic only (no dataset content)
        and is always returned, so a reviewer can see which tier answered.
        """
        config: QualityConfig = getattr(CONFIG, "quality", QualityConfig())
        report: Dict[str, Any] = {
            "enabled": bool(config.enabled),
            "applied": False,
            "contract": contract.as_dict() if contract else None,
            "original_violations": [],
            "original_response_checks": {},
            "violations": [],
            "repaired": False,
            "fallback_used": False,
            "final_ok": True,
        }
        if not config.enabled or contract is None or not contract.is_actionable:
            return response, report

        inputs = [
            state.get("str_context"),
            state.get("file_content"),
            state.get("dataset_structure"),
            str(state.get("raw_history") or ""),
            state.get("user_input"),
        ]
        retrieval = state.get("retrieval")
        retrieval_text = getattr(retrieval, "text", "") or ""
        gate = check_contract(response, contract, inputs=inputs,
                              evidence=retrieval_text)
        report["original_violations"] = [v.as_dict() for v in gate.violations]
        report["original_response_checks"] = _response_checks(response)
        if gate.ok:
            return response, report

        report["applied"] = True
        candidate = deterministic_repair(response, contract, gate)
        gate = check_contract(candidate, contract, inputs=inputs,
                              evidence=retrieval_text)
        report["repaired"] = candidate != response

        attempts = 0
        while not gate.ok and attempts < config.repair_attempts:
            candidate = self._repair_with_llm(state, candidate, gate, contract)
            gate = check_contract(candidate, contract, inputs=inputs,
                                  evidence=retrieval_text)
            attempts += 1
            report["repaired"] = True

        if not gate.ok and config.fallback_enabled:
            candidate = render_grounded_template(contract)
            gate = check_contract(candidate, contract, inputs=inputs,
                                  evidence=retrieval_text)
            report["fallback_used"] = True

        report["violations"] = [v.as_dict() for v in gate.violations]
        report["final_ok"] = gate.ok
        return candidate, report

    def _repair_with_llm(
        self,
        state: Dict[str, Any],
        text: str,
        gate: Any,
        contract: AnswerContract,
    ) -> str:
        """One constrained repair attempt; keeps every other claim intact."""
        messages = [
            SystemMessage(
                content=render_repair_instructions(contract, gate.violations)
            ),
            HumanMessage(
                content=(
                    f"Original question: {state.get('user_input', '')}\n\n"
                    f"Answer to repair:\n\n{text}\n\n"
                    "Rewrite the answer now, fixing only the listed violations."
                )
            ),
        ]
        with suspend_prompt_log():
            result = self._llm_runnable().invoke(messages)
        return result.content if isinstance(result, AIMessage) else str(result)

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
        gate_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Structured record of *why* the answer looks the way it does.

        Contains record ids, relevance/applicability decisions, rejection
        reasons and the contract/gate outcome — never the dataset's own
        content (no file digests, no trees, no history), so it is safe to
        log or return.
        """
        return {
            "retrieval": _retrieval_diagnostics(retrieval),
            "evidence": {
                "history": bool(history),
                "dataset_structure": bool(dataset_structure),
                "file_content": bool(file_content),
            },
            "response_checks": _response_checks(response),
            "gate": gate_report or {"enabled": False},
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
        contract: Optional[AnswerContract] = None,
    ) -> str:
        """Assemble the system half of the prompt.

        Kept as the public API (``benchmark_timing.py`` and the tests
        call it directly); the rendering itself lives in
        :func:`prompts.render_system_content`, so the wording has one
        home and the pipeline and this method cannot drift apart.

        The conversation appears twice on purpose: as real chat turns
        (see :data:`prompts.BIDS_PROMPT` and the LLM layer) *and* as a
        section here. The chat turns are what keeps a re-asked question
        from reading as a fresh start; the section keeps what was said
        in view while the answer is composed, for models that weight
        system text more heavily than their own earlier messages. Both
        render the same turns, so the prompt builder must not invent a
        second version of them.

        ``dataset_structure`` and ``file_content`` are pre-compiled by
        the client — a condensed tree and a file digest — and are left
        out entirely when absent, so a finding with no dataset behind
        it costs no lines here.
        """
        return render_system_content(
            system_prompt=system_prompt,
            context=context,
            retrieval=retrieval,
            action_unavailable=action_unavailable,
            unclear=unclear,
            history=history,
            dataset_structure=dataset_structure,
            file_content=file_content,
            contract=contract,
        )
