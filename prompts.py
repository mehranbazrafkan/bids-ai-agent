"""Prompt architecture for the BIDS agent.

Two things live here, deliberately separated:

* the *content* — what the model is told (the capability boundary, how
  to treat each evidence section, the output format). This wording is
  the product of the Phase-1 regression work (hallucinated rules,
  invented values, fabricated recording purposes) and must stay
  verbatim; ``tests/test_prompts.py`` pins its key claims.
  Content is plain Python: no LLM is involved in deciding what goes
  into the prompt.

* the *structure* — how the conversation is shaped for the model
  (system message, chat history, the current question). That is the
  ``ChatPromptTemplate`` below: a named, declarative artifact the
  planner composes in its LCEL pipeline.

All evidence (issue, dataset tree, file digest, retrieved knowledge,
quarantined records, earlier turns) is rendered into the single
system message by :func:`render_system_content`; the history also
travels as real chat turns between the system message and the
question — the same duplication the previous implementation had,
because it is what keeps a re-asked question reading as a second
question rather than a fresh start.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from answer_contract import AnswerContract, render_contract
from system_prompt import BIDS_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Section texts (verbatim from the original planner)
# ---------------------------------------------------------------------------

#: Banner for action requests in the explanation-only phase: execution is
#: disclosed as unavailable, and the answer must still be grounded.
ACTION_UNAVAILABLE_SECTION = (
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

#: Banner for requests the classifier could not route confidently.
INTENT_UNCERTAIN_SECTION = (
    "[Intent Uncertain]\n"
    "The classifier could not confidently determine whether the "
    "user wants an explanation or an action. Answer helpfully "
    "as an explanation-only assistant based on the provided "
    "context, and end with one short clarifying question if "
    "the user's goal is genuinely ambiguous."
)

#: Appended to the quarantine block of records that were retrieved but
#: ruled out by their own applicability selectors.
REJECTION_INSTRUCTION = (
    "These records were retrieved by keyword overlap, but their own "
    "applicability conditions conflict with this issue's file. Do not use "
    "them to explain this issue; mention them, if at all, only to say why "
    "they do not apply."
)


def _evidence_class(assessment: Any) -> str:
    """The grounding class, with a fallback for older assessment doubles."""
    return (
        getattr(assessment, "evidence_class", "")
        or getattr(assessment, "relevance", "")
        or "background"
    )


def evidence_summary(retrieval: Any) -> str:
    """Deterministic count of each grounding class, plus the missing state.

    The summary makes "no direct record" visible before the records
    themselves; a small model can then anchor on the state instead of
    treating the closest background match as the answer.
    """
    if retrieval is None or not getattr(retrieval, "found", False):
        return ""
    counter = getattr(retrieval, "evidence_counts", None)
    if callable(counter):
        counts = counter()
    else:
        # Lightweight test doubles only need ``assessments``/``rejected``.
        counts = {
            "direct": 0,
            "supporting": 0,
            "background": 0,
            "inapplicable": len(getattr(retrieval, "rejected", ()) or ()),
        }
        for assessment in getattr(retrieval, "assessments", ()) or ():
            key = _evidence_class(assessment)
            counts[key] = counts.get(key, 0) + 1
    lines = ["[Evidence Summary]"]
    lines.append(
        f"- Direct evidence: {counts['direct']} record(s) — documents the "
        "reported rule or the exact constraint named by it."
    )
    lines.append(
        f"- Supporting evidence: {counts['supporting']} record(s) — "
        "explains the affected field, a named constraint, or application "
        "behaviour; it does not establish the reported rule."
    )
    lines.append(
        f"- Background only: {counts['background']} record(s) — related "
        "context; never use it as the issue's definition or cause."
    )
    if counts["inapplicable"]:
        lines.append(
            f"- Inapplicable (quarantined): {counts['inapplicable']} "
            "record(s) — excluded by their own applicability conditions."
        )
    if counts["direct"]:
        lines.append("- Missing: none for the reported rule.")
    else:
        lines.append(
            "- Missing: no direct record establishes the reported rule. "
            "Say so explicitly; supporting/background records cannot fill "
            "that gap."
        )
    return "\n".join(lines)


def retrieval_note(retrieval: Any) -> str:
    """How the model must treat each grounding class of the evidence.

    Rendered right after the records themselves so the classification is
    acted on, not just displayed: supporting records explain a field or
    constraint but never the rule, background records may never become the
    issue's definition, and an absence of direct records must be stated
    rather than papered over with the closest matching document.
    """
    classes = [_evidence_class(a) for a in retrieval.assessments]
    direct = [a for a, c in zip(retrieval.assessments, classes) if c == "direct"]
    supporting = [a for a, c in zip(retrieval.assessments, classes)
                  if c == "supporting"]
    background = [a for a, c in zip(retrieval.assessments, classes)
                  if c == "background"]
    lines = ["[Retrieval Note]"]
    if direct:
        lines.append(
            "- Records marked \"Evidence role: Direct\" document this "
            "issue's rule or the exact constraint the message refers to; "
            "use them to support statements about this issue."
        )
    else:
        lines.append(
            "- None of the records above is marked \"Directly relevant\": "
            "the knowledge base does not directly document this issue's "
            "rule. State that explicitly and explain only what the issue "
            "details and file content show — do not present a supporting "
            "or background record as the definition, cause, or rule behind "
            "this issue."
        )
    if supporting:
        lines.append(
            "- Records marked \"Evidence role: Supporting\" explain the "
            "affected field, the exact format/constraint named in the "
            "message, or BIDS-Manager application behaviour. Use them only "
            "for that narrow purpose. Do not present them as documentation "
            "that the reported rule exists, and do not import their rule "
            "text or selectors into this issue."
        )
    if background:
        lines.append(
            "- Records marked \"Evidence role: Background only\" are context "
            "only (examples, unrelated rules, general file rules). Do not "
            "use them to define the issue, apply their file conditions "
            "(suffix, extension, selectors), or merge their behaviour into "
            "this explanation."
        )
    if retrieval.quality == "lexical":
        lines.append(
            "- The best match only shares vocabulary with the issue: treat "
            "every record above as weak background at most."
        )
    elif retrieval.quality == "semantic":
        lines.append(
            "- The best match was selected by embedding similarity, not by "
            "an exact rule or field identifier. Similarity alone is not "
            "proof: the grounding roles above still decide what each "
            "record may establish."
        )
    lines.append("")
    lines.append("[Grounding and Provenance Rules]")
    lines.append(
        "- Attribute every factual claim to the source that establishes it: "
        "the BIDS specification/schema (normative), the BIDS Validator "
        "implementation, BIDS-Manager application behaviour, the supplied "
        "dataset (issue, file content, tree), supporting/background "
        "knowledge, or your own inference."
    )
    lines.append(
        "- Never upgrade a validator or BIDS-Manager finding into a BIDS "
        "requirement. Never attribute a background record's statement to "
        "the BIDS specification."
    )
    lines.append(
        "- Cite a knowledge record by its ID when a claim depends on it. "
        "Never cite a record marked \"Not applicable\"."
    )
    lines.append(
        "- Claims about the dataset may use only values present in the "
        "supplied issue, file content, or tree. Label reasoning of your own "
        "as an inference (for example, \"likely\" or \"this suggests\")."
    )
    lines.append(
        "- If the supplied evidence does not establish a fact, say it cannot "
        "be determined; do not fill the gap with a plausible rule or value."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System-content rendering (deterministic, no model)
# ---------------------------------------------------------------------------


def render_system_content(
    system_prompt: str,
    context: str,
    retrieval: Optional[Any],
    action_unavailable: bool = False,
    unclear: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    dataset_structure: Optional[str] = None,
    file_content: Optional[str] = None,
    contract: Optional[AnswerContract] = None,
) -> str:
    """Assemble the system half of the prompt.

    The system instructions come first. The issue, the dataset around it
    and the retrieved records are then presented as *data* the model may not
    treat as instructions. The per-request answer checklist is appended
    *last*, after every evidence section, so it is the final thing the model
    reads before composing its answer.

    The conversation appears twice on purpose: as real chat turns (see
    :data:`BIDS_PROMPT`) *and* as a section here. The chat turns are what
    keeps a re-asked question from reading as a fresh start; the section
    keeps what was said in view while the answer is composed, for models
    that weight system text more heavily than their own earlier messages.
    Both render the same turns, so this function must not invent a second
    version of them.

    ``dataset_structure`` and ``file_content`` are pre-compiled by the
    client — a condensed tree and a file digest — and are left out
    entirely when absent, so a finding with no dataset behind it costs no
    lines here.

    Ported verbatim from ``Planner.build_system_prompt``; the section
    layout and wording are pinned by the regression tests.
    """
    parts: List[str] = [f"System: {system_prompt}"]

    if action_unavailable:
        parts.append(ACTION_UNAVAILABLE_SECTION)

    if unclear:
        parts.append(INTENT_UNCERTAIN_SECTION)

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
                + REJECTION_INSTRUCTION
            )
        parts.append(section)
    else:
        parts.append(
            "[Retrieved BIDS Knowledge]\n"
            f"{evidence_summary(retrieval)}\n\n"
            f"{retrieval.text}\n\n{retrieval_note(retrieval)}"
        )
        if retrieval.rejection_text:
            parts.append(
                "[Retrieved Records — Not Applicable]\n"
                f"{retrieval.rejection_text}\n{REJECTION_INSTRUCTION}"
            )

    if history:
        # The same turns that travel as chat messages, rendered here as
        # well: see the docstring above.
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

    if contract is not None and contract.is_actionable:
        # Last, so the model checks its draft against the deterministic
        # obligations immediately before answering.
        parts.append(render_contract(contract))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chat history conversion
# ---------------------------------------------------------------------------


def to_chat_history(history: Optional[List[Dict[str, str]]]) -> List[Any]:
    """Turn a ``{"role", "content"}`` transcript into chat messages.

    Only the two roles a conversation is made of — and only something to
    say — survive: a template handed any other role or an empty string
    here produces a turn the model cannot answer to (the same filter the
    LLM layer used to apply at the model boundary).
    """
    turns: List[Any] = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "")
        if not content.strip():
            continue
        if role == "user":
            turns.append(HumanMessage(content=content))
        elif role == "assistant":
            turns.append(AIMessage(content=content))
    return turns


# ---------------------------------------------------------------------------
# The BIDS chat prompt
# ---------------------------------------------------------------------------

#: The conversation shape of one BIDS request: the assembled system
#: message (instructions + all evidence sections), the earlier turns as
#: real chat messages, then the user's current question.
BIDS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_content}"),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{user_input}"),
    ]
)
