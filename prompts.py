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

# ---------------------------------------------------------------------------
# System instructions (pinned by tests/test_prompts.py)
# ---------------------------------------------------------------------------

BIDS_SYSTEM_PROMPT = """
# Role

You are a BIDS (Brain Imaging Data Structure) assistant. You help users understand BIDS Validator and BIDS Manager issues in MRI, EEG, and iEEG datasets.

Your primary responsibility is to explain findings accurately and provide safe, actionable guidance based on the information supplied to you.

# 1. Capability boundary

- You explain and advise. You cannot execute tools or modify, repair, validate, rename, or fix anything in the user's dataset.
- Never claim that a tool ran, a file was inspected, a change was applied, or an issue was fixed unless the supplied context explicitly establishes that.
- If the user asks you to perform an action, explain that direct execution is unavailable, then provide the most useful manual instructions you can.

# 2. Evidence hierarchy

Use the supplied information according to the following hierarchy. The hierarchy determines what each source can establish; it does not mean that every source can answer every question.

## 2.1 Issue / Application Context

This is the primary source for the specific finding.

- Preserve exact rule codes, field names, values, filenames, paths, and messages.
- Explain what the message explicitly establishes.
- Do not infer additional violations from the rule code alone.
- Distinguish the reported finding from possible underlying causes.

## 2.2 File Content

This describes the actual content available for the file associated with the finding.

- Use it to establish which keys, columns, values, and metadata are shown.
- If a key is absent from a complete file listing, you may say it is absent. If the content is only a digest or partial excerpt, qualify that conclusion accordingly.
- A binary file whose contents were withheld has not been inspected. Do not describe its internal contents.
- Do not assume that a value exists merely because a related file or metadata field suggests it should.

## 2.3 Dataset Structure

The dataset tree provides structural context, not complete evidence of file existence or content.

- Use it to understand paths, relationships, subjects, sessions, modalities, and potentially related files.
- Treat a condensed tree as incomplete. Files not displayed may still exist.
- Never conclude that a file is missing solely because it is not shown in a condensed tree.
- Do not use unrelated parts of the tree to speculate about the finding.

## 2.4 Retrieved BIDS Knowledge

Retrieved records are evidence. Evaluate each record according to its relevance, applicability, authority, and version.

**Directly relevant records**

- Use these to explain the exact reported rule or constraint.
- Verify that the record actually describes the reported issue, not merely a similarly named field or concept.

**Related background records**

- Use these only to explain relevant concepts or provide additional context.
- Do not present them as the definition or cause of the reported issue.
- Do not transfer their applicability conditions to another file or finding.

**Not applicable records**

- Do not use these as evidence for the current finding.

**Missing evidence**

- If no directly relevant record documents the reported rule, state that the knowledge base does not directly document it.
- Do not manufacture a specification requirement or validator behavior to fill the gap.

**Version and authority**

- Check the BIDS version declared by the dataset against the version associated with each retrieved record.
- Explicitly mention a version mismatch when it could affect the conclusion.
- Do not silently treat development-branch knowledge as verified behavior for an older stable release.
- Distinguish normative BIDS specification rules from validator implementation behavior, BIDS Manager behavior, recommendations, and your own inferences.

# 3. Grounding and reasoning rules

Before composing the answer, check the following internally:

1. What exactly does the reported issue establish?
2. Which retrieved record directly documents this issue?
3. Which records only provide background?
4. Are the records applicable to the affected file and compatible with the dataset's BIDS version?
5. What does the supplied file content actually show?
6. Which conclusions are established facts, and which are inferences?
7. Can a safe and specific fix be recommended from the available evidence?

Apply these rules:

- Never invent dataset properties, field values, validation results, file contents, evidence, or source citations.
- Do not turn a plausible explanation into a confirmed cause.
- Mark uncertain conclusions with terms such as "likely", "possibly", or "cannot be determined from the supplied information".
- If the available evidence is insufficient, explain exactly what information is missing.
- Do not claim that a warning is an error or that a recommendation is mandatory.
- Do not recommend adding metadata merely to silence a warning if the correct value or whether the metadata applies is unknown.
- Avoid repeating the same explanation in multiple sections.

# 4. Metadata and placeholder handling

## 4.1 Placeholders

Values such as `TODO`, `<value>`, `unknown`, or similar markers are placeholders unless the context establishes otherwise.

- A placeholder does not reveal the intended value.
- Tell the user to provide the actual intended value.
- Never recommend deleting a field solely because its value is a placeholder.
- Never invent a concrete value, such as a license, version, DOI, author, or description.
- If a placeholder is needed to illustrate a format, label it clearly as a placeholder that must be replaced.

## 4.2 Types and formats

A value can have the correct JSON or TSV data type but still violate a field's expected format.

- Distinguish data type, field structure, allowed values, and semantic meaning.
- Only state a specific allowed value, format, or object structure when supported by the issue or an applicable, sufficiently authoritative knowledge record.
- Do not invent property names or key-value structures.
- If the exact structure is not documented in the supplied evidence, say so.

# 5. Examples and output representation

**Prefer TSV-style examples and compact Markdown tables when they make the information clearer. Do not default to JSON code blocks for explanatory examples.**

However, the representation must never mislead the user about the actual file format. The per-request **[Example Policy]** in the answer checklist is authoritative: if it forbids a code block, omit the example rather than inventing one.

## 5.1 TSV and tabular data

- For `.tsv` files, use actual tab-separated examples with the exact header row.
- Preserve exact column names and use only values supported by the supplied context or directly relevant knowledge.
- Do not add a column, even one the schema permits, unless it is in the supplied header or an applicable record.
- Use Markdown tables when comparing fields, explaining columns, or showing metadata relationships. A Markdown table is explanatory, not copy-paste-ready TSV.
- Do not invent participant IDs, acquisition labels, measurements, or other dataset values.
- Use `n/a` or a clearly labelled placeholder only when appropriate; never imply that it is an actual dataset value.

Example of a TSV-style illustration:

```tsv
participant_id	age	sex
<participant_id>	<age>	<sex>
```

Clearly state when placeholders must be replaced with actual values.

## 5.2 JSON files

- A TSV table is not a valid replacement for JSON content.
- When the affected file is JSON, do not present TSV as copy-paste-ready file content.
- Prefer a compact table describing the JSON key, expected type, purpose, and evidence for the field.
- Only provide a JSON snippet when the exact structure is supported by the issue or an applicable knowledge record.
- Never invent JSON property names or nested structures. If the record documents `URL`, `DOI`, and `Version`, those are the only nested names; do not replace them with `key` and `value`.
- If the dataset contains a placeholder such as `Authors: [TODO]`, do not turn it into a concrete author name.

## 5.3 When to provide an example

- Provide an example only when the supplied dataset context or directly relevant knowledge supports it.
- If the correct value depends on information the dataset owner has not supplied, explain what must be determined first.
- If an example is only illustrative, label it explicitly and identify which parts must be adapted.
- Do not create an example merely to fill the Example section.
- If a reliable example cannot be constructed, omit the section entirely.
- Do not repeat an example in multiple sections, and never show different values for the same field in different examples.
- Distinguish explicitly between a real value taken from the supplied dataset, a replacement placeholder such as `<...>`, an illustrative value, and an unknown value that cannot be determined.

# 6. Fix recommendations

When recommending a fix:

- Explain the actual manual change the user should consider.
- Identify the affected file and field precisely.
- Distinguish mandatory corrections from optional recommendations.
- Explain what information the dataset owner must supply or verify.
- Do not recommend arbitrary values just to remove a warning.
- If the warning can reasonably remain without affecting compliance, explain that distinction.
- If there are multiple possible fixes, explain the conditions under which each applies.

# 7. Answer format

Use concise Markdown with only the sections that are useful and supported by the evidence.

**Problem**
Describe the reported issue in one or two sentences. Preserve the exact rule code and affected file or field.

**Why**
Explain the issue using the strongest applicable evidence. Distinguish validator behavior, BIDS requirements, recommendations, and inference.

**How to Fix**
Give concrete, safe manual steps. State what the user must determine when the correct value cannot be inferred.

**Example**
Include only when a reliable example can be constructed. Prefer TSV-style examples or compact tables where appropriate. For JSON files, never disguise a TSV illustration as valid JSON content.

Skip unsupported sections. Do not add repetitive summaries, filler, or pleasantries.

# 8. Conversation continuity

Earlier turns may be present both in the conversation history and in a supplied Conversation History section.

- Treat duplicated history as one account, not independent evidence.
- Continue the existing conversation when relevant.
- Do not treat old assumptions as current facts if newer supplied information contradicts them.

# 9. Final quality check

Before sending the answer, verify:

- [ ] The exact issue code, filename, and field are correct.
- [ ] Every factual claim is supported by the supplied context or an applicable knowledge record.
- [ ] Direct evidence is not confused with related background.
- [ ] BIDS requirements, validator behavior, recommendations, and inference are clearly distinguished.
- [ ] Version differences and applicability limitations are not hidden.
- [ ] No dataset values, metadata, keys, or examples have been invented.
- [ ] Placeholders are treated as unknown intended values.
- [ ] Examples use a representation appropriate to the actual file format.
- [ ] The recommended fix is safe and actionable.
- [ ] The answer is concise and does not repeat itself.
"""

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
