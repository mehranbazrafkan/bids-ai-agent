"""Tests for the prompt architecture (prompts.py).

Two concerns are pinned here:

* the *content* — the system instructions are re-exported unchanged from
  where the production agent reads them (``agent.system_prompt``), so
  the Phase-1 regression wording has exactly one home;
* the *structure* — the ``BIDS_PROMPT`` message shape (system, chat
  history, question), the history conversion, and the deterministic
  section layout of :func:`render_system_content`.
"""

import agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompts import (
    ACTION_UNAVAILABLE_SECTION,
    BIDS_PROMPT,
    BIDS_SYSTEM_PROMPT,
    INTENT_UNCERTAIN_SECTION,
    REJECTION_INSTRUCTION,
    render_system_content,
    retrieval_note,
    to_chat_history,
)


class FakeAssessment:
    def __init__(self, relevance):
        self.relevance = relevance


class FakeRetrieval:
    def __init__(self, found=True, quality="rule_exact", text="EVIDENCE",
                 relevance=None, rejection_text=None):
        self.found = found
        self.quality = quality
        self.text = text
        self.rejection_text = rejection_text
        self.assessments = [FakeAssessment(relevance or "direct")]
        self.rejected = []


# ---------------------------------------------------------------------------
# Content: one home for the wording
# ---------------------------------------------------------------------------


def test_system_prompt_is_the_single_source_of_truth():
    """agent.system_prompt must BE the prompts constant, not a copy.

    A second copy could drift; an identity check rules that out and
    keeps ``from agent import system_prompt`` a pure re-export.
    """
    assert agent.system_prompt is BIDS_SYSTEM_PROMPT


def test_system_prompt_keeps_the_capability_boundary():
    # The wording the Phase-1 regression work pinned (see
    # tests/test_regression_scenarios.py for the prompt-level pins).
    assert BIDS_SYSTEM_PROMPT.lstrip().startswith(
        "You are a BIDS (Brain Imaging Data Structure) assistant."
    )
    assert "CAPABILITY BOUNDARY — EXPLANATION ONLY" in BIDS_SYSTEM_PROMPT
    assert "**Problem**" in BIDS_SYSTEM_PROMPT
    assert "**Why**" in BIDS_SYSTEM_PROMPT
    assert "**How to Fix**" in BIDS_SYSTEM_PROMPT
    assert "**Example**" in BIDS_SYSTEM_PROMPT


def test_section_constants_are_stable():
    assert ACTION_UNAVAILABLE_SECTION.startswith(
        "[Action Requested — Execution Not Available]"
    )
    assert INTENT_UNCERTAIN_SECTION.startswith("[Intent Uncertain]")
    assert "Do not use them to explain this issue" in REJECTION_INSTRUCTION


# ---------------------------------------------------------------------------
# Structure: the BIDS_PROMPT message shape
# ---------------------------------------------------------------------------


def test_prompt_shape_is_system_history_question():
    history = [HumanMessage(content="h"), AIMessage(content="a")]
    value = BIDS_PROMPT.invoke(
        {"system_content": "SYS", "user_input": "Q", "history": history}
    )
    assert value.to_messages() == [
        SystemMessage(content="SYS"),
        HumanMessage(content="h"),
        AIMessage(content="a"),
        HumanMessage(content="Q"),
    ]


def test_prompt_without_history_is_system_question():
    value = BIDS_PROMPT.invoke(
        {"system_content": "SYS", "user_input": "Q", "history": []}
    )
    assert value.to_messages() == [
        SystemMessage(content="SYS"),
        HumanMessage(content="Q"),
    ]


def test_prompt_ignores_extra_state_keys():
    # The pipeline hands the prepared state dict to the template; the
    # evidence fields must not be required variables.
    value = BIDS_PROMPT.invoke(
        {
            "system_content": "SYS",
            "user_input": "Q",
            "history": [],
            "retrieval": object(),
            "decision": object(),
            "action_unavailable": False,
            "unclear": False,
            "dataset_structure": None,
            "file_content": None,
        }
    )
    assert value.to_messages() == [
        SystemMessage(content="SYS"),
        HumanMessage(content="Q"),
    ]


# ---------------------------------------------------------------------------
# to_chat_history
# ---------------------------------------------------------------------------


def test_to_chat_history_keeps_only_user_assistant_turns():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "nope"},          # not a chat role
        {"role": "user", "content": "   "},             # nothing to say
        "not a dict",                                    # malformed
        {"role": "User", "content": "q2"},              # role is case/whitespace-normalized
    ]
    assert to_chat_history(history) == [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
    ]


def test_to_chat_history_none_is_empty():
    assert to_chat_history(None) == []
    assert to_chat_history([]) == []


# ---------------------------------------------------------------------------
# render_system_content — deterministic section layout
# ---------------------------------------------------------------------------


def test_section_order_with_everything_present():
    text = render_system_content(
        system_prompt="SYS",
        context="CTX",
        retrieval=FakeRetrieval(),
        action_unavailable=True,
        unclear=True,
        history=[{"role": "user", "content": "q"}],
        dataset_structure="tree",
        file_content="digest",
    )
    # The layout the regression suite pins: instructions first, banners,
    # the issue, the dataset evidence, the knowledge, then the turns.
    assert text.index("System: SYS") == 0
    assert text.index("[Action Requested — Execution Not Available]") < \
        text.index("[Intent Uncertain]") < \
        text.index("[Issue / Application Context]") < \
        text.index("[Dataset Structure") < \
        text.index("[File Content") < \
        text.index("[Retrieved BIDS Knowledge]") < \
        text.index("[Conversation History")
    assert "CTX" in text
    assert "tree" in text
    assert "digest" in text
    assert "User: q" in text


def test_absent_evidence_sections_cost_no_lines():
    text = render_system_content(system_prompt="SYS", context="CTX",
                                 retrieval=FakeRetrieval())
    assert "[Action Requested" not in text
    assert "[Intent Uncertain]" not in text
    assert "[Dataset Structure" not in text
    assert "[File Content" not in text
    assert "[Conversation History" not in text


def test_retrieval_outcomes_are_rendered_distinctly():
    # found
    found = render_system_content("SYS", "CTX", FakeRetrieval())
    assert "EVIDENCE" in found
    assert "[Retrieval Note]" in found
    # miss
    missed = render_system_content(
        "SYS", "CTX", FakeRetrieval(found=False, text="")
    )
    assert ("No relevant knowledge-base record was found for this issue"
            in missed)
    assert "EVIDENCE" not in missed
    # not performed
    skipped = render_system_content("SYS", "CTX", None)
    assert ("No retrieval was performed (no issue context provided)"
            in skipped)


def test_quarantine_block_appears_only_with_rejections():
    with_rej = render_system_content(
        "SYS", "CTX",
        FakeRetrieval(rejection_text="REJ-RECORD"),
    )
    assert "[Retrieved Records — Not Applicable]" in with_rej
    assert "REJ-RECORD" in with_rej
    assert REJECTION_INSTRUCTION in with_rej

    clean = render_system_content("SYS", "CTX", FakeRetrieval())
    assert "[Retrieved Records — Not Applicable]" not in clean


def test_retrieval_note_tracks_relevance_and_quality():
    assert 'None of the records above is marked "Directly relevant"' in \
        retrieval_note(FakeRetrieval(relevance="background"))
    assert 'None of the records above is marked "Directly relevant"' not in \
        retrieval_note(FakeRetrieval(relevance="direct"))
    # lexical quality adds the weak-background line
    assert "treat every record above as weak background" in \
        retrieval_note(FakeRetrieval(quality="lexical"))
