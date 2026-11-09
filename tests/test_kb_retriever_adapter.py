"""Tests for KBRetriever — the LangChain facade over the knowledge base.

The facade must be transparent: every structured call delegates to the
core :class:`retriever.Retriever` unchanged (same outcomes, same
assessments), and the LangChain interface (``_get_relevant_documents``
/ ``invoke``) reports the kept records as one ``Document`` each — with
a miss staying an empty list, never invented content.

Loading the KB is CPU-only and fast, so these run in the default suite.
"""

import os

import pytest

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "knowledge-base")

from config import RetrievalConfig
from kb_retriever import KBRetriever
from retriever import Retriever

ISSUE = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array",
    "field": "ReferencesAndLinks",
}


@pytest.fixture(scope="module")
def kb():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


@pytest.fixture(scope="module")
def kb_retriever(kb):
    return KBRetriever(core=kb)


# ---------------------------------------------------------------------------
# Delegation — the structured contract is unchanged
# ---------------------------------------------------------------------------


def test_wraps_the_core(kb, kb_retriever):
    assert kb_retriever.core is kb


def test_retrieve_issue_structured_delegates(kb, kb_retriever):
    via_facade = kb_retriever.retrieve_issue_structured(
        ISSUE, user_question="Explain this"
    )
    via_core = kb.retrieve_issue_structured(ISSUE, user_question="Explain this")
    assert via_facade is via_core or (
        via_facade.found == via_core.found
        and via_facade.quality == via_core.quality
        and [a.id for a in via_facade.assessments]
        == [a.id for a in via_core.assessments]
    )
    assert via_facade.found is True
    assert via_facade.quality == "rule_exact"


def test_retrieve_structured_delegates(kb, kb_retriever):
    assert kb_retriever.retrieve_structured("HEDVersion") is not None
    assert kb_retriever.retrieve_structured("Flibbertigibbet zzz").found is False


def test_retrieve_text_delegates(kb, kb_retriever):
    query = "ReferencesAndLinks must be array"
    assert kb_retriever.retrieve(query) == kb.retrieve(query)
    assert kb_retriever.retrieve_issue(ISSUE) == kb.retrieve_issue(ISSUE)


def test_introspection_delegates(kb, kb_retriever):
    assert kb_retriever.get_record_count() == kb.get_record_count() == 3505
    assert kb_retriever.get_record_types() == kb.get_record_types()


def test_explain_delegates(kb, kb_retriever):
    assert kb_retriever.explain("HEDVersion", top_k=2) == kb.explain(
        "HEDVersion", top_k=2
    )


# ---------------------------------------------------------------------------
# The LangChain interface
# ---------------------------------------------------------------------------


def test_invoke_returns_one_document_per_kept_record(kb_retriever):
    result = kb_retriever.core.retrieve_structured("ReferencesAndLinks")
    docs = kb_retriever.invoke("ReferencesAndLinks")
    assert result.found
    assert len(docs) == len(result.assessments)
    for doc in docs:
        assert doc.page_content.startswith("=== Knowledge Item")
        assert set(doc.metadata) == {
            "id", "title", "match", "relevance", "applicability", "score",
        }
    # same records, same order, as the structured outcome
    assert [d.metadata["id"] for d in docs] == [
        a.id for a in result.assessments
    ]


def test_miss_returns_no_documents(kb_retriever):
    # The flat interface has no "miss" marker; an empty list IS the miss,
    # and nothing is ever invented to fill it.
    assert kb_retriever.invoke("Flibbertigibbet zzz") == []


def test_rejected_records_never_become_documents(kb_retriever):
    """The LangChain path keeps the same quarantine discipline as the
    prompt path: a record ruled out by its own applicability selectors
    is never served as a document, whatever the query."""
    query = "HEDVersion events"
    docs = kb_retriever.invoke(query)
    result = kb_retriever.core.retrieve_structured(query)
    rejected_ids = {a.id for a in result.rejected}
    kept_ids = {d.metadata["id"] for d in docs}
    assert rejected_ids.isdisjoint(kept_ids)
