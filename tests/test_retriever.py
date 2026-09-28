"""Tests for the BIDS knowledge-base retriever."""

import json
import os
import tempfile

import pytest

from config import RetrievalConfig
from retriever import (
    MSG_EMPTY_KB,
    MSG_EMPTY_QUERY,
    MSG_NO_KB,
    MSG_NO_MATCH,
    KnowledgeRecord,
    Retriever,
    Scorer,
    _UNSET,
    _parse_query,
)

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "knowledge-base")


def make_retriever(**kwargs):
    if "knowledge_base_dir" not in kwargs:
        kwargs["knowledge_base_dir"] = KB_DIR
    config = RetrievalConfig(**kwargs)
    return Retriever(config=config)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loads_enriched_kb_by_default():
    r = make_retriever()
    assert r.get_record_count() == 1266
    types = r.get_record_types()
    assert "MetadataRule" in types
    assert "TabularRule" in types


def test_knowledge_file_override():
    r = make_retriever(knowledge_file="knowledge.jsonl")
    assert r.get_record_count() == 1272


def test_missing_kb_dir_returns_stable_message(tmp_path):
    r = Retriever(data_dir=str(tmp_path))
    assert r.retrieve("anything") == MSG_NO_KB.format(tmp_path)


def test_source_paths_are_repo_relative():
    r = make_retriever()
    for rec in r.records:
        path = rec.source.get("path", "")
        assert path.startswith("./"), path
        assert "\\" not in path, path
        assert "Users" not in path, path
        assert not path.startswith("C:"), path


# ---------------------------------------------------------------------------
# Retrieval behaviour
# ---------------------------------------------------------------------------


def test_retrieve_plain_query_returns_formatted_items():
    out = make_retriever().retrieve("What is the task entity?", top_k=1)
    assert "Knowledge Item" in out
    assert "ID:" in out


def test_retrieve_empty_query():
    assert make_retriever().retrieve("") == MSG_EMPTY_QUERY
    assert make_retriever().retrieve("   ") == MSG_EMPTY_QUERY


def test_retrieve_no_match_stable_message():
    out = make_retriever().retrieve("zzzzqqqq xxxyyy")
    assert out == MSG_NO_MATCH.format("zzzzqqqq xxxyyy")


def test_retrieve_issue_hits_exact_rule():
    issue = {
        "severity": "err",
        "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
        "message": "ReferencesAndLinks must be array",
        "field": "ReferencesAndLinks",
    }
    out = make_retriever().retrieve_issue(issue, user_question="Explain this to me.")
    assert "JSON_SCHEMA_VALIDATION_ERROR" in out


def test_tabular_issue_hits_tabular_rule():
    issue = {
        "path": "participants.tsv",
        "severity": "err",
        "suffix": "participants",
        "issues": [{"severity": "err",
                    "rule_id": "TSV_VALUE_INCORRECT_TYPE",
                    "message": "column 'age': value '020Y' is not valid",
                    "field": "age"}],
    }
    out = make_retriever().retrieve_issue(issue)
    assert "age" in out
    assert "participants" in out.lower()


def test_top_k_returns_multiple_distinct_items():
    out = make_retriever().retrieve("subject session dataset", top_k=3)
    ids = [line[len("ID: "):] for line in out.splitlines() if line.startswith("ID: ")]
    assert len(ids) == 3
    assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------


def test_parse_query_recognizes_serialized_issue():
    sq = _parse_query(
        "Explain this. severity:err rule_id:JSON_SCHEMA_VALIDATION_ERROR "
        "message:ReferencesAndLinks must be array field:ReferencesAndLinks"
    )
    assert sq["rule_id"] == "JSON_SCHEMA_VALIDATION_ERROR"
    assert sq["field"] == "ReferencesAndLinks"
    assert sq["severity"] == "err"
    assert "Explain this." in sq["general_text"]


def test_parse_query_ignores_plain_text():
    assert _parse_query("What is the task entity?") == {}


def test_parse_query_lifts_nested_issue():
    sq = _parse_query(
        "Directing. path:participants.tsv severity:err suffix:participants "
        "issues:[severity:err rule_id:TSV_VALUE_INCORRECT_TYPE"
        " message:value not valid field:age]"
    )
    assert sq["rule_id"] == "TSV_VALUE_INCORRECT_TYPE"
    assert sq["field"] == "age"


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------


def test_normalize_splits_camel_case():
    assert KnowledgeRecord._normalize("ReferencesAndLinks") == "references and links"
    assert KnowledgeRecord._normalize("JSONSchema") == "json schema"


def test_scorer_rewards_exact_rule_id():
    r = make_retriever()
    # find the record carrying that rule code
    targets = [i for i, rec in enumerate(r.records)
               if isinstance(rec.requirements, dict)
               and rec.requirements.get("code") == "JSON_SCHEMA_VALIDATION_ERROR"]
    assert targets, "expected a record with requirement code JSON_SCHEMA_VALIDATION_ERROR"
    idx = targets[0]
    query = ("severity:err rule_id:JSON_SCHEMA_VALIDATION_ERROR "
             "message:invalid schema")
    bd = Scorer.explain(
        query, r.records[idx], r.blocks[idx], r.identities[idx], r._idf,
        structured_query=_parse_query(query),
        precomputed=r._precomputed[idx],
    )
    bonus_types = [s["type"] for s in bd["structured"]]
    assert "rule_id_exact" in bonus_types


def test_relationships_and_adjacency_loaded():
    r = make_retriever()
    assert len(r.edges) > 0
    assert sum(len(v) for v in r._adjacency.values()) > 0


# ---------------------------------------------------------------------------
# Lazy loading of raw_content
# ---------------------------------------------------------------------------


def _write_tiny_kb(directory):
    lines = [
        {"id": "r1", "knowledge_type": "Check", "title": "Rule one",
         "summary": "first record", "source": {"file": "a.yaml",
         "path": "./raw-bids-spec/a.yaml"}, "raw_content": {"x": 1}},
        {"id": "r2", "knowledge_type": "Check", "title": "Rule two",
         "summary": "second record", "source": {"file": "b.yaml",
         "path": "./raw-bids-spec/b.yaml"}, "raw_content": {"y": 2}},
    ]
    kb = os.path.join(directory, "kb")
    os.makedirs(kb)
    path = os.path.join(kb, "enriched_knowledge.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return kb


@pytest.mark.parametrize(
    "lazy,index_raw",
    [(True, False), (False, True)],
)
def test_lazy_raw_content_modes(tmp_path, lazy, index_raw):
    kb = _write_tiny_kb(tmp_path)
    r = make_retriever(knowledge_base_dir=kb,
                       knowledge_file="enriched_knowledge.jsonl",
                       lazy_raw_content=lazy,
                       index_raw_content=index_raw)
    rec = r.records[0]

    if lazy:
        # raw_content is not held in memory before first access.
        assert rec._raw_cache is _UNSET
        # and it is excluded from the search index.
        labels = [label for label, _t, _w in r.blocks[0]]
        assert "raw_content" not in labels
    else:
        assert rec._raw_cache == {"x": 1}
        labels = [label for label, _t, _w in r.blocks[0]]
        assert "raw_content" in labels

    # The property resolves the value in both modes.
    assert rec.raw_content == {"x": 1}
    # After a lazy load it is cached on the record.
    if lazy:
        assert rec._raw_cache == {"x": 1}


def test_retrieve_still_works_with_lazy_raw(tmp_path):
    kb = _write_tiny_kb(tmp_path)
    r = make_retriever(knowledge_base_dir=kb,
                       knowledge_file="enriched_knowledge.jsonl")
    out = r.retrieve("first")
    assert "Rule one" in out