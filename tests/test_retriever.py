"""Tests for the BIDS knowledge-base retriever."""

import json
import os
import re

import pytest

from config import RetrievalConfig
from retriever import (
    _UNSET,
    MSG_EMPTY_QUERY,
    MSG_NO_KB,
    MSG_NO_MATCH,
    KnowledgeRecord,
    Retriever,
    Scorer,
    _parse_query,
)

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "knowledge-base")

# Windows drive prefix — must never appear in a repo-relative path.
DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def make_retriever(**kwargs):
    if "knowledge_base_dir" not in kwargs:
        kwargs["knowledge_base_dir"] = KB_DIR
    config = RetrievalConfig(**kwargs)
    return Retriever(config=config)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loads_kb_by_default():
    r = make_retriever()
    # The single-file knowledge base: every record the retriever serves.
    assert r.get_record_count() == 3505
    types = r.get_record_types()
    assert "MetadataRule" in types
    assert "TabularRule" in types


def test_record_ids_are_unique_and_namespaced():
    r = make_retriever()
    ids = [rec.id for rec in r.records]
    assert len(ids) == len(set(ids))
    assert all(rid.startswith("kb_") for rid in ids)


def test_sequence_name_record_present():
    """The SequenceName field record comes from the spec's metadata
    objects, with repo-relative provenance and its own definition."""
    r = make_retriever()
    by_id = {rec.id: rec for rec in r.records}
    seq = by_id.get("kb_concept_d8c545b081")
    assert seq is not None, "SequenceName record missing"
    assert seq.title == "metadata field: SequenceName"
    assert seq.source.get("path") == "src/schema/objects/metadata.yaml"
    assert seq.raw_content["definition"]["name"] == "SequenceName"


def test_knowledge_file_override(tmp_path):
    kb = _write_tiny_kb(tmp_path, filename="custom_kb.jsonl")
    r = make_retriever(knowledge_base_dir=kb, knowledge_file="custom_kb.jsonl")
    assert r.get_record_count() == 2


def test_missing_kb_dir_returns_stable_message(tmp_path):
    r = Retriever(data_dir=str(tmp_path))
    assert r.retrieve("anything") == MSG_NO_KB.format(tmp_path)


def test_source_paths_are_repo_relative():
    r = make_retriever()
    for rec in r.records:
        path = rec.source.get("path", "")
        assert path, rec.id
        # repository-relative: no "./" prefix, no absolute or
        # machine-specific paths anywhere in the knowledge base
        assert not path.startswith("./"), path
        assert not path.startswith("/"), path
        assert "\\" not in path, path
        assert "Users" not in path, path
        assert not DRIVE_LETTER.match(path), path


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
    # The validator's TSV_VALUE_INCORRECT_TYPE issue documents this
    # column-type failure and ranks as evidence for it.
    assert "kb_validationissue_1e69723610" in out
    assert "TSV_VALUE_INCORRECT_TYPE" in out
    assert "age" in out


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
               if "JSON_SCHEMA_VALIDATION_ERROR" in rec.rule_codes]
    assert targets, "expected a record with rule code JSON_SCHEMA_VALIDATION_ERROR"
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


def test_no_relationship_machinery_is_loaded():
    """The single-file knowledge base carries no relationship graph: the
    old ``edges``/``_adjacency`` structures must be gone for good."""
    r = make_retriever()
    assert not hasattr(r, "edges")
    assert not hasattr(r, "_adjacency")


# ---------------------------------------------------------------------------
# Lazy loading of raw_content
# ---------------------------------------------------------------------------


def _write_tiny_kb(directory, filename="enriched_knowledge.jsonl"):
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
    path = os.path.join(kb, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(line) + "\n" for line in lines)
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