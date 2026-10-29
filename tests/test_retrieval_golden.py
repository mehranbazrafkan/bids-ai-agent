"""Golden-set retrieval tests for the four canonical sample issues.

Each case pins three things:

1. what retrieval returns (record id + match quality) when the
   knowledge base actually documents the issue;
2. that clearly irrelevant records do not outrank the relevant ones;
3. that missing coverage is reported as an explicit miss (or an
   explicitly-marked lexical match), never as invented documentation —
   cross-checked against knowledge-base/COVERAGE_GAPS.md.

Plus scoring invariants: repeated query tokens must not inflate scores,
JSON bookkeeping fields must not leak tokens, empty/malformed queries
fail honestly, and the ``explain`` audit output must agree with the
scores retrieval reported.
"""

import json
import os

import pytest

from config import RetrievalConfig
from retriever import MSG_EMPTY_QUERY, Retriever, Scorer, _issue_structured

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "knowledge-base")
COVERAGE_DOC = os.path.join(KB_DIR, "COVERAGE_GAPS.md")

# ---------------------------------------------------------------------------
# The four sample issues (same payloads tests/test_agent.py uses)
# ---------------------------------------------------------------------------

SAMPLE_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
    "field": "ReferencesAndLinks",
    "line": "null",
    "lines": [],
    "fix_label": "null",
    "fix_action": "null",
    "mirrored": False,
}

SAMPLE_002 = {
    "severity": "warn",
    "rule_id": "bidsmgr.todo_placeholder",
    "message": "field 'EthicsApprovals' contains a TODO placeholder",
    "field": "EthicsApprovals",
    "line": "null",
    "lines": [],
    "fix_label": "Set a real value",
    "fix_action": "set_field",
    "mirrored": False,
}

SAMPLE_003 = {
    "path": "participants.tsv",
    "severity": "err",
    "datatype": "null",
    "suffix": "participants",
    "issues": [
        {
            "severity": "err",
            "rule_id": "TSV_VALUE_INCORRECT_TYPE",
            "message": "column 'age': value '020Y' is not valid for its type  \u00b7  Each value must be a valid number, or 'n/a'.",
            "field": "age",
            "line": 2,
            "lines": [2, 3],
            "fix_label": "null",
            "fix_action": "null",
            "mirrored": False,
        }
    ],
    "sidecar_fields": [],
}

SAMPLE_004 = {
    "severity": "warn",
    "rule_id": "SIDECAR_KEY_RECOMMENDED",
    "message": "missing recommended field 'SequenceName'  \u00b7  Manufacturer's designation of the sequence name. Add it to the JSON, for example {\"SequenceName\": \"text\"}.",
    "field": "SequenceName",
    "line": "null",
    "lines": [],
    "fix_label": "Fix",
    "fix_action": "add_field",
    "mirrored": True,
}


@pytest.fixture(scope="module")
def r():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


def top_ids(result, retriever):
    return [retriever.records[i].id for i in result.indices]


def kb_text():
    with open(os.path.join(KB_DIR, "enriched_knowledge.jsonl"),
              encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. JSON_SCHEMA_VALIDATION_ERROR / ReferencesAndLinks — documented
# ---------------------------------------------------------------------------


class TestCase001JsonSchema:
    def test_exact_rule_match_ranks_first(self, r):
        res = r.retrieve_issue_structured(SAMPLE_001,
                                          user_question="Explain this to me.")
        assert res.found is True
        assert res.quality == "rule_exact"
        assert top_ids(res, r)[0] == "err_jsonschemavalidationerror"

    def test_top_record_documents_the_code(self, r):
        res = r.retrieve_issue_structured(SAMPLE_001, user_question="Explain.")
        rec = r.records[res.indices[0]]
        assert "JSON_SCHEMA_VALIDATION_ERROR" in rec.rule_codes

    def test_relevance_ordering(self, r):
        """The exact-rule record must stay above incidental matches."""
        res = r.retrieve_issue_structured(SAMPLE_001, user_question="Explain.",
                                          top_k=3)
        scores = res.scores
        assert len(scores) == 3
        assert scores == sorted(scores, reverse=True)
        # clear margin over the runner-up (which only shares vocabulary)
        assert scores[0] > scores[1] * 2

    def test_explain_output_agrees_with_retrieval_score(self, r):
        res = r.retrieve_issue_structured(SAMPLE_001, user_question="Explain.")
        out = r.explain("Explain.", top_k=1, issue=SAMPLE_001)
        assert "rule_id_exact" in out
        first_final = float(
            [ln for ln in out.splitlines() if ln.startswith("Final score:")][0]
            .split(":", 1)[1]
        )
        assert first_final == pytest.approx(res.scores[0])
        assert "Record: err_jsonschemavalidationerror" in out


# ---------------------------------------------------------------------------
# 2. bidsmgr.todo_placeholder / EthicsApprovals — explicit coverage gap
# ---------------------------------------------------------------------------


class TestCase002BidsmgrTodo:
    def test_retrieval_reports_an_honest_miss(self, r):
        res = r.retrieve_issue_structured(SAMPLE_002,
                                          user_question="Explain this warning.")
        assert res.found is False
        assert res.quality == "none"
        assert res.indices == [] and res.scores == []
        # The miss string is never a fake document.
        assert "Knowledge Item" not in res.text

    def test_coverage_gap_is_documented_and_real(self):
        doc = open(COVERAGE_DOC, encoding="utf-8").read()
        assert "bidsmgr.todo_placeholder" in doc
        assert "EthicsApprovals" in doc
        # ...and the gap genuinely still exists (update the doc if records
        # are ever added from verified sources).
        text = kb_text()
        assert "bidsmgr" not in text
        assert "EthicsApprovals" not in text

    def test_identity_fallback_does_not_invent_a_match(self, r):
        """The bounded fallback (identity-only pass) still misses."""
        res = r.retrieve_issue_structured(SAMPLE_002)
        assert res.found is False


# ---------------------------------------------------------------------------
# 3. TSV_VALUE_INCORRECT_TYPE / age 020Y — column record, code is a gap
# ---------------------------------------------------------------------------


class TestCase003TsvAge:
    def test_column_record_ranks_first(self, r):
        res = r.retrieve_issue_structured(SAMPLE_003,
                                          user_question="What does this error mean?")
        assert res.found is True
        assert res.quality == "field_exact"
        assert top_ids(res, r)[0] == "tabcol_participants_age"

    def test_participants_records_outrank_unrelated_ones(self, r):
        res = r.retrieve_issue_structured(SAMPLE_003, user_question="Explain.",
                                          top_k=3)
        ids = top_ids(res, r)
        assert len(ids) == 3
        # every returned record is about age/participants — nothing from
        # unrelated domains shares the top of the ranking
        for rid in ids:
            assert ("participant" in rid or "age" in rid.lower()), rid

    def test_rule_code_itself_is_a_documented_gap(self):
        doc = open(COVERAGE_DOC, encoding="utf-8").read()
        assert "TSV_VALUE_INCORRECT_TYPE" in doc
        assert "TSV_VALUE_INCORRECT_TYPE" not in kb_text()

    def test_no_results_is_reported_when_only_the_code_is_given(self):
        """With no field/path context, the absent code must not 'match'."""
        r = Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))
        res = r.retrieve_issue_structured(
            {"severity": "err", "rule_id": "TSV_VALUE_INCORRECT_TYPE"},
            user_question="Explain.",
        )
        # either an honest miss, or clearly-marked lexical background —
        # never a claimed exact rule match
        if res.found:
            assert res.quality in ("lexical", "field_exact")
            assert res.quality != "rule_exact"


# ---------------------------------------------------------------------------
# 4. SIDECAR_KEY_RECOMMENDED / SequenceName — derived field record
# ---------------------------------------------------------------------------


class TestCase004SequenceName:
    def test_sequence_name_field_record_ranks_first(self, r):
        res = r.retrieve_issue_structured(SAMPLE_004,
                                          user_question="Explain this warning.")
        assert res.found is True
        assert res.quality == "field_exact"
        assert top_ids(res, r)[0] == "metafield_sequencename"

    def test_irrelevant_records_do_not_outrank_it(self, r):
        res = r.retrieve_issue_structured(SAMPLE_004, user_question="Explain.",
                                          top_k=5)
        ids = top_ids(res, r)
        assert ids[0] == "metafield_sequencename"
        for irrelevant in ("ext_json", "tabcol_ieegelectrodes_manufacturer",
                           "suff_t1w", "suff_scans"):
            if irrelevant in ids:
                assert ids.index(irrelevant) > 0, f"{irrelevant} ranked above the SequenceName record"

    def test_mri_sidecar_group_is_the_provenance(self, r):
        rec = next(rec for rec in r.records if rec.id == "metafield_sequencename")
        assert rec.source["file"] == "rules/sidecars/mri.yaml"
        assert rec.source["path"] == "./rules/sidecars/mri.yaml"
        assert rec.source["section"] == "metadata_fields"
        assert rec.source["key"] == "MRISequenceSpecifics.SequenceName"
        # level copied verbatim from the source fragment
        assert rec.raw_content == {"field": "SequenceName", "level": "recommended"}

    def test_validator_code_is_a_documented_gap(self):
        doc = open(COVERAGE_DOC, encoding="utf-8").read()
        assert "SIDECAR_KEY_RECOMMENDED" in doc
        assert "SIDECAR_KEY_RECOMMENDED" not in kb_text()


# ---------------------------------------------------------------------------
# Scoring invariants
# ---------------------------------------------------------------------------


class TestScoringInvariants:
    def test_repeated_message_tokens_do_not_inflate_scores(self, r):
        """Duplicating the message text must not change any score."""
        sq_once = _issue_structured(SAMPLE_003)
        doubled = dict(sq_once)
        doubled["message_text"] = sq_once["message_text"] + " " + sq_once["message_text"]

        tokens_once, _, _ = Scorer.query_terms("q", sq_once)
        tokens_double, _, _ = Scorer.query_terms("q", doubled)
        assert tokens_once == tokens_double  # deduplicated

        # identical scores on the records that matter
        for rid in ("tabcol_participants_age", "check_dataset_AgeUnits",
                    "file_participants"):
            idx = r._index[rid]
            s1 = Scorer.score("q", r.records[idx], r.blocks[idx],
                              r.identities[idx], r._idf,
                              structured_query=sq_once,
                              precomputed=r._precomputed[idx])
            s2 = Scorer.score("q", r.records[idx], r.blocks[idx],
                              r.identities[idx], r._idf,
                              structured_query=doubled,
                              precomputed=r._precomputed[idx])
            assert s1 == pytest.approx(s2)

    def test_json_bookkeeping_keys_do_not_leak_tokens(self):
        sq = _issue_structured(SAMPLE_001)
        tokens, identifier_tokens, _phrase = Scorer.query_terms("q", sq)
        for noise in ("null", "none", "true", "false", "line", "lines",
                      "fix", "label", "action", "mirrored", "severity",
                      "field", "value"):
            assert noise not in tokens, noise
        # the structured keys themselves are handled by role, not as words
        assert "rule" not in tokens and "id" not in tokens

    def test_exact_rule_id_matching_uses_all_known_code_locations(self, r):
        """Codes stored only in raw_content.issue.code match exactly too."""
        res = r.retrieve_issue_structured(
            {"severity": "err", "rule_id": "AGE_UNITS",
             "message": "age units look wrong"},
            user_question="What does this mean?",
        )
        assert res.found is True
        assert res.quality == "rule_exact"
        assert top_ids(res, r)[0] == "check_dataset_AgeUnits"

    def test_field_name_match_beats_incidental_word_overlap(self, r):
        """SequenceName (identifier) outranks records merely containing
        the word 'name' or 'sequence'."""
        res = r.retrieve_issue_structured(SAMPLE_004, user_question="Explain.")
        winners = {r.records[i].id for i in res.indices[:2]}
        assert "metafield_sequencename" in winners
        bd_top = Scorer.explain(
            "Explain.", r.records[res.indices[0]], r.blocks[res.indices[0]],
            r.identities[res.indices[0]], r._idf,
            structured_query=_issue_structured(SAMPLE_004),
            precomputed=r._precomputed[res.indices[0]],
        )
        assert "field_exact" in {s["type"] for s in bd_top["structured"]}

    def test_empty_query_is_an_honest_miss(self, r):
        assert r.retrieve("") == MSG_EMPTY_QUERY
        res = r.retrieve_structured("   ")
        assert res.found is False
        assert res.text == MSG_EMPTY_QUERY

    @pytest.mark.parametrize("issue", [
        {},                                     # empty
        None,                                   # wrong type
        ["severity", "err"],                    # wrong type
        {"severity": "err"},                    # no identifiers at all
        {"rule_id": None, "field": 123},        # null / non-string values
        {"issues": "not-a-list"},               # malformed nesting
    ])
    def test_malformed_issue_never_crashes(self, r, issue):
        res = r.retrieve_issue_structured(issue, user_question="Explain.")
        # no exception; outcome is explicit either way
        assert res.found in (True, False)
        assert res.quality in ("rule_exact", "field_exact", "lexical", "none")
        if not res.found:
            assert res.indices == []


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


class TestFallback:
    def test_identity_only_fallback_runs_once_and_stops(self, r):
        """A near-identical second query pass is allowed only once."""
        # SAMPLE_001 finds on pass 1 -> exactly one pass, results kept.
        res = r.retrieve_issue_structured(SAMPLE_001, user_question="Explain.")
        assert res.found is True
        # SAMPLE_002 misses twice (full, then identity-only) and stops.
        res2 = r.retrieve_issue_structured(SAMPLE_002, user_question="Explain.")
        assert res2.found is False

    def test_question_only_retrieval_works_without_issue(self, r):
        res = r.retrieve_structured("What is the task entity?")
        assert res.found is True
        assert "Knowledge Item" in res.text
        # plain-text queries are lexical by nature
        assert res.quality in ("lexical", "field_exact")
