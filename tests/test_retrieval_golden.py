"""Golden-set retrieval tests for the four canonical sample issues.

Each case pins three things:

1. what retrieval returns (record id + match quality) when the
   knowledge base actually documents the issue;
2. that clearly irrelevant records do not outrank the relevant ones;
3. that missing coverage is reported honestly — an undocumented rule
   (``bidsmgr.todo_placeholder``) gets background-only evidence with no
   "direct" label, never invented documentation. Coverage claims are
   cross-checked against the knowledge base file itself.

Plus scoring invariants: repeated query tokens must not inflate scores,
JSON bookkeeping fields must not leak tokens, empty/malformed queries
fail honestly, and the ``explain`` audit output must agree with the
scores retrieval reported.
"""

import os

import pytest

from config import RetrievalConfig
from retriever import MSG_EMPTY_QUERY, Retriever, Scorer, _issue_structured

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "knowledge-base")
KB_FILE = os.path.join(KB_DIR, "knowledge_base.jsonl")

# Golden record ids in the single-file knowledge base.
GOLDEN_001_TOP = "kb_validationissue_5b87f24aab"
GOLDEN_003_TOP = "kb_validationissue_1e69723610"
GOLDEN_004_TOP3 = [
    "kb_concept_d8c545b081",       # metadata field: SequenceName
    "kb_concept_ed2a0e8248",       # SequenceName (glossary)
    "kb_validationissue_44c046515d",  # SIDECAR_KEY_RECOMMENDED (direct)
]

# ---------------------------------------------------------------------------
# The four sample issues (same payloads tests/test_agent.py uses)
# ---------------------------------------------------------------------------

SAMPLE_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  ·  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
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
            "message": "column 'age': value '020Y' is not valid for its type  ·  Each value must be a valid number, or 'n/a'.",
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
    "message": "missing recommended field 'SequenceName'  ·  Manufacturer's designation of the sequence name. Add it to the JSON, for example {\"SequenceName\": \"text\"}.",
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
    with open(KB_FILE, encoding="utf-8") as f:
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
        assert top_ids(res, r)[0] == GOLDEN_001_TOP

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
        # The runner-up is itself a structured field match (the
        # ReferencesAndLinks field concept), so a 2x ratio would be the
        # wrong demand — what matters is that the exact-rule record
        # leads it with a clear margin and is labelled direct.
        assert scores[0] > scores[1]
        assert scores[0] >= scores[1] * 1.3
        assert res.assessments[0].match == "rule_exact"
        assert res.assessments[0].relevance == "direct"

    def test_explain_output_agrees_with_retrieval_score(self, r):
        res = r.retrieve_issue_structured(SAMPLE_001, user_question="Explain.")
        out = r.explain("Explain.", top_k=1, issue=SAMPLE_001)
        assert "rule_id_exact" in out
        first_final = float(
            [ln for ln in out.splitlines() if ln.startswith("Final score:")][0]
            .split(":", 1)[1]
        )
        assert first_final == pytest.approx(res.scores[0])
        assert f"Record: {GOLDEN_001_TOP}" in out


# ---------------------------------------------------------------------------
# 2. bidsmgr.todo_placeholder / EthicsApprovals — the rule itself is
#    undocumented: background yes, "direct" never
# ---------------------------------------------------------------------------


class TestCase002BidsmgrTodo:
    def test_background_only_no_direct_record(self, r):
        """The field is documented; the bidsmgr rule is not.

        Retrieval may surface EthicsApprovals background records, but
        none of them documents ``bidsmgr.todo_placeholder``, so nothing
        may be labelled as that rule's documentation.
        """
        res = r.retrieve_issue_structured(SAMPLE_002,
                                          user_question="Explain this warning.")
        assert res.found is True  # background context exists
        assert res.quality == "field_exact"
        assert res.has_direct is False
        assert all(a.relevance != "direct" for a in res.assessments)

    def test_the_rule_code_is_still_absent_from_the_kb(self):
        text = kb_text()
        assert "bidsmgr" not in text        # the rule id has no record
        assert "EthicsApprovals" in text    # ...but the field does

    def test_fallback_pass_still_finds_no_direct_record(self, r):
        """The bounded fallback (identity-only pass) yields background
        at most: no pass may invent documentation for a code the
        knowledge base does not carry."""
        res = r.retrieve_issue_structured(SAMPLE_002)
        assert res.found is True
        assert res.has_direct is False
        assert all(a.relevance != "direct" for a in res.assessments)


# ---------------------------------------------------------------------------
# 3. TSV_VALUE_INCORRECT_TYPE / age 020Y — exact rule record ranks first
# ---------------------------------------------------------------------------


class TestCase003TsvAge:
    def test_exact_rule_record_ranks_first(self, r):
        res = r.retrieve_issue_structured(SAMPLE_003,
                                          user_question="What does this error mean?")
        assert res.found is True
        assert res.quality == "rule_exact"
        assert top_ids(res, r)[0] == GOLDEN_003_TOP

    def test_age_records_outrank_unrelated_ones(self, r):
        res = r.retrieve_issue_structured(SAMPLE_003, user_question="Explain.",
                                          top_k=3)
        ids = top_ids(res, r)
        assert len(ids) == 3
        # every returned record is about age / TSV values — nothing from
        # an unrelated domain shares the top of the ranking
        for rid in ids:
            rec = r.records[r._index[rid]]
            hay = (rec.title or "").lower()
            assert "age" in hay or "tsv" in hay, rid

    def test_rule_code_is_documented_in_the_kb(self):
        # The KB now carries the validator's TSV_VALUE_INCORRECT_TYPE
        # issue, so retrieval reports a real exact match instead of a
        # declared gap.
        assert "TSV_VALUE_INCORRECT_TYPE" in kb_text()

    def test_code_only_query_finds_the_documented_code(self, r):
        """With no field/path context the exact code still resolves —
        the code exists in the KB, so an exact match is the honest
        outcome, not an invented one."""
        res = r.retrieve_issue_structured(
            {"severity": "err", "rule_id": "TSV_VALUE_INCORRECT_TYPE"},
            user_question="Explain.",
        )
        assert res.found is True
        assert res.quality == "rule_exact"
        assert top_ids(res, r)[0] == GOLDEN_003_TOP


# ---------------------------------------------------------------------------
# 4. SIDECAR_KEY_RECOMMENDED / SequenceName — spec field record
# ---------------------------------------------------------------------------


class TestCase004SequenceName:
    def test_sequence_name_field_record_ranks_first(self, r):
        res = r.retrieve_issue_structured(SAMPLE_004,
                                          user_question="Explain this warning.")
        assert res.found is True
        assert res.quality == "field_exact"
        assert top_ids(res, r)[0] == "kb_concept_d8c545b081"

    def test_irrelevant_records_do_not_outrank_it(self, r):
        res = r.retrieve_issue_structured(SAMPLE_004, user_question="Explain.",
                                          top_k=5)
        ids = top_ids(res, r)
        assert ids[:3] == GOLDEN_004_TOP3
        # the remaining ranks are still sidecar/JSON context — nothing
        # from an unrelated domain cracked the top of the ranking
        for rid in ids[3:]:
            rec = r.records[r._index[rid]]
            hay = (rec.title or "").lower()
            assert "sidecar" in hay or "json" in hay, rid

    def test_spec_source_is_the_provenance(self, r):
        rec = r.records[r._index["kb_concept_d8c545b081"]]
        # Provenance points at the BIDS specification source file,
        # repo-relative with no "./" prefix.
        assert rec.source["file"] == "src/schema/objects/metadata.yaml"
        assert rec.source["path"] == "src/schema/objects/metadata.yaml"
        assert not rec.source["path"].startswith("./")
        # The raw fragment is the field's own definition.
        assert rec.raw_content["definition"]["name"] == "SequenceName"
        assert rec.requirements is None

    def test_sidecar_issue_code_is_documented_in_the_kb(self):
        # The validator's SIDECAR_KEY_RECOMMENDED issue is in the KB —
        # and indeed ranks as the direct record for SAMPLE_004.
        assert "SIDECAR_KEY_RECOMMENDED" in kb_text()


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
        for rid in ("kb_validationissue_1e69723610",   # the TSV age rule
                    "kb_validationissue_cdcb00840e",   # AGE_UNITS
                    "kb_concept_847180d3ba"):          # dataset file: participants
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
        """Codes stored in a record's rule_codes match exactly — both
        the bare validator form and the legacy ``rules.checks.*`` form."""
        res = r.retrieve_issue_structured(
            {"severity": "err", "rule_id": "AGE_UNITS",
             "message": "age units look wrong"},
            user_question="What does this mean?",
        )
        assert res.found is True
        assert res.quality == "rule_exact"
        top = top_ids(res, r)[0]
        assert top == "kb_validationissue_cdcb00840e"
        rec = r.records[r._index[top]]
        assert "AGE_UNITS" in rec.rule_codes

    def test_field_name_match_beats_incidental_word_overlap(self, r):
        """SequenceName (identifier) outranks records merely containing
        the word 'name' or 'sequence'."""
        res = r.retrieve_issue_structured(SAMPLE_004, user_question="Explain.")
        winners = {r.records[i].id for i in res.indices[:2]}
        assert "kb_concept_d8c545b081" in winners
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
        """At most two passes; a miss stays a miss afterwards."""
        # SAMPLE_001 finds on pass 1 -> results kept, no second pass.
        res = r.retrieve_issue_structured(SAMPLE_001, user_question="Explain.")
        assert res.found is True
        # An identifier absent from the KB still misses after the
        # identity-only fallback pass: the bounded walk stops with an
        # explicit outcome instead of looping or inventing a match.
        res2 = r.retrieve_issue_structured(
            {"field": "Flibbertigibbet"}, user_question="Explain.")
        assert res2.found is False
        assert res2.quality == "none"
        assert res2.assessments == []

    def test_question_only_retrieval_works_without_issue(self, r):
        res = r.retrieve_structured("What is the task entity?")
        assert res.found is True
        assert "Knowledge Item" in res.text
        # plain-text queries are lexical by nature
        assert res.quality in ("lexical", "field_exact")
