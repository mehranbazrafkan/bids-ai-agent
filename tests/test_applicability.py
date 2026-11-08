"""Unit tests for applicability.py — deterministic selector evaluation.

The applicability stage answers one question without any LLM involved:

    Given a record's BIDS selectors and the issue's file context, does
    the record apply to this issue's file?

Design rules pinned here:

* **Conservative**: an un-evaluatable selector is *unknown* and never
  causes a rejection (unknown never rejects).
* **Reject only on definite conflict**: a record is inapplicable only
  when at least one selector evaluates to False.
* **AND semantics**: BIDS schema selectors all must hold, so one False
  rules the record out while one True is not enough to keep it alone.
"""

import pytest

from applicability import (
    IssueScope,
    build_issue_scope,
    classify_selectors,
    evaluate_selector,
    extract_selectors,
    infer_suffix,
)

# ---------------------------------------------------------------------------
# Scope construction
# ---------------------------------------------------------------------------


class TestBuildIssueScope:
    def test_path_is_decomposed(self):
        scope = build_issue_scope({"path": "sub-01/sub-01_task-rest_physio.json"})
        assert scope.basename == "sub-01_task-rest_physio.json"
        assert scope.extension == ".json"
        assert scope.stem == "sub-01_task-rest_physio"
        assert scope.suffix == "physio"          # inferred from entity name
        assert scope.suffix_source == "inferred"
        assert "sub" in scope.entities and "task" in scope.entities

    def test_non_entity_stem_keeps_whole_name_as_suffix(self):
        # ``dataset_description`` must NOT degrade to ``description`` —
        # that would wrongly satisfy selectors like suffix == 'description'.
        scope = build_issue_scope({"path": "dataset_description.json"})
        assert scope.suffix == "dataset_description"
        assert scope.extension == ".json"

    def test_issue_suffix_wins_over_inference(self):
        scope = build_issue_scope({"path": "sub-01_events.tsv", "suffix": "events"})
        assert scope.suffix == "events"
        assert scope.suffix_source == "issue"

    def test_nii_gz_uses_compound_extension(self):
        scope = build_issue_scope({"path": "sub-01_T1w.nii.gz"})
        assert scope.extension == ".nii.gz"
        assert scope.stem == "sub-01_T1w"
        assert scope.suffix == "T1w"

    def test_nested_issues_are_merged(self):
        scope = build_issue_scope({
            "path": "sub-01/sub-01_task-rest_physio.json",
            "issues": [{"suffix": "physio", "datatype": "func"}],
        })
        assert scope.datatype == "func"
        assert scope.suffix == "physio"
        assert scope.suffix_source == "issue"

    def test_empty_issue_is_not_known(self):
        scope = build_issue_scope({})
        assert not scope.known
        assert scope.path is None and scope.suffix is None

    def test_non_dict_issue_is_not_known(self):
        assert not build_issue_scope(None).known
        assert not build_issue_scope(["severity", "err"]).known

    def test_null_string_values_are_ignored(self):
        scope = build_issue_scope({"path": None, "datatype": "null"})
        assert scope.datatype is None


class TestInferSuffix:
    @pytest.mark.parametrize("stem,expected", [
        ("sub-01_task-rest_physio", "physio"),
        ("sub-01_T1w", "T1w"),
        ("dataset_description", "dataset_description"),
        ("participants", "participants"),
        ("CITATION", "CITATION"),
        ("", None),
    ])
    def test_infer_suffix(self, stem, expected):
        assert infer_suffix(stem) == expected


# ---------------------------------------------------------------------------
# Selector evaluation (three-valued: True / False / None)
# ---------------------------------------------------------------------------


class TestEvaluateSelector:
    def test_suffix_equality(self):
        events = build_issue_scope({"path": "sub-01_task-rest_events.tsv",
                                    "suffix": "events"})
        assert evaluate_selector('suffix == "events"', events) is True
        assert evaluate_selector('suffix == "physio"', events) is False
        assert evaluate_selector('suffix != "physio"', events) is True

    def test_unknown_suffix_is_none_not_false(self):
        # No suffix in the context at all -> the selector cannot be
        # evaluated: None (unknown), never False (which would reject).
        scope = IssueScope(extension=".dat")
        assert evaluate_selector('suffix == "events"', scope) is None

    def test_extension_equality(self):
        scope = build_issue_scope({"path": "sub-01_events.tsv"})
        assert evaluate_selector("extension == '.tsv'", scope) is True
        assert evaluate_selector("extension == '.json'", scope) is False

    def test_path_compares_by_file_name(self):
        scope = build_issue_scope({"path": "my_dataset/dataset_description.json"})
        assert evaluate_selector('path == "/dataset_description.json"', scope) is True
        assert evaluate_selector('path == "/events.tsv"', scope) is False

    def test_path_unknown_is_none(self):
        scope = IssueScope()  # no path at all
        assert evaluate_selector('path == "/dataset_description.json"', scope) is None

    def test_intersects_suffix(self):
        scope = build_issue_scope({"path": "sub-01_dwi.nii.gz", "suffix": "dwi"})
        assert evaluate_selector('intersects([suffix], ["dwi", "epi"])', scope) is True
        assert evaluate_selector('intersects([suffix], ["bold", "sbref"])', scope) is False

    def test_entity_presence(self):
        scope = build_issue_scope({"path": "atlas-MNI152_description.json"})
        assert evaluate_selector('"atlas" in entities', scope) is True
        assert evaluate_selector('"run" in entities', scope) is False
        assert evaluate_selector("entities.atlas", scope) is True
        assert evaluate_selector("entities.run", scope) is False

    def test_entity_selector_without_path_is_unknown(self):
        scope = IssueScope(suffix="description")  # no path -> entities unknown
        assert evaluate_selector('"atlas" in entities', scope) is None
        assert evaluate_selector("entities.atlas", scope) is None

    def test_match_extension_regex(self):
        nii = build_issue_scope({"path": "sub-01_T1w.nii.gz"})
        tsv = build_issue_scope({"path": "sub-01_events.tsv"})
        # KB selectors arrive JSON-unescaped: '\\.' in the file -> '\.'
        sel = r"match(extension, '^\.nii(\.gz)?$')"
        assert evaluate_selector(sel, nii) is True
        assert evaluate_selector(sel, tsv) is False

    def test_bad_regex_is_unknown_not_a_crash(self):
        scope = build_issue_scope({"path": "x.tsv"})
        assert evaluate_selector(r"match(extension, '([bad')", scope) is None

    def test_unmodelled_selector_is_unknown(self):
        # Sidecar/state/function-call selectors are out of scope: they
        # must evaluate to unknown, never to a rejection.
        scope = build_issue_scope({"path": "sub-01_events.tsv", "suffix": "events"})
        for sel in (
            'sidecar in ["scans"]',
            'dataset.modalities.contains("mr")',
            'has_nifti',
            'nifti_header.TE == 0.01',
        ):
            assert evaluate_selector(sel, scope) is None, sel

    def test_empty_selector_is_unknown(self):
        assert evaluate_selector("", build_issue_scope({"path": "x.tsv"})) is None


# ---------------------------------------------------------------------------
# Classification (AND semantics, conservative)
# ---------------------------------------------------------------------------


class TestClassifySelectors:
    def test_no_selectors_is_unknown(self):
        result = classify_selectors((), build_issue_scope({"path": "x.tsv"}))
        assert result.status == "unknown"
        assert not result.is_inapplicable

    def test_unknown_scope_never_rejects(self):
        # No file context: even a strict selector must not rule the
        # record out — we simply cannot tell.
        result = classify_selectors(['suffix == "events"'], IssueScope())
        assert result.status == "unknown"
        assert not result.is_inapplicable

    def test_all_selectors_hold_is_applicable(self):
        scope = build_issue_scope({"path": "sub-01_task-rest_events.tsv",
                                   "suffix": "events"})
        result = classify_selectors(
            ['suffix == "events"', 'extension == ".tsv"'], scope
        )
        assert result.status == "applicable"
        assert result.checked == ('suffix == "events"', 'extension == ".tsv"')

    def test_one_false_selector_rules_the_record_out(self):
        # AND semantics: dataset_description.json fails suffix == 'events'
        # even though extension == '.json' would hold for a JSON file.
        scope = build_issue_scope({"path": "dataset_description.json"})
        result = classify_selectors(
            ['suffix == "events"', 'extension == ".json"'], scope
        )
        assert result.status == "inapplicable"
        assert result.is_inapplicable
        assert 'suffix == "events"' in result.reason

    def test_partial_match_with_unknown_is_applicable_not_rejected(self):
        scope = build_issue_scope({"path": "sub-01_task-rest_events.tsv",
                                   "suffix": "events"})
        # One holds, one cannot be evaluated (modality): keep the record.
        result = classify_selectors(
            ['suffix == "events"', 'intersects([modality], ["mr"])'], scope
        )
        assert result.status == "applicable"
        assert result.unverified == ('intersects([modality], ["mr"])',)

    def test_all_unknown_is_unknown(self):
        scope = build_issue_scope({"path": "dataset_description.json"})
        result = classify_selectors(['sidecar in ["scans"]'], scope)
        assert result.status == "unknown"

    def test_selector_conflict_reason_names_the_file(self):
        scope = build_issue_scope({"path": "dataset_description.json"})
        result = classify_selectors(["entities.atlas"], scope)
        assert result.status == "inapplicable"
        assert "dataset_description.json" in result.reason

    def test_string_selector_is_accepted(self):
        scope = build_issue_scope({"path": "dataset_description.json"})
        result = classify_selectors('path == "/dataset_description.json"', scope)
        assert result.status == "applicable"

    def test_malformed_selector_is_unknown(self):
        scope = build_issue_scope({"path": "x.tsv"})
        assert classify_selectors(123, scope).status == "unknown"
        assert classify_selectors(None, scope).status == "unknown"


# ---------------------------------------------------------------------------
# Selector extraction from raw KB records
# ---------------------------------------------------------------------------


class TestExtractSelectors:
    def test_from_rule_definition(self):
        raw = {"rule_definition": {"selectors": ['suffix == "events"']}}
        assert extract_selectors({"raw_content": raw}) == ('suffix == "events"',)

    def test_from_original_rule_definition(self):
        raw = {"original_rule_definition": {"selectors": ["path == '/x'"]}}
        assert extract_selectors({"raw_content": raw}) == ("path == '/x'",)

    def test_from_nested_definition(self):
        raw = {"issue_definition": {"selectors": ["extension == '.tsv'"]}}
        assert extract_selectors({"raw_content": raw}) == ("extension == '.tsv'",)

    def test_missing_raw_or_selectors_is_empty(self):
        assert extract_selectors({}) == ()
        assert extract_selectors({"raw_content": None}) == ()
        assert extract_selectors({"raw_content": {"rule_definition": {}}}) == ()

    def test_non_list_selectors_are_ignored(self):
        raw = {"rule_definition": {"selectors": "suffix == 'events'"}}
        assert extract_selectors({"raw_content": raw}) == ()
