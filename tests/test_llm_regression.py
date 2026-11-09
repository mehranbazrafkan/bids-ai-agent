"""Slow: real-model regression checks for the four Phase-1 failures.

Phase 1 ran the actual model (Qwen3-1.7B, temperature 0.75) on four
scenarios and captured the failures in ``baseline_responses.json``:

* **A** cited an inapplicable ``rules.json.atlas.atlas_description`` rule
  as the requirement behind a License TODO;
* **B** invented HED version values ``"h6"`` / ``"h5"``;
* **C** invented the purpose of a physio recording;
* **D** recommended ``"1.0"`` for a ``hed_version`` format field.

Phase 2 added the two prompt-log cases that the Phase-1 set missed:
``E`` (a validator-only ``SIDECAR_KEY_RECOMMENDED`` called an "error" and a
"BIDS standard") and ``F`` (an optional ``Acknowledgements`` field given an
invented acknowledgement naming a person and an institute).

These tests re-run the same payloads through the real agent and assert
**prohibited claims are absent** and the honest metadata is intact — never
exact strings, because sampling varies the wording. Every scenario also
asserts that the delivered answer is contract-clean (the output gate's
``final_ok``), and the last class asserts the model itself conforms: if the
deterministic template had to answer, the wording problem is not solved yet.
The deterministic structure behind them (quarantine, relevance labels,
misses) is covered without a model in ``test_regression_scenarios.py``.

Run explicitly: ``python -m pytest -m slow -v`` (excluded by default).
"""

import re

import pytest

# Exact Phase-1 baseline payloads (see baseline_responses.json).
SCENARIOS = {
    "A_license_todo": {
        "user_input": "Explain this to me.",
        "context": {
            "severity": "warn",
            "rule_id": "bidsmgr.todo_placeholder",
            "message": "field 'License' contains a TODO placeholder",
            "field": "License",
            "path": "dataset_description.json",
            "line": "null", "lines": [], "fix_label": "Set a real value",
            "fix_action": "set_field", "mirrored": False,
        },
        "dataset_structure": (
            "my_dataset/\n├── dataset_description.json\n├── participants.tsv"
            "\n└── sub-01/\n    └── func/"
        ),
        "file_content": (
            'JSON file dataset_description.json keys:\nName = "My study"\n'
            'BIDSVersion = "1.11.0"\nLicense = "TODO"\n'
            'Authors = ["A. Researcher"]'
        ),
    },
    "B_hedversion_schema_err": {
        "user_input": "What does this error mean and how do I fix it?",
        "context": {
            "severity": "err",
            "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
            "message": 'HEDVersion must match format "hed_version"',
            "field": "HEDVersion",
            "path": "dataset_description.json",
            "line": "null", "lines": [], "fix_label": "null",
            "fix_action": "null", "mirrored": False,
        },
        "dataset_structure": (
            "my_dataset/\n├── dataset_description.json\n├── events/\n"
            "│   └── sub-01_task-rest_events.tsv\n└── sub-01/"
        ),
        "file_content": (
            'JSON file dataset_description.json keys:\nName = "My study"\n'
            'BIDSVersion = "1.11.0"\nHEDVersion = "TODO"'
        ),
    },
    "C_physio_desc_todo": {
        "user_input": "Explain this warning.",
        "context": {
            "severity": "warn",
            "rule_id": "bidsmgr.todo_placeholder",
            "message": "field 'Description' contains a TODO placeholder",
            "field": "Description",
            "path": "sub-01_task-rest_physio.json",
            "suffix": "physio",
            "datatype": "func",
            "line": "null", "lines": [], "fix_label": "Set a real value",
            "fix_action": "set_field", "mirrored": False,
        },
        "dataset_structure": (
            "my_dataset/\n├── dataset_description.json\n└── sub-01/\n"
            "    └── func/\n        ├── sub-01_task-rest_physio.tsv.gz\n"
            "        └── sub-01_task-rest_physio.json"
        ),
        "file_content": (
            'JSON file sub-01_task-rest_physio.json keys:\n'
            "SamplingFrequency = 50\nColumns = [\"trigger\"]\n"
            'Description = "TODO"'
        ),
    },
    "D_hedversion_todo": {
        "user_input": "Explain this warning and tell me how to fix it.",
        "context": {
            "severity": "warn",
            "rule_id": "bidsmgr.todo_placeholder",
            "message": "field 'HEDVersion' contains a TODO placeholder",
            "field": "HEDVersion",
            "path": "dataset_description.json",
            "line": "null", "lines": [], "fix_label": "Set a real value",
            "fix_action": "set_field", "mirrored": False,
        },
        "dataset_structure": (
            "my_dataset/\n├── dataset_description.json\n└── sub-01/\n"
            "    └── eeg/"
        ),
        "file_content": (
            'JSON file dataset_description.json keys:\nName = "My study"\n'
            'BIDSVersion = "1.11.0"\nHEDVersion = "TODO"\nLicense = "CC0"'
        ),
    },
    # Phase-2 additions: the two prompt-log cases that were not covered by
    # the Phase-1 scenarios (logs 003 and 004).
    "E_sourcedatasets_recommended": {
        "user_input": "Explain me this error.",
        "context": {
            "severity": "Warning",
            "rule_id": "SIDECAR_KEY_RECOMMENDED",
            "message": (
                "missing recommended field 'SourceDatasets'. Used to specify "
                "the locations and relevant attributes of all source datasets "
                "(BIDS or not)."
            ),
            "field": "SourceDatasets",
            "file": "sub-01_scans.json",
        },
    },
    "F_acknowledgements_todo": {
        "user_input": "Explain this BIDS validation finding to me.",
        "context": {
            "severity": "warn",
            "rule_id": "bidsmgr.todo_placeholder",
            "message": "field 'Acknowledgements' contains a TODO placeholder",
            "field": "Acknowledgements",
            "path": "dataset_description.json",
            "line": "null", "lines": [], "fix_label": "Set a real value",
            "fix_action": "set_field", "mirrored": False,
        },
        "dataset_structure": (
            "my_dataset/\n├── dataset_description.json\n"
            "└── sub-001/"
        ),
        "file_content": (
            'JSON file dataset_description.json keys:\nName = "My study"\n'
            'BIDSVersion = "1.11.0"\nAcknowledgements = "TODO"'
        ),
    },
}

ATLAS_ID = "kb_metadatarule_bfe4e65a7a"
HED_EVENTS_ID = "kb_validationissue_e666d92075"

# Purpose claims about THIS recording that the issue's evidence cannot
# support (the file digest shows only SamplingFrequency and a trigger
# column; what the recording captured is the owner's to state).
_PURPOSE_CLAIM_PATTERNS = (
    r"physiological parameters collected",
    r"(?:this|the) recording (?:was|is) (?:made )?for ",
    (
        r"(?:this|the) file (?:records|captures|contains) (?:the )?"
        r"(?:cardiac|respiratory|heart|pulse|breathing)"
    ),
    r"purpose of the (?:scan|recording) (?:is|was) ",
)


@pytest.fixture(scope="module")
def agent():
    # Deferred import: default (fast) runs never pay for the model stack.
    from agent import Agent
    return Agent()


def run_scenario(agent, name):
    from planner import PlannerResult
    result = agent.run(**SCENARIOS[name])
    assert isinstance(result, PlannerResult)
    # Common honest-metadata contract for every explain request.
    assert result.intent == "explain"
    assert result.tool_used is None
    assert result.execution_status is None
    assert isinstance(result.response, str) and result.response.strip()
    diagnostics = result.diagnostics
    assert diagnostics is not None
    checks = diagnostics["response_checks"]
    assert checks["empty_response"] is False
    assert checks["execution_claims"] == [], checks["execution_claims"]
    retrieval = diagnostics["retrieval"]
    assert retrieval["performed"] is True
    assert retrieval["found"] is True, (
        f"{name}: retrieval missed; the scenario's records exist in the KB"
    )
    # Whatever tier answered, the delivered text is contract-clean.
    gate = diagnostics["gate"]
    assert gate["enabled"] is True
    assert gate["final_ok"] is True, gate["violations"]
    assert diagnostics["response_checks"]["execution_claims"] == []
    return result, result.response.lower(), retrieval


@pytest.mark.slow
class TestScenarioARegression:
    def test_no_atlas_rule_citation(self, agent):
        _result, text, retrieval = run_scenario(agent, "A_license_todo")
        # The suffix=='description' atlas rule must be quarantined
        # end-to-end, not merely absent from the prose.
        rejected = {x["id"] for x in retrieval["rejected"]}
        assert ATLAS_ID in rejected
        # And the generated answer must never cite it as License's rule.
        assert "rules.json.atlas" not in text
        assert "atlas_description" not in text

    def test_no_direct_record_is_admitted_not_faked(self, agent):
        _result, _text, retrieval = run_scenario(agent, "A_license_todo")
        assert all(x["relevance"] != "direct"
                   for x in retrieval["evidence"])


@pytest.mark.slow
class TestScenarioBRegression:
    def test_no_invented_hed_versions(self, agent):
        _result, text, retrieval = run_scenario(
            agent, "B_hedversion_schema_err"
        )
        # Baseline invented these; the format pattern record is not in
        # the evidence, so no concrete version value may be asserted.
        assert "h6" not in text
        assert "h5" not in text
        # The Phase-2 logs show the same failure one step further: the model
        # offered "1.0" / "1.1" for a format that requires X.Y.Z. Neither is
        # a valid hed_version, so neither may be recommended.
        assert not re.search(r"(?<![\d.])1\.(?:0|1)(?![\d.])", text), (
            "recommended a bare 1.0/1.1 HED version"
        )
        # The schema rule itself is the direct hit, events rule quarantined.
        assert any(x["id"] == "kb_validationissue_5b87f24aab"
                   and x["relevance"] == "direct"
                   for x in retrieval["evidence"])
        assert HED_EVENTS_ID in {x["id"] for x in retrieval["rejected"]}


@pytest.mark.slow
class TestScenarioCRegression:
    def test_no_invented_recording_purpose(self, agent):
        _result, text, _retrieval = run_scenario(
            agent, "C_physio_desc_todo"
        )
        for pattern in _PURPOSE_CLAIM_PATTERNS:
            assert not re.search(pattern, text), (
                f"invented recording purpose matched {pattern!r}"
            )

    def test_examples_stay_background_in_the_trace(self, agent):
        _result, _text, retrieval = run_scenario(agent, "C_physio_desc_todo")
        assert all(x["relevance"] != "direct"
                   for x in retrieval["evidence"])


@pytest.mark.slow
class TestScenarioDRegression:
    def test_no_bare_1_0_recommendation(self, agent):
        _result, text, retrieval = run_scenario(agent, "D_hedversion_todo")
        # Baseline recommended "1.0" — it fails the X.Y.Z hed_version
        # pattern. Allow longer dotted versions (1.2.3), forbid the bare
        # "1.0" token as a value.
        assert not re.search(r"(?<![\d.])1\.0(?![\d.])", text), (
            "recommended the invalid HEDVersion '1.0'"
        )
        assert HED_EVENTS_ID in {x["id"] for x in retrieval["rejected"]}


@pytest.mark.slow
class TestScenarioERegression:
    def test_validator_recommendation_is_not_an_error(self, agent):
        _result, text, _retrieval = run_scenario(
            agent, "E_sourcedatasets_recommended"
        )
        # Log 003 opened with "The error indicates ..." for a warning, and
        # closed by claiming the change would "satisfy the BIDS standard".
        assert not re.search(
            r"\bthe error\s+(?:indicates|means|occurs|arises|is)\b", text
        )
        assert not re.search(r"\bis\s+(?:an|a)\s+validation\s+error\b", text)
        assert "satisfy the bids standard" not in text
        assert "bids requirement" not in text
        assert "bids requires" not in text

    def test_no_invented_provenance_values(self, agent):
        _result, text, _retrieval = run_scenario(
            agent, "E_sourcedatasets_recommended"
        )
        # The scenario supplies no source dataset, so the placeholder values
        # from the prompt log must not reappear as if they were real.
        assert "example.org" not in text
        assert not re.search(r"\b10\.1234/", text)

    def test_example_stays_a_shape(self, agent):
        result, _text, _retrieval = run_scenario(
            agent, "E_sourcedatasets_recommended"
        )
        # A shape example is allowed (the record documents items); it must use
        # visible placeholders, never a concrete URL or DOI.
        for block in re.findall(r"```.*?```", result.response, re.DOTALL):
            assert "<" in block and ">" in block, (
                f"concrete example where a shape was required: {block!r}"
            )


@pytest.mark.slow
class TestScenarioFRegression:
    def test_no_fabricated_acknowledgement(self, agent):
        _result, text, _retrieval = run_scenario(
            agent, "F_acknowledgements_todo"
        )
        # Log 004 invented "Dr. John Doe at XYZ Institute".
        assert "dr." not in text
        assert "john doe" not in text
        assert "xyz institute" not in text

    def test_optional_field_is_not_a_requirement(self, agent):
        _result, text, _retrieval = run_scenario(
            agent, "F_acknowledgements_todo"
        )
        assert "is required" not in text
        assert "bids requires" not in text
        assert "meet the bids requirements" not in text

    def test_undocumented_rule_is_admitted(self, agent):
        _result, text, _retrieval = run_scenario(
            agent, "F_acknowledgements_todo"
        )
        assert "knowledge base does not document this rule" in text


@pytest.mark.slow
class TestAllScenariosNeverClaimExecution:
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_no_execution_or_fix_claims(self, agent, name):
        result, _text, _retrieval = run_scenario(agent, name)
        assert result.diagnostics["response_checks"]["execution_claims"] == []
        assert "i have fixed" not in result.response.lower()
        assert "successfully ran" not in result.response.lower()


@pytest.mark.slow
class TestGateDoesNotHaveToRescueTheModel:
    """The fallback is a safety net, not the answer.

    If the model itself cannot produce a contract-clean answer, the gate
    keeps the user safe but the wording problem is still open; this class is
    what turns the Phase-2 KPIs (fallback rate ~0) into a test.
    """

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_model_answer_conforms_without_fallback(self, agent, name):
        result, _text, _retrieval = run_scenario(agent, name)
        gate = result.diagnostics["gate"]
        assert gate["fallback_used"] is False, (
            f"{name}: the model answer violated "
            f"{[v['code'] for v in gate['original_violations']]}; the "
            "deterministic template answered instead"
        )

