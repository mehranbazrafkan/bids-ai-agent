"""Slow: real-model regression checks for the four Phase-1 failures.

Phase 1 ran the actual model (Qwen3-1.7B, temperature 0.75) on four
scenarios and captured the failures in ``baseline_responses.json``:

* **A** cited an inapplicable ``rules.json.atlas.atlas_description`` rule
  as the requirement behind a License TODO;
* **B** invented HED version values ``"h6"`` / ``"h5"``;
* **C** invented the purpose of a physio recording;
* **D** recommended ``"1.0"`` for a ``hed_version`` format field.

These tests re-run the same payloads through the real agent and assert
**prohibited claims are absent** and the honest metadata is intact — never
exact strings, because sampling at temperature 0.75 varies the wording.
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
class TestAllScenariosNeverClaimExecution:
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_no_execution_or_fix_claims(self, agent, name):
        result, _text, _retrieval = run_scenario(agent, name)
        assert result.diagnostics["response_checks"]["execution_claims"] == []
        assert "i have fixed" not in result.response.lower()
        assert "successfully ran" not in result.response.lower()
