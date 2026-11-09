"""The deterministic answer contract (``answer_contract.py``).

These tests pin the judgements that used to be left to the model: the BIDS
level of the field, the authority behind the finding, whether the value is a
placeholder, and whether an example may be shown. Getting any of them wrong is
what produced the Phase-2 failures, so they are asserted per recorded case.
"""

import os

import pytest

from answer_contract import (
    AUTHORITY_MANAGER,
    AUTHORITY_VALIDATOR,
    NO_DIRECT_SENTENCE,
    AnswerContract,
    build_contract,
    derive_bids_level,
    normalise_severity,
    render_contract,
)
from config import RetrievalConfig
from retriever import Retriever
from tests.data.output_fixtures import CASES

KB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge-base",
)


@pytest.fixture(scope="module")
def kb():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


def contract_for(kb, case):
    retrieval = kb.retrieve_issue_structured(
        case["issue"], user_question="Explain this."
    )
    return build_contract(case["issue"], retrieval, case["file_content"])


# ---------------------------------------------------------------------------
# The four recorded cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_contract_matches_the_recorded_case(kb, case):
    contract = contract_for(kb, case)
    for key, expected in case["expected_contract"].items():
        assert getattr(contract, key) == expected, (
            f"{case['name']}: {key} should be {expected!r}"
        )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_contract_renders_every_obligation(kb, case):
    contract = contract_for(kb, case)
    text = render_contract(contract)
    assert "[ANSWER CONTRACT" in text
    assert contract.rule_id in text
    if contract.field:
        assert contract.field in text
    if contract.requires_no_direct_statement:
        assert NO_DIRECT_SENTENCE in text
    if contract.bids_level and contract.bids_level != "required":
        assert "do NOT call it required" in text
    assert ("example: not allowed" in text
            if not contract.example_allowed
            else "example: shape only" in text)


def test_license_is_recommended_not_required(kb):
    """The exact inversion in log 001: the spec says recommended."""
    contract = contract_for(kb, CASES[0])
    assert contract.bids_level == "recommended"


def test_acknowledgements_is_optional(kb):
    """The exact inversion in log 004: the spec says optional."""
    contract = contract_for(kb, CASES[3])
    assert contract.bids_level == "optional"


def test_sourcedatasets_authority_is_validator(kb):
    """The exact inversion in log 003: validator-only rule, not the spec."""
    contract = contract_for(kb, CASES[2])
    assert contract.authority == AUTHORITY_VALIDATOR


def test_hed_version_example_may_be_a_shape_but_never_a_value(kb):
    contract = contract_for(kb, CASES[1])
    assert contract.example_allowed is True
    assert contract.concrete_value_allowed is False


# ---------------------------------------------------------------------------
# Derivation rules
# ---------------------------------------------------------------------------


def test_manager_prefix_wins_over_the_records():
    contract = build_contract({
        "severity": "warn",
        "rule_id": "bidsmgr.todo_placeholder",
        "field": "License",
    })
    assert contract.authority == AUTHORITY_MANAGER


def test_no_identifiers_means_no_contract():
    assert build_contract({}) is None
    assert build_contract({"severity": "warn"}) is None
    assert build_contract({"message": "something happened"}) is None


def test_unknown_level_is_not_invented():
    assert derive_bids_level("SomeUnknownField", []) is None
    contract = build_contract({
        "severity": "warn",
        "rule_id": "SOMETHING_NEW",
        "field": "SomeUnknownField",
    })
    assert contract.bids_level is None


def test_placeholder_detected_from_file_digest():
    contract = build_contract(
        {"rule_id": "TABLE_VALUE_IS_TODO", "field": "Description"},
        None,
        'JSON file x.json keys:\nDescription = "TODO"',
    )
    assert contract.placeholder is True
    assert any("Description" in item for item in contract.must_state_unknown)


def test_no_retrieval_means_no_no_direct_requirement():
    contract = build_contract({
        "severity": "err",
        "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
        "field": "HEDVersion",
    })
    assert contract.requires_no_direct_statement is False


def test_severity_normalisation():
    assert normalise_severity("warn") == "warning"
    assert normalise_severity("err") == "error"
    assert normalise_severity("Warning") == "warning"
    assert normalise_severity("") == ""


def test_actionable_requires_an_identifier():
    assert AnswerContract().is_actionable is False
    assert AnswerContract(rule_id="X").is_actionable is True
    assert AnswerContract(field="License").is_actionable is True
