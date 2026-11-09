"""The output gate (``output_gate.py``) and its pipeline enforcement.

The four Phase-2 prompt-log outputs are frozen in
``tests/data/output_fixtures.py``. Each one must still be blocked, with the
same violation codes; the grounded template must always pass its own gate; and
the planner must never deliver a violating answer to the user.
"""

import os

import pytest
from langchain_core.messages import AIMessage

from answer_contract import build_contract
from answer_repair import deterministic_repair, render_grounded_template
from config import RetrievalConfig
from output_gate import check, response_checks
from planner import Planner
from retriever import Retriever
from tests.data.output_fixtures import CASES

KB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge-base",
)

CLEAN_ANSWERS = {
    "license_todo": """**Problem**
The field `License` in `dataset_description.json` contains a TODO placeholder.

**Why**
The knowledge base does not document this rule. BIDS lists `License` as recommended.

**How to Fix**
Replace the TODO with the license the dataset owner intends.""",
    "hedversion_format": """**Problem**
`HEDVersion` does not match the `hed_version` format.

**Why**
The validator reports `JSON_SCHEMA_VALIDATION_ERROR`.

**How to Fix**
Set `HEDVersion` to a value accepted by the `hed_version` format. The HED
schema version in use must be determined first.""",
    "sourcedatasets_recommended": """**Problem**
The warning reports that `SourceDatasets` is missing from `sub-01_scans.json`.

**Why**
This is a validator recommendation, not a specification requirement.

**How to Fix**
Add `SourceDatasets` only if it applies to this sidecar, and give the source
dataset's real URL, DOI, or version.""",
    "acknowledgements_todo": """**Problem**
The field `Acknowledgements` in `dataset_description.json` contains a TODO placeholder.

**Why**
The knowledge base does not document this rule. BIDS lists `Acknowledgements` as optional.

**How to Fix**
Replace the TODO with the acknowledgement text the dataset owner intends.""",
}


@pytest.fixture(scope="module")
def kb():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


def contract_for(kb, case):
    retrieval = kb.retrieve_issue_structured(
        case["issue"], user_question="Explain this."
    )
    return build_contract(case["issue"], retrieval, case["file_content"])


class NoLaya:
    def classify_intent(self, user_input, context=""):
        raise AssertionError("classification is disabled in this test")


class SpyLLM:
    """Always answers with ``response`` — including the repair request."""

    def __init__(self, response="STUB"):
        self.response = response
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        return AIMessage(content=self.response)


# ---------------------------------------------------------------------------
# The recorded outputs must still be blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_recorded_output_produces_the_expected_violations(kb, case):
    contract = contract_for(kb, case)
    gate = check(case["output"], contract,
                 inputs=[case["file_content"]])
    assert set(gate.codes) == case["expected_codes"], (
        f"{case['name']}: got {sorted(gate.codes)}"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_clean_answer_is_not_flagged(kb, case):
    contract = contract_for(kb, case)
    gate = check(CLEAN_ANSWERS[case["name"]], contract,
                 inputs=[case["file_content"]])
    assert gate.ok, [v.as_dict() for v in gate.violations]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_grounded_template_passes_its_own_gate(kb, case):
    contract = contract_for(kb, case)
    template = render_grounded_template(contract)
    gate = check(template, contract, inputs=[case["file_content"]])
    assert gate.ok, [v.as_dict() for v in gate.violations]
    # And it carries the contract's obligations.
    if contract.requires_no_direct_statement:
        assert "knowledge base does not document this rule" in template.lower()
    if contract.bids_level and contract.bids_level != "required":
        assert "is required" not in template.lower()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_deterministic_repair_removes_mechanical_violations(kb, case):
    contract = contract_for(kb, case)
    gate = check(case["output"], contract, inputs=[case["file_content"]])
    repaired = deterministic_repair(case["output"], contract, gate)
    after = check(repaired, contract, inputs=[case["file_content"]])
    # Whatever cannot be fixed mechanically is left for the repair call or the
    # template; the mechanical codes must be gone.
    assert "example_not_allowed" not in after.codes
    assert "concrete_value_not_allowed" not in after.codes
    if "missing_no_direct_declaration" in gate.codes:
        assert "missing_no_direct_declaration" not in after.codes


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_execution_claim_is_a_violation(kb):
    contract = contract_for(kb, CASES[1])
    gate = check("I have fixed the issue for you.", contract)
    assert "execution_claim" in gate.codes


def test_prose_version_recommendation_is_flagged(kb):
    """The JSON example is not the only place a version can be invented."""
    contract = contract_for(kb, CASES[1])
    gate = check(
        "For example, common versions might be `1.0`, `1.1`, depending on "
        "the setup.",
        contract,
    )
    assert "unsupported_version_value" in gate.codes


def test_bids_version_is_not_the_field_value(kb):
    contract = contract_for(kb, CASES[1])
    gate = check(
        "BIDS version 1.11.0 introduced this check; the HED version remains "
        "to be determined.",
        contract,
    )
    assert "unsupported_version_value" not in gate.codes


def test_disclaimer_is_not_a_violation(kb):
    contract = contract_for(kb, CASES[1])
    gate = check(
        "I cannot modify your dataset. Set HEDVersion to a value accepted by "
        "the hed_version format.",
        contract,
    )
    assert gate.ok, gate.codes


def test_empty_answer_is_flagged(kb):
    contract = contract_for(kb, CASES[1])
    assert "empty_response" in check("   \n ", contract).codes


def test_response_checks_shape_is_unchanged():
    checks = response_checks("I have fixed the issue.")
    assert set(checks) == {"empty_response", "execution_claims"}
    assert checks["execution_claims"] == ["claims to have changed the dataset"]


def test_named_entity_from_the_evidence_is_allowed(kb):
    contract = contract_for(kb, CASES[3])
    gate = check(
        "Contact Dr. Jane Roe at Example University about this field.",
        contract,
        inputs=["Please ask Dr. Jane Roe at Example University."],
    )
    assert "fabricated_named_entity" not in gate.codes


def test_concrete_value_from_the_evidence_is_allowed(kb):
    contract = contract_for(kb, CASES[0])
    gate = check(
        '**Problem**\nThe file sets "License": "CC BY 4.0".',
        contract,
        inputs=['License: "CC BY 4.0"'],
    )
    assert "concrete_value_not_allowed" not in gate.codes


# ---------------------------------------------------------------------------
# Pipeline enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_planner_never_delivers_the_violating_answer(kb, case):
    llm = SpyLLM(case["output"])
    planner = Planner(llm=llm, retriever=kb, laya=NoLaya())
    result = planner.execute(
        system_prompt="SYSTEM",
        user_input="Explain this.",
        context=case["issue"],
        file_content=case["file_content"],
    )
    contract = contract_for(kb, case)
    final = check(result.response, contract, inputs=[case["file_content"]])
    assert final.ok, [v.as_dict() for v in final.violations]
    assert result.diagnostics["gate"]["applied"] is True
    assert (result.diagnostics["gate"]["repaired"]
            or result.diagnostics["gate"]["fallback_used"])
    assert result.response != case["output"]


def test_clean_answer_costs_no_extra_model_call(kb):
    llm = SpyLLM(CLEAN_ANSWERS["license_todo"])
    planner = Planner(llm=llm, retriever=kb, laya=NoLaya())
    result = planner.execute(
        system_prompt="SYSTEM",
        user_input="Explain this.",
        context=CASES[0]["issue"],
        file_content=CASES[0]["file_content"],
    )
    assert len(llm.calls) == 1
    assert result.response == CLEAN_ANSWERS["license_todo"]
    assert result.diagnostics["gate"]["applied"] is False


def test_gate_is_diagnostic_when_disabled(kb, monkeypatch):
    from config import CONFIG

    monkeypatch.setattr(CONFIG.quality, "enabled", False)
    llm = SpyLLM(CASES[0]["output"])
    planner = Planner(llm=llm, retriever=kb, laya=NoLaya())
    result = planner.execute(
        system_prompt="SYSTEM",
        user_input="Explain this.",
        context=CASES[0]["issue"],
        file_content=CASES[0]["file_content"],
    )
    assert result.response == CASES[0]["output"]
    assert result.diagnostics["gate"]["enabled"] is False


# ---------------------------------------------------------------------------
# The prompt log must show the request the user actually made
# ---------------------------------------------------------------------------


class _LoggingBackend:
    """A backend that saves the prompt log exactly as ``llm_client.LLM`` does."""

    model_name = "fake-backend"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def chat(self, system_prompt, user_prompt, enable_thinking=None,
             history=None):
        import prompt_log

        self.calls += 1
        prompt_log.save(system_prompt, user_prompt=user_prompt,
                        output=self.answer)
        return self.answer


def test_repair_does_not_overwrite_the_prompt_log(kb, tmp_path, monkeypatch):
    """The repair pass is a second call for the same request, not a new one.

    Before this was fixed, the repair system prompt (``[Answer Repair]``)
    replaced the real system prompt in ``prompt_log.txt``, so the file showed
    a repair instruction where the agent's evidence-rich prompt should be.
    """
    import prompt_log
    from local_chat_model import LocalChatModel

    monkeypatch.setattr(prompt_log, "DEFAULT_PATH", tmp_path / "prompt_log.txt")
    backend = _LoggingBackend(CASES[0]["output"])
    planner = Planner(
        llm=LocalChatModel(llm=backend), retriever=kb, laya=NoLaya()
    )
    result = planner.execute(
        system_prompt="MAIN SYSTEM PROMPT",
        user_input="Explain this.",
        context=CASES[0]["issue"],
        file_content=CASES[0]["file_content"],
    )
    text = (tmp_path / "prompt_log.txt").read_text(encoding="utf-8")
    assert backend.calls == 2                     # main generation + repair
    assert result.diagnostics["gate"]["repaired"] is True
    assert "MAIN SYSTEM PROMPT" in text
    assert "[Answer Repair]" not in text
