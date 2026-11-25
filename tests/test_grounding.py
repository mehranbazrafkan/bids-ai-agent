"""Evidence grounding and provenance.

These tests pin the deterministic half of the grounding contract added around
the knowledge base:

* every retrieved record gets a role — direct, supporting, background, or
  inapplicable — and a "missing direct evidence" state is explicit;
* authority comes only from the record that documents the reported rule, not
  from the closest background field definition;
* the prompt tells the model what each role may establish and forbids
  presenting a background record as the issue's definition;
* provenance (authority, source id, file, revision, versions) survives into
  the prompt, diagnostics and answer contract;
* the output gate rejects invented rule codes and unsupported attribution.

The real KB is used for realistic cases (``TOO_FEW_AUTHORS`` and
``JSON_KEY_RECOMMENDED``); synthetic records cover conflicts that the current
single-file KB cannot place in one evidence set.
"""

import os
from types import SimpleNamespace

import pytest

from answer_contract import (
    AUTHORITY_SPEC,
    AUTHORITY_UNKNOWN,
    AUTHORITY_VALIDATOR,
    build_contract,
    render_contract,
)
from config import RetrievalConfig
from output_gate import check
from prompts import evidence_summary, retrieval_note
from retriever import (
    Formatter,
    KnowledgeRecord,
    RecordAssessment,
    RetrievalResult,
    Retriever,
)

KB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge-base",
)

TOO_FEW_AUTHORS = {
    "severity": "warn",
    "rule_id": "TOO_FEW_AUTHORS",
    "message": (
        "The 'Authors' field of 'dataset_description.json' should contain "
        "an array of values - with one author per value."
    ),
    "field": "Authors",
    "path": "dataset_description.json",
}

JSON_KEY_RECOMMENDED = {
    "severity": "warn",
    "rule_id": "JSON_KEY_RECOMMENDED",
    "message": "A JSON file is missing a key listed as recommended.",
    "field": "SequenceName",
    "path": "sub-01_task-rest_bold.json",
}

# A rule the KB does not carry. The field is documented, so supporting field
# material exists; no record documents the rule itself.
MISSING_RULE = {
    "severity": "warn",
    "rule_id": "FUTURE_BIDS_RULE",
    "message": "a future check failed on the Authors field",
    "field": "Authors",
    "path": "dataset_description.json",
}

CONTRADICTORY = {
    "severity": "err",
    "rule_id": "rules.json.atlas.atlas_description",
    "message": "atlas description issue",
    "field": "Atlas",
    "path": "dataset_description.json",
}


@pytest.fixture(scope="module")
def kb():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


def assess(record, evidence_class, relevance=None):
    """A RecordAssessment carrying the fields the contract/prompt read."""
    return RecordAssessment(
        id=record.id,
        title=record.title,
        score=1.0,
        match="field_exact",
        applicability="applicable",
        applicability_reason="selectors hold or are unknown",
        relevance=relevance or (
            "direct" if evidence_class == "direct" else "background"
        ),
        relevance_reason="test assessment",
        record=record,
        evidence_class=evidence_class,
        evidence_reason={
            "direct": "documents the reported rule",
            "supporting": "documents the affected field only",
            "background": "context only",
        }.get(evidence_class, "test"),
        provenance=record.provenance(),
    )


def retrieval_of(*assessments, quality="field_exact", rejected=()):
    return RetrievalResult(
        found=True,
        quality=quality,
        indices=[],
        scores=[],
        text="",
        assessments=list(assessments),
        rejected=list(rejected),
    )


def synthetic_record(
    record_id,
    *,
    level="required",
    severity="warning",
    version="1.10.0",
    authority="normative_specification",
):
    return KnowledgeRecord({
        "id": record_id,
        "knowledge_type": "MetadataRule",
        "title": f"License rule {record_id}",
        "summary": "License is a metadata field.",
        "retrieval_text": "License metadata field",
        "facts": {
            "issue_code": "FUTURE_RULE",
            "field_levels": {"License": level},
            "severity": severity,
        },
        "authority": authority,
        "scope": {
            "authority": authority,
            "knowledge_domain": "metadata_rule",
        },
        "bids_version": version,
        "source": {
            "source_id": "test-source",
            "file": "src/schema/rules/sidecars/license.yaml",
            "revision": "abc123def456",
            "line_start": 1,
            "line_end": 4,
        },
        "raw_content": {"rule_definition": {"fields": {"License": {
            "level": level,
        }}}},
    })


# ---------------------------------------------------------------------------
# Prompt construction: deterministic role and missing-state text
# ---------------------------------------------------------------------------


def _fake_assessment(evidence_class, record_id):
    return SimpleNamespace(
        id=record_id,
        evidence_class=evidence_class,
        relevance="direct" if evidence_class == "direct" else "background",
        evidence_reason="reason",
        relevance_reason="reason",
        provenance={"record_id": record_id},
    )


def test_evidence_summary_distinguishes_all_classes():
    retrieval = SimpleNamespace(
        found=True,
        quality="rule_exact",
        assessments=[
            _fake_assessment("direct", "kb_direct"),
            _fake_assessment("supporting", "kb_support"),
            _fake_assessment("background", "kb_back"),
        ],
        rejected=[_fake_assessment("inapplicable", "kb_reject")],
        evidence_counts=lambda: {
            "direct": 1, "supporting": 1, "background": 1,
            "inapplicable": 1,
        },
    )
    summary = evidence_summary(retrieval)
    assert "Direct evidence: 1" in summary
    assert "Supporting evidence: 1" in summary
    assert "Background only: 1" in summary
    assert "Inapplicable (quarantined): 1" in summary
    assert "Missing: none" in summary


def test_background_is_never_a_definition_in_the_note():
    retrieval = SimpleNamespace(
        found=True,
        quality="lexical",
        assessments=[
            _fake_assessment("supporting", "kb_support"),
            _fake_assessment("background", "kb_back"),
        ],
        rejected=[],
        evidence_counts=lambda: {
            "direct": 0, "supporting": 1, "background": 1,
            "inapplicable": 0,
        },
    )
    note = retrieval_note(retrieval)
    assert 'None of the records above is marked "Directly relevant"' in note
    assert "Evidence role: Supporting" in note
    assert "Do not present them as documentation" in note
    assert "Evidence role: Background only" in note
    assert "do not use them to define the issue" in note.lower()


# ---------------------------------------------------------------------------
# Realistic KB cases
# ---------------------------------------------------------------------------


def test_too_few_authors_has_specification_authority(kb):
    result = kb.retrieve_issue_structured(
        TOO_FEW_AUTHORS, user_question="Explain this warning."
    )
    assert result.has_direct is True
    direct = [a for a in result.assessments
              if a.evidence_class == "direct"]
    assert [a.id for a in direct] == ["kb_validationissue_f73c42d966"]
    assert direct[0].provenance["authority"] == AUTHORITY_SPEC
    assert direct[0].provenance["authority_label"].startswith(
        "BIDS specification"
    )
    assert any(a.evidence_class == "supporting" for a in result.assessments)

    contract = build_contract(
        TOO_FEW_AUTHORS, result,
        file_content="BIDSVersion: 1.11.1",
    )
    assert contract.direct_evidence is True
    assert contract.authority == AUTHORITY_SPEC
    assert "kb_validationissue_f73c42d966" in contract.direct_evidence_ids
    assert contract.supporting_evidence_ids
    checklist = render_contract(contract)
    assert "BIDS specification rule" in checklist
    assert "supporting record(s)" in checklist
    assert "NOT the reported rule" in checklist


def test_json_key_recommended_stays_validator_authority(kb):
    result = kb.retrieve_issue_structured(
        JSON_KEY_RECOMMENDED, user_question="Explain this warning."
    )
    assert result.has_direct is True
    direct = [a for a in result.assessments
              if a.evidence_class == "direct"]
    assert [a.id for a in direct] == ["kb_validationissue_f3224f4fd5"]
    assert direct[0].provenance["authority"] == AUTHORITY_VALIDATOR
    # A supporting field definition from the specification must not upgrade
    # the *rule* to a specification requirement.
    contract = build_contract(
        JSON_KEY_RECOMMENDED, result,
        file_content="BIDSVersion: 1.11.1",
    )
    assert contract.authority == AUTHORITY_VALIDATOR
    assert "validator check" in render_contract(contract)
    assert "BIDS specification rule" not in render_contract(contract)


def test_missing_direct_evidence_is_explicit_and_never_borrows_authority(kb):
    result = kb.retrieve_issue_structured(
        MISSING_RULE, user_question="Explain this warning."
    )
    assert result.found is True
    assert result.has_direct is False
    assert result.missing_direct is True
    assert result.evidence_counts()["direct"] == 0
    assert all(a.evidence_class in ("supporting", "background")
               for a in result.assessments)

    contract = build_contract(
        MISSING_RULE, result,
        file_content="BIDSVersion: 1.11.1",
    )
    assert contract.requires_no_direct_statement is True
    # Supporting field records are authoritative for the *field*, not for the
    # missing rule; the contract must not inherit their specification status.
    assert contract.authority == AUTHORITY_UNKNOWN
    assert contract.source_authorities
    assert "FUTURE_BIDS_RULE" in render_contract(contract)
    assert "no direct record" in render_contract(contract)
    assert "The knowledge base does not document this rule." in (
        render_contract(contract)
    )


def test_background_only_contradictory_case_is_not_direct(kb):
    result = kb.retrieve_issue_structured(
        CONTRADICTORY, user_question="Explain this."
    )
    assert result.found is True
    assert result.has_direct is False
    assert result.rejected, "the atlas rule must be quarantined"
    assert all(a.evidence_class in ("background", "inapplicable")
               for a in result.assessments)
    contract = build_contract(CONTRADICTORY, result)
    assert contract.direct_evidence_ids == ()
    assert contract.background_evidence_ids
    checklist = render_contract(contract)
    assert "background record(s)" in checklist
    assert "context only; never use" in checklist


# ---------------------------------------------------------------------------
# Provenance and version handling
# ---------------------------------------------------------------------------


def test_provenance_survives_prompt_diagnostics_and_contract(kb):
    result = kb.retrieve_issue_structured(
        TOO_FEW_AUTHORS, user_question="Explain."
    )
    direct = next(a for a in result.assessments
                  if a.evidence_class == "direct")
    provenance = direct.as_dict()["provenance"]
    assert provenance["record_id"] == direct.id
    assert provenance["source_id"] == "bids-specification-repo"
    assert provenance["file"] == "src/schema/rules/checks/hints.yaml"
    assert provenance["revision"].startswith("8973fdf")
    assert provenance["bids_version"]
    assert provenance["source_ref"].startswith(
        "source=bids-specification-repo, file=src/schema/rules/checks/hints.yaml"
    )

    rendered = Formatter.format(direct.record, 1, assessment=direct)
    assert "Evidence role: Direct" in rendered
    assert "Authority: BIDS specification/schema (normative)" in rendered
    assert "Source: src/schema/rules/checks/hints.yaml" in rendered
    assert "Trace: source id: bids-specification-repo" in rendered
    assert "revision: 8973fdf11d3c" in rendered


def test_version_mismatch_is_reported_but_not_invented(kb):
    result = kb.retrieve_issue_structured(
        TOO_FEW_AUTHORS, user_question="Explain."
    )
    old = build_contract(
        TOO_FEW_AUTHORS, result, file_content="BIDSVersion: 1.8.0"
    )
    assert old.dataset_bids_version == "1.8.0"
    assert any("1.11.3-dev" in note and "1.8.0" in note
               for note in old.version_notes)
    assert "BIDS version:" in render_contract(old)

    same = build_contract(
        TOO_FEW_AUTHORS, result, file_content="BIDSVersion: 1.11.3"
    )
    assert same.version_notes == ()


def test_conflicting_evidence_is_detected_deterministically():
    required = synthetic_record("kb_req", level="required", severity="warning",
                                version="1.10.0")
    recommended = synthetic_record(
        "kb_rec", level="recommended", severity="error", version="1.11.0",
        authority="validator_implementation",
    )
    result = retrieval_of(
        assess(required, "supporting"),
        assess(recommended, "supporting"),
    )
    contract = build_contract(
        {"rule_id": "FUTURE_RULE", "field": "License"}, result
    )
    kinds = {conflict.kind for conflict in contract.conflicts}
    assert {"level", "severity", "version", "authority"} <= kinds
    checklist = render_contract(contract)
    assert "conflicting evidence (level)" in checklist
    assert "conflicting evidence (severity)" in checklist
    assert "conflicting evidence (version)" in checklist
    assert "do not merge or silently pick one" in checklist


# ---------------------------------------------------------------------------
# Output gate: unsupported claims and misattribution
# ---------------------------------------------------------------------------


def test_gate_rejects_an_invented_rule_citation(kb):
    result = kb.retrieve_issue_structured(
        JSON_KEY_RECOMMENDED, user_question="Explain."
    )
    contract = build_contract(JSON_KEY_RECOMMENDED, result)
    answer = (
        "**Problem**\n"
        "The file violates FAKE_RULE_CODE.\n\n"
        "**Why**\n"
        "The knowledge base says so.\n\n"
        "**How to Fix**\n"
        "Add the missing key.\n\n"
        "**Example**\n"
        "None."
    )
    gate = check(answer, contract, inputs=[], evidence=result.text)
    assert "unsupported_rule_citation" in gate.codes


def test_gate_rejects_attribution_to_an_absent_source(kb):
    license_todo = {
        "severity": "warn",
        "rule_id": "bidsmgr.todo_placeholder",
        "message": "field 'License' contains a TODO placeholder",
        "field": "License",
        "path": "dataset_description.json",
    }
    result = kb.retrieve_issue_structured(
        license_todo, user_question="Explain."
    )
    contract = build_contract(license_todo, result)
    assert AUTHORITY_VALIDATOR not in contract.source_authorities
    answer = (
        "**Problem**\n"
        "The License field contains a TODO placeholder.\n\n"
        "**Why**\n"
        "According to the BIDS validator, License is required.\n\n"
        "**How to Fix**\n"
        "Set a real license value.\n\n"
        "**Example**\n"
        "None."
    )
    gate = check(answer, contract, inputs=[], evidence=result.text)
    assert "authority_misattribution" in gate.codes


def test_gate_allows_a_supported_claim_and_citation(kb):
    result = kb.retrieve_issue_structured(
        TOO_FEW_AUTHORS, user_question="Explain."
    )
    contract = build_contract(
        TOO_FEW_AUTHORS, result,
        file_content="BIDSVersion: 1.11.1",
    )
    answer = (
        "**Problem**\n"
        "The Authors field contains only one author entry.\n\n"
        "**Why**\n"
        "The BIDS specification documents TOO_FEW_AUTHORS as a warning and "
        "recommends one author per array entry.\n\n"
        "**How to Fix**\n"
        "Split the single author string into separate array entries.\n\n"
        "**Example**\n"
        "None."
    )
    gate = check(answer, contract, inputs=[], evidence=result.text)
    assert "unsupported_rule_citation" not in gate.codes
    assert "authority_misattribution" not in gate.codes
