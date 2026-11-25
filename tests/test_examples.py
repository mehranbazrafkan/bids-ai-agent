"""Format-aware examples: no invented structure or values.

The two realistic cases are the ones from the report:

* ``TOO_FEW_AUTHORS`` with ``Authors: [TODO]`` must not turn a placeholder
  into ``John Doe``.
* ``JSON_KEY_RECOMMENDED`` for ``SourceDatasets`` must not replace the
  documented ``URL``/``DOI``/``Version`` structure with ``key``/``value``.

A synthetic tabular rule covers TSV output because the current issue fixtures
do not include a table finding.
"""

import os

import pytest

from answer_contract import build_contract, render_contract
from answer_repair import deterministic_repair, render_grounded_template
from config import RetrievalConfig
from output_gate import check
from retriever import (
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

SOURCE_DATASETS = {
    "severity": "warn",
    "rule_id": "JSON_KEY_RECOMMENDED",
    "message": "A JSON file is missing a key listed as recommended.",
    "field": "SourceDatasets",
    "path": "sub-01_scans.json",
}

TSV_ISSUE = {
    "severity": "warn",
    "rule_id": "TSV_COLUMN_ORDER",
    "message": "participant_id must be the first column",
    "field": "participant_id",
    "path": "participants.tsv",
}

TSV_DIGEST = (
    "TSV table: participants.tsv\n"
    "participant_id\tage\tsex\n"
    "sub-01\t<age>\t<sex>"
)

AUTHORS_DIGEST = "JSON document:\n  Authors: [TODO]"


@pytest.fixture(scope="module")
def kb():
    return Retriever(config=RetrievalConfig(knowledge_base_dir=KB_DIR))


def contract_for(kb, issue, file_content=None):
    retrieval = kb.retrieve_issue_structured(
        issue, user_question="Explain this.", top_k=8,
    )
    return build_contract(issue, retrieval, file_content)


def synthetic_tabular_record():
    return KnowledgeRecord({
        "id": "kb_tabularrule_test",
        "knowledge_type": "TabularRule",
        "title": "Tabular rule participants",
        "summary": "participants.tsv has participant_id, age and sex columns.",
        "retrieval_text": "participants.tsv participant_id age sex",
        "facts": {
            "initial_columns": ["participant_id", "age", "sex"],
            "required_columns": ["participant_id"],
            "columns": {
                "participant_id": "required",
                "age": "optional",
                "sex": "optional",
            },
        },
        "authority": "normative_specification",
        "scope": {"authority": "normative_specification"},
        "bids_version": "1.11.1",
        "source": {
            "source_id": "test-source",
            "file": "src/schema/rules/tabular_data/participants.yaml",
            "revision": "abc123",
        },
        "raw_content": {
            "rule_definition": {
                "initial_columns": ["participant_id", "age", "sex"],
                "columns": {
                    "participant_id": "required",
                    "age": "optional",
                    "sex": "optional",
                },
            },
        },
    })


def tsv_contract():
    record = synthetic_tabular_record()
    assessment = RecordAssessment(
        id=record.id,
        title=record.title,
        score=1.0,
        match="field_exact",
        applicability="applicable",
        applicability_reason="selectors hold",
        relevance="direct",
        relevance_reason="documents the table",
        record=record,
        evidence_class="direct",
        evidence_reason="documents the tabular rule",
    )
    retrieval = RetrievalResult(
        found=True,
        quality="field_exact",
        indices=[],
        scores=[],
        text="",
        assessments=[assessment],
        rejected=[],
    )
    return build_contract(TSV_ISSUE, retrieval, TSV_DIGEST)


def example_answer(example: str) -> str:
    return (
        "**Problem**\n"
        "The finding concerns the affected field.\n\n"
        "**Why**\n"
        "The supplied evidence reports the finding.\n\n"
        "**How to Fix**\n"
        "Use a value the dataset owner supplies.\n\n"
        "**Example**\n"
        f"{example}"
    )


# ---------------------------------------------------------------------------
# Case A: TOO_FEW_AUTHORS
# ---------------------------------------------------------------------------


def test_authors_contract_documents_only_array_shape(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    assert contract.example_policy.file_format == "json"
    assert contract.example_allowed is True
    assert "Authors" in contract.example_policy.keys
    checklist = render_contract(contract)
    assert "[Example Policy" in checklist
    assert "allowed property names" in checklist
    assert "replacement placeholder" in checklist
    assert "illustrative value" in checklist
    assert "unknown" in checklist


def test_authors_placeholder_is_not_replaced_with_invented_names(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        example_answer('```json\n{"Authors": ["John Doe", "Jane Smith"]}\n```'),
        contract,
        inputs=[AUTHORS_DIGEST],
    )
    assert "invented_example_value" in gate.codes


def test_authors_explicit_placeholder_is_allowed(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        example_answer('```json\n{"Authors": ["<author name>"]}\n```'),
        contract,
        inputs=[AUTHORS_DIGEST],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


def test_prose_author_names_are_flagged_too(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        example_answer("Add John Doe and Jane Smith as authors."),
        contract,
        inputs=[AUTHORS_DIGEST],
    )
    assert "fabricated_named_entity" in gate.codes


def test_authors_example_can_be_omitted(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        "**Problem**\nThe field contains a TODO placeholder.\n\n"
        "**How to Fix**\nAsk the dataset owner which authors to list.",
        contract,
        inputs=[AUTHORS_DIGEST],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


def test_supplied_author_values_are_real_not_illustrative(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        example_answer('```json\n{"Authors": ["Jane Roe", "John Smith"]}\n```'),
        contract,
        inputs=[AUTHORS_DIGEST, "Jane Roe", "John Smith"],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


# ---------------------------------------------------------------------------
# Case B: JSON_KEY_RECOMMENDED / SourceDatasets
# ---------------------------------------------------------------------------


def test_sourcedatasets_contract_exposes_documented_keys(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    assert contract.example_policy.file_format == "json"
    assert contract.example_allowed is True
    for key in ("SourceDatasets", "URL", "DOI", "Version"):
        assert key in contract.example_policy.keys


def test_json_must_not_invent_key_value_structure(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    gate = check(
        example_answer(
            '```json\n{"SourceDatasets": [{"key": "value"}]}\n```'
        ),
        contract,
        inputs=["BIDSVersion: 1.11.1"],
    )
    assert "invented_example_property" in gate.codes


def test_json_documented_structure_with_placeholders_is_allowed(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    gate = check(
        example_answer(
            "```json\n"
            '{"SourceDatasets": [{"URL": "<URL>", "DOI": "<DOI>", '
            '"Version": "<version>"}]}\n'
            "```"
        ),
        contract,
        inputs=["BIDSVersion: 1.11.1"],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


def test_json_example_values_must_be_supplied_or_placeholders(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    gate = check(
        example_answer(
            "```json\n"
            '{"SourceDatasets": [{"URL": "http://example.org", '
            '"DOI": "10.1234/abc", "Version": "1.0"}]}\n'
            "```"
        ),
        contract,
        inputs=["BIDSVersion: 1.11.1"],
    )
    assert "invented_example_value" in gate.codes


def test_tsv_is_not_valid_json_file_content(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    gate = check(
        example_answer("```tsv\nparticipant_id\tage\nsub-01\t30\n```"),
        contract,
        inputs=["BIDSVersion: 1.11.1"],
    )
    assert "format_mismatch" in gate.codes


def test_json_answer_can_use_a_markdown_table(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    gate = check(
        example_answer(
            "| Key | Expected type | Purpose | Evidence |\n"
            "| --- | --- | --- | --- |\n"
            "| `URL` | string | source location | field definition |"
        ),
        contract,
        inputs=["BIDSVersion: 1.11.1"],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


def test_key_table_does_not_hide_invented_properties(kb):
    contract = contract_for(kb, SOURCE_DATASETS, "BIDSVersion: 1.11.1")
    gate = check(
        example_answer(
            "| Key | Expected type | Purpose | Evidence |\n"
            "| --- | --- | --- | --- |\n"
            "| `key` | string | invented | none |"
        ),
        contract,
        inputs=["BIDSVersion: 1.11.1"],
    )
    assert "invented_example_property" in gate.codes


# ---------------------------------------------------------------------------
# TSV format
# ---------------------------------------------------------------------------


def test_tsv_contract_uses_exact_supplied_columns():
    contract = tsv_contract()
    assert contract.example_policy.file_format == "tsv"
    assert contract.example_policy.tsv_block_allowed is True
    assert contract.example_policy.columns == ("participant_id", "age", "sex")
    assert "participant_id" in render_contract(contract)


def test_tsv_example_preserves_columns_and_placeholders():
    contract = tsv_contract()
    gate = check(
        example_answer(
            "```tsv\nparticipant_id\tage\tsex\n"
            "<participant_id>\t<age>\t<sex>\n```"
        ),
        contract,
        inputs=[TSV_DIGEST],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


def test_tsv_example_rejects_extra_columns():
    contract = tsv_contract()
    gate = check(
        example_answer(
            "```tsv\nparticipant_id\tage\tsex\tgroup\n"
            "<participant_id>\t<age>\t<sex>\t<group>\n```"
        ),
        contract,
        inputs=[TSV_DIGEST],
    )
    assert "invented_tsv_column" in gate.codes


def test_tsv_example_rejects_json_content():
    contract = tsv_contract()
    gate = check(
        example_answer('```json\n{"participant_id": "<id>"}\n```'),
        contract,
        inputs=[TSV_DIGEST],
    )
    assert "format_mismatch" in gate.codes


def test_tsv_example_rejects_invented_cell_values():
    contract = tsv_contract()
    gate = check(
        example_answer(
            "```tsv\nparticipant_id\tage\tsex\nsub-99\t42\tF\n```"
        ),
        contract,
        inputs=[TSV_DIGEST],
    )
    assert "invented_example_value" in gate.codes


# ---------------------------------------------------------------------------
# Placeholders, repetition and repair
# ---------------------------------------------------------------------------


def test_todo_and_unknown_are_placeholders(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        example_answer(
            '```json\n{"Authors": ["TODO", "unknown", "[TODO]"]}\n```'
        ),
        contract,
        inputs=[AUTHORS_DIGEST],
    )
    assert gate.ok, [v.as_dict() for v in gate.violations]


def test_repeated_example_is_flagged(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    block = '```json\n{"Authors": ["<author name>"]}\n```'
    gate = check(
        example_answer(block + "\n\n" + block),
        contract,
        inputs=[AUTHORS_DIGEST],
    )
    assert "duplicate_example" in gate.codes


def test_conflicting_examples_are_flagged(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    gate = check(
        example_answer(
            '```json\n{"Authors": ["Jane Roe"]}\n```\n\n'
            '```json\n{"Authors": ["John Doe"]}\n```'
        ),
        contract,
        inputs=[AUTHORS_DIGEST, "Jane Roe"],
    )
    assert "contradictory_example" in gate.codes


def test_unsupported_example_is_removed_then_omitted(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    output = example_answer('```json\n{"Authors": ["John Doe"]}\n```')
    gate = check(output, contract, inputs=[AUTHORS_DIGEST])
    repaired = deterministic_repair(output, contract, gate)
    assert "```" not in repaired
    assert "John Doe" not in repaired
    assert "**Example**" not in repaired


def test_grounded_template_never_invents_an_example(kb):
    contract = contract_for(kb, TOO_FEW_AUTHORS, AUTHORS_DIGEST)
    template = render_grounded_template(contract)
    assert "```" not in template
    assert "John Doe" not in template
