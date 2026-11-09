"""Tests for the quality harness (``eval_quality.py``).

The harness is only useful if it is reproducible: the frozen quality set is
well-formed, the Wilson interval is correct at the boundaries, the recorded
Phase-2 baseline scores exactly 50%, and the deterministic template scores
8/8 on every recorded case.
"""

import json
import os

import pytest

import eval_quality as eq
from answer_contract import build_contract
from answer_repair import render_grounded_template
from tests.data.output_fixtures import CASES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def retriever():
    return eq.make_retriever()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_wilson_interval_boundaries():
    low, high = eq.wilson(0, 4)
    assert low == 0.0
    assert 0.45 < high < 0.55

    low, high = eq.wilson(4, 4)
    assert 0.45 < low < 0.55
    assert high == 1.0

    assert eq.wilson(0, 0) == (0.0, 0.0)


def test_wilson_interval_midpoint():
    low, high = eq.wilson(5, 10)
    assert abs(low - 0.2366) < 0.01
    assert abs(high - 0.7634) < 0.01


# ---------------------------------------------------------------------------
# The frozen quality set
# ---------------------------------------------------------------------------


def test_quality_set_is_stratified():
    items = eq.load_quality_set(eq.DEFAULT_SET)
    assert len(items) == 36
    assert len({item["id"] for item in items}) == 36

    families = {}
    for item in items:
        families[item["family"]] = families.get(item["family"], 0) + 1
    assert families == {
        "validator_spec": 10,
        "validator_impl": 20,
        "manager_todo": 6,
    }


def test_every_item_has_a_retrievable_contract(retriever):
    items = eq.load_quality_set(eq.DEFAULT_SET)
    for item in items:
        retrieval = retriever.retrieve_issue_structured(
            item["issue"], user_question=item["user_input"]
        )
        assert retrieval.found, item["id"]
        contract = build_contract(
            item["issue"], retrieval, item.get("file_content")
        )
        assert contract is not None, item["id"]
        # The safety net must itself be perfect for every finding, otherwise
        # the fallback tier can deliver a violating answer.
        score = eq.score_answer(
            item,
            render_grounded_template(contract),
            contract,
            evidence=retrieval.text,
        )
        assert all(score.criteria.values()), (
            item["id"],
            [name for name, ok in score.criteria.items() if not ok],
        )


def test_audit_reports_no_retrieval_misses(retriever, capsys):
    items = eq.load_quality_set(eq.DEFAULT_SET)
    assert eq.run_audit(items, retriever) == 0
    output = capsys.readouterr().out
    assert "retrieval miss              : 0" in output


# ---------------------------------------------------------------------------
# The recorded baseline
# ---------------------------------------------------------------------------


def test_baseline_is_fifty_percent_and_fabricates_once(retriever):
    _scores, summary = eq.score_baseline(retriever)
    assert summary["items"] == 4
    assert summary["overall_rate"] == 0.5
    assert summary["fabrication_failures"] == 1
    assert summary["criteria"]["contract_clean"]["passed"] == 0
    assert summary["criteria"]["declares_unknown"]["passed"] == 0


def test_baseline_file_is_not_stale(retriever):
    path = os.path.join(HERE, "baseline_scores.json")
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)

    _scores, summary = eq.score_baseline(retriever)
    assert stored["rubric"] == list(eq.CRITERIA)
    assert stored["summary"] == summary


def test_the_fabrication_is_the_acknowledgements_case(retriever):
    scores, _summary = eq.score_baseline(retriever)
    by_id = {score.item_id: score for score in scores}
    assert by_id["acknowledgements_todo"].criteria[
        "no_fabricated_entity"
    ] is False
    assert by_id["license_todo"].criteria["no_fabricated_entity"] is True


# ---------------------------------------------------------------------------
# The target: the deterministic template must be perfect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_grounded_template_scores_eight_of_eight(retriever, case):
    retrieval = retriever.retrieve_issue_structured(
        case["issue"], user_question="Explain this."
    )
    contract = build_contract(
        case["issue"], retrieval, case.get("file_content")
    )
    item = {
        "id": case["name"],
        "family": "recorded_phase2",
        "file_content": case.get("file_content"),
        "user_input": "Explain this.",
    }
    score = eq.score_answer(item, render_grounded_template(contract), contract)
    assert all(score.criteria.values()), [
        name for name, passed in score.criteria.items() if not passed
    ]
