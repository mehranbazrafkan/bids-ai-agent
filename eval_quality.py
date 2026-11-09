#!/usr/bin/env python3
"""Offline/online quality evaluation for the BIDS explain agent.

Scores an answer against the same :mod:`answer_contract` / :mod:`output_gate`
machinery the production path uses, on a frozen stratified set of findings
(``tests/data/quality_set.jsonl``). Eight independent binary criteria, each
reported with a Wilson 95% confidence interval, so a claim like "90% correct"
can be checked instead of asserted.

Three modes:

* ``--replay responses.json`` - score recorded answers (no model, fast).
* ``--live`` - run the real agent on every item (needs the model stack).
* ``--baseline`` - score the four recorded Phase-2 prompt-log outputs and
  write ``baseline_scores.json`` (this is the number the work started from).
* ``--audit-kb`` - report retrieval/contract coverage of the quality set.

Examples::

    python eval_quality.py --baseline
    python eval_quality.py --replay responses.json --min-score 0.9
    python eval_quality.py --live --save-responses responses.json
    python eval_quality.py --audit-kb

The exit code is non-zero when the overall rate is below ``--min-score`` or
when any answer fabricates a person/institution, unless ``--baseline`` is
used (the baseline is expected to fail).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from answer_contract import NO_DIRECT_SENTENCE, build_contract
from config import RetrievalConfig
from output_gate import check
from retriever import Retriever

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(HERE, "tests", "data", "quality_set.jsonl")
DEFAULT_KB = os.path.join(HERE, "knowledge-base")
BASELINE_FILE = os.path.join(HERE, "baseline_scores.json")

#: The eight binary criteria. ``contract_clean`` is the strict aggregate.
CRITERIA = (
    "contract_clean",
    "no_execution_claim",
    "no_fabricated_entity",
    "modality_correct",
    "severity_correct",
    "no_unsupported_value",
    "declares_unknown",
    "structure_complete",
)

SECTION_MARKERS = ("**problem**", "**why**", "**how to fix**")

UNKNOWN_MARKERS = (
    "cannot be determined",
    "not documented",
    "does not document",
    "unknown",
    "depends on",
    "the dataset owner",
    "the owner",
    "not specified",
    "must be determined",
    "cannot tell",
    "no information",
    "not enough information",
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (no SciPy)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ItemScore:
    item_id: str
    family: str
    criteria: Dict[str, bool] = field(default_factory=dict)
    codes: Tuple[str, ...] = ()

    @property
    def passed(self) -> int:
        return sum(1 for value in self.criteria.values() if value)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.item_id,
            "family": self.family,
            "criteria": self.criteria,
            "violations": list(self.codes),
        }


def _declares_unknown(text: str, contract: Any) -> bool:
    lowered = (text or "").lower()
    if getattr(contract, "requires_no_direct_statement", False):
        if NO_DIRECT_SENTENCE.lower() in lowered:
            return True
    if getattr(contract, "must_state_unknown", ()):
        return any(marker in lowered for marker in UNKNOWN_MARKERS)
    return True


def score_answer(
    item: Dict[str, Any],
    text: str,
    contract: Any,
    evidence: str = "",
) -> ItemScore:
    """Bin one answer into the eight criteria."""
    inputs = [item.get("file_content"), item.get("dataset_structure"),
              item.get("user_input")]
    gate = check(text or "", contract, inputs=inputs, evidence=evidence)
    codes = set(gate.codes)

    score = ItemScore(
        item_id=item.get("id", "?"),
        family=item.get("family", "?"),
        codes=tuple(sorted(codes)),
    )
    lowered = (text or "").lower()
    score.criteria = {
        "contract_clean": gate.ok,
        "no_execution_claim": "execution_claim" not in codes,
        "no_fabricated_entity": "fabricated_named_entity" not in codes,
        "modality_correct": not {
            "modality_overclaim", "bids_requirement_overclaim",
        } & codes,
        "severity_correct": "severity_mismatch" not in codes,
        "no_unsupported_value": not {
            "concrete_value_not_allowed",
            "example_not_allowed",
            "unsupported_version_value",
        } & codes,
        "declares_unknown": _declares_unknown(text, contract),
        "structure_complete": bool(lowered.strip()) and all(
            marker in lowered for marker in SECTION_MARKERS
        ),
    }
    return score


def summarise(scores: Sequence[ItemScore]) -> Dict[str, Any]:
    total = len(scores)
    summary: Dict[str, Any] = {}
    for criterion in CRITERIA:
        passed = sum(1 for score in scores if score.criteria.get(criterion))
        low, high = wilson(passed, total)
        summary[criterion] = {
            "passed": passed,
            "total": total,
            "rate": round(passed / total, 4) if total else 0.0,
            "wilson_95": [round(low, 4), round(high, 4)],
        }
    rates = [
        summary[criterion]["rate"] for criterion in CRITERIA if total
    ]
    return {
        "items": total,
        "criteria": summary,
        "overall_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "fabrication_failures": total - summary["no_fabricated_entity"]["passed"],
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_quality_set(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_responses(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "responses" in payload:
        payload = payload["responses"]
    if not isinstance(payload, dict):
        raise SystemExit("responses file must be a JSON object of id -> text")
    return {str(key): str(value) for key, value in payload.items()}


def make_retriever() -> Retriever:
    return Retriever(
        config=RetrievalConfig(knowledge_base_dir=DEFAULT_KB)
    )


def evaluate(
    items: Sequence[Dict[str, Any]],
    responses: Dict[str, str],
    retriever: Retriever,
) -> List[ItemScore]:
    scores: List[ItemScore] = []
    for item in items:
        key = item.get("id", "")
        if key not in responses:
            raise SystemExit(f"no response for item {key!r}")
        retrieval = retriever.retrieve_issue_structured(
            item.get("issue", {}),
            user_question=item.get("user_input", ""),
        )
        contract = build_contract(
            item.get("issue", {}), retrieval, item.get("file_content")
        )
        scores.append(score_answer(
            item, responses[key], contract, evidence=retrieval.text
        ))
    return scores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(summary: Dict[str, Any], detailed: bool = False) -> None:
    print()
    print(f"items: {summary['items']}")
    print(f"{'criterion':<26} {'pass':>7}  {'rate':>6}  wilson 95%")
    print("-" * 64)
    for criterion in CRITERIA:
        row = summary["criteria"][criterion]
        low, high = row["wilson_95"]
        print(
            f"{criterion:<26} {row['passed']:>3}/{row['total']:<3} "
            f"{row['rate'] * 100:>6.1f}%  [{low * 100:.0f}%, {high * 100:.0f}%]"
        )
    print("-" * 64)
    print(f"{'overall (mean)':<26} {'':>7}  {summary['overall_rate'] * 100:>6.1f}%")
    print(f"fabrication failures: {summary['fabrication_failures']}")
    print()


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Baseline: the four recorded prompt-log outputs
# ---------------------------------------------------------------------------


def score_baseline(
    retriever: Retriever,
) -> Tuple[List[ItemScore], Dict[str, Any]]:
    """Score the four recorded prompt-log outputs against the rubric."""
    from tests.data.output_fixtures import CASES

    scores: List[ItemScore] = []
    for case in CASES:
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
        scores.append(score_answer(
            item, case["output"], contract, evidence=retrieval.text
        ))
    return scores, summarise(scores)


def run_baseline(retriever: Retriever) -> int:
    scores, summary = score_baseline(retriever)
    payload = {
        "generated_by": "eval_quality.py --baseline",
        "source": "the four recorded Phase-2 prompt logs (Test-0002)",
        "rubric": list(CRITERIA),
        "per_item": [score.as_dict() for score in scores],
        "summary": summary,
    }
    write_json(BASELINE_FILE, payload)
    print_report(summary)
    print(f"wrote {BASELINE_FILE}")
    return 0


# ---------------------------------------------------------------------------
# KB coverage audit
# ---------------------------------------------------------------------------


def run_audit(items: Sequence[Dict[str, Any]], retriever: Retriever) -> int:
    missing: List[str] = []
    no_contract: List[str] = []
    no_direct: List[str] = []
    unknown_level: List[str] = []

    for item in items:
        retrieval = retriever.retrieve_issue_structured(
            item.get("issue", {}),
            user_question=item.get("user_input", ""),
        )
        contract = build_contract(
            item.get("issue", {}), retrieval, item.get("file_content")
        )
        if not retrieval.found:
            missing.append(item["id"])
            continue
        if contract is None:
            no_contract.append(item["id"])
            continue
        if not contract.direct_evidence:
            no_direct.append(item["id"])
        if contract.field and contract.bids_level is None:
            unknown_level.append(item["id"])

    total = len(items)
    print()
    print(f"KB coverage audit over {total} findings")
    print("-" * 64)
    print(f"retrieval miss              : {len(missing)}")
    print(f"no contract                 : {len(no_contract)}")
    print(f"no direct record (declared) : {len(no_direct)}")
    print(f"field found, level unknown  : {len(unknown_level)}")
    print("-" * 64)
    if missing:
        print("missing records for:")
        for item in missing:
            print(f"  - {item}")
    if unknown_level:
        print("field-level BIDS level unavailable for:")
        for item in unknown_level:
            print(f"  - {item}")
    # A retrieval miss is a real KB gap; a declared no-direct case is the
    # contract working as intended, so it does not fail the audit.
    return 1 if missing else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default=DEFAULT_SET,
                        help="quality set JSONL (default: %(default)s)")
    parser.add_argument("--replay", help="score these recorded answers (JSON)")
    parser.add_argument("--live", action="store_true",
                        help="run the real agent on every item")
    parser.add_argument("--save-responses",
                        help="with --live, write the answers to this JSON")
    parser.add_argument("--baseline", action="store_true",
                        help="score the four recorded Phase-2 outputs")
    parser.add_argument("--audit-kb", action="store_true",
                        help="report KB coverage of the quality set")
    parser.add_argument("--min-score", type=float, default=0.9,
                        help="minimum overall rate (default: %(default)s)")
    parser.add_argument("--json", help="write the full report to this file")
    args = parser.parse_args(argv)

    retriever = make_retriever()

    if args.baseline:
        return run_baseline(retriever)

    items = load_quality_set(args.set)

    if args.audit_kb:
        return run_audit(items, retriever)

    if args.live:
        from agent import Agent

        agent = Agent()
        responses: Dict[str, str] = {}
        for item in items:
            result = agent.run(
                user_input=item.get("user_input", "Explain this."),
                context=item.get("issue", {}),
                dataset_structure=item.get("dataset_structure"),
                file_content=item.get("file_content"),
            )
            responses[item["id"]] = result.response
            print(f"  {item['id']}: {len(result.response)} chars")
        if args.save_responses:
            write_json(args.save_responses, {"responses": responses})
            print(f"wrote {args.save_responses}")
    elif args.replay:
        responses = load_responses(args.replay)
    else:
        parser.error("choose --live, --replay, --baseline or --audit-kb")

    scores = evaluate(items, responses, retriever)
    summary = summarise(scores)
    print_report(summary)

    if args.json:
        write_json(args.json, {
            "rubric": list(CRITERIA),
            "summary": summary,
            "per_item": [score.as_dict() for score in scores],
        })
        print(f"wrote {args.json}")

    failed = summary["overall_rate"] < args.min_score
    if summary["fabrication_failures"]:
        print(
            f"FAIL: {summary['fabrication_failures']} answer(s) fabricated a "
            "person or institution"
        )
        failed = True
    if failed:
        print(
            f"FAIL: overall {summary['overall_rate'] * 100:.1f}% < "
            f"{args.min_score * 100:.0f}%"
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
