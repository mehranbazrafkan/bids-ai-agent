"""Evaluate Laya's intent classification against a labelled set.

Usage (from the repository root):

    python eval_intent.py                        # full local model
    python eval_intent.py --limit 10             # first 10 items only
    python eval_intent.py --laya-mode disabled   # control run: everything is
                                                  # "unclear" (the dead-classifier
                                                  # baseline; no model is loaded)

The dataset is `tests/data/intent_eval.jsonl`: {"id", "text", "label"} with
labels explain / fix / unclear.  The report contains overall accuracy, a
confusion matrix and every failure with the reported confidence/source.

The accuracy on this small hand-built set is a smoke signal, not a quality
score; it exists to catch a dead or badly broken classifier (the control run
scores 12/51 = 0.24, since most items are explain/fix).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from laya_decision import INTENT_UNCLEAR, LayaDecision, LayaDecisionEngine  # noqa: E402

LABELS = ("explain", "fix", "unclear")
DEFAULT_DATASET = ROOT / "tests" / "data" / "intent_eval.jsonl"


class _DeadClassifier:
    """Control condition: returns 'unclear' for everything, never loads a model."""

    @staticmethod
    def classify_intent(user_input: str, context: str = "") -> LayaDecision:
        return LayaDecision(INTENT_UNCLEAR, 0.0, "fallback")


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "text", "label"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {sorted(missing)}")
            if row["label"] not in LABELS:
                raise ValueError(f"{path}:{line_no}: bad label {row['label']!r}")
            items.append(row)
    return items


def run(dataset: Path, limit: int | None, laya_mode: str) -> int:
    items = load_items(dataset)
    if limit is not None:
        items = items[:limit]

    engine: object
    if laya_mode == "disabled":
        engine = _DeadClassifier()
    else:
        engine = LayaDecisionEngine()

    confusion = Counter()
    per_class_total = Counter()
    per_class_ok = Counter()
    failures: list[tuple[str, str, str, LayaDecision]] = []

    for item in items:
        decision = engine.classify_intent(item["text"])  # type: ignore[attr-defined]
        predicted = decision.intent
        confusion[(item["label"], predicted)] += 1
        per_class_total[item["label"]] += 1
        if predicted == item["label"]:
            per_class_ok[item["label"]] += 1
        else:
            failures.append((item["id"], item["label"], predicted, decision))

    total = len(items)
    correct = sum(per_class_ok.values())
    accuracy = correct / total if total else 0.0

    shown = dataset.relative_to(ROOT) if dataset.is_relative_to(ROOT) else dataset
    print(f"dataset : {shown}")
    print(f"items   : {total}  (laya mode: {laya_mode})")
    print(f"accuracy: {correct}/{total} = {accuracy:.3f}")
    print("per-class recall:")
    for label in LABELS:
        total_l = per_class_total[label]
        ok_l = per_class_ok[label]
        rate = f"{ok_l / total_l:.2f}" if total_l else "n/a"
        print(f"  {label:8s} {ok_l:3d}/{total_l:3d}  {rate}")
    print("confusion matrix (rows = gold, columns = predicted):")
    print("        " + "  ".join(f"{p:>8s}" for p in LABELS))
    for gold in LABELS:
        cells = [f"{confusion[(gold, p)]:8d}" for p in LABELS]
        print(f"  {gold:>6s}" + "  ".join(cells))
    if failures:
        print(f"failures ({len(failures)}):")
        for item_id, gold, predicted, decision in failures:
            print(
                f"  {item_id:32s} gold={gold:8s} got={predicted:8s} "
                f"(confidence={decision.confidence:.3f}, source={decision.source})"
            )
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--laya-mode", default="full", choices=["full", "disabled"])
    args = parser.parse_args()
    return run(args.dataset, args.limit, args.laya_mode)


if __name__ == "__main__":
    raise SystemExit(main())
