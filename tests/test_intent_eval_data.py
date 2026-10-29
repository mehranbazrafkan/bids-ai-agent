"""Fast validation of the intent evaluation dataset (no model required).

The dataset powers `eval_intent.py` and the optional `slow` test in
test_agent.py; these checks keep it well-formed without loading Laya.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval_intent import DEFAULT_DATASET, LABELS, load_items

MIN_ITEMS = 30
MIN_PER_LABEL = 8


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return load_items(DEFAULT_DATASET)


def test_dataset_exists_and_is_jsonl():
    assert DEFAULT_DATASET.exists(), f"missing {DEFAULT_DATASET}"
    with DEFAULT_DATASET.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            json.loads(line)  # raises on malformed lines


def test_dataset_has_enough_items_per_label(items):
    counts = Counter(item["label"] for item in items)
    assert len(items) >= MIN_ITEMS, f"only {len(items)} items"
    for label in LABELS:
        assert counts[label] >= MIN_PER_LABEL, f"only {counts[label]} {label} items"


def test_ids_unique_and_texts_non_trivial(items):
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for item in items:
        assert item["text"].strip(), f"empty text in {item['id']}"
        # Deliberately tiny: short greetings ("Hi there", "ping") are valid
        # unclear-class items; only reject empty/near-empty strings.
        assert len(item["text"].strip()) >= 4, f"too-short text in {item['id']}"


def test_no_label_leaks_into_text(items):
    """The text must not contain the answer (e.g. the word "explain" as a label hint).

    Paraphrases are fine; an explicit `intent=...` marker would make the eval
    trivially gameable, so forbid it.
    """
    for item in items:
        lowered = item["text"].lower()
        for label in LABELS:
            assert f"intent={label}" not in lowered, f"label leak in {item['id']}"
            assert f"label={label}" not in lowered, f"label leak in {item['id']}"


def test_default_dataset_points_at_repo_file():
    assert DEFAULT_DATASET.is_absolute()
    assert "tests" in DEFAULT_DATASET.parts


# --- slow: runs the real local Laya model -----------------------------------
# The bar is a *dead-classifier* regression check, not a quality claim: the
# trivial "everything is unclear" baseline scores 12/51 = 0.235 on this set,
# and the previous entropy-confidence gate (measured 0.31-0.35) failed here.
# The current probability gate measured ~0.63 (two runs, 0.627/0.627); 0.45
# leaves headroom for run-to-run model variation while still requiring the
# classifier to actually work. Report real numbers from `python eval_intent.py`,
# not from this threshold.


@pytest.mark.slow
def test_laya_beats_dead_classifier_baseline(capsys):
    import eval_intent

    exit_code = eval_intent.run(DEFAULT_DATASET, limit=None, laya_mode="full")
    out = capsys.readouterr().out
    accuracy_line = next(line for line in out.splitlines() if line.startswith("accuracy:"))
    correct, total = accuracy_line.split(":")[1].split("=")[0].strip().split("/")
    accuracy = int(correct) / int(total)
    assert accuracy >= 0.45, f"accuracy {accuracy:.3f} on eval set; output:\n{out}"
    assert exit_code in (0, 1)


def test_disabled_mode_is_the_dead_classifier_baseline(capsys):
    """The control condition must be all-unclear (no model, no surprises)."""
    import eval_intent

    exit_code = eval_intent.run(DEFAULT_DATASET, limit=None, laya_mode="disabled")
    out = capsys.readouterr().out
    assert "accuracy: 12/51 = 0.235" in out, out
    assert exit_code == 1
