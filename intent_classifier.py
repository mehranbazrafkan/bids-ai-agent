"""Fast encoder-based intent classifier using zero-shot classification."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import pipeline

from model_manager import manager as model_manager

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_PIPELINE_NAME = "intent-classifier"


def _load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model=_MODEL_NAME,
        device=device,
    )


class IntentClassifier:
    """Zero-shot text classifier for EXPLAIN vs FIX intent detection.

    Uses a small encoder model (~200 MB) instead of the full LLM for
    classification.  The model is loaded on demand and offloaded after
    each call to keep VRAM free for the main LLM.
    """

    LABELS = ["explain", "fix"]
    HYPOTHESIS_TEMPLATE = "The user wants to {} the issue."

    def classify(self, user_input: str, context: str = "") -> Optional[str]:
        """Classify user intent as 'explain' or 'fix'."""
        clf = model_manager.get(_PIPELINE_NAME, _load_pipeline)

        text = f"{user_input}\nContext: {context}" if context else user_input

        result = clf(
            text,
            candidate_labels=self.LABELS,
            hypothesis_template=self.HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        model_manager.offload_current()

        if top_score < 0.5:
            return None
        return top_label
