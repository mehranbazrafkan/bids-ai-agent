"""
Laya-based decision engine for the BIDS AI agent.

Replaces all LLM-based decision-making with Laya's fast, non-autoregressive
System 1 decision engine. The LLM is kept ONLY for output generation
(explanations and summarization).

Decisions handled by Laya:
  1. Intent classification — "explain" vs "fix"
  2. Tool selection — which tool to call for a given issue
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

# Suppress Laya's RuntimeWarning (e.g. tokenizer parallelism)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import laya


class LayaDecisionEngine:
    """Wraps Laya for all agent decision-making.

    Uses a single Laya checkpoint loaded once and reused across all
    decision calls. Laya is non-autoregressive — a single forward pass
    (~33 ms on GPU) — so it is dramatically faster than an LLM for
    classification / routing decisions.
    """

    def __init__(self, model: str = "convaiinnovations/laya"):
        self._agent = laya.load(model)

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, user_input: str, context: str = "") -> Optional[str]:
        """Classify the user's intent as 'explain' or 'fix'.

        Returns None when Laya cannot make a confident decision.
        """
        state = f"{user_input}\nContext: {context}" if context else user_input

        questions = {
            "intent": {
                "type": "choice",
                "instructions": "What does the user want to do with this issue?",
                "criteria": {
                    "explain": (
                        "The user wants to understand, learn about, or get an "
                        "explanation of the issue. They are not asking for a fix."
                    ),
                    "fix": (
                        "The user wants to resolve, repair, correct, rename, or "
                        "otherwise fix the issue."
                    ),
                },
            }
        }

        result = self._agent.predict(state, questions)
        answer = result["answers"]["intent"]
        choice = answer.get("choice")
        confidence = answer.get("answer_confidence", 0.0)

        if choice in ("explain", "fix") and confidence >= 0.5:
            return choice
        return None

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def select_tool(self, user_input: str, context: str = "") -> Optional[str]:
        """Decide which tool should handle the given issue.

        Returns the tool name, or None when no tool matches confidently.
        """
        state = f"{user_input}\nContext: {context}" if context else user_input

        questions = {
            "tool": {
                "type": "choice",
                "instructions": "Which tool should be used to fix this issue?",
                "criteria": {
                    "validate": (
                        "The issue is about validating the dataset, checking "
                        "BIDS compliance, or verifying structure and metadata."
                    ),
                    "repair_metadata": (
                        "The issue is about repairing, correcting, or fixing "
                        "metadata fields, values, or JSON/TSV content."
                    ),
                    "rename_subject": (
                        "The issue is about renaming subjects, sessions, or "
                        "directories to match BIDS naming conventions."
                    ),
                },
            }
        }

        result = self._agent.predict(state, questions)
        answer = result["answers"]["tool"]
        choice = answer.get("choice")
        confidence = answer.get("answer_confidence", 0.0)

        if choice in ("validate", "repair_metadata", "rename_subject") and confidence >= 0.5:
            return choice
        return None
