"""Render one request-side prompt as a plain-text log.

The desktop app exports the prompt of every validation finding for review
and fine-tuning. The record from :meth:`agent.Agent.build_prompt` is enough
to reconstruct what a real question would send, but JSON is awkward to read
and to feed to tooling that expects one prompt per file. This module keeps
the on-disk shape in one place: the complete system message, the user
question, and deliberately no generated answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_prompt_log(record: Mapping[str, Any]) -> str:
    """Format a prompt record as the app's per-finding ``.log`` file.

    ``system_prompt`` already contains the instructions and every evidence
    section (issue, file digest, dataset tree, retrieved knowledge and the
    answer checklist). ``user_prompt`` is the question that follows it.
    There is no output section: this is the request side of a fine-tuning
    pair, not a conversation.
    """
    system = str(record.get("system_prompt") or "").strip()
    user = str(record.get("user_prompt") or "").strip()
    return (
        "[System prompt]\n"
        "---------------\n"
        f"{system}\n\n"
        "[User prompt]\n"
        "-------------\n"
        f"{user}\n"
    )


__all__ = ["render_prompt_log"]
