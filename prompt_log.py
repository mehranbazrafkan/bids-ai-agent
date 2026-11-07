"""Write the most recent system prompt out to a text file.

The desktop app sends the question, the finding's context, the dataset
tree and the file digest as separate fields; the planner assembles them
into the system prompt that is then handed to the model. None of that is
visible from the app, so this writes it down for inspection -- but only
the latest one: every request overwrites the file, and nothing else is
ever written. No chat turns, no question, no rendered text: a
conversation lives in RAM while it runs and nowhere else.

The file lands next to ``main.py`` (``bidsmgr/ai_agent/prompt_log.txt``)
whatever the working directory is.

Turn it off with ``BIDS_AGENT_LLM_SAVE_PROMPTS=false`` or
``{"llm": {"save_prompts": false}}`` in ``config.json``. Writing is
wrapped in a ``try``: a broken path or a full disk must never cost the
user an answer.

This module is a temporary inspection aid: remove it together with the
one call in ``llm_client`` once testing is done.
"""

from __future__ import annotations

from pathlib import Path

from config import CONFIG

#: Next to ``main.py`` -- the service spawns the agent with that as its
#: working directory, but ``python main.py`` run from anywhere should
#: land in the same place, so this is anchored on the module rather than
#: on the cwd.
DEFAULT_PATH = Path(__file__).resolve().parent / "prompt_log.txt"


def save(system_prompt: str, *, path: Path | None = None) -> Path | None:
    """Replace the log with the latest system prompt.

    Only the system prompt is written -- no conversation -- and only the
    latest one: the file is overwritten rather than appended to, so it
    always shows exactly what the most recent request was built from.
    Returns the file written, or ``None`` when the log is switched off
    or writing failed.
    """
    if not bool(getattr(CONFIG.llm, "save_prompts", True)):
        return None
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(system_prompt, encoding="utf-8")
    except OSError:
        # A debug aid must never break an answer.
        return None
    return target
