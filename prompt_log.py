"""Write the most recent request and answer out to a text file.

The desktop app sends the question, the finding's context, the dataset
tree and the file digest as separate fields; the planner assembles them
into the system prompt that is then handed to the model along with the
user's own question. None of that is visible from the app, so this
writes it down for inspection -- but only the latest one: every request
overwrites the file. It holds three labelled parts: the system prompt
exactly as the planner assembled it, the user prompt (the question the
user actually asked) and the agent output (what the model generated).
No chat history is written: a conversation lives in RAM while it runs
and nowhere else.

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


def _render(system_prompt: str, user_prompt: str, output: str) -> str:
    """The file's layout: three labelled parts, latest request only."""
    return (
        f"[System prompt]\n{system_prompt}\n"
        "\n"
        f"[User prompt]\n{user_prompt}\n"
        "\n"
        f"[Agent output]\n{output}\n"
    )


def save(
    system_prompt: str,
    *,
    user_prompt: str = "",
    output: str = "",
    path: Path | None = None,
) -> Path | None:
    """Replace the log with the latest system prompt, user prompt and
    agent output.

    Only those three labelled parts are written -- no conversation --
    and only the latest ones: the file is overwritten rather than
    appended to, so it always shows exactly what the most recent
    request was built from and what came back. Returns the file written,
    or ``None`` when the log is switched off or writing failed.
    """
    if not bool(getattr(CONFIG.llm, "save_prompts", True)):
        return None
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _render(system_prompt, user_prompt, output),
            encoding="utf-8",
        )
    except OSError:
        # A debug aid must never break an answer.
        return None
    return target
