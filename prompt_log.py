"""Write the most recent request and answer out to a text file.

The desktop app sends the question, the finding's context, the dataset
tree and the file digest as separate fields; the planner assembles them
into the system prompt that is then handed to the model along with the
user's own question. None of that is visible from the app, so this
writes it down for inspection -- but only the latest one: every request
overwrites the file. It holds three labelled parts: the system prompt
exactly as the planner assembled it, the user prompt (the question the
user actually asked) and the agent output (what the model generated).
When the quality gate changes that output -- a deterministic repair, a
model repair or the grounded template -- a fourth part is added,
``[Final response]``, holding the text the caller actually received.
Both are kept on purpose: the pair shows what the model said and what
the gate did to it. No chat history is written: a conversation lives in
RAM while it runs and nowhere else.

The file lands next to ``main.py`` (``bidsmgr/ai_agent/prompt_log.txt``)
whatever the working directory is.

One request may reach the model more than once: the planner's answer-repair
pass is a second call. That call runs inside :func:`suspended`, so it cannot
overwrite the record of the request the user actually made. The planner then
calls :func:`annotate_final` once the gate has settled on the text it returns,
which is why the file can carry a ``[Final response]`` part the model never
produced.

Turn it off with ``BIDS_AGENT_LLM_SAVE_PROMPTS=false`` or
``{"llm": {"save_prompts": false}}`` in ``config.json``. Writing is
wrapped in a ``try``: a broken path or a full disk must never cost the
user an answer.

This module is a temporary inspection aid: remove it together with the
one call in ``llm_client`` once testing is done.
"""

from __future__ import annotations

import contextlib
import contextvars
from pathlib import Path
from typing import Optional, Tuple

from config import CONFIG

#: Next to ``main.py`` -- the service spawns the agent with that as its
#: working directory, but ``python main.py`` run from anywhere should
#: land in the same place, so this is anchored on the module rather than
#: on the cwd.
DEFAULT_PATH = Path(__file__).resolve().parent / "prompt_log.txt"

#: Set while a second model call for the *same* user request runs (the
#: answer-repair pass). Such a call must not overwrite the record of the
#: request the user actually made. A ContextVar rather than a global so
#: concurrent requests cannot suppress each other.
_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prompt_log_suppressed", default=False
)

#: The parts of the record currently on disk, so the planner can add the
#: final answer without having to hand the same three strings back. A module
#: global rather than a ContextVar: the model call runs inside LangChain's
#: thread pool, whose contexts are copies, so a value set there would never
#: reach the planner again. The file is a single slot anyway -- the latest
#: request replaces the previous one -- and this record follows the same rule.
_LAST: Optional[Tuple[Path, str, str, str]] = None


@contextlib.contextmanager
def suspended():
    """Do not write the log while this context is active.

    Used around the planner's repair call: it is a second generation for
    the same request, and the file is meant to show the system prompt the
    answer was originally built from.
    """
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)



def _render(
    system_prompt: str,
    user_prompt: str,
    output: str,
    *,
    final: str = "",
    note: str = "",
) -> str:
    """The file's layout: labelled parts, latest request only.

    ``final`` is the answer the caller received; it is omitted while the
    quality gate has changed nothing, so an untouched answer is written once
    and the file keeps its three-part reading.
    """
    text = (
        f"[System prompt]\n{system_prompt}\n"
        "\n"
        f"[User prompt]\n{user_prompt}\n"
        "\n"
        f"[Agent output]\n{output}\n"
    )
    if final:
        note_line = f"{note}\n" if note.strip() else ""
        text += f"\n[Final response]\n{note_line}{final}\n"
    return text


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
    request was built from and what came back. A later
    :func:`annotate_final` adds the fourth part when the quality gate
    changed the answer. Calls made inside :func:`suspended` (the
    planner's repair pass) are ignored. Returns the file written, or
    ``None`` when the log is switched off, the call is suspended, or
    writing failed.
    """
    global _LAST
    if _SUPPRESSED.get():
        return None
    if not bool(getattr(CONFIG.llm, "save_prompts", True)):
        return None
    target = Path(path) if path is not None else DEFAULT_PATH
    # Until this request's record is on disk, there is nothing for
    # :func:`annotate_final` to attach an answer to -- a stale record from an
    # earlier request must not be extended with this request's answer.
    _LAST = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _render(system_prompt, user_prompt, output),
            encoding="utf-8",
        )
    except OSError:
        # A debug aid must never break an answer.
        return None
    _LAST = (target, system_prompt, user_prompt, output)
    return target


def annotate_final(final: str, *, note: str = "") -> Path | None:
    """Add the answer that was actually returned, after the quality gate.

    ``[Agent output]`` stays the model's own text; this is the second half of
    the pair, so a repair can be seen rather than guessed at. ``note`` is a
    short one-line account of what the gate did (codes, tier); it is written
    under the ``[Final response]`` heading.

    Does nothing when no record was written for this request, when the gate
    left the answer untouched (nothing to compare), or when the log is off.
    """
    if _SUPPRESSED.get():
        return None
    if not bool(getattr(CONFIG.llm, "save_prompts", True)):
        return None
    last = _LAST
    if last is None:
        return None
    target, system_prompt, user_prompt, output = last
    if (final or "") == output:
        return None
    try:
        target.write_text(
            _render(
                system_prompt,
                user_prompt,
                output,
                final=final or "",
                note=note,
            ),
            encoding="utf-8",
        )
    except OSError:
        return None
    return target
