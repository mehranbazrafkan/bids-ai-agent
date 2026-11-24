"""Capture the most recent request and answer in memory, for the API.

The desktop app sends the question, the finding's context, the dataset
tree and the file digest as separate fields; the planner assembles them
into the system prompt that is then handed to the model along with the
user's own question. None of that is visible from the app, so this
module records it -- and the generation metadata that goes with it --
in memory, and ``POST /predict`` returns the record to the client.

This used to write ``prompt_log.txt`` next to ``main.py``: a file the
user had to find by hand, overwritten on every request, sitting inside
the package directory. The record now travels in the ``/predict``
response instead (under ``log``, with ``model``, ``prompt_tokens``,
``generated_tokens`` and ``generation_seconds`` beside it), so the
desktop app can show it per message; the agent writes nothing.

One record, the most recent request, held at module scope under a lock.
It holds the system prompt exactly as the planner assembled it, the user
prompt (the question the user actually asked), the agent output (what
the model generated) and how the answer was produced (model, token
counts, seconds). No chat history is recorded here: the agent keeps
none, and any copy of a conversation on disk belongs to the desktop
app's chat-log folder, not to this process.

The record is module-level rather than a ContextVar on purpose: the
planner's LCEL pipeline runs its answer branch in an executor thread, so
a value set there never propagates back to the thread that calls
:func:`snapshot`. A global record is exactly what the old file was -- one
"most recent request" -- with the same concurrency caveat: two requests
in flight at once overwrite each other, which a local, single-user
desktop app does not do.

One request may reach the model more than once: the planner's
answer-repair pass is a second call. That call runs inside
:func:`suspended`, so it cannot overwrite the record of the request the
user actually made. Short classification calls never record at all.

A failure here never costs an answer: recording is wrapped so a broken
record cannot raise into the generation path.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import threading

#: The record of the most recent request, under :data:`_LOCK`. A plain
#: dict rather than a ContextVar: the answer branch of the planner's
#: pipeline runs in an executor thread, and a ContextVar set there never
#: comes back to the thread that snapshots the record.
_RECORD: dict | None = None
_LOCK = threading.Lock()

#: Set while a second model call for the *same* user request runs (the
#: answer-repair pass). Such a call must not overwrite the record of the
#: request the user actually made. A ContextVar rather than a global so
#: concurrent requests cannot suppress each other; the repair pass runs
#: inline in the request's own thread, which is what makes this work.
_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prompt_log_suppressed", default=False
)


@contextlib.contextmanager
def suspended():
    """Do not touch the record while this context is active.

    Used around the planner's repair call: it is a second generation for
    the same request, and the record is meant to show the system prompt
    the answer was originally built from.
    """
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)


def reset() -> None:
    """Forget any record left over from a previous request.

    ``POST /predict`` calls this before the agent runs, so a request
    whose model is never called (the clarification shortcut) reports no
    log rather than the previous request's.
    """
    global _RECORD
    with _LOCK:
        _RECORD = None


def begin(system_prompt: str, user_prompt: str) -> None:
    """Start the record for this request: the two prompt halves.

    Called by the generation path with the system prompt exactly as the
    planner assembled it and the user's own question. Calls made inside
    :func:`suspended` (the repair pass) are ignored, so the record keeps
    the original request.
    """
    if _SUPPRESSED.get():
        return
    global _RECORD
    with _LOCK:
        _RECORD = {
            "system_prompt": str(system_prompt or ""),
            "user_prompt": str(user_prompt or ""),
        }


def record_generation(
    output: str,
    *,
    model: str = "",
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    seconds: float | None = None,
) -> None:
    """Fill in what the model generated and how it was produced.

    Called by the generation path after decoding. ``model`` is the
    checkpoint that wrote the answer, the token counts cover the prompt
    handed to it and the completion it produced, and ``seconds`` is the
    wall-clock time of the whole generation (template, encode, generate,
    decode). Calls made inside :func:`suspended` are ignored.
    """
    if _SUPPRESSED.get():
        return
    global _RECORD
    with _LOCK:
        if _RECORD is None:
            # ``begin`` was never called for this request (a direct call
            # to the LLM outside /predict). Recording only the output
            # would present half a picture as the whole one, so start
            # the record here rather than lose it.
            _RECORD = {"system_prompt": "", "user_prompt": ""}
        _RECORD.update({
            "output": str(output or ""),
            "model": str(model or ""),
            "prompt_tokens": (
                int(prompt_tokens) if prompt_tokens is not None else None
            ),
            "generated_tokens": (
                int(generated_tokens) if generated_tokens is not None else None
            ),
            "generation_seconds": (
                round(float(seconds), 3) if seconds is not None else None
            ),
        })


def snapshot() -> dict | None:
    """The record of the most recent request, or ``None``.

    ``POST /predict`` calls this after the agent returns and attaches it
    to the response. A copy is returned so the client's payload cannot
    reach back into the record while the next request builds.
    """
    with _LOCK:
        return copy.deepcopy(_RECORD) if _RECORD is not None else None


__all__ = ["begin", "record_generation", "reset", "snapshot", "suspended"]
