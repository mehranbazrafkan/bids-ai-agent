"""What the agent is doing right now, for a client that is waiting on it.

``POST /predict`` does not answer until the answer exists, so from the
outside a request that is reading a checkpoint into memory and one that
is writing the answer look exactly the same: like a request that has not
returned. The first request of a session loads the model, which can take
a minute on a CPU, and a desktop app showing "working for 0 ... 60
seconds" during that time is reporting the loader's work as the model's.

This module is the small piece of process-wide truth behind
``GET /status``: which of the three phases the process is in, and, once
generation has begun, how long it has been going. A client polls it from
beside the blocking request and starts its clock when the phase turns to
``generating``.

It also remembers how long the load itself took, from the first byte of
the checkpoint to the first generated token. The client's clock does not
run during that time (it is not generation), but the answer's record
reports it afterwards: a cold first answer is slow for two different
reasons, and the user is entitled to see both numbers.

The state belongs to the process, not to a request -- one model, one
device, one answer at a time -- and is written from inference threads
while Flask's other threads read it, hence the lock. Like
:mod:`prompt_log`, two simultaneous requests would overwrite each other's
phase; a local, single-user desktop app does not do that.
"""

from __future__ import annotations

import threading
import time

#: No request is being worked on. Also where a request ends, answered or
#: not: a phase that outlives its request would be a clock nobody stops.
IDLE = "idle"
#: A request needs the model and has not got it in memory yet. This is
#: the part of the wait that is not generation.
LOADING = "loading"
#: The model is resident and is writing (an intent answer, the answer
#: itself, a repair). The client's clock starts here.
GENERATING = "generating"

_LOCK = threading.Lock()
_PHASE = IDLE
#: ``time.time()`` of the first generation in this request, or ``None``.
#: Fixed once per request rather than per call, so the second model call
#: of a pipeline (a repair, the answer after a classified intent) does not
#: restart the clock halfway through.
_GENERATION_STARTED: float | None = None
#: ``time.time()`` when the checkpoint began to be read, or ``None``.
_LOAD_STARTED: float | None = None
#: How long that read took, once generation began. ``None`` until then,
#: and for a request whose model was already in memory.
_LOAD_SECONDS: float | None = None


def loading() -> None:
    """A request is about to load the checkpoint.

    Leaves the phase at ``GENERATING`` if generation has already begun:
    a load in the middle of an answer (there is none today, but the phase
    must never run backwards) is not the client's cue to stop counting.
    """
    global _PHASE, _LOAD_STARTED
    with _LOCK:
        if _PHASE != GENERATING:
            _PHASE = LOADING
            if _LOAD_STARTED is None:
                _LOAD_STARTED = time.time()


def generating() -> None:
    """The model has started writing output.

    The first call in a request fixes the moment the clock starts; later
    calls leave it alone, so the measured time is the request's, not the
    last call's. It is also the moment the load is over: the distance
    from :func:`loading` to here is what the answer's record reports
    separately.
    """
    global _PHASE, _GENERATION_STARTED, _LOAD_SECONDS
    with _LOCK:
        if _GENERATION_STARTED is None:
            _GENERATION_STARTED = time.time()
            if _LOAD_STARTED is not None:
                _LOAD_SECONDS = max(0.0, _GENERATION_STARTED - _LOAD_STARTED)
        _PHASE = GENERATING


def finish() -> None:
    """The request is over, whatever came of it."""
    global _PHASE, _GENERATION_STARTED, _LOAD_STARTED, _LOAD_SECONDS
    with _LOCK:
        _PHASE = IDLE
        _GENERATION_STARTED = None
        _LOAD_STARTED = None
        _LOAD_SECONDS = None


def load_seconds() -> float:
    """How long the checkpoint took to read before generation began.

    ``0.0`` when no load was needed — the model was already in memory.
    Read it before :func:`finish`, which ends the request and clears the
    measurement with it.
    """
    with _LOCK:
        return float(_LOAD_SECONDS or 0.0)


def snapshot() -> dict:
    """The phase as ``GET /status`` reports it.

    ``generating_for`` is a duration rather than a timestamp, so the
    client never has to trust its clock to agree with this process's:
    it anchors its own counter at *now minus this many seconds*.
    """
    with _LOCK:
        phase = _PHASE
        started = _GENERATION_STARTED
    elapsed = None
    if phase == GENERATING and started is not None:
        elapsed = max(0.0, time.time() - started)
    return {"phase": phase, "generating_for": elapsed}
