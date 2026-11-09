"""Step-level observability for the BIDS pipeline.

The planner runs each request through a small LCEL pipeline. Attaching
a :class:`StepLogger` to it records the name and wall time of every
step that started, finished or failed — evidence preparation, prompt
rendering, the LLM generation itself, result assembly — into the
``bids_agent.pipeline`` logger, so a slow or odd request can be
diagnosed from the log without re-running it against the model.

The handler records *names and durations only*. It never logs step
inputs or outputs: those carry the dataset's content, and the
diagnostics contract (see ``Planner._diagnostics``) is that such
content never leaves the structured, internal-only trace.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler

#: One line per pipeline step: which step, how long.
logger = logging.getLogger("bids_agent.pipeline")


class StepLogger(BaseCallbackHandler):
    """Log each pipeline step's name and duration.

    The LangChain runtime reports a run's name in the ``name`` keyword
    of the start callbacks (older runtimes used ``run_name``); both are
    read so the handler keeps working across versions.
    """

    def __init__(self, log: Optional[logging.Logger] = None) -> None:
        self._log = log or logger
        # run_id -> (step name, start time); child runs of one step all
        # share the parent's timing window, so a step is popped on its
        # own end/error.
        self._starts: Dict[uuid.UUID, Tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _step_name(kwargs: Dict[str, Any]) -> str:
        name = kwargs.get("name") or kwargs.get("run_name")
        return str(name) if name else "step"

    def _start(self, run_id: uuid.UUID, kwargs: Dict[str, Any]) -> None:
        self._starts[run_id] = (self._step_name(kwargs), time.perf_counter())

    def _end(self, run_id: uuid.UUID) -> None:
        entry = self._starts.pop(run_id, None)
        if entry is None:
            return
        name, started = entry
        self._log.info("pipeline step %s: %.3fs", name, time.perf_counter() - started)

    def _error(self, run_id: uuid.UUID, error: BaseException) -> None:
        entry = self._starts.pop(run_id, None)
        if entry is None:
            return
        name, started = entry
        self._log.warning(
            "pipeline step %s failed after %.3fs: %s",
            name, time.perf_counter() - started, error,
        )

    # ------------------------------------------------------------------
    # Callback hooks (chains cover lambdas and dict branches; the LLM
    # hooks keep the generation step separately visible).
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, kwargs)

    def on_chain_end(self, outputs: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._error(run_id, error)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, kwargs)

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._error(run_id, error)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        # Chat models (the production LocalChatModel) report through this
        # hook in langchain-core 1.x, not on_llm_start.
        self._start(run_id, kwargs)

    def on_chat_model_end(
        self, response: Any, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        self._end(run_id)

    def on_chat_model_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        self._error(run_id, error)
