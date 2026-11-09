"""LangChain chat model wrapping the local LLM backend.

The BIDS agent's generation backend (``llm_client.LLM``) speaks a flat
``chat(system_prompt, user_prompt, history=...)`` interface and owns the
whole model lifecycle — lazy loading, quantization, the ModelManager,
the prompt log. The LangChain world speaks *messages*: a system
message, the earlier turns, the current question.

This adapter is the seam between the two. The planner's LCEL pipeline
hands it a message list; it reassembles the flat backend call from it —

* every system message  -> the ``system_prompt`` string,
* the turns before the last user turn -> ``history``,
* the final user turn   -> ``user_prompt``,

— and nothing else changes: the same backend, the same prompt
byte-for-byte, the same logging. The wrapper is deliberately thin; any
behavior that matters for the Phase-1 regression work (the chat shape,
the template, the generation parameters) stays in ``llm_client.LLM``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


def _as_text(content: Any) -> str:
    """Message content as plain text (content is a str in this project)."""
    return content if isinstance(content, str) else str(content or "")


class LocalChatModel(BaseChatModel):
    """LangChain chat model for the local LLM backend.

    Parameters mirror the backend's own knobs only where the caller may
    want to override them; unset values mean "the backend decides" (its
    config, captured at construction), which is exactly what the
    non-LangChain path does.
    """

    # The backend (an ``llm_client.LLM``) is a plain class with its own
    # lifecycle, so it is held as an untyped field rather than a typed
    # pydantic model field: the wrapper only needs ``.chat(...)`` and
    # ``.model_name``, and tests may substitute a duck-typed fake.
    llm: Any
    #: Reported in generation info; empty means "the backend's model".
    model_name: str = ""
    #: Passed through to ``LLM.chat``; ``None`` keeps the backend's own
    #: configured value (the default for the production wiring).
    enable_thinking: Optional[bool] = None

    @property
    def _llm_type(self) -> str:
        return "bids-local-chat"

    # ------------------------------------------------------------------
    # LangChain entry point
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text, history, user_prompt = self._decompose(messages)
        if not user_prompt.strip():
            # A conversation without a current question cannot be turned
            # into a backend call (the flat signature needs one), so say
            # so instead of generating against an empty prompt.
            raise ValueError(
                "LocalChatModel requires the final message to be a user "
                "message with content"
            )
        raw = self.llm.chat(
            system_text,
            user_prompt,
            self.enable_thinking,
            history=history or None,
        )
        generation = ChatGeneration(
            message=AIMessage(content=raw),
            generation_info={
                "model": self.model_name or self.llm.model_name,
            },
        )
        return ChatResult(generations=[generation])

    # ------------------------------------------------------------------
    # Message -> flat backend call
    # ------------------------------------------------------------------

    @staticmethod
    def _decompose(
        messages: Sequence[BaseMessage],
    ) -> tuple[str, List[Dict[str, str]], str]:
        """Map a chat message list onto the flat backend signature.

        Returns ``(system_prompt, history, user_prompt)``. The last user
        message is the live question; every earlier user/assistant turn
        becomes a ``{"role", "content"}`` history entry in order, which
        is exactly the shape ``LLM._generate`` expects.
        """
        system_parts: List[str] = []
        turns: List[Dict[str, str]] = []
        pending_user = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(_as_text(msg.content))
            elif isinstance(msg, HumanMessage):
                if pending_user:
                    # A new user turn closes the previous one: the earlier
                    # question is history now, this one is the live one.
                    turns.append({"role": "user", "content": pending_user})
                pending_user = _as_text(msg.content)
            elif isinstance(msg, AIMessage):
                if pending_user:
                    turns.append({"role": "user", "content": pending_user})
                    pending_user = ""
                turns.append({"role": "assistant", "content": _as_text(msg.content)})
        return "\n\n".join(system_parts), turns, pending_user

    # ------------------------------------------------------------------
    # Legacy flat interface (benchmark_timing.py and old call sites)
    # ------------------------------------------------------------------

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """One prompt, in the old flat spelling, through the new path.

        Builds the same message list the pipeline would build and runs
        it through :meth:`invoke`, so the backend call — and therefore
        the prompt bytes — is identical to the pre-LangChain code path.
        """
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for turn in history or []:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip().lower()
            content = str(turn.get("content") or "")
            if not content.strip():
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_prompt))
        return self.invoke(messages).content
