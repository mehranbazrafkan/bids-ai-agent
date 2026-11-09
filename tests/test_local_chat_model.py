"""Tests for LocalChatModel — the LangChatModel seam over the backend.

The wrapper's whole job is to map a LangChain message list onto the
flat backend call ``LLM.chat(system_prompt, user_prompt, history=...)``
without changing a single prompt byte. These tests pin that mapping
with a recording fake backend — no model, no GPU, no Laya.
"""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from local_chat_model import LocalChatModel


class FakeBackend:
    """Records the flat call; answers with a canned string."""

    def __init__(self, answer="BACKEND ANSWER", model_name="fake/model"):
        self.answer = answer
        self.model_name = model_name
        self.calls = []

    def chat(self, system_prompt, user_prompt, enable_thinking=None,
             history=None):
        self.calls.append(
            (system_prompt, user_prompt, enable_thinking, history)
        )
        return self.answer


def make_model(backend):
    return LocalChatModel(llm=backend)


# ---------------------------------------------------------------------------
# Message list -> flat backend call
# ---------------------------------------------------------------------------


def test_decompose_maps_system_history_question():
    backend = FakeBackend()
    model = make_model(backend)
    out = model.invoke([
        SystemMessage(content="SYS"),
        HumanMessage(content="H1"),
        AIMessage(content="A1"),
        HumanMessage(content="H2"),
    ])
    assert out.content == "BACKEND ANSWER"
    system, user, thinking, history = backend.calls[0]
    assert system == "SYS"
    assert user == "H2"
    assert thinking is None
    # earlier turns, in order, in the backend's own dict shape
    assert history == [
        {"role": "user", "content": "H1"},
        {"role": "assistant", "content": "A1"},
    ]


def test_decompose_single_question_has_no_history():
    backend = FakeBackend()
    make_model(backend).invoke([
        SystemMessage(content="SYS"),
        HumanMessage(content="Q"),
    ])
    system, user, thinking, history = backend.calls[0]
    assert (system, user) == ("SYS", "Q")
    assert history is None  # not an empty list: the backend filters either way


def test_decompose_multiple_system_messages_join():
    backend = FakeBackend()
    make_model(backend).invoke([
        SystemMessage(content="S1"),
        SystemMessage(content="S2"),
        HumanMessage(content="Q"),
    ])
    assert backend.calls[0][0] == "S1\n\nS2"


def test_decompose_ai_message_closes_the_pending_user_turn():
    backend = FakeBackend()
    make_model(backend).invoke([
        SystemMessage(content="SYS"),
        HumanMessage(content="H1"),
        AIMessage(content="A1"),
        AIMessage(content="A2"),   # two answers in a row: both history
        HumanMessage(content="Q"),
    ])
    assert backend.calls[0][3] == [
        {"role": "user", "content": "H1"},
        {"role": "assistant", "content": "A1"},
        {"role": "assistant", "content": "A2"},
    ]


def test_result_is_an_ai_message_with_generation_info():
    backend = FakeBackend(model_name="the/model")
    result = make_model(backend)._generate(
        [SystemMessage(content="S"), HumanMessage(content="Q")]
    )
    generation = result.generations[0]
    assert isinstance(generation.message, AIMessage)
    assert generation.message.content == "BACKEND ANSWER"
    # which model answered, for the record
    assert generation.generation_info["model"] == "the/model"


def test_missing_user_message_raises():
    model = make_model(FakeBackend())
    with pytest.raises(ValueError, match="user message"):
        model.invoke([SystemMessage(content="SYS")])
    with pytest.raises(ValueError, match="user message"):
        model.invoke([HumanMessage(content="H1"), AIMessage(content="A1")])


def test_llm_type_reports_the_wrapper():
    assert make_model(FakeBackend())._llm_type == "bids-local-chat"


def test_enable_thinking_override_is_forwarded():
    backend = FakeBackend()
    LocalChatModel(llm=backend, enable_thinking=False).invoke(
        [SystemMessage(content="S"), HumanMessage(content="Q")]
    )
    assert backend.calls[0][2] is False


# ---------------------------------------------------------------------------
# The legacy flat interface (benchmark_timing.py call shape)
# ---------------------------------------------------------------------------


def test_generate_response_shim_matches_the_message_path():
    """The old flat spelling must produce the SAME backend call.

    That is what keeps ``benchmark_timing.py`` (and any old call site)
    measuring the identical prompt the pipeline sends.
    """
    flat = FakeBackend()
    LocalChatModel(llm=flat).generate_response(
        "SYS", "Q?", history=[
            {"role": "user", "content": "h1"},
            {"role": "assistant", "content": "a1"},
        ],
    )

    messages = FakeBackend()
    make_model(messages).invoke([
        SystemMessage(content="SYS"),
        HumanMessage(content="h1"),
        AIMessage(content="a1"),
        HumanMessage(content="Q?"),
    ])

    assert flat.calls[0] == messages.calls[0]


def test_generate_response_shim_returns_the_answer():
    backend = FakeBackend(answer="ANS")
    out = make_model(backend).generate_response("S", "Q", history=None)
    assert out == "ANS"


def test_generate_response_shim_filters_non_chat_roles():
    backend = FakeBackend()
    make_model(backend).generate_response("S", "Q", history=[
        {"role": "system", "content": "skip me"},
        {"role": "user", "content": "  "},
        {"role": "user", "content": "kept"},
    ])
    assert backend.calls[0][3] == [{"role": "user", "content": "kept"}]
