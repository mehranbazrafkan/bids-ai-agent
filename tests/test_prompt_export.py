"""The plain-text prompt log used by the dataset export.

The route and the desktop exporter are tested separately; this pins the
file shape itself, especially the deliberate absence of an answer.
"""

from __future__ import annotations

from prompt_export import render_prompt_log


def test_renders_the_system_prompt_then_the_user_prompt():
    text = render_prompt_log({
        "system_prompt": "System: INSTRUCTIONS\n\n[Issue]\nerr",
        "user_prompt": "Explain this error to me.",
    })

    assert text == (
        "[System prompt]\n"
        "---------------\n"
        "System: INSTRUCTIONS\n\n[Issue]\nerr\n\n"
        "[User prompt]\n"
        "-------------\n"
        "Explain this error to me.\n"
    )


def test_has_no_output_section():
    text = render_prompt_log({
        "system_prompt": "S",
        "user_prompt": "Q",
        "output": "a generated answer",
    })

    assert "[Agent output]" not in text
    assert "a generated answer" not in text
