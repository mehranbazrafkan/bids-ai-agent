"""Repair and fallback for answers that violate their contract.

Three layers, cheapest first:

1. :func:`deterministic_repair` — mechanical fixes that need no model: add the
   mandated "the knowledge base does not document this rule" sentence, strip
   an example the evidence does not support, replace an invented value with a
   visible placeholder, drop an empty section.
2. a single model repair pass (the prompt lives in
   :func:`render_repair_instructions`) for what remains.
3. :func:`render_grounded_template` — an answer built only from the contract,
   correct by construction. It is the floor of the system: if two repair
   attempts cannot produce a conforming answer, the user gets this instead of
   a fluent violation.

Nothing here invents a dataset property. The template can only say what the
contract already establishes.
"""

from __future__ import annotations

import re
from typing import Sequence

from answer_contract import (
    AUTHORITY_MANAGER,
    AUTHORITY_SPEC,
    AUTHORITY_VALIDATOR,
    AnswerContract,
    NO_DIRECT_SENTENCE,
    render_contract,
)
from output_gate import GateResult, GateViolation, neutralise_version_values

_FIELD_ASSIGN_RE_TEMPLATE = (
    r'(?P<prefix>["\']?{field}["\']?\s*[:=]\s*["\'])(?P<value>[^"\']*)(?P<suffix>["\'])'
)


def _placeholder_for(field_name: str) -> str:
    return f"<{field_name or 'value'} to be provided>"


def _strip_fences(text: str) -> str:
    return re.sub(r"\n?```.*?```\n?", "\n", text, flags=re.DOTALL)


def _replace_concrete_value(text: str, contract: AnswerContract) -> str:
    """Replace an invented value with a visible placeholder, keep the shape."""
    if not contract.field:
        return text
    pattern = re.compile(
        _FIELD_ASSIGN_RE_TEMPLATE.format(field=re.escape(contract.field)),
        re.IGNORECASE,
    )

    def _sub(match: re.Match) -> str:
        value = match.group("value").strip()
        lowered = value.lower()
        is_placeholder = (
            not value
            or (value.startswith("<") and value.endswith(">"))
            or lowered in {"todo", "null", "none", "text", "value", "string",
                           "example", "n/a"}
            or lowered.startswith("your ")
        )
        if is_placeholder:
            return match.group(0)
        return (
            match.group("prefix") + _placeholder_for(contract.field)
            + match.group("suffix")
        )

    return pattern.sub(_sub, text)


def _remove_empty_sections(text: str) -> str:
    """Drop a heading that is immediately followed by another heading/EOF."""
    lines = text.splitlines()
    kept: list[str] = []
    heading = re.compile(r"^\s*\*\*(Problem|Why|How to Fix|Example)\*\*\s*$")
    for index, line in enumerate(lines):
        if heading.match(line):
            following = next(
                (candidate for candidate in lines[index + 1:] if candidate.strip()),
                "",
            )
            if not following or heading.match(following):
                continue
        kept.append(line)
    return "\n".join(kept)


def _append_unknowns(text: str, contract: AnswerContract) -> str:
    if "what cannot be determined" in text.lower():
        return text
    lines = [f"- {item}" for item in contract.must_state_unknown]
    if not lines:
        return text
    return (
        text.rstrip()
        + "\n\n**What cannot be determined**\n"
        + "\n".join(lines)
    )


def deterministic_repair(
    text: str,
    contract: AnswerContract,
    gate: GateResult,
) -> str:
    """Apply every mechanical fix the violations call for."""
    codes = set(gate.codes)
    repaired = text or ""

    if "example_not_allowed" in codes:
        repaired = _remove_empty_sections(_strip_fences(repaired))

    if "concrete_value_not_allowed" in codes:
        repaired = _replace_concrete_value(repaired, contract)

    if "unsupported_version_value" in codes:
        repaired = neutralise_version_values(repaired)

    if "empty_section" in codes:
        repaired = _remove_empty_sections(repaired)

    if "missing_no_direct_declaration" in codes:
        repaired = repaired.rstrip() + "\n\n" + NO_DIRECT_SENTENCE
        repaired = _append_unknowns(repaired, contract)

    if "duplicate_section" in codes or "empty_section" in codes:
        repaired = _remove_empty_sections(repaired)

    return repaired.strip()


# ---------------------------------------------------------------------------
# Model repair
# ---------------------------------------------------------------------------


def render_repair_instructions(
    contract: AnswerContract,
    violations: Sequence[GateViolation],
) -> str:
    """The system half of the single repair request."""
    rules = "\n".join(f"- {v.code}: {v.detail}" for v in violations)
    return (
        "[Answer Repair]\n"
        "Your previous answer violates the answer contract below. Rewrite it "
        "so every violation is fixed and nothing else changes: keep the "
        "manual steps, keep the explanation, do not add new claims, do not "
        "add an example unless the contract allows one.\n\n"
        f"Violations to fix:\n{rules}\n\n"
        f"{render_contract(contract)}"
    )


# ---------------------------------------------------------------------------
# Grounded template (correct by construction)
# ---------------------------------------------------------------------------


def _severity_word(contract: AnswerContract) -> str:
    return "warning" if contract.severity == "warning" else (
        contract.severity or "finding"
    )


def render_grounded_template(contract: AnswerContract) -> str:
    """A complete answer built only from the contract.

    Used as the final tier: it cannot fabricate a value, invert the BIDS
    level, call a warning an error, or cite a rule the evidence does not
    support.
    """
    field = f"`{contract.field}`" if contract.field else "the reported field"
    where = f" in `{contract.path}`" if contract.path else ""
    severity = _severity_word(contract)

    if contract.placeholder:
        problem = (
            f"The field {field}{where} contains a TODO placeholder, so a real "
            "value has not been supplied yet."
        )
    elif contract.message:
        problem = f"The {severity} reports: {contract.message}"
    else:
        problem = f"The {severity} concerns {field}{where}."

    why: list[str] = []
    if contract.authority == AUTHORITY_MANAGER:
        why.append(
            "This is a dataset-manager check, not a BIDS specification "
            "requirement."
        )
    elif contract.authority == AUTHORITY_VALIDATOR:
        why.append(
            "This is a validator check, not itself a BIDS specification "
            "requirement."
        )
    elif contract.authority == AUTHORITY_SPEC:
        why.append("This is a BIDS specification rule.")
    if contract.bids_level:
        why.append(f"BIDS lists {field} as {contract.bids_level}.")
    if contract.requires_no_direct_statement:
        why.append(NO_DIRECT_SENTENCE)
    elif contract.direct_evidence and contract.rule_id:
        why.append(f"The retrieved record documents rule `{contract.rule_id}`.")

    if contract.placeholder:
        how = (
            f"Replace the TODO in {field} with the value the dataset owner "
            "intends."
        )
    elif contract.message:
        how = (
            f"Correct the value of {field} so it satisfies the constraint in "
            "the reported finding."
        )
    else:
        how = f"Correct the value of {field}."

    unknown = contract.must_state_unknown or (
        "no further detail can be determined from the supplied evidence",
    )

    parts = [
        "**Problem**",
        problem,
        "",
        "**Why**",
        " ".join(why) if why else (
            "The supplied evidence does not establish a cause."
        ),
        "",
        "**How to Fix**",
        how,
        "",
        "**What cannot be determined**",
        "\n".join(f"- {item}" for item in unknown),
    ]
    return "\n".join(parts)
