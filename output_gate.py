"""Deterministic validation of a generated answer against its contract.

The gate is the enforcement half of :mod:`answer_contract`: the contract says
what the answer may claim, the gate checks whether it did. It is pure regular
expression work over the answer text plus the contract — no model, no
retrieval, no disk — so it is fast, reproducible, and testable with the four
recorded Phase-1 outputs.

Violations are *findings*, not rewrites. ``planner`` decides what to do with
them (deterministic repair, one model repair, deterministic template), which
keeps this module a pure predicate and the policy in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from answer_contract import NO_DIRECT_SENTENCE, AnswerContract

# ---------------------------------------------------------------------------
# Capability boundary: claims that a tool ran or the dataset was changed.
#
# Deliberately conservative — disclaimers such as "I cannot modify your
# dataset" must never match. Moved here from ``planner`` so the gate owns
# every output constraint; ``planner._response_checks`` delegates to keep the
# old import path and its diagnostics shape stable.
# ---------------------------------------------------------------------------

EXECUTION_CLAIM_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(
        r"\b(?:i\s+(?:have\s+|already\s+)?|i'?ve\s+(?:already\s+)?)"
        r"(?:fixed|repaired|renamed|modified|updated|corrected|patched)\b",
        re.IGNORECASE,
    ), "claims to have changed the dataset"),
    (re.compile(r"\bi\s+(?:have\s+|already\s+)?ran\b", re.IGNORECASE),
     "claims to have run something"),
    (re.compile(
        r"\bsuccessfully\s+(?:fixed|repaired|renamed|corrected|updated)\b",
        re.IGNORECASE,
    ), "claims a successful fix"),
    (re.compile(
        r"\b(?:validation|check|repair|update)s?\s+"
        r"(?:completed|finished)\s+successfully\b",
        re.IGNORECASE,
    ), "claims a successful validation run"),
    (re.compile(
        r"\b(?:the|your|this)\s+(?:issue|error|problem)\s+"
        r"(?:has been|was|is now)\s+(?:fixed|resolved|corrected|updated)\b",
        re.IGNORECASE,
    ), "claims the issue was already fixed"),
    (re.compile(
        r"\bi\s+(?:have\s+)?(?:applied|made)\s+(?:the\s+)?"
        r"(?:fix|change|correction|update)s?\b",
        re.IGNORECASE,
    ), "claims to have applied a fix"),
    (re.compile(
        r"\b(?:the\s+)?(?:tool|script|validator)\s+"
        r"(?:ran|executed|completed)\b",
        re.IGNORECASE,
    ), "claims a tool/validator ran"),
)


def execution_claims(text: str) -> list[str]:
    """Labels for every capability-boundary claim found in ``text``."""
    return [
        label
        for pattern, label in EXECUTION_CLAIM_PATTERNS
        if pattern.search(text or "")
    ]


def response_checks(text: str) -> Dict[str, Any]:
    """The pre-existing, flag-only checks (stable diagnostics shape)."""
    body = text or ""
    return {
        "empty_response": not body.strip(),
        "execution_claims": execution_claims(body),
    }


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateViolation:
    code: str
    detail: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class GateResult:
    violations: Tuple[GateViolation, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(v.code for v in self.violations)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [v.as_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

_NO_DIRECT_RE = re.compile(
    r"knowledge\s+base\s+(?:does\s+not|doesn'?t|cannot|can'?t)\s+"
    r"(?:directly\s+)?(?:document|cover|describe|contain)"
    r"|knowledge\s+base\s+has\s+no\s+(?:direct\s+)?(?:record|documentation)"
    r"|no\s+directly\s+relevant\s+(?:record|documentation)",
    re.IGNORECASE,
)

_REQUIRED_CLAIM_RE = re.compile(
    r"\b(?:is|are)\s+(?:considered\s+|a\s+|an\s+)?required\b"
    r"|\b(?:the|a)\s+required\s+(?:field|value|key|metadata)\b"
    r"|\brequired\s+(?:by|in)\s+(?:the\s+)?BIDS\b",
    re.IGNORECASE,
)

_BIDS_OVERCLAIM_RE = re.compile(
    r"\b(?:meet|satisf(?:y|ies))\s+(?:the\s+)?BIDS\s+"
    r"(?:standard|requirements?|specification)\b"
    r"|(?<!whether )\bBIDS\s+(?:requires|mandates)\b",
    re.IGNORECASE,
)

_WRONG_SEVERITY_RE = re.compile(
    r"\b(?:this|the)\s+error\s+(?:indicates|means|occurs|arises|is)\b"
    r"|\bis\s+(?:an|a)\s+validation\s+error\b",
    re.IGNORECASE,
)

_NAMED_ENTITY_RE = re.compile(
    r"\bDr\.\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?"
    r"|\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\s+"
    r"(?:Institute|University|Hospital|Centre|Center|Laboratory|Lab)\b"
)

_SECTION_RE = re.compile(
    r"^\s*\*\*(Problem|Why|How to Fix|Example)\*\*\s*$", re.MULTILINE
)

# A bare calibration/format version (1.0, 1.2.3). BIDS/schema/validator
# versions are facts about the specification, not the field's value, so they
# are excluded; the field's own value has no such prefix.
_VERSION_RE = re.compile(
    r"(?<![\w.])(?<!BIDS )(?<!BIDSVersion )(?<!Version )(?<!version )"
    r"(?<!schema )(?<!validator )(\d+\.\d+(?:\.\d+)?)(?![\w.])"
)

# Values that are structure, not a claim about this dataset's value.
_PLACEHOLDER_VALUES = frozenset({
    "", "...", "…", "todo", "null", "none", "text", "value", "string",
    "example", "n/a", "your value", "placeholder",
})


def _is_placeholder_value(value: str) -> bool:
    text = (value or "").strip().strip("\"'")
    lowered = text.lower()
    if lowered in _PLACEHOLDER_VALUES:
        return True
    if text.startswith("<") and text.endswith(">"):
        return True
    if lowered.startswith("your "):
        return True
    if "todo" in lowered:
        return True
    return False


def _field_value_claims(text: str, field_name: str) -> list[str]:
    """Quoted/assigned values the answer gives to ``field_name``."""
    pattern = re.compile(
        rf'["\']?{re.escape(field_name)}["\']?\s*[:=]\s*["\']([^"\']*)["\']',
        re.IGNORECASE,
    )
    return [m.group(1) for m in pattern.finditer(text)]


def _is_version_field(contract: AnswerContract) -> bool:
    return (
        "version" in (contract.field or "").lower()
        or "format" in (contract.message or "").lower()
    )


def _version_claims(
    text: str,
    contract: AnswerContract,
    supplied: str,
    evidence: str,
) -> list[str]:
    """Bare version values offered for a field that fixes none.

    This is the Phase-2 failure that a field-assignment check misses: the
    model repaired the JSON example but still wrote "common versions might be
    1.0, 1.1" in prose. Only lines that mention the field or a version are
    inspected, and BIDS/schema/validator versions are excluded by pattern.
    """
    if not _is_version_field(contract) or contract.concrete_value_allowed:
        return []
    evidence_lower = (evidence or "").lower()
    claims: list[str] = []
    for line in (text or "").splitlines():
        lowered = line.lower()
        if (contract.field or "").lower() not in lowered and "version" not in lowered:
            continue
        for match in _VERSION_RE.finditer(line):
            value = match.group(1)
            if value.lower() in supplied:
                continue
            if contract.direct_evidence and re.search(
                rf"(?<![\w.]){re.escape(value)}(?![\w.])", evidence_lower
            ):
                continue
            claims.append(value)
    return claims


def neutralise_version_values(text: str) -> str:
    """Replace bare version literals on version lines with ``<version>``.

    Used by the deterministic repair for an ``unsupported_version_value``
    finding: the sentence survives, the invented value does not.
    """
    lines: list[str] = []
    for line in (text or "").splitlines():
        if "version" in line.lower():
            line = _VERSION_RE.sub("<version>", line)
        lines.append(line)
    return "\n".join(lines)


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _first_region(text: str, limit: int = 700) -> str:
    return (text or "")[:limit]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def check(
    text: str,
    contract: AnswerContract | None,
    inputs: Sequence[Any] = (),
    evidence: str = "",
) -> GateResult:
    """Validate ``text`` against ``contract``.

    ``inputs`` are the evidence strings the answer was allowed to rely on.
    They are used only to avoid flagging a value or a name that the issue,
    file digest, dataset tree or conversation actually supplied. ``evidence``
    is the retrieved knowledge text; a value it documents is allowed when the
    contract rests on a direct record.
    """
    body = text or ""
    violations: list[GateViolation] = []

    for label in execution_claims(body):
        violations.append(GateViolation("execution_claim", label))

    if not body.strip():
        violations.append(GateViolation("empty_response", "the answer is empty"))
        return GateResult(tuple(violations))

    if contract is None:
        return GateResult(tuple(violations))

    supplied = " \n ".join(
        str(item) for item in inputs if item
    ).lower()

    # G2 — an undocumented rule must be declared, not papered over.
    if contract.requires_no_direct_statement:
        if (NO_DIRECT_SENTENCE.lower() not in body.lower()
                and not _NO_DIRECT_RE.search(body)):
            violations.append(GateViolation(
                "missing_no_direct_declaration",
                "the knowledge base documents no record for this rule, but "
                "the answer does not say so",
            ))

    # G1 — a concrete value for the issue's field, when the evidence does not
    # fix one. This is the check that stops invented HED versions, license
    # names and acknowledgements.
    if contract.field and not contract.concrete_value_allowed:
        for value in _field_value_claims(body, contract.field):
            if _is_placeholder_value(value):
                continue
            if value.strip().lower() in supplied:
                continue
            violations.append(GateViolation(
                "concrete_value_not_allowed",
                f"proposes {contract.field}={value!r}; the evidence does not "
                "determine a concrete value",
            ))

    # G1b — a bare version value in prose. The JSON example check above misses
    # "common versions might be 1.0, 1.1", which is the same invented claim.
    if contract.field and not contract.concrete_value_allowed:
        for value in _version_claims(body, contract, supplied, evidence):
            violations.append(GateViolation(
                "unsupported_version_value",
                f"offers {value!r} as a {contract.field} value; the evidence "
                "does not determine one",
            ))
            break

    # G4 — no worked example at all when the field has no documented shape.
    if not contract.example_allowed and _FENCE_RE.search(body):
        violations.append(GateViolation(
            "example_not_allowed",
            "the answer contains a code example, but the evidence does not "
            "document the field's structure",
        ))

    # G3 — required / recommended / optional must match the metadata rule.
    if contract.field and contract.bids_level in ("recommended", "optional"):
        for paragraph in _paragraphs(body):
            if contract.field.lower() not in paragraph.lower():
                continue
            if _REQUIRED_CLAIM_RE.search(paragraph):
                violations.append(GateViolation(
                    "modality_overclaim",
                    f"calls {contract.field} required, but BIDS lists it as "
                    f"{contract.bids_level}",
                ))
                break

    if (contract.bids_level in ("recommended", "optional")
            or contract.authority != "spec"):
        if _BIDS_OVERCLAIM_RE.search(body):
            violations.append(GateViolation(
                "bids_requirement_overclaim",
                "presents the finding as a BIDS requirement",
            ))

    # G3b — a manager/validator warning is not an error.
    if contract.severity in ("warning", "info"):
        if _WRONG_SEVERITY_RE.search(_first_region(body)):
            violations.append(GateViolation(
                "severity_mismatch",
                "calls this warning an error",
            ))

    # G5 — named people/institutions the evidence never supplied.
    for match in _NAMED_ENTITY_RE.finditer(body):
        name = match.group(0).strip()
        if name.lower() not in supplied:
            violations.append(GateViolation(
                "fabricated_named_entity",
                f"names {name!r}, which no supplied evidence mentions",
            ))
            break

    # G6 — duplicated or empty output sections.
    headings = [(m.group(1), m.start(), m.end())
                for m in _SECTION_RE.finditer(body)]
    seen = set()
    for name, _start, _end in headings:
        if name in seen:
            violations.append(GateViolation(
                "duplicate_section", f"section **{name}** appears twice",
            ))
            break
        seen.add(name)
    for index, (_name, _start, end) in enumerate(headings):
        next_start = headings[index + 1][1] if index + 1 < len(headings) else len(body)
        if not body[end:next_start].strip():
            violations.append(GateViolation(
                "empty_section",
                f"section **{headings[index][0]}** has no content",
            ))
            break

    return GateResult(tuple(violations))
