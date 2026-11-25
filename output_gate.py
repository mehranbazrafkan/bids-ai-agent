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

import json
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
_FENCE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_JSON_KEY_RE = re.compile(r'"([^"\n]+)"\s*:')
_JSON_QUOTED_VALUE_RE = re.compile(r'"([^"\n]+)"\s*:\s*"([^"]*)"')
_JSON_ARRAY_RE = re.compile(r'"([^"\n]+)"\s*:\s*\[([^\]]*)\]', re.DOTALL)
_PARAGRAPH_VALUE_RE = re.compile(r'"([^"\n]+)"\s*:\s*([^,\n}\]]+)')
_MARKDOWN_SEPARATOR_RE = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$", re.MULTILINE
)

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

# A validator/rule code: uppercase segments joined by underscores. The
# mixed-case names of BIDS fields (HEDVersion, SourceDatasets) are deliberately
# not matched, and the issue's own rule is always allowed. Segments after the
# first need at least two characters so identifiers such as ``CC_BY_4_0`` are
# not mistaken for a rule code.
_RULE_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]{2,})+\b")

# Code-like identifiers that are not validator/manager rules.
_NON_RULE_CODE_PREFIXES = (
    "CC_", "SPDX_", "DOI_", "ISBN_", "ISSN_", "RRID_", "ORCID_",
)

# Explicit attribution of a claim to a source. Only ``source + reporting
# verb`` forms are matched, so a neutral mention ("the validator may report")
# is not a finding.
_AUTHORITY_ATTRIBUTIONS = (
    ("validator", re.compile(
        r"\b(?:BIDS\s+)?validator\s+(?:reports|flags|requires|says|states|"
        r"considers|treats)\b"
        r"|\baccording\s+to\s+the\s+(?:BIDS\s+)?validator\b",
        re.IGNORECASE,
    )),
    ("manager", re.compile(
        r"\bBIDS-Manager\s+(?:reports|flags|requires|says|states|treats|"
        r"writes|checks)\b"
        r"|\baccording\s+to\s+BIDS-Manager\b",
        re.IGNORECASE,
    )),
    ("spec", re.compile(
        r"\bBIDS\s+(?:specification|spec|standard)\s+(?:requires|states|"
        r"mandates|says|defines|specifies|lists)\b"
        r"|\baccording\s+to\s+the\s+BIDS\s+(?:specification|spec|standard)\b",
        re.IGNORECASE,
    )),
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
    "", "...", "…", "todo", "tbd", "null", "none", "text", "value", "string",
    "example", "n/a", "your value", "placeholder", "unknown",
})


def _is_placeholder_value(value: str) -> bool:
    text = (value or "").strip().strip("\"'")
    lowered = text.lower()
    if lowered in _PLACEHOLDER_VALUES:
        return True
    if lowered.strip("[](){}").strip() in _PLACEHOLDER_VALUES:
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


def _is_markdown_table_fence(language: str, content: str) -> bool:
    """A table is an explanation, not a claim to be copy-pasted as a file."""
    if language in ("markdown", "md"):
        return True
    if "\t" in content:
        return False
    return bool(_MARKDOWN_SEPARATOR_RE.search(content))


def _code_example_fences(text: str) -> list[tuple[str, str]]:
    """Fenced examples, excluding Markdown tables."""
    result: list[tuple[str, str]] = []
    for match in _FENCE_BLOCK_RE.finditer(text or ""):
        language = match.group(1).strip().lower()
        content = match.group(2)
        if _is_markdown_table_fence(language, content):
            continue
        result.append((language, content))
    return result


def _walk_json(node: Any, keys: set[str], values: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(str(key))
            _walk_json(value, keys, values)
    elif isinstance(node, list):
        for item in node:
            _walk_json(item, keys, values)
    elif isinstance(node, (str, int, float, bool)):
        values.add(str(node))


def _json_example_parts(content: str) -> tuple[set[str], set[str]] | None:
    """Keys and scalar values in a JSON document or a bare key-value snippet."""
    stripped = (content or "").strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return None
        keys: set[str] = set()
        values: set[str] = set()
        _walk_json(parsed, keys, values)
        return keys, values

    keys = set(_JSON_KEY_RE.findall(content or ""))
    values: set[str] = set()
    for match in _JSON_QUOTED_VALUE_RE.finditer(content or ""):
        values.add(match.group(2))
    for match in _JSON_ARRAY_RE.finditer(content or ""):
        if match.group(1) in keys:
            values.update(re.findall(r'"([^"]*)"', match.group(2)))
    for match in _PARAGRAPH_VALUE_RE.finditer(content or ""):
        if match.group(1) not in keys:
            continue
        value = match.group(2).strip().strip("\"'")
        if value and not value.startswith(("[", "{")):
            values.add(value)
    return keys, values


def _looks_like_tsv(content: str) -> bool:
    return "\t" in (content or "") and not (content or "").lstrip().startswith(
        ("{", "[")
    )


def _field_array_examples(text: str, field_name: str) -> list[tuple[str, ...]]:
    """Each complete array example for an array-valued field.

    Multiple entries in one ``Authors`` array are one example, not conflicting
    values. Comparing whole arrays still catches two different arrays shown in
    different sections.
    """
    if not field_name:
        return []
    pattern = re.compile(
        rf'"{re.escape(field_name)}"\s*:\s*\[([^\]]*)\]', re.DOTALL
    )
    examples: list[tuple[str, ...]] = []
    for match in pattern.finditer(text or ""):
        group = match.group(1)
        # An array of objects (for example SourceDatasets) contains nested
        # property names, not values of the affected field.
        if "{" in group or "[" in group:
            continue
        values = tuple(sorted(
            value.strip().lower()
            for value in re.findall(r'"([^"]*)"', group)
            if value.strip() and not _is_placeholder_value(value)
        ))
        if values:
            examples.append(values)
    return examples


def _table_property_violations(
    text: str,
    contract: AnswerContract,
    policy: Any,
) -> list[GateViolation]:
    """Property names listed in a Markdown key table.

    A table is the preferred JSON explanation, but it is still an assertion
    about the file's property names. A table that says ``key`` when the
    evidence documents ``URL``/``DOI``/``Version`` is the table form of the
    same invented structure.
    """
    if policy.file_format != "json":
        return []
    headings = list(_SECTION_RE.finditer(text or ""))
    table_parts: list[str] = []
    for index, heading in enumerate(headings):
        if heading.group(1) not in ("Example", "How to Fix"):
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        table_parts.append(text[heading.end():end])
    table_text = "\n".join(table_parts) or (text or "")

    key_headers = {
        "key", "json key", "property", "field", "name", "metadata key",
    }
    lines = table_text.splitlines()
    bad: set[str] = set()
    index = 0
    while index < len(lines):
        if index + 1 >= len(lines):
            break
        header_line = lines[index]
        separator = lines[index + 1]
        if "|" not in header_line or not _MARKDOWN_SEPARATOR_RE.match(separator):
            index += 1
            continue
        header = [
            cell.strip().strip("`* ").lower()
            for cell in header_line.strip().strip("|").split("|")
        ]
        key_columns = [
            position for position, name in enumerate(header)
            if name in key_headers
        ]
        index += 2
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            cells = [
                cell.strip()
                for cell in lines[index].strip().strip("|").split("|")
            ]
            for position in key_columns:
                if position >= len(cells):
                    continue
                name = cells[position].strip().strip("`* ")
                if (
                    name
                    and re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", name)
                    and name not in policy.keys
                ):
                    bad.add(name)
            index += 1
        index += 1

    if not bad:
        return []
    return [GateViolation(
        "invented_example_property",
        "the key table invents JSON property name(s) not documented for this "
        "field: " + ", ".join(sorted(bad)[:5]),
    )]


_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
_NAME_STOPWORDS = frozenset({
    "authors", "bids", "dataset", "example", "fields", "fix", "hed",
    "json", "license", "manager", "metadata", "problem", "source",
    "tsv", "validator", "version", "why",
})


def _prose_example_violations(
    text: str,
    contract: AnswerContract,
    supplied: str,
) -> list[GateViolation]:
    """Names in an Authors example that no supplied context contains."""
    if (contract.field or "").lower() != "authors":
        return []
    headings = list(_SECTION_RE.finditer(text or ""))
    example_parts: list[str] = []
    for index, heading in enumerate(headings):
        if heading.group(1) not in ("Example", "How to Fix"):
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        example_parts.append(text[heading.end():end])
    if not example_parts:
        return []
    example_text = "\n".join(example_parts)
    for match in _PERSON_NAME_RE.finditer(example_text):
        name = match.group(0)
        words = {word.lower() for word in name.split()}
        if words & _NAME_STOPWORDS:
            continue
        if re.search(
            rf"(?<![\w.]){re.escape(name.lower())}(?![\w.])", supplied
        ):
            continue
        return [GateViolation(
            "fabricated_named_entity",
            f"names {name!r}, which no supplied evidence mentions",
        )]
    return []


def _example_value_allowed(
    value: str,
    contract: AnswerContract,
    supplied: str,
    evidence: str,
) -> bool:
    text = (value or "").strip()
    if _is_placeholder_value(text):
        return True
    allowed_values = {
        item.lower() for item in contract.example_policy.allowed_values
    }
    if text.lower() in allowed_values:
        return True
    lowered = text.lower()
    if supplied and re.search(
        rf"(?<![\w.]){re.escape(lowered)}(?![\w.])", supplied
    ):
        return True
    if contract.concrete_value_allowed and evidence:
        if re.search(
            rf"(?<![\w.]){re.escape(text)}(?![\w.])",
            evidence,
            re.IGNORECASE,
        ):
            return True
    return False


def _example_violations(
    text: str,
    contract: AnswerContract,
    supplied: str,
    evidence: str,
) -> list[GateViolation]:
    """Validate code-block examples against the request's example policy."""
    policy = getattr(contract, "example_policy", None)
    if not contract.example_allowed or policy is None:
        if policy is not None and policy.file_format == "json":
            return _table_property_violations(text, contract, policy)
        return []

    violations: list[GateViolation] = []
    blocks = _code_example_fences(text)
    seen: set[str] = set()

    # A shell/editing command is a fix instruction, not a file example.
    non_example_languages = {
        "bash", "sh", "shell", "powershell", "ps1", "console", "cmd",
        "python", "py",
    }

    for language, content in blocks:
        if language in non_example_languages:
            continue
        normalized = re.sub(r"\s+", " ", content).strip()
        if normalized in seen:
            violations.append(GateViolation(
                "duplicate_example",
                "the same example appears more than once in the answer",
            ))
            continue
        seen.add(normalized)

        if policy.file_format == "json":
            if language == "tsv" or _looks_like_tsv(content):
                violations.append(GateViolation(
                    "format_mismatch",
                    "the affected file is JSON, but the example is "
                    "tab-separated content",
                ))
                continue
            if language and language not in (
                "json", "jsonc", "json5", "javascript", "js",
                "text", "plaintext",
            ):
                violations.append(GateViolation(
                    "format_mismatch",
                    f"the affected file is JSON, but the example is fenced as "
                    f"{language!r}",
                ))
                continue
            parts = _json_example_parts(content)
            if parts is None:
                violations.append(GateViolation(
                    "invalid_json_example",
                    "the example is presented as JSON but is not valid JSON",
                ))
                continue
            keys, values = parts
            if not keys:
                violations.append(GateViolation(
                    "invalid_json_example",
                    "the JSON example contains no property names",
                ))
                continue
            extra_keys = [key for key in keys if key not in policy.keys]
            if extra_keys:
                violations.append(GateViolation(
                    "invented_example_property",
                    "the example invents JSON property name(s) not documented "
                    "for this field: " + ", ".join(sorted(extra_keys)[:5]),
                ))
            bad_values = [
                value for value in values
                if not _example_value_allowed(
                    value, contract, supplied, evidence,
                )
            ]
            if bad_values:
                violations.append(GateViolation(
                    "invented_example_value",
                    "the example contains value(s) not present in the supplied "
                    "dataset or documented as allowed: "
                    + ", ".join(sorted(bad_values)[:5]),
                ))

        elif policy.file_format == "tsv":
            if language == "json" or content.lstrip().startswith(("{", "[")):
                violations.append(GateViolation(
                    "format_mismatch",
                    "the affected file is TSV, but the example is JSON content",
                ))
                continue
            if language and language not in ("tsv", "text", "plaintext"):
                violations.append(GateViolation(
                    "format_mismatch",
                    f"the affected file is TSV, but the example is fenced as "
                    f"{language!r}",
                ))
                continue
            lines = [line for line in content.splitlines() if line.strip()]
            if not lines:
                continue
            header = [cell.strip() for cell in lines[0].split("\t")]
            if len(header) < 2:
                violations.append(GateViolation(
                    "format_mismatch",
                    "the TSV example has no tab-separated header",
                ))
                continue
            extra_columns = [
                column for column in header if column not in policy.columns
            ]
            if extra_columns:
                violations.append(GateViolation(
                    "invented_tsv_column",
                    "the example invents TSV column(s) not documented or "
                    "supplied: " + ", ".join(sorted(extra_columns)[:5]),
                ))
                continue
            bad_values: list[str] = []
            for row in lines[1:]:
                cells = row.split("\t")
                if len(cells) != len(header):
                    violations.append(GateViolation(
                        "format_mismatch",
                        "a TSV row does not match the header's column count",
                    ))
                    break
                bad_values.extend(
                    cell for cell in cells
                    if not _example_value_allowed(
                        cell, contract, supplied, evidence,
                    )
                )
            if bad_values:
                violations.append(GateViolation(
                    "invented_example_value",
                    "the TSV example contains value(s) not present in the "
                    "supplied dataset: "
                    + ", ".join(sorted(set(bad_values))[:5]),
                ))
        else:
            violations.append(GateViolation(
                "format_mismatch",
                "the affected file format is not established, so a "
                "file-content example cannot be presented",
            ))

    if policy.file_format == "json":
        violations.extend(_table_property_violations(text, contract, policy))

    if contract.field:
        scalar_values = {
            value.strip().lower()
            for value in _field_value_claims(text, contract.field)
            if value.strip() and not _is_placeholder_value(value)
        }
        array_values = _field_array_examples(text, contract.field)
        if len(scalar_values) > 1 or len(set(array_values)) > 1:
            violations.append(GateViolation(
                "contradictory_example",
                f"the answer gives more than one different example for "
                f"{contract.field}",
            ))

    violations.extend(_prose_example_violations(text, contract, supplied))
    return violations


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

    raw_supplied = " \n ".join(str(item) for item in inputs if item)
    supplied = raw_supplied.lower()

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
    # A fenced Markdown table is explanatory, not a file-content example.
    if not contract.example_allowed and _code_example_fences(body):
        violations.append(GateViolation(
            "example_not_allowed",
            "the answer contains a code example, but the evidence does not "
            "document the field's structure",
        ))

    # G4b — format-aware example validation: exact JSON properties/TSV
    # columns, no invented values, no TSV masquerading as JSON.
    violations.extend(_example_violations(body, contract, supplied, evidence))

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

    # G5b — a rule code the answer did not get from the issue or evidence.
    allowed_codes = set()
    if contract.rule_id:
        allowed_codes.add(contract.rule_id.upper())
    for source_text in (raw_supplied, evidence or ""):
        allowed_codes.update(
            code.upper() for code in _RULE_CODE_RE.findall(source_text)
        )
    for code in _RULE_CODE_RE.findall(body):
        if code.upper() in allowed_codes:
            continue
        if code.upper().startswith(_NON_RULE_CODE_PREFIXES):
            continue
        violations.append(GateViolation(
            "unsupported_rule_citation",
            f"cites code {code!r}, which neither the issue nor the "
            "retrieved evidence supplies",
        ))
        break

    # G5c — attribution to a source the evidence does not contain. This is the
    # provenance version of a hallucination: "the BIDS validator says" when
    # only a BIDS-Manager record was retrieved. Neutral mentions of the
    # validator are allowed; only explicit reporting verbs are checked.
    source_authorities = set(
        getattr(contract, "source_authorities", ()) or ()
    )
    if source_authorities:
        for authority, pattern in _AUTHORITY_ATTRIBUTIONS:
            if authority in source_authorities:
                continue
            if authority == contract.authority:
                continue
            if pattern.search(body):
                violations.append(GateViolation(
                    "authority_misattribution",
                    f"attributes the claim to {authority}, which no retrieved "
                    "record with that authority supports",
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
