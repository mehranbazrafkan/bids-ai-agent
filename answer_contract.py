"""The per-request answer contract.

The model used to be asked to *derive* what the evidence implies: whether the
field is required, recommended or optional; whether the finding is a BIDS
requirement or only a validator/manager warning; whether the knowledge base
documents the rule; whether a concrete value may be proposed. Phase-1 showed
that a small model does not do this reliably, so those judgements are computed
here — deterministically, from the issue and the retrieved records — and handed
to the model as explicit slots.

Nothing in this module calls a model. Every field of :class:`AnswerContract`
is derived from data the planner already holds, so the same issue always
produces the same contract, and :mod:`output_gate` can validate the generated
answer against it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Reuse the retriever's issue flattening so nested findings behave identically
# in retrieval and in the contract (the innermost issue wins).
from retriever import _issue_structured

#: Exact sentence the answer must contain when no record documents the rule.
NO_DIRECT_SENTENCE = (
    "The knowledge base does not document this rule."
)

#: Authority classes, in the vocabulary the prompt uses.
AUTHORITY_SPEC = "spec"
AUTHORITY_VALIDATOR = "validator"
AUTHORITY_MANAGER = "manager"
AUTHORITY_UNKNOWN = "unknown"

_LEVELS = ("required", "recommended", "optional")

# Field-specific things the owner must supply. Kept short on purpose: the
# contract is a directive block, not another essay.
_UNKNOWNS: Dict[str, Tuple[str, ...]] = {
    "HEDVersion": (
        "the HED schema version in use",
        "a value accepted by the hed_version format",
    ),
    "License": (
        "which license the dataset owner intends",
    ),
    "Acknowledgements": (
        "what acknowledgement text, if any, the dataset owner wants to include",
    ),
    "Authors": (
        "which authors and affiliations the dataset owner intends",
    ),
    "Funding": (
        "which funding sources the dataset owner intends",
    ),
    "EthicsApprovals": (
        "which ethics approvals the dataset owner intends",
    ),
    "DatasetDOI": (
        "the dataset DOI, if one has been assigned",
    ),
    "Description": (
        "the description the dataset owner intends",
    ),
}


def normalise_severity(value: Any) -> str:
    """``err``/``error`` -> ``error``; ``warn``/``warning`` -> ``warning``."""
    text = str(value or "").strip().lower()
    if text in ("err", "error", "fatal"):
        return "error"
    if text in ("warn", "warning"):
        return "warning"
    if text in ("info", "information"):
        return "info"
    return text


def _is_placeholder_rule(rule_id: str, field_name: str,
                         file_content: Optional[str]) -> bool:
    rid = (rule_id or "").lower()
    if "todo" in rid or "placeholder" in rid:
        return True
    if field_name and file_content:
        # The compiled digest renders ``License: TODO`` / ``License = "TODO"``.
        pattern = re.compile(
            rf"\b{re.escape(field_name)}\b\s*[:=]\s*[\"']?\s*TODO\b",
            re.IGNORECASE,
        )
        if pattern.search(file_content):
            return True
    return False


def _rule_fields(record: Any) -> Dict[str, Any]:
    """``rule_definition.fields`` of a MetadataRule record, when present."""
    raw = getattr(record, "raw_content", None)
    if not isinstance(raw, dict):
        return {}
    definition = raw.get("rule_definition")
    if not isinstance(definition, dict):
        definition = raw.get("definition")
    if not isinstance(definition, dict):
        return {}
    fields = definition.get("fields")
    return fields if isinstance(fields, dict) else {}


def _field_definition(record: Any, field_name: str) -> Dict[str, Any]:
    """The metadata-field definition of ``field_name``, when present."""
    raw = getattr(record, "raw_content", None)
    if not isinstance(raw, dict):
        return {}
    definition = raw.get("definition")
    if not isinstance(definition, dict):
        return {}
    if str(definition.get("name") or "") != field_name:
        return {}
    return definition


def _level_of(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("level")
    text = str(value or "").strip().lower()
    return text if text in _LEVELS else None


def derive_bids_level(field_name: str, records: Iterable[Any]) -> Optional[str]:
    """The field's BIDS level from the metadata rule that governs it.

    Only a field explicitly listed by a retrieved metadata rule counts. When
    no such rule is available the level is ``None`` — the gate then refuses to
    police "required", because inventing a level would be worse than saying
    nothing.
    """
    if not field_name:
        return None
    for record in records:
        fields = _rule_fields(record)
        if field_name in fields:
            level = _level_of(fields[field_name])
            if level:
                return level
    return None


def _authority_of(
    rule_id: str,
    records: Sequence[Any],
    direct_records: Sequence[Any] = (),
) -> str:
    if (rule_id or "").lower().startswith("bidsmgr."):
        return AUTHORITY_MANAGER
    # The record that documents *this* rule decides the authority; a
    # background field definition must not upgrade a validator warning into a
    # specification requirement.
    for group in (direct_records, records):
        for record in group:
            scope = getattr(record, "scope", None)
            if not isinstance(scope, dict):
                continue
            authority = str(scope.get("authority") or "").lower()
            domain = str(scope.get("knowledge_domain") or "").lower()
            if ("validator" in authority or "validator" in domain
                    or scope.get("implementation_only") is True):
                return AUTHORITY_VALIDATOR
            if "specification" in authority or "normative" in authority:
                return AUTHORITY_SPEC
    return AUTHORITY_UNKNOWN


def _has_structural_definition(field_name: str, records: Sequence[Any]) -> bool:
    """Whether a record documents the field's *shape* (not just its type).

    ``{"HEDVersion": "<version>"}`` is a shape example and is allowed;
    ``{"HEDVersion": "1.0"}`` is a concrete claim and is not. A bare
    ``type: string`` is not enough — that is exactly where invented
    acknowledgements and license names come from.
    """
    if not field_name:
        return False
    structural = ("anyOf", "oneOf", "enum", "items", "format")
    for record in records:
        definition = _field_definition(record, field_name)
        if any(key in definition for key in structural):
            return True
        allowed = getattr(record, "allowed_values", None)
        if allowed:
            return True
    return False


def _has_enumerated_values(field_name: str, records: Sequence[Any]) -> bool:
    """Whether the records fix an exhaustive set of valid values."""
    if not field_name:
        return False
    for record in records:
        allowed = getattr(record, "allowed_values", None)
        if isinstance(allowed, (list, tuple, dict)) and allowed:
            return True
        definition = _field_definition(record, field_name)
        if "enum" in definition:
            return True
    return False


def _unknowns_for(contract: "AnswerContract") -> Tuple[str, ...]:
    items: List[str] = []
    if contract.field and contract.placeholder:
        items.extend(_UNKNOWNS.get(
            contract.field,
            (f"the value the dataset owner intends for '{contract.field}'",),
        ))
    elif contract.field:
        items.append(
            f"the intended value for '{contract.field}'"
        )
    if contract.bids_level is None and contract.field:
        items.append(
            f"whether BIDS treats '{contract.field}' as required, "
            "recommended, or optional"
        )
    # Preserve order, drop duplicates, keep the block short.
    seen = set()
    unique: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique[:2])


@dataclass(frozen=True)
class AnswerContract:
    """What the answer is allowed, required, and forbidden to say."""

    rule_id: str = ""
    field: str = ""
    path: str = ""
    message: str = ""
    severity: str = ""
    bids_level: Optional[str] = None
    authority: str = AUTHORITY_UNKNOWN
    direct_evidence: bool = False
    placeholder: bool = False
    example_allowed: bool = False
    concrete_value_allowed: bool = False
    requires_no_direct_statement: bool = False
    mandatory_sentence: str = NO_DIRECT_SENTENCE
    must_state_unknown: Tuple[str, ...] = ()

    # -- rendering ------------------------------------------------------

    @property
    def is_actionable(self) -> bool:
        """Whether there is enough information to police an answer."""
        return bool(self.rule_id or self.field)

    def as_dict(self) -> Dict[str, Any]:
        """Contract surface safe for diagnostics (no dataset content)."""
        return {
            "rule_id": self.rule_id,
            "field": self.field,
            "severity": self.severity,
            "bids_level": self.bids_level,
            "authority": self.authority,
            "direct_evidence": self.direct_evidence,
            "placeholder": self.placeholder,
            "example_allowed": self.example_allowed,
            "concrete_value_allowed": self.concrete_value_allowed,
            "requires_no_direct_statement": self.requires_no_direct_statement,
        }


def build_contract(
    issue: Any,
    retrieval: Any = None,
    file_content: Optional[str] = None,
) -> Optional[AnswerContract]:
    """Derive the answer contract for one issue.

    ``retrieval`` is a :class:`retriever.RetrievalResult` (or ``None``); the
    records behind its assessments are used to read the field's BIDS level and
    the field's structural definition. Returns ``None`` when the issue carries
    no usable identifier (nothing to police).
    """
    if not isinstance(issue, dict) or not issue:
        return None

    structured = _issue_structured(issue)
    rule_id = str(structured.get("rule_id") or "")
    field_name = str(structured.get("field") or "")
    path = str(structured.get("path") or "")
    message = str(structured.get("message_text") or "")
    severity = normalise_severity(structured.get("severity"))

    if not rule_id and not field_name:
        return None

    records: List[Any] = []
    direct_records: List[Any] = []
    direct_evidence = False
    if retrieval is not None:
        direct_evidence = bool(getattr(retrieval, "has_direct", False))
        for assessment in getattr(retrieval, "assessments", ()) or ():
            record = getattr(assessment, "record", None)
            if record is not None:
                records.append(record)
                if getattr(assessment, "relevance", "") == "direct":
                    direct_records.append(record)

    bids_level = derive_bids_level(field_name, records)
    authority = _authority_of(rule_id, records, direct_records)
    placeholder = _is_placeholder_rule(rule_id, field_name, file_content)
    example_allowed = _has_structural_definition(field_name, records)
    concrete_value_allowed = _has_enumerated_values(field_name, records)

    requires_no_direct = bool(
        retrieval is not None
        and getattr(retrieval, "found", False)
        and not direct_evidence
    )

    contract = AnswerContract(
        rule_id=rule_id,
        field=field_name,
        path=path,
        message=message,
        severity=severity,
        bids_level=bids_level,
        authority=authority,
        direct_evidence=direct_evidence,
        placeholder=placeholder,
        example_allowed=example_allowed,
        concrete_value_allowed=concrete_value_allowed,
        requires_no_direct_statement=requires_no_direct,
        must_state_unknown=(),
    )
    return replace(contract, must_state_unknown=_unknowns_for(contract))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_AUTHORITY_TEXT = {
    AUTHORITY_MANAGER: "dataset-manager check, not a BIDS requirement",
    AUTHORITY_VALIDATOR: "validator check, not itself a BIDS requirement",
    AUTHORITY_SPEC: "BIDS specification rule",
    AUTHORITY_UNKNOWN: "authority not established by the evidence",
}


def render_contract(contract: AnswerContract) -> str:
    """Render the contract as a compact pre-send checklist.

    The planner appends this block *last*, after every evidence section, so the
    model checks its draft against it immediately before replying. It is a
    directive, not prose: the model must not copy it into the answer.
    """
    lines: List[str] = ["[ANSWER CHECKLIST — verify, do not copy]"]
    target = ""
    if contract.rule_id and contract.field:
        target = f"rule `{contract.rule_id}`, field `{contract.field}`"
    elif contract.rule_id:
        target = f"rule `{contract.rule_id}`"
    elif contract.field:
        target = f"field `{contract.field}`"
    if contract.severity:
        word = "warning" if contract.severity == "warning" else contract.severity
        note = (
            f'say "{word}", not "error"' if word == "warning"
            else f"severity: {word}"
        )
        target = f"{target} — {note}" if target else note
    if target:
        lines.append(f"- [ ] {target}")
    if contract.bids_level:
        suffix = (
            " — do NOT call it required"
            if contract.bids_level != "required" else ""
        )
        lines.append(f"- [ ] BIDS level: {contract.bids_level}{suffix}")
    if contract.authority != AUTHORITY_UNKNOWN:
        lines.append(f"- [ ] authority: {_AUTHORITY_TEXT[contract.authority]}")
    if contract.direct_evidence:
        lines.append(
            "- [ ] direct record — use it for the rule being reported"
        )
    elif contract.requires_no_direct_statement:
        lines.append(
            "- [ ] no direct record — include: "
            f'"{contract.mandatory_sentence}"'
        )
    value_rules: List[str] = []
    if contract.placeholder:
        value_rules.append("placeholder — no concrete replacement value")
    value_rules.append(
        "example: shape only (<...>), never a value"
        if contract.example_allowed
        else "example: not allowed — omit it"
    )
    lines.append("- [ ] " + "; ".join(value_rules))
    if contract.must_state_unknown:
        lines.append(
            "- [ ] state as unknown: " + "; ".join(contract.must_state_unknown)
        )
    return "\n".join(lines)
