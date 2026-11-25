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
from retriever import _issue_structured, authority_class

#: Exact sentence the answer must contain when no record documents the rule.
NO_DIRECT_SENTENCE = (
    "The knowledge base does not document this rule."
)

#: Authority classes, in the vocabulary the prompt uses.
AUTHORITY_SPEC = "spec"
AUTHORITY_DOCUMENTATION = "documentation"
AUTHORITY_VALIDATOR = "validator"
AUTHORITY_MANAGER = "manager"
AUTHORITY_DATASET_EXAMPLE = "dataset_example"
AUTHORITY_SCHEMA_ARCHIVE = "schema_archive"
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
    """The authority of the record that documents *this rule*.

    Only a direct record may establish authority. A background field
    definition must not upgrade a validator or BIDS-Manager finding into a
    specification requirement, and a retrieval miss leaves the authority
    unknown rather than borrowing the closest record's.
    """
    if (rule_id or "").lower().startswith("bidsmgr."):
        return AUTHORITY_MANAGER
    for record in direct_records:
        scope = getattr(record, "scope", None)
        if isinstance(scope, dict) and scope.get("implementation_only") is True:
            return AUTHORITY_VALIDATOR
        value = getattr(record, "authority", "")
        if not value and isinstance(scope, dict):
            value = scope.get("authority")
        resolved = authority_class(value)
        if resolved != AUTHORITY_UNKNOWN:
            return resolved
    return AUTHORITY_UNKNOWN


def _record_rule_codes(record: Any) -> Tuple[str, ...]:
    codes = getattr(record, "rule_codes", ()) or ()
    return tuple(str(code) for code in codes if str(code).strip())


def _record_severity(record: Any) -> str:
    facts = getattr(record, "facts", None)
    if isinstance(facts, dict) and facts.get("severity"):
        return normalise_severity(facts.get("severity"))
    return normalise_severity(getattr(record, "severity", ""))


def _version_key(value: Any) -> Tuple[int, ...]:
    """Comparable major/minor/patch tuple; ``-dev`` sorts with its release."""
    text = str(value or "").strip().lower()
    text = text.split("-", 1)[0].replace("v", "", 1)
    parts: List[int] = []
    for piece in text.split("."):
        match = re.match(r"^(\d+)", piece)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


@dataclass(frozen=True)
class EvidenceConflict:
    """One deterministic provenance/evidence inconsistency.

    ``kind`` is ``level`` (same field, different BIDS level), ``severity``
    (same rule, different severity), ``version`` (same rule/field documented
    for incompatible releases), or ``authority`` (same rule documented by
    sources with different standing).
    """

    kind: str
    detail: str
    records: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "records": list(self.records),
        }


def _detect_conflicts(records: Sequence[Any], field_name: str) -> Tuple[EvidenceConflict, ...]:
    """Conservative conflicts that the selected evidence actually supports."""
    conflicts: List[EvidenceConflict] = []
    usable = [r for r in records
              if getattr(r, "knowledge_type", "") != "Example"]

    # Same rule code, different severity or incompatible BIDS versions.
    by_code: Dict[str, List[Any]] = {}
    for record in usable:
        for code in _record_rule_codes(record):
            by_code.setdefault(code.lower(), []).append(record)
    for code, group in by_code.items():
        if len(group) < 2:
            continue
        severities = {_record_severity(r) for r in group} - {""}
        if len(severities) > 1:
            conflicts.append(EvidenceConflict(
                "severity",
                f"rule '{code}' is recorded with different severities: "
                + ", ".join(sorted(severities)),
                tuple(sorted({str(getattr(r, "id", "")) for r in group})),
            ))
        versions = {}
        for record in group:
            key = _version_key(getattr(record, "bids_version", ""))
            if key:
                versions[key] = str(getattr(record, "bids_version", ""))
        if len(versions) > 1:
            conflicts.append(EvidenceConflict(
                "version",
                f"rule '{code}' is documented for different BIDS releases: "
                + ", ".join(sorted(versions.values())),
                tuple(sorted({str(getattr(r, "id", "")) for r in group})),
            ))
        authorities = {
            authority_class(getattr(r, "authority", ""))
            for r in group
        } - {AUTHORITY_UNKNOWN}
        if len(authorities) > 1:
            conflicts.append(EvidenceConflict(
                "authority",
                f"rule '{code}' is documented by sources with different "
                "standing: " + ", ".join(sorted(authorities)),
                tuple(sorted({str(getattr(r, "id", "")) for r in group})),
            ))

    # Same affected field, different documented BIDS level.
    if field_name:
        defining = [r for r in usable if (
            hasattr(r, "defines_field") and r.defines_field(field_name)
        )]
        levels: Dict[str, List[Any]] = {}
        for record in defining:
            level = record.field_level(field_name)
            if level:
                levels.setdefault(level, []).append(record)
        if len(levels) > 1:
            detail = "; ".join(
                f"{level}: " + ", ".join(
                    str(getattr(r, "id", "")) for r in group
                )
                for level, group in sorted(levels.items())
            )
            conflicts.append(EvidenceConflict(
                "level",
                f"field '{field_name}' has conflicting BIDS levels — {detail}",
                tuple(sorted({str(getattr(r, "id", "")) for r in defining})),
            ))
    return tuple(conflicts)


def _dataset_bids_version(issue: Any, file_content: Optional[str]) -> str:
    """The dataset's declared BIDS version, when the supplied data carries it."""
    if isinstance(issue, dict):
        for key in ("bids_version", "BIDSVersion", "dataset_bids_version"):
            value = issue.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if file_content:
        match = re.search(
            r"\bBIDSVersion\b\s*[:=]\s*[\"']?([0-9][0-9A-Za-z.\-+]*)",
            file_content,
        )
        if match:
            return match.group(1).strip().rstrip("\"',")
    return ""


def _version_notes(
    records: Sequence[Any],
    dataset_version: str,
) -> Tuple[str, ...]:
    """Version differences worth stating before the answer is composed."""
    if not dataset_version:
        return ()
    dataset_key = _version_key(dataset_version)
    if not dataset_key:
        return ()
    notes: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for record in records:
        record_version = str(getattr(record, "bids_version", "") or "")
        record_key = _version_key(record_version)
        if not record_key or record_key[:2] == dataset_key[:2]:
            continue
        marker = (str(getattr(record, "id", "")), record_version)
        if marker in seen:
            continue
        seen.add(marker)
        notes.append(
            f"record {marker[0]} documents its rule in BIDS {record_version}; "
            f"the dataset declares BIDS {dataset_version}"
        )
    return tuple(notes)


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
    # Provenance/grounding surface (record ids from the assessments).
    direct_evidence_ids: Tuple[str, ...] = ()
    supporting_evidence_ids: Tuple[str, ...] = ()
    background_evidence_ids: Tuple[str, ...] = ()
    source_authorities: Tuple[str, ...] = ()
    conflicts: Tuple[EvidenceConflict, ...] = ()
    dataset_bids_version: str = ""
    version_notes: Tuple[str, ...] = ()

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
            "direct_evidence_ids": list(self.direct_evidence_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "background_evidence_ids": list(self.background_evidence_ids),
            "source_authorities": list(self.source_authorities),
            "conflicts": [c.as_dict() for c in self.conflicts],
            "dataset_bids_version": self.dataset_bids_version,
            "version_notes": list(self.version_notes),
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
    direct_ids: List[str] = []
    supporting_ids: List[str] = []
    background_ids: List[str] = []
    authorities: List[str] = []
    direct_evidence = False
    if retrieval is not None:
        direct_evidence = bool(getattr(retrieval, "has_direct", False))
        for assessment in getattr(retrieval, "assessments", ()) or ():
            record = getattr(assessment, "record", None)
            if record is None:
                continue
            records.append(record)
            evidence_class = (
                getattr(assessment, "evidence_class", "")
                or getattr(assessment, "relevance", "")
            )
            record_id = str(getattr(record, "id", "") or "")
            if evidence_class == "direct":
                direct_records.append(record)
                if record_id:
                    direct_ids.append(record_id)
            elif evidence_class == "supporting" and record_id:
                supporting_ids.append(record_id)
            elif record_id:
                background_ids.append(record_id)
            scope = getattr(record, "scope", None)
            resolved = authority_class(
                getattr(record, "authority", "")
                or (scope.get("authority") if isinstance(scope, dict) else "")
            )
            if resolved != AUTHORITY_UNKNOWN and resolved not in authorities:
                authorities.append(resolved)

    bids_level = derive_bids_level(field_name, records)
    authority = _authority_of(rule_id, records, direct_records)
    placeholder = _is_placeholder_rule(rule_id, field_name, file_content)
    example_allowed = _has_structural_definition(field_name, records)
    concrete_value_allowed = _has_enumerated_values(field_name, records)
    dataset_version = _dataset_bids_version(issue, file_content)
    version_notes = _version_notes(direct_records, dataset_version)
    conflicts = _detect_conflicts(records, field_name)

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
        direct_evidence_ids=tuple(direct_ids),
        supporting_evidence_ids=tuple(supporting_ids),
        background_evidence_ids=tuple(background_ids),
        source_authorities=tuple(authorities),
        conflicts=conflicts,
        dataset_bids_version=dataset_version,
        version_notes=version_notes,
    )
    return replace(contract, must_state_unknown=_unknowns_for(contract))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_AUTHORITY_TEXT = {
    AUTHORITY_MANAGER: "dataset-manager check, not a BIDS requirement",
    AUTHORITY_VALIDATOR: "validator check, not itself a BIDS requirement",
    AUTHORITY_SPEC: "BIDS specification rule",
    AUTHORITY_DOCUMENTATION: "BIDS documentation, not a normative rule",
    AUTHORITY_DATASET_EXAMPLE: "illustrative dataset example",
    AUTHORITY_SCHEMA_ARCHIVE: "archived BIDS schema snapshot",
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
        authority = _AUTHORITY_TEXT[contract.authority]
        if contract.direct_evidence_ids:
            authority += (
                " (direct record: "
                + ", ".join(contract.direct_evidence_ids)
                + ")"
            )
        lines.append(f"- [ ] authority: {authority}")
    if contract.direct_evidence:
        lines.append(
            "- [ ] direct record — use it for the rule being reported"
        )
    elif contract.requires_no_direct_statement:
        lines.append(
            "- [ ] no direct record — include: "
            f'"{contract.mandatory_sentence}"'
        )
    if contract.supporting_evidence_ids:
        lines.append(
            "- [ ] supporting record(s) "
            + ", ".join(contract.supporting_evidence_ids)
            + " — may explain the field, format constraint, or application "
            "behaviour, but NOT the reported rule"
        )
    if contract.background_evidence_ids:
        lines.append(
            "- [ ] background record(s) "
            + ", ".join(contract.background_evidence_ids)
            + " — context only; never use as the issue's definition, cause, "
            "or rule"
        )
    for conflict in contract.conflicts:
        lines.append(
            f"- [ ] conflicting evidence ({conflict.kind}): "
            f"{conflict.detail} — attribute each claim to its record; do not "
            "merge or silently pick one"
        )
    for note in contract.version_notes:
        lines.append(
            f"- [ ] BIDS version: {note} — state the difference if it could "
            "change the answer"
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
