#!/usr/bin/env python3
"""
Derive per-field sidecar metadata records from grouped extraction records.

Why
---
The original extraction pipeline (``extract_kb.py``, not committed to this
repository) produced ONE record per *group* key of
``rules/sidecars/*.yaml`` — e.g. ``meta_mrisequencespecifics`` — with the
individual metadata fields (``SequenceName``, ``EchoTime``, ... ) buried
inside ``raw_content.fields``. As a result the knowledge base had no
direct, searchable record for any individual sidecar field: a query about
the recommended ``SequenceName`` field, for example, could not match any
record's title, id, or summary, and unrelated records ranked above the
grouped MRI sidecar metadata that actually documents the field.

This script closes that gap using ONLY material already present in the
knowledge base itself (the verbatim ``raw_content.fields`` fragments
extracted from the BIDS specification YAML). It derives one
``MetadataRule`` record per distinct field name:

* ``title``/``id`` name the field directly, so field-name queries match it
  the same way they match ``tabcol_*`` column records;
* ``summary``/``retrieval_text`` state the field name, its BIDS level
  (required/recommended/optional/deprecated), and the group it comes from;
* verbatim source text (``description_addendum``, ``level_addendum``,
  validator ``issue`` code/message, group selectors) is copied as-is —
  nothing is invented or reworded;
* ``source`` keeps the group's provenance and extends ``key`` to
  ``<group_key>.<field>``;
* ``raw_content`` is the field's original fragment, so a validator issue
  attached to the field (e.g. ``B0_FIELD_IDENTIFIER_RECOMMENDED``) remains
  available for exact rule-code matching.

Deduplication: a field that appears in several groups (``Manufacturer``,
``FlipAngle``, ...) yields ONE record — the first occurrence in file
order. The group records themselves are untouched, so alternative group
contexts remain documented there.

Scope: only ``enriched_knowledge.jsonl`` (the file loaded at runtime).
``knowledge.jsonl`` stays the pristine base extraction.

Idempotent: records whose id already exists are skipped, so re-running
appends nothing. Existing lines are preserved byte-for-byte.

Usage::

    python knowledge-base/derive_sidecar_field_records.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_FILE = "enriched_knowledge.jsonl"

# Fields that carry no retrieval signal of their own if ever present.
# (Not expected in sidecar YAML, kept defensively.)
SKIP_FIELDS = frozenset({"selectors"})


def _slug(field: str) -> str:
    return "".join(ch for ch in field.lower() if ch.isalnum())


def _level_of(spec) -> str:
    if isinstance(spec, str):
        return spec.strip()
    if isinstance(spec, dict):
        return str(spec.get("level", "") or "").strip()
    return ""


def _flatten_selectors(selectors) -> str:
    if not isinstance(selectors, list) or not selectors:
        return ""
    return "; ".join(str(s) for s in selectors)


def build_record(group: dict, field: str, spec, used_ids: set) -> dict:
    """Build one derived record from a group record and one field entry."""
    source = group.get("source") or {}
    src_file = str(source.get("file", "") or "").replace("\\", "/")
    src_path = str(source.get("path", "") or "").replace("\\", "/")
    group_key = str(source.get("key", "") or group.get("id", ""))
    level = _level_of(spec)
    required = level == "required"

    base_id = f"metafield_{_slug(field)}"
    rec_id = base_id
    if rec_id in used_ids:
        rec_id = f"{base_id}_{_slug(str(group.get('id', '')))}"
    suffix_n = 2
    while rec_id in used_ids:
        rec_id = f"{base_id}_{suffix_n}"
        suffix_n += 1

    # -- summary: purely factual (field, level, group) -------------------
    summary = f"BIDS sidecar metadata field '{field}'"
    if level:
        summary += f" (level: {level})"
    if group_key:
        summary += f", from group '{group_key}'"
    summary += "."

    # -- retrieval_text: facts plus verbatim source fragments ------------
    parts = [summary]
    if src_file:
        parts.append(f"Defined in {src_file}.")
    raw = group.get("raw_content")
    selectors = ""
    if isinstance(raw, dict):
        selectors = _flatten_selectors(raw.get("selectors"))
        if selectors:
            parts.append(f"Group selectors: {selectors}.")
    if isinstance(spec, dict):
        desc = spec.get("description_addendum")
        if isinstance(desc, str) and desc.strip():
            parts.append(f"Source note: {desc.strip()}")
        lvl_add = spec.get("level_addendum")
        if isinstance(lvl_add, str) and lvl_add.strip():
            parts.append(f"Level note: {lvl_add.strip()}")
        issue = spec.get("issue")
        if isinstance(issue, dict) and issue.get("code"):
            msg = str(issue.get("message", "") or "").strip()
            parts.append(f"Associated validator issue {issue['code']}: {msg}")
    retrieval_text = " ".join(parts)

    # -- raw_content: the field's original fragment ----------------------
    raw_content = {"field": field}
    if isinstance(spec, dict):
        raw_content.update(spec)
    else:
        raw_content["level"] = spec

    requirements: dict = {"required": required}
    if level:
        requirements["level"] = level

    return {
        "id": rec_id,
        "knowledge_type": "MetadataRule",
        "title": f"Metadata Field: {field}",
        "summary": summary,
        "retrieval_text": retrieval_text,
        "scope": {
            "metadata_fields": [field],
            "group": group_key,
            "applicable_to": [],
        },
        "conditions": [{"required": required}],
        "requirements": [requirements],
        "allowed_values": None,
        "unit": None,
        "bids_version": str(group.get("bids_version", "") or ""),
        "schema_version": str(group.get("schema_version", "") or ""),
        "source": {
            "file": src_file,
            "path": src_path,
            "section": "metadata_fields",
            "key": f"{group_key}.{field}" if group_key else field,
        },
        "raw_content": raw_content,
        "derived_from": "sidecar_field_derivation",
    }


def derive_records(lines: list[str]) -> tuple[list[str], dict]:
    """Return (new_lines, stats). Existing lines are preserved as-is."""
    existing_ids: set = set()
    parsed: list[tuple[int, dict]] = []
    for i, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            if rec.get("id"):
                existing_ids.add(rec["id"])
            parsed.append((i, rec))

    used_ids = set(existing_ids)
    derived: list[dict] = []
    seen_fields: set = set()
    duplicates = 0
    groups = 0

    for _i, group in parsed:
        raw = group.get("raw_content")
        if not isinstance(raw, dict) or not isinstance(raw.get("fields"), dict):
            continue
        groups += 1
        for field, spec in raw["fields"].items():
            if not isinstance(field, str) or not field or field in SKIP_FIELDS:
                continue
            if field in seen_fields:
                duplicates += 1  # dedupe: one record per field name, file order
                continue
            seen_fields.add(field)
            rec = build_record(group, field, spec, used_ids)
            used_ids.add(rec["id"])
            derived.append(rec)

    stats = {
        "groups_scanned": groups,
        "unique_fields": len(derived),
        "duplicate_occurrences_skipped": duplicates,
    }
    if not derived:
        return list(lines), stats

    out = list(lines)
    out.extend(json.dumps(rec, ensure_ascii=False) for rec in derived)
    return out, stats


def process_file(path: str, dry_run: bool = False) -> dict:
    with open(path, "rb") as f:
        head = f.read(4096)
    eol = "\r\n" if b"\r\n" in head else "\n"

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    # Idempotence: if derived records already exist, nothing to append.
    already = sum(1 for ln in lines if '"derived_from": "sidecar_field_derivation"' in ln)
    if already:
        return {"appended": 0, "skipped_existing": already}

    new_lines, stats = derive_records(lines)
    appended = len(new_lines) - len(lines)
    stats["appended"] = appended
    if not dry_run and appended:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(eol.join(new_lines) + eol)
        os.replace(tmp, path)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be added without writing")
    args = parser.parse_args()

    path = os.path.join(HERE, TARGET_FILE)
    if not os.path.exists(path):
        print(f"missing: {path}")
        return 1
    stats = process_file(path, dry_run=args.dry_run)
    print(f"{TARGET_FILE}: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
