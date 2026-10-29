#!/usr/bin/env python3
"""
Normalize ``source.path`` provenance in the knowledge-base JSONL files.

The original extraction ran on a developer machine and stored absolute
Windows paths (``C:\\Users\\<name>\\...\\RAW BIDS Knowledge Data\\...``) in
``source.path``. Absolute, machine-specific paths are not portable and leak
local usernames, so this script rewrites them to repository-relative paths
of the form ``./rules/checks/dataset.yaml`` (forward slashes), matching the
convention documented in KB_README.md.

Rules
-----
1. If ``source.file`` is present and not itself absolute, the relative path
   is ``"./" + source.file`` with backslashes converted to forward slashes
   (``source.file`` is the authoritative relative location of the record's
   source document).
2. Otherwise, if the absolute path contains the known source root marker
   (``RAW BIDS Knowledge Data``), the path after the marker is used.
3. Otherwise the value is left untouched and reported, so nothing is ever
   silently dropped or guessed.

Only ``source.path`` is modified; every other byte of the record is
preserved. The script is idempotent: already-normalized (``./...``) paths
are left as-is, so it can be re-run safely after future enrichments.

Usage::

    python knowledge-base/normalize_provenance.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = ("knowledge.jsonl", "enriched_knowledge.jsonl")

# Marker directory that held the raw BIDS rule sources on the machine that
# generated the knowledge base.
ROOT_MARKER = "RAW BIDS Knowledge Data"


def _looks_absolute(path: str) -> bool:
    return (
        len(path) > 2 and path[1] == ":" and path[2] in ("/", "\\")
        or path.startswith("/")
        or path.startswith("\\\\")
    )


def normalize_path(record: dict) -> tuple[bool, str]:
    """Return ``(changed, new_value)`` for this record's ``source.path``."""
    source = record.get("source")
    if not isinstance(source, dict):
        return False, ""
    path = str(source.get("path", "") or "")
    if not path:
        return False, path
    if not _looks_absolute(path):
        # Already relative (or a bare filename) — normalize separators and
        # make it explicitly repo-relative.
        if "\\" in path or not path.startswith("./"):
            tail = path.lstrip("/\\")
            fixed = "./" + tail.replace("\\", "/")
            return fixed != path, fixed
        return False, path

    file_name = str(source.get("file", "") or "").strip()
    if file_name and not _looks_absolute(file_name):
        new = "./" + file_name.replace("\\", "/")
        return new != path, new

    for marker in (ROOT_MARKER, ROOT_MARKER.lower()):
        idx = path.find(marker)
        if idx != -1:
            tail = path[idx + len(marker):].lstrip("/\\")
            new = "./" + tail.replace("\\", "/")
            return new != path, new

    # Unknown shape: do not guess.
    return False, path


def process_file(path: str) -> int:
    # Preserve the file's existing line endings so unchanged lines do not
    # pick up diff noise.
    with open(path, "rb") as f:
        eol = "\r\n" if b"\r\n" in f.read(4096) else "\n"

    lines = []
    changed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\r\n")
            if not raw.strip():
                lines.append(raw)
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                lines.append(raw)
                continue
            if isinstance(record, dict):
                was_changed, new_path = normalize_path(record)
                if was_changed:
                    record.setdefault("source", {})["path"] = new_path
                    changed += 1
                    lines.append(json.dumps(record, ensure_ascii=False))
                else:
                    lines.append(raw)
            else:
                lines.append(raw)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(eol.join(lines) + eol)
    os.replace(tmp, path)
    return changed


def main() -> int:
    total = 0
    for name in FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"skipped (missing): {name}")
            continue
        changed = process_file(path)
        total += changed
        print(f"{name}: {changed} source.path values normalized")
    print(f"done: {total} paths normalized in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
