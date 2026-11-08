"""Applicability assessment for retrieved BIDS knowledge records.

A knowledge record often carries *selectors* — the conditions under which
the documented rule applies (BIDS schema semantics: **all** selectors of
a record must hold). Example::

    ["suffix == "events"", "extension == ".tsv""]

Whether such a record explains a concrete validation issue depends on the
file the issue concerns. A rule about ``events.tsv`` says nothing about
``dataset_description.json``, no matter how often the same field name
appears in both.

This module answers one deterministic question, with no LLM involved::

    Given a record's selectors and the issue's file context, is the
    record applicable, inapplicable, or can we not tell?

Design rules:

* **Conservative**: a selector that cannot be evaluated from the
  available context evaluates to ``None`` (unknown) — unknown never
  causes a record to be rejected.
* **Reject only on a definite conflict**: a record is inapplicable only
  when at least one selector evaluates to *false*.
* **No knowledge-base changes**: selectors are read from the record data
  already loaded by the retriever.

Supported selector shapes (everything else evaluates to unknown):

* ``suffix == "x"`` / ``suffix != "x"``
* ``extension == ".tsv"`` / ``extension != ".tsv"``
* ``datatype == "func"`` / ``datatype != "func"``
* ``path == "/dataset_description.json"`` (compared by file name)
* ``intersects([suffix], ["dwi", "epi"])`` / ``intersects([extension], ...)``
* ``"x" in entities`` / ``entities.x``
* ``match(extension, '^\\.nii(\\.gz)?$')``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

# BIDS entity keys that can prefix a file stem ("sub-01_task-rest_physio").
# Used only to decide whether a file name looks BIDS-entity-structured, so
# the list is intentionally broad but not required to be exhaustive: an
# unknown entity prefix just makes suffix inference fall back to the whole
# stem, which is the conservative direction for non-entity files.
_ENTITY_PREFIX_RE = re.compile(
    r"^(?:sub|ses|task|acq|ce|rec|dir|run|mod|echo|flip|inv|mt|part|proc|"
    r"space|res|den|label|from|to|mode|hemi|sample|stain|chunk|tracksys|"
    r"recording|split|desc|dataset|sample)[0-9]*-",
    re.IGNORECASE,
)

# selector == "value" / selector != "value" (either quote style)
_EQ_RE = re.compile(
    r"""^\s*(suffix|extension|datatype|path)\s*(==|!=)\s*["']([^"']*)["']\s*$"""
)
# intersects([suffix], ["a", "b"]) — the two forms seen in the schema
_INTERSECTS_RE = re.compile(
    r"""^\s*intersects\(\s*\[\s*(suffix|extension|datatype|modality)\s*\]\s*,\s*\[([^\]]*)\]\s*\)\s*$"""
)
# "x" in entities  /  entities.x
_ENTITY_IN_RE = re.compile(r"""^\s*["']([^"']+)["']\s+in\s+entities\s*$""")
_ENTITY_RE = re.compile(r"""^\s*entities\.([A-Za-z_][A-Za-z0-9_]*)\s*$""")
# match(extension, "regex")
_MATCH_EXT_RE = re.compile(
    r"""^\s*match\(\s*extension\s*,\s*["'](.*)["']\s*\)\s*$""", re.DOTALL
)

# Splitting a selector value list: ["dwi", "epi"]
_LIST_ITEM_RE = re.compile(r"""["']([^"']*)["']""")


@dataclass(frozen=True)
class IssueScope:
    """The file context an issue is about, extracted from the issue dict.

    Any component may be ``None`` when the issue does not carry enough
    information — ``None`` means *cannot tell*, never *does not match*.
    """

    path: str | None = None
    basename: str | None = None
    extension: str | None = None      # ".json", ".tsv", ...
    suffix: str | None = None         # "events", "physio", ...
    suffix_source: str | None = None  # "issue" | "inferred" | None
    datatype: str | None = None
    stem: str | None = None
    entities: tuple[str, ...] = ()       # entity keys present in the file name

    @property
    def known(self) -> bool:
        return bool(self.path or self.suffix or self.datatype)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "nan"):
        return None
    return text


def infer_suffix(stem: str) -> str | None:
    """Best-effort BIDS suffix from a file stem.

    Entity-structured names (``sub-01_task-rest_physio``) end with the
    suffix as their last ``_`` token when that token is not itself an
    entity pair. Names without an entity prefix (``dataset_description``,
    ``participants``) are their own suffix — taking only the last token
    would turn ``dataset_description`` into ``description`` and wrongly
    match rules that apply to ``*_description.json`` files.
    """
    stem = stem.strip()
    if not stem:
        return None
    tokens = stem.split("_")
    if len(tokens) > 1 and _ENTITY_PREFIX_RE.match(tokens[0]):
        last = tokens[-1]
        if "-" not in last:
            return last
    return stem


def _entities_of(basename: str) -> tuple[str, ...]:
    """Entity keys present in a BIDS file name (``sub-01_task-rest_...``)."""
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    keys: list[str] = []
    for token in stem.split("_"):
        if "-" in token:
            key = token.split("-", 1)[0].lower()
            if key and key.isalpha():
                keys.append(key)
    return tuple(dict.fromkeys(keys))


def build_issue_scope(issue: Any) -> IssueScope:
    """Build the file scope from an issue dict (nested issues merged)."""
    if not isinstance(issue, dict):
        return IssueScope()

    combined: dict[str, Any] = dict(issue)
    inner = issue.get("issues")
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        for key, value in inner[0].items():
            combined.setdefault(key, value)

    path = _clean(combined.get("path"))
    datatype = _clean(combined.get("datatype"))
    suffix = _clean(combined.get("suffix"))

    basename: str | None = None
    extension: str | None = None
    stem: str | None = None
    entities: tuple[str, ...] = ()
    if path:
        posix = path.replace("\\", "/")
        basename = PurePosixPath(posix).name
        if "." in basename:
            stem, ext = basename.rsplit(".", 1)
            extension = "." + ext.lower()
            # BIDS imaging files use the compound extension ".nii.gz";
            # seeing only ".gz" would wrongly fail selectors such as
            # match(extension, '^\\.nii(\\.gz)?$').
            if extension == ".gz" and stem.lower().endswith(".nii"):
                stem = stem[: -len(".nii")]
                extension = ".nii.gz"
        else:
            stem = basename
        entities = _entities_of(basename)

    suffix_source = None
    if not suffix and stem:
        inferred = infer_suffix(stem)
        if inferred:
            # Entity-structured names yield their last non-entity token;
            # non-entity names (participants, dataset_description) keep the
            # whole stem, which is their BIDS suffix.
            suffix = inferred
            suffix_source = "inferred"
    elif suffix:
        suffix_source = "issue"

    return IssueScope(
        path=path,
        basename=basename,
        extension=extension,
        suffix=suffix,
        suffix_source=suffix_source,
        datatype=datatype,
        stem=stem,
        entities=entities,
    )


def _eval_equality(key: str, op: str, value: str, scope: IssueScope) -> bool | None:
    if key == "extension":
        known = scope.extension
    elif key == "datatype":
        known = scope.datatype
    elif key == "path":
        # Schema paths are dataset-root-relative ("/x/y.tsv"); the issue
        # path may be relative from anywhere. File-name equality is the
        # only comparison that is safe in both directions.
        if not (scope.basename and value):
            return None
        sel_name = PurePosixPath(value.replace("\\", "/")).name
        return scope.basename == sel_name if op == "==" else scope.basename != sel_name
    else:  # suffix
        known = scope.suffix
    if known is None:
        return None
    return known == value if op == "==" else known != value


def evaluate_selector(selector: str, scope: IssueScope) -> bool | None:
    """Evaluate one selector against the issue scope.

    Returns ``True`` (holds), ``False`` (definitely does not hold) or
    ``None`` (cannot be determined from the available context).
    """
    sel = str(selector or "").strip()
    if not sel:
        return None

    m = _EQ_RE.match(sel)
    if m:
        return _eval_equality(m.group(1), m.group(2), m.group(3), scope)

    m = _INTERSECTS_RE.match(sel)
    if m:
        key = m.group(1)
        items = [v.strip().strip("\"'") for v in _LIST_ITEM_RE.findall(m.group(2))]
        if key == "extension":
            known = scope.extension
        elif key == "datatype":
            known = scope.datatype
        elif key == "suffix":
            known = scope.suffix
        else:
            return None  # modality: not derivable from the issue context
        if known is None:
            return None
        return known in items

    m = _ENTITY_IN_RE.match(sel)
    if m:
        if not scope.path:
            return None
        return m.group(1) in scope.entities

    m = _ENTITY_RE.match(sel)
    if m:
        if not scope.path:
            return None
        return m.group(1).lower() in scope.entities

    m = _MATCH_EXT_RE.match(sel)
    if m:
        if scope.extension is None:
            return None
        pattern = m.group(1)
        try:
            return re.search(pattern, scope.extension) is not None
        except re.error:
            return None

    # Anything involving sidecar/associations/nifti_header/dataset state or
    # function calls we do not model: unknown, never a rejection.
    return None


@dataclass(frozen=True)
class ApplicabilityResult:
    """Outcome of checking a record's selectors against an issue scope."""

    status: str                       # "applicable" | "inapplicable" | "unknown"
    reason: str = ""
    checked: tuple[str, ...] = ()     # selectors that evaluated definitively
    unverified: tuple[str, ...] = ()  # selectors that could not be evaluated

    @property
    def is_inapplicable(self) -> bool:
        return self.status == "inapplicable"


def classify_selectors(
    selectors: Any, scope: IssueScope
) -> ApplicabilityResult:
    """Classify a record's applicability from its selector list.

    BIDS schema semantics: **all** selectors must hold, so a single
    selector that definitely fails rules the record out for this issue.
    """
    if not selectors:
        return ApplicabilityResult("unknown", "record carries no applicability selectors")

    if isinstance(selectors, str):
        selectors = [selectors]
    if not isinstance(selectors, (list, tuple)):
        return ApplicabilityResult("unknown", "record carries no applicability selectors")

    if not scope.known:
        return ApplicabilityResult(
            "unknown",
            "the issue provides no file context (path/suffix/datatype) to check selectors against",
        )

    checked: list[str] = []
    unverified: list[str] = []
    for selector in selectors:
        if not isinstance(selector, str):
            unverified.append(str(selector))
            continue
        verdict = evaluate_selector(selector, scope)
        if verdict is False:
            return ApplicabilityResult(
                "inapplicable",
                f"selector `{selector.strip()}` does not hold for "
                f"{scope.path or scope.datatype or scope.suffix or 'this file'}",
                tuple(checked),
                tuple(unverified),
            )
        if verdict is True:
            checked.append(selector.strip())
        else:
            unverified.append(selector.strip())

    if checked:
        detail = "; ".join(checked)
        note = (
            f"unverified conditions remain: {', '.join(unverified)}"
            if unverified else "all selectors hold"
        )
        return ApplicabilityResult("applicable", f"{detail} ({note})",
                                  tuple(checked), tuple(unverified))
    return ApplicabilityResult(
        "unknown",
        "selectors could not be checked against the available file context"
        + (f" (unverified: {', '.join(unverified)})" if unverified else ""),
        tuple(checked),
        tuple(unverified),
    )


def extract_selectors(data: dict[str, Any]) -> tuple[str, ...]:
    """Pull the selector list out of a raw knowledge-base record.

    Read at load time from the data already in hand (the record's
    ``raw_content`` is still materialized during ``KnowledgeRecord``
    construction, before lazy loading takes over).
    """
    raw = data.get("raw_content")
    if not isinstance(raw, dict):
        return ()
    for key in ("rule_definition", "original_rule_definition"):
        node = raw.get(key)
        if isinstance(node, dict):
            selectors = node.get("selectors")
            if isinstance(selectors, list) and selectors:
                return tuple(str(s) for s in selectors)
    # Some records nest the definition one level deeper.
    for value in raw.values():
        if isinstance(value, dict):
            selectors = value.get("selectors")
            if isinstance(selectors, list) and selectors:
                return tuple(str(s) for s in selectors)
    return ()
