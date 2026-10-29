# Knowledge-Base Coverage Gaps

This file records knowledge the retriever is *expected* not to find, so
tests and readers can distinguish "not in the knowledge base" from
"retrieval failed". It is verified against the JSONL files in this
directory by `tests/test_retrieval_golden.py`.

## Source situation

* The knowledge base was generated from the BIDS specification YAML tree
  (`RAW BIDS Knowledge Data/`, see `source.path` provenance) by an
  extraction script (`extract_kb.py`) that was **never committed to this
  repository**, and the original source tree **no longer exists on this
  machine** (the absolute paths it recorded pointed at a temporary
  Desktop folder). The committed, reproducible artifacts are:
  * `knowledge.jsonl` — the base extraction (1272 records),
  * `enriched_knowledge.jsonl` — base extraction + web-enriched records
    (1266) + `derive_sidecar_field_records.py` output (335),
  * `normalize_provenance.py` / `derive_sidecar_field_records.py` —
    reproducible, idempotent post-processing scripts.
* Validator/manager issue codes that came from *other tools* (the
  JavaScript bids-validator, a BIDS manager) were never part of the BIDS
  specification YAML, so no record for them can be regenerated from
  material in this repository.

## Confirmed gaps (as of the current files)

| Missing item | Kind | Why it is missing | What would be needed |
|---|---|---|---|
| `bidsmgr.todo_placeholder` (e.g. for `EthicsApprovals` TODO placeholders) | BIDS-manager rule code | The BIDS manager's rule sources are not in this repository or its history | The BIDS-manager source that defines `bidsmgr.*` rules, run through the same extraction schema |
| `TSV_VALUE_INCORRECT_TYPE` (e.g. `participants.tsv` age value `020Y`) | bids-validator issue code | The JS bids-validator is not part of the BIDS specification YAML used for extraction | bids-validator issue definitions (code + message), extracted into `requirements.code`-style records |
| `SIDECAR_KEY_RECOMMENDED` (e.g. missing recommended `SequenceName`) | bids-validator issue code | Same as above | Same as above |
| Direct documentation for `EthicsApprovals` | BIDS field | The extraction source contains no `dataset_description.json` sidecar field group; `EthicsApprovals` appears nowhere in any record's `raw_content` | The BIDS specification's `dataset_description` sidecar rules |
| Validator messages/levels attached to the codes above | validator metadata | Follows from the missing codes | Same sources as above |

## What IS covered for those sample issues (added this phase)

* **`SequenceName`** — now has a direct record
  (`metafield_sequencename`, level `recommended`,
  `rules/sidecars/mri.yaml`), derived by
  `derive_sidecar_field_records.py` from the field's own verbatim source
  fragment inside `meta_mrisequencespecifics.raw_content.fields`.
* **`participants.tsv` age column** — `tabcol_participants_age` (column
  constraints) plus the related checks `check_dataset_AgeUnits` and
  `check_privacy_CheckAge89`.
* **`JSON_SCHEMA_VALIDATION_ERROR`** — `err_jsonschemavalidationerror`
  (`requirements.code`), matched exactly by rule id.
* **158 records** expose a validator/manager rule code via
  `requirements.code` or `raw_content.issue.code`, and both are indexed
  for exact rule-id matching (e.g. `AGE_UNITS` →
  `check_dataset_AgeUnits`).
* **335 sidecar field records** (one per distinct metadata field name in
  the grouped `raw_content.fields`) give every field present in the
  source material a directly searchable record — `EchoTime`,
  `RepetitionTime`, `Manufacturer`, `SequenceName`, …

## Policy

Missing codes are reported honestly: retrieval returns an explicit
no-result (or only clearly-marked lexical background), the planner tells
the model that no relevant record was found, and the response must not
invent a rule definition. No records in this knowledge base are
hand-written to make retrieval pass; derived records copy verbatim
source fragments only.
