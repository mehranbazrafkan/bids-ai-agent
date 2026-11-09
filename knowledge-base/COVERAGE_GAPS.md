# Knowledge-base coverage gaps

This file records what the knowledge base does *not* cover, so a missing record
is never mistaken for a fact the agent may invent. It is updated by running:

```bash
python eval_quality.py --audit-kb
```

The audit runs the frozen quality set (`tests/data/quality_set.jsonl`, 36
findings) through the real retriever and the answer-contract builder, then
reports:

- retrieval misses (no record clears the relevance floor);
- findings for which no contract can be built;
- findings with no direct record (the answer must say so);
- fields whose BIDS level is not established by the retrieved rule.

## Current result (quality set, n = 36)

| check | count |
| --- | --- |
| retrieval misses | **0** |
| no contract | **0** |
| no direct record (required to be declared) | **6** |
| field found, BIDS level unknown | **1** (`Description`) |

The six no-direct findings are the `bidsmgr.todo_placeholder` cases. That rule
is an agent-side dataset-manager check, not BIDS knowledge, so there is
deliberately no knowledge-base record for it. `answer_contract` marks these as
`requires_no_direct_statement`, and every answer must contain "The knowledge
base does not document this rule." This is the documented behaviour, not a gap
to fill.

`Description` appears in several rules with different levels depending on the
file, so the retrieved evidence does not establish a single level. The contract
therefore leaves `bids_level` unset and the gate does not police
required/recommended wording for that field. Inventing a level would be worse.

## Gaps investigated and found already covered

| suspected gap | actual record(s) |
| --- | --- |
| no `metadata field: License` record | `kb_concept_07e35d327b` (`metadata field: License`, normative schema, `RECOMMENDED` abbreviations) |
| no scans-sidecar record for `SourceDatasets` | `kb_concept_c2d0acbbaa` (`metadata field: SourceDatasets`), `kb_concept_56cbc072ad` (`Scans file`), `kb_tabularrule_fd6dcd9f95` (`Tabular data rule Scans`) |
| no rule giving `License` / `Acknowledgements` / `HEDVersion` levels | `kb_metadatarule_5dad7e66a6` (`Metadata rule dataset_description`): `License` recommended, `Acknowledgements` optional, `HEDVersion` recommended |

The recorded Phase-2 failures were therefore **not** retrieval failures: every
case retrieved the correct rule or concept. They were generation failures
(inverting the level, calling a warning an error, inventing a value), which are
now prevented by the contract, the gate and the grounded fallback rather than
by adding duplicate records.

## Why no synthetic records were added

`knowledge_base.jsonl` is generated from verified BIDS sources; every record
carries a repository, revision and retrieval timestamp. Hand-writing a record
without that provenance would undermine the guarantee that the KB contains
only sourced material. Where a rule genuinely has no source (the manager's
`TODO` check), the honest answer is "the knowledge base does not document this
rule" — which is exactly what the contract requires.
