# BIDS Knowledge Base – Generation Summary

## Version Information
- **BIDS Version**: 1.11.2-dev
- **Schema Version**: 2.0.0-dev

## What Was Extracted
The BIDS specification YAML files were processed into **1272 atomic knowledge records** and
**271 relationship edges** (`knowledge.jsonl`; 1266 unique ids — the original
extraction emitted 6 duplicate ids, preserved as-is).

The record the retriever actually loads at runtime is
**`enriched_knowledge.jsonl`** (`config.RetrievalConfig.knowledge_file`):
**1601 records** = the 1266 unique base records + **335 derived sidecar
field records** (see “Derived records” below). `knowledge.jsonl` is left
untouched as the reproducible base extraction.

### Knowledge Categories
| Category | Count |
|----------|-------|
| Association | 13 |
| Check | 131 |
| Concept | 240 |
| Definition | 36 |
| DirectoryRule | 33 |
| Enum | 218 |
| Error | 22 |
| FileSpecification | 180 |
| MetadataRule | 184 |
| Relationship | 11 |
| TabularRule | 180 |
| Template | 17 |
| Version | 2 |
| Warning | 5 |

**Total**: 1272 records

## How Knowledge Was Categorized
Each source file was classified and knowledge extracted according to its semantic content:

| Source Directory | Category | Description |
|------------------|----------|-------------|
| `meta/` | Version, Association, Template | Specification metadata, file associations, filename templates |
| `objects/` | Concept, Definition | BIDS object definitions (entities, suffixes, datatypes, modalities, etc.) |
| `rules/` | FilenameRule, DirectoryRule, EntityRule | Validation rules, directory layouts, entity orderings |
| `rules/checks/` | Check, Error, Warning | Validation checks and their severity levels |
| `rules/sidecars/` | MetadataRule | JSON sidecar field definitions and requirements |
| `rules/tabular_data/` | TabularRule | TSV/CSV column definitions and constraints |
| `rules/files/` | FileSpecification | File type, suffix, extension, and entity rules |
| `BIDS_VERSION`, `SCHEMA_VERSION` | Version | Version information for reproducibility |

## Relationships
Relationships were constructed using a controlled vocabulary:

| Relation | Description |
|----------|-------------|
| `defines` | A source defines or introduces a concept |
| `covers` | A modality covers a set of datatypes |
| `applies_to` | A rule applies to specific file types |
| `triggers` | A check triggers an error/warning |
| `maps_to_metadata` | An entity maps to a JSON metadata field |
| `templates` | A template defines filename structure |
| `validates` | An expression test validates engine behavior |
| `requires` | A rule requires certain fields or values |

## Derived records (enriched_knowledge.jsonl)

`derive_sidecar_field_records.py` appends one directly searchable
`MetadataRule` record per distinct metadata field found in the grouped
`raw_content.fields` of the local records (e.g. `SequenceName`,
`EchoTime`, `Manufacturer`). The records copy the field's own verbatim
source fragment — nothing is written by hand — and the script is
**idempotent**: re-running it does not duplicate records (verified by
`tests/test_retriever.py`).

Run it from the repository root:

```bash
python knowledge-base/derive_sidecar_field_records.py
```

This exists because the grouped sidecar records (e.g.
`meta_mrisequencespecifics`) previously hid individual fields from exact
field matching, leaving gaps such as `SequenceName` (see
`COVERAGE_GAPS.md`).

## Provenance
Every knowledge record contains:
- `source.file` – Original YAML filename
- `source.path` – Relative path to the source file (relative to the repo root, e.g. `./raw-bids-spec/...`)
- `source.section` – YAML section/key where information was found
- `source.key` – Specific key within the section
- `bids_version` / `schema_version` – For version-aware retrieval
- `raw_content` – The original YAML fragment for traceability

The original extraction ran on a different machine and stored **absolute
Windows paths** in `source.path`. `normalize_provenance.py` rewrites them
to repository-relative `./...` paths (using `source.file`, or the known
source-root marker in the absolute path) and is idempotent — already
normalized paths are left alone:

```bash
python knowledge-base/normalize_provenance.py
```

## Coverage Gaps

Some issues a client may ask about cannot be covered from material in
this repository (validator/manager rule codes whose sources were never
part of the BIDS specification YAML, and fields absent from the original
source tree). Those are documented — not fabricated — in
**`COVERAGE_GAPS.md`**, which lists each missing item, why it is missing,
what source would be needed, and what *is* covered for the sample issues.
`tests/test_retrieval_golden.py` asserts both the covered lookups and the
documented misses.

## What Could Not Be Classified
0 sections were not confidently classified. These were either:
- Empty or whitespace-only YAML sections
- Nested structures that did not represent atomic knowledge
- Metadata about the specification's construction rather than user-facing BIDS rules

## Inference
All knowledge records were marked as `confidence: explicit`. No inferred knowledge was added.
Inference is available as a future enhancement but not used in this extraction.

## Conflicts
0 potential conflicts were found between source files.
Conflicting records preserve both definitions and are flagged in their metadata.

## File Descriptions
- **knowledge.jsonl** – One JSON object per line; 1272 atomic knowledge records (1266 unique ids; base extraction, not loaded at runtime)
- **enriched_knowledge.jsonl** – Runtime knowledge base: 1266 unique base records + 335 derived sidecar field records = 1601
- **relationships.jsonl** – One relationship per line; graph edges connecting concepts
- **sources.jsonl** – Inventory of all source files and their contribution to the knowledge base
- **processing_report.json** – Summary statistics and quality metrics
- **derive_sidecar_field_records.py** – Idempotent derivation of per-field sidecar records into `enriched_knowledge.jsonl`
- **normalize_provenance.py** – Idempotent normalization of absolute `source.path` values to repo-relative paths
- **COVERAGE_GAPS.md** – Documented coverage gaps (what the KB cannot cover and why)
- **KB_README.md** – This file (human-readable summary)

## Usage Notes
This knowledge base is designed for:
1. **Exact retrieval**: Entity names, suffixes, metadata fields, error codes
2. **Semantic retrieval**: Natural-language questions via `retrieval_text`
3. **Relationship traversal**: Graph queries through `relationships.jsonl`

The raw BIDS source files remain the authoritative reference.
This knowledge base is a structured, machine-readable intermediate representation.
