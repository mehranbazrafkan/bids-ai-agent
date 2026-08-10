# BIDS Knowledge Base – Generation Summary

## Version Information
- **BIDS Version**: 1.11.2-dev
- **Schema Version**: 2.0.0-dev

## What Was Extracted
The BIDS specification YAML files were processed into **1272 atomic knowledge records** and
**271 relationship edges**.

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

## Provenance
Every knowledge record contains:
- `source.file` – Original YAML filename
- `source.path` – Full path to source file
- `source.section` – YAML section/key where information was found
- `source.key` – Specific key within the section
- `bids_version` / `schema_version` – For version-aware retrieval
- `raw_content` – The original YAML fragment for traceability

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
- **knowledge.jsonl** – One JSON object per line; 121+ unique atomic knowledge records
- **relationships.jsonl** – One relationship per line; graph edges connecting concepts
- **sources.jsonl** – Inventory of all source files and their contribution to the knowledge base
- **processing_report.json** – Summary statistics and quality metrics
- **KB_README.md** – This file (human-readable summary)

## Usage Notes
This knowledge base is designed for:
1. **Exact retrieval**: Entity names, suffixes, metadata fields, error codes
2. **Semantic retrieval**: Natural-language questions via `retrieval_text`
3. **Relationship traversal**: Graph queries through `relationships.jsonl`

The raw BIDS source files remain the authoritative reference.
This knowledge base is a structured, machine-readable intermediate representation.
