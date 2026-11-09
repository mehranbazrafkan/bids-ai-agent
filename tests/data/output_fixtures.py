"""The four recorded agent outputs from ``Test-0002`` (Phase-2 baseline).

Each entry is the exact prompt-log payload plus the exact answer the agent
gave, and the gate violations that answer must produce. These are frozen on
purpose: a change that stops catching them is a regression, whatever else it
improves.
"""

FILE_DIGEST = """JSON document, 1,264 characters:
  Name: test
  BIDSVersion: 1.11.1
  DatasetType: raw
  License: TODO
  Authors: [TODO]
  Acknowledgements: TODO
  HowToAcknowledge: TODO
  Funding: [TODO]
  EthicsApprovals: [TODO]
  ReferencesAndLinks: [TODO]
  DatasetDOI: TODO
  Description: TODO
  HEDVersion: TODO"""

CASES = [
    {
        "name": "license_todo",
        "issue": {
            "severity": "warn",
            "rule_id": "bidsmgr.todo_placeholder",
            "message": "field 'License' contains a TODO placeholder",
            "field": "License",
            "path": "dataset_description.json",
        },
        "file_content": FILE_DIGEST,
        "expected_codes": {
            "missing_no_direct_declaration",
            "example_not_allowed",
            "modality_overclaim",
        },
        "expected_contract": {
            "bids_level": "recommended",
            "authority": "manager",
            "placeholder": True,
            "example_allowed": False,
            "concrete_value_allowed": False,
            "direct_evidence": False,
        },
        "output": """**Problem**
The field `'License'` in the file `dataset_description.json` contains a TODO placeholder, which indicates that the value has not been set yet. This is a warning according to the BIDS validator.

**Why**
This happens because the `License` field in `dataset_description.json` is expected to contain a value, but it currently holds a TODO placeholder. According to the BIDS specification, this field is considered required and should be filled in with the appropriate license information.

**How to Fix**
The user needs to provide a real value for the `'License'` field in `dataset_description.json`. For example, they could include a license filename like `LICENSE`, specifying the name of the license file in the dataset.

**Example**
If the user wants to specify a license file, they might update the `dataset_description.json` entry under the `'License'` field as follows:

```json
{
...
"License": "LICENSE",
...
}
```

Here, `"LICENSE"` represents the name of the license file, and it should match the actual file in the dataset directory. This ensures consistency between the specified license in the metadata and the actual file.""",
    },
    {
        "name": "hedversion_format",
        "issue": {
            "severity": "err",
            "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
            "message": 'HEDVersion must match format "hed_version"',
            "field": "HEDVersion",
            "path": "dataset_description.json",
        },
        "file_content": FILE_DIGEST,
        "expected_codes": {
            "concrete_value_not_allowed",
            "unsupported_version_value",
        },
        "expected_contract": {
            "bids_level": "recommended",
            "authority": "validator",
            "placeholder": True,
            "example_allowed": True,
            "concrete_value_allowed": False,
            "direct_evidence": True,
        },
        "output": """**Problem**
The `HEDVersion` field in `dataset_description.json` is not matching the expected format specified by the BIDS schema. The validation error indicates that the value should conform to the `"hed_version"` format, which is a string with a specific structure.

**Why**
This issue arises due to the absence of a properly formatted `HEDVersion` value in the dataset. According to the BIDS specification, `HEDVersion` must follow the pattern `"hed_version"` and should contain valid version identifiers. Since the current value is unspecified or unformatted, it fails the schema check.

**How to Fix**
1. Identify the correct version number(s) for the HED schema being used. For example, common versions might be `"1.0"`, `"1.1"`, etc., depending on the setup.
2. Update the `HEDVersion` field in `dataset_description.json` to reflect the chosen version.
3. Ensure that any other related files (like `.json` or `.tsv`) containing `HEDVersion` also have consistent formatting.

**Example**
A correctly formatted `HEDVersion` could look like:
```json
"HEDVersion": "1.0"
```

Please ensure that the value matches the expected format and aligns with the version of the HED schema used in your dataset.""",
    },
    {
        "name": "sourcedatasets_recommended",
        "issue": {
            "severity": "Warning",
            "rule_id": "SIDECAR_KEY_RECOMMENDED",
            "message": (
                "missing recommended field 'SourceDatasets'. Used to specify "
                "the locations and relevant attributes of all source datasets "
                "(BIDS or not)."
            ),
            "field": "SourceDatasets",
            "file": "sub-01_scans.json",
        },
        "file_content": None,
        "expected_codes": {"severity_mismatch", "bids_requirement_overclaim"},
        "expected_contract": {
            "bids_level": None,
            "authority": "validator",
            "placeholder": False,
            "example_allowed": True,
            "concrete_value_allowed": False,
            "direct_evidence": True,
        },
        "output": """**Problem**
The error indicates that the file `sub-01_scans.json` is missing the recommended field `'SourceDatasets'`. According to the BIDS specification, this field should be included in the JSON sidecar of data files to specify the locations and relevant attributes of all source datasets (including non-BIDS datasets).

**Why**
This issue arises because the JSON sidecar for the data file lacks the `'SourceDatasets'` key, which is considered important for tracking and referencing source data. While the BIDS specification doesn't mandate this field specifically, it is strongly recommended as part of good practices for reproducibility and traceability.

**How to Fix**
To resolve this issue, the user needs to add the `'SourceDatasets'` key to the JSON sidecar of the file `sub-01_scans.json`. This involves modifying the JSON structure of the file to ensure it includes this key with appropriate values such as URLs, DOIs, or versions.

For example, a correctly formatted entry might look like this:

```json
{
  "SourceDatasets": [
    {
      "URL": "http://example.org/data",
      "DOI": "10.1234/abc123",
      "Version": "1.0"
    }
  ]
}
```

Please replace the placeholders (`http://example.org/data`) with the actual URL(s), DOI(s), and version(s) of the source datasets.

**Example**
If the current file has no `'SourceDatasets'` key, adding it as described above will satisfy the BIDS standard and address the error. Ensure that any existing entries in `"SourceDatasets"` follow the expected format and contain valid keys like `"URL"`, `"DOI"`, and `"Version"`.""",
    },
    {
        "name": "acknowledgements_todo",
        "issue": {
            "severity": "warn",
            "rule_id": "bidsmgr.todo_placeholder",
            "message": "field 'Acknowledgements' contains a TODO placeholder",
            "field": "Acknowledgements",
            "path": "dataset_description.json",
        },
        "file_content": FILE_DIGEST,
        "expected_codes": {
            "missing_no_direct_declaration",
            "concrete_value_not_allowed",
            "example_not_allowed",
            "bids_requirement_overclaim",
            "fabricated_named_entity",
        },
        "expected_contract": {
            "bids_level": "optional",
            "authority": "manager",
            "placeholder": True,
            "example_allowed": False,
            "concrete_value_allowed": False,
            "direct_evidence": False,
        },
        "output": """**Problem**
The field `'Acknowledgements'` in the file `dataset_description.json` contains a TODO placeholder, which means it has not been set to a real value yet according to the BIDS validation findings.

**Why**
This issue arises because the `Acknowledgements` field is defined in the BIDS metadata schema as a metadata field that should contain text acknowledging contributions beyond those listed in `Authors` or `Funding`. However, the validation result indicates that the field currently holds a TODO placeholder, meaning it hasn't been filled in by the dataset creator.

**How to Fix**
The user needs to provide a real value for the `'Acknowledgements'` field in `dataset_description.json`. This could be done manually by editing the file and replacing the TODO placeholder with appropriate acknowledgment text. For instance, if the dataset involves collaborations or institutional support, the user should include such information here.

**Example**
A corrected example might look like:
```json
{
  "Acknowledgements": "We gratefully acknowledge the technical support provided by Dr. John Doe at XYZ Institute."
}
```

Please ensure the value added aligns with the project's goals and standards. Once updated, the field will meet the BIDS requirements and avoid further validation warnings.""",
    },
]
