from agent import Agent
import json

# sample_issue_001 = {
#   "severity": "err",
#   "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
#   "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
#   "field": "ReferencesAndLinks",
#   "line": "null",
#   "lines": [],
#   "fix_label": "null",
#   "fix_action": "null",
#   "mirrored": False
# }

# sample_issue_002 = {
#   "severity": "warn",
#   "rule_id": "bidsmgr.todo_placeholder",
#   "message": "field 'EthicsApprovals' contains a TODO placeholder",
#   "field": "EthicsApprovals",
#   "line": "null",
#   "lines": [],
#   "fix_label": "Set a real value",
#   "fix_action": "set_field",
#   "mirrored": False
# }

# sample_issue_003 = {
#   "path": "participants.tsv",
#   "severity": "err",
#   "datatype": "null",
#   "suffix": "participants",
#   "issues": [
#     {
#       "severity": "err",
#       "rule_id": "TSV_VALUE_INCORRECT_TYPE",
#       "message": "column 'age': value '020Y' is not valid for its type  \u00b7  Each value must be a valid number, or 'n/a'.",
#       "field": "age",
#       "line": 2,
#       "lines": [
#         2,
#         3
#       ],
#       "fix_label": "null",
#       "fix_action": "null",
#       "mirrored": False
#     }
#   ],
#   "sidecar_fields": []
# }

sample_issue_004 = {
  "severity": "warn",
  "rule_id": "SIDECAR_KEY_RECOMMENDED",
  "message": "missing recommended field 'SequenceName'  \u00b7  Manufacturer's designation of the sequence name. Add it to the JSON, for example {\"SequenceName\": \"text\"}.",
  "field": "SequenceName",
  "line": "null",
  "lines": [],
  "fix_label": "Fix",
  "fix_action": "add_field",
  "mirrored": True
}

# sample_issue_001 = """
# {
#   "severity": "err",
#   "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
#   "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
#   "field": "ReferencesAndLinks",
#   "line": null,
#   "lines": [],
#   "fix_label": null,
#   "fix_action": null,
#   "mirrored": false
# }
# """

# sample_issue_001 = {
#   "severity": "err",
#   "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
#   "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
#   "field": "ReferencesAndLinks",
#   "line": "null",
#   "lines": [],
#   "fix_label": "null",
#   "fix_action": "null",
#   "mirrored": False
# }

user_prompt = "Explain this to me."

agent = Agent()
response = agent.run(user_input=user_prompt, context=sample_issue_004)
# response = agent.run(user_input=user_prompt, context=json.loads(sample_issue_001))
print(response)