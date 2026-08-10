from agent import Agent

sample_warning = {
  "severity": "warn",
  "rule_id": "bidsmgr.todo_placeholder",
  "message": "field 'License' contains a TODO placeholder",
  "field": "License",
  "line": "",
  "lines": [],
  "fix_label": "Set a real value",
  "fix_action": "set_field",
  "mirrored": False
}

user_prompt = "Please explain this warning to me."

agent = Agent()
response = agent.run(user_input=user_prompt, context=sample_warning)
print(response)