# All the tools from the main application will be registered here for the agent to use
from typing import Any, Dict, List

def validate(**kwargs):
    # Placeholder for the validate tool
    return "Validation completed."

def repair_metadata(**kwargs):
    # Placeholder for the repair_metadata tool
    return "Metadata repaired."

def rename_subject(**kwargs):
    # Placeholder for the rename_subject tool
    return "Subject renamed."

registered_tools = []

def agent_tool(name=None, description=None):
    """
    Decorator to register a function as a tool for the agent.
    """
    def decorator(func):
        registered_tools.append({
            "function": func,
            "name": name or func.__name__,
            "description": description or func.__doc__,
        })
        return func
    return decorator

class ToolRegistry:
    def __init__(self):
        self._tools = {
            "validate": validate,
            "repair_metadata": repair_metadata,
            "rename_subject": rename_subject,
        }

    def execute(self, name, **kwargs):
        tool = self._tools[name]
        return tool(**kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns metadata for registered tools (useful for LLM function calling)."""
        return [{"name": name, "doc": func.__doc__} for name, func in self._tools.items()]