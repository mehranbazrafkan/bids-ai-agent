"""
Tool registry — infrastructure for a future tool-execution phase.

The current phase is explanation-only: **no tool is executed**. The
placeholder tools that used to return hardcoded success messages ("The
validation completed", "Metadata repaired", ...) have been removed
because the planner previously summarized their output as if real work
had happened.

The registry and the ``agent_tool`` decorator are intentionally kept so
that a future phase can register real tools (validate, repair_metadata,
rename_subject, ...) and dispatch them from the planner without an
architectural rewrite. Calling ``execute`` while no tool is registered
raises :class:`ToolExecutionUnavailable` instead of pretending to work.
"""

from typing import Any, Callable, Dict, List, Optional


class ToolExecutionUnavailable(Exception):
    """Raised when tool execution is requested but no tools are enabled."""


registered_tools: List[Dict[str, Any]] = []


def agent_tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register a function as a tool for the agent.

    Registered tools are collected in ``registered_tools`` and can be
    handed to :class:`ToolRegistry` when a tool-execution phase is
    introduced.
    """

    def decorator(func: Callable):
        registered_tools.append({
            "function": func,
            "name": name or func.__name__,
            "description": description or func.__doc__,
        })
        return func

    return decorator


class ToolRegistry:
    def __init__(self, tools: Optional[Dict[str, Callable]] = None):
        # In the explanation-only phase this defaults to an *empty*
        # registry: nothing is executable.
        self._tools: Dict[str, Callable] = dict(tools or {})

    @property
    def tools(self) -> Dict[str, Callable]:
        return self._tools

    def register(self, name: str, func: Callable) -> None:
        self._tools[name] = func

    def execute(self, name: str, **kwargs) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolExecutionUnavailable(
                f"No tool named {name!r} is registered. Tool execution is "
                "not available in the current explanation-only phase."
            )
        return tool(**kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns metadata for registered tools (useful for LLM function calling)."""
        return [
            {"name": name, "doc": getattr(func, "__doc__", "")}
            for name, func in self._tools.items()
        ]