"""
Tools module for Reflexion Agent.
Provides safe tool execution for calculator, web search, and extensible tool framework.
"""

from reflexion_agent.tools.base import (
    BaseTool,
    AsyncBaseTool,
    ToolResult,
    ToolError,
    ToolExecutionError,
    ToolValidationError,
)

from reflexion_agent.tools.calculator import CalculatorTool
from reflexion_agent.tools.web_search import WebSearchTool

from reflexion_agent.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    register_custom_tool,
    create_tools_from_config,
    _BUILT_IN_TOOLS,
)

__all__ = [
    # Base classes
    "BaseTool",
    "AsyncBaseTool",
    "ToolResult",
    "ToolError",
    "ToolExecutionError",
    "ToolValidationError",
    
    # Built-in tools
    "CalculatorTool",
    "WebSearchTool",
    
    # Registry
    "ToolRegistry",
    "get_tool_registry",
    "register_custom_tool",
    "create_tools_from_config",
    "_BUILT_IN_TOOLS",
]