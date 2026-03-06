from typing import Dict, Type, Optional, List, Any
import importlib

from reflexion_agent.observability import get_logger
from reflexion_agent.config import EnvSettings
from reflexion_agent.tools.base import BaseTool, AsyncBaseTool
from reflexion_agent.tools.calculator import CalculatorTool
from reflexion_agent.tools.web_search import WebSearchTool

logger = get_logger(__name__)


# Built-in tools registry
_BUILT_IN_TOOLS: Dict[str, Type[BaseTool]] = {
    "calculator": CalculatorTool,
    "web_search": WebSearchTool,
}

# Custom tools registry (can be extended at runtime)
_CUSTOM_TOOLS: Dict[str, Type[BaseTool]] = {}


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self, env_settings: Optional[EnvSettings] = None):
        """
        Initialize tool registry.
        
        Args:
            env_settings: Environment settings for tool configuration
        """
        self.env_settings = env_settings
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False
    
    async def initialize(self, enabled_tools: Optional[List[str]] = None):
        """
        Initialize tools based on configuration.
        
        Args:
            enabled_tools: List of tool names to enable (if None, enable all)
        """
        if enabled_tools is None:
            # Enable all built-in tools by default
            enabled_tools = list(_BUILT_IN_TOOLS.keys())
        
        # Initialize enabled tools
        for tool_name in enabled_tools:
            await self.register_tool(tool_name)
        
        self._initialized = True
        logger.info("tool_registry_initialized", tools=list(self._tools.keys()))
    
    async def register_tool(self, tool_name: str, tool_class: Optional[Type[BaseTool]] = None):
        """
        Register and initialize a tool.
        
        Args:
            tool_name: Name of the tool
            tool_class: Tool class (if None, look up in built-in registry)
            
        Raises:
            ValueError: If tool not found
        """
        # Get tool class if not provided
        if tool_class is None:
            if tool_name in _BUILT_IN_TOOLS:
                tool_class = _BUILT_IN_TOOLS[tool_name]
            elif tool_name in _CUSTOM_TOOLS:
                tool_class = _CUSTOM_TOOLS[tool_name]
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        
        # Create tool instance with environment settings if needed
        if tool_name == "web_search" and self.env_settings:
            # Configure web search with API keys if available
            tool = tool_class(
                api_key=self.env_settings.get_api_key("tavily") if hasattr(self.env_settings, 'get_api_key') else None
            )
        else:
            tool = tool_class()
        
        # Initialize async tools
        if isinstance(tool, AsyncBaseTool):
            await tool.initialize()
        
        self._tools[tool_name] = tool
        logger.debug("tool_registered", tool=tool_name, class_name=tool_class.__name__)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return [tool.schema() for tool in self._tools.values()]
    
    async def run_tool(self, name: str, **kwargs) -> Any:
        """Run a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return await tool.run(**kwargs)
    
    async def cleanup(self):
        """Clean up all tools."""
        for tool in self._tools.values():
            if isinstance(tool, AsyncBaseTool):
                await tool.cleanup()
        
        self._tools.clear()
        self._initialized = False
        logger.info("tool_registry_cleaned_up")
    
    def __getitem__(self, name: str) -> BaseTool:
        """Dictionary-style access to tools."""
        tool = self.get_tool(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")
        return tool
    
    def __contains__(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools


# Global registry instance
_default_registry: Optional[ToolRegistry] = None


def get_tool_registry(env_settings: Optional[EnvSettings] = None) -> ToolRegistry:
    """
    Get or create the default tool registry.
    
    Args:
        env_settings: Environment settings for tool configuration
        
    Returns:
        ToolRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(env_settings)
    return _default_registry


def register_custom_tool(name: str, tool_class: Type[BaseTool]):
    """
    Register a custom tool globally.
    
    Args:
        name: Tool name
        tool_class: Tool class
    """
    if not issubclass(tool_class, BaseTool):
        raise ValueError("Tool class must subclass BaseTool")
    
    _CUSTOM_TOOLS[name] = tool_class
    logger.info("custom_tool_registered", tool=name, class_name=tool_class.__name__)


def create_tools_from_config(config_tools: List[str], env_settings: Optional[EnvSettings] = None) -> ToolRegistry:
    """
    Create and initialize tools from configuration.
    
    Args:
        config_tools: List of tool names from config
        env_settings: Environment settings
        
    Returns:
        Initialized ToolRegistry
    """
    registry = ToolRegistry(env_settings)
    
    # This is async - need to be called with await in async context
    # Return registry and let caller initialize
    return registry
