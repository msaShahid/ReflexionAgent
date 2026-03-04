# src/reflexion_agent/utils/exceptions.py
class ReflexionError(Exception):
    """Base exception for Reflexion Agent."""
    pass


class ConfigurationError(ReflexionError):
    """Configuration related errors."""
    pass


class LLMError(ReflexionError):
    """LLM related errors."""
    pass


class MemoryError(ReflexionError):
    """Memory storage related errors."""
    pass


class ToolError(ReflexionError):
    """Tool execution related errors."""
    pass