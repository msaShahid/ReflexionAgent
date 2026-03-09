class ReflexionError(Exception):
    """Base exception for Reflexion Agent."""
    pass

class ConfigurationError(ReflexionError):
    """Configuration related errors."""
    pass

class LLMError(ReflexionError):
    """Base class for LLM-related errors."""
    pass

class ProviderError(LLMError):
    """Provider-specific errors (API keys, connection issues)."""
    pass

class ModelError(LLMError):
    """Model-specific errors (context length, invalid parameters)."""
    pass

class RateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class MemoryError(ReflexionError):
    """Memory storage related errors."""
    pass

class ToolError(ReflexionError):
    """Base class for tool-related errors."""
    pass

class ToolExecutionError(ToolError):
    """Error during tool execution."""
    pass

class ToolValidationError(ToolError):
    """Error in tool input validation."""
    pass

class ActorError(ReflexionError):
    """Actor component errors."""
    pass

class EvaluatorError(ReflexionError):
    """Evaluator component errors."""
    pass

class ReflectorError(ReflexionError):
    """Reflector component errors."""
    pass

class LoopError(ReflexionError):
    """Reflexion loop errors."""
    pass