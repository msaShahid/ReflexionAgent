"""
Reflexion Agent - Production-ready AI agent with episodic + reflection memory.

A sophisticated agent that learns from its mistakes through a cycle of:
1. Act: Generate an answer
2. Evaluate: Score the answer
3. Reflect: Learn from failures
4. Repeat: Improve with accumulated knowledge
"""

__version__ = "1.0.0"

from reflexion_agent.agent import (
    ReflexionLoop,
    ReflexionResult,
    StopReason,
    Actor,
    Evaluator,
    Reflector,
    create_agent,
    create_agent_async,
)

from reflexion_agent.config import (
    Settings,
    EnvSettings,
    get_settings,
    get_config_for_env,
)

from reflexion_agent.providers import (
    BaseLLMProvider,
    Message,
    LLMResponse,
    AnthropicProvider,
    OpenAIProvider,
)

from reflexion_agent.memory import (
    Episode,
    ReflectionMemory,
    ShortTermMemory,
    BaseEpisodicStore,
    BaseReflectionStore,
)

from reflexion_agent.tools import (
    BaseTool,
    ToolResult,
    CalculatorTool,
    WebSearchTool,
    ToolRegistry,
)

from reflexion_agent.observability import (
    configure_logging,
    configure_tracing,
    get_logger,
    span,
)

from reflexion_agent.utils.exceptions import ReflexionError

# Convenience function for quick usage
async def quick_run(task: str, env: str = "development") -> ReflexionResult:
    """
    Quick one-shot task execution.
    
    Args:
        task: Task to perform
        env: Environment (development, staging, production)
        
    Returns:
        ReflexionResult
        
    Example:
        >>> import asyncio
        >>> from reflexion_agent import quick_run
        >>> result = asyncio.run(quick_run("What is 2+2?"))
        >>> print(result.final_answer)
        4
    """
    from reflexion_agent.main import run
    return await run(task, env=env)


__all__ = [
    # Version
    "__version__",
    
    # Main API
    "ReflexionLoop",
    "ReflexionResult",
    "StopReason",
    "quick_run",
    
    # Agent components
    "Actor",
    "Evaluator", 
    "Reflector",
    "create_agent",
    "create_agent_async",
    
    # Configuration
    "Settings",
    "EnvSettings",
    "get_settings",
    "get_config_for_env",
    
    # LLM Providers
    "BaseLLMProvider",
    "Message",
    "LLMResponse",
    "AnthropicProvider",
    "OpenAIProvider",
    
    # Memory
    "Episode",
    "ReflectionMemory",
    "ShortTermMemory",
    "BaseEpisodicStore",
    "BaseReflectionStore",
    
    # Tools
    "BaseTool",
    "ToolResult",
    "CalculatorTool",
    "WebSearchTool",
    "ToolRegistry",
    
    # Observability
    "configure_logging",
    "configure_tracing",
    "get_logger",
    "span",
    
    # Exceptions
    "ReflexionError",
]