from typing import Optional

from reflexion_agent.config import Settings, EnvSettings
from reflexion_agent.providers.factory import build_all_providers
from reflexion_agent.memory.factory import build_all_memories
from reflexion_agent.tools.registry import ToolRegistry, get_tool_registry
from reflexion_agent.agent.actor import Actor
from reflexion_agent.agent.evaluator import Evaluator
from reflexion_agent.agent.reflector import Reflector
from reflexion_agent.agent.reflexion_loop import ReflexionLoop
from reflexion_agent.utils.exceptions import ConfigurationError
from reflexion_agent.observability import get_logger

logger = get_logger(__name__)


def create_agent(
    settings: Settings,
    env_settings: EnvSettings,
    enable_tools: bool = True
) -> ReflexionLoop:
    """
    Create a fully configured reflexion agent.
    
    Args:
        settings: Application settings
        env_settings: Environment settings with API keys
        enable_tools: Whether to enable tools
        
    Returns:
        Configured ReflexionLoop
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    logger.info("creating_reflexion_agent", name=settings.agent.name)
    
    try:
        # Build providers
        providers = build_all_providers(settings.llm, env_settings)
        
        # Build memory stores
        memories = build_all_memories(settings.memory)
        
        # Build tool registry if enabled
        tool_registry = None
        if enable_tools and settings.tools.enabled:
            tool_registry = get_tool_registry(env_settings)
            # Initialize tools (async, but we're in sync context)
            # In practice, you'd call await tool_registry.initialize()
            # We'll handle this in the main application
        
        # Create components
        actor = Actor(
            provider=providers['actor'],
            tool_registry=tool_registry
        )
        
        evaluator = Evaluator(
            provider=providers['evaluator'],
            stopping_score=settings.agent.stopping_score
        )
        
        reflector = Reflector(
            provider=providers['reflector']
        )
        
        # Create main loop
        loop = ReflexionLoop(
            actor=actor,
            evaluator=evaluator,
            reflector=reflector,
            episodic_store=memories['episodic'],
            reflection_store=memories['reflection'],
            settings=settings
        )
        
        logger.info("reflexion_agent_created")
        return loop
        
    except KeyError as e:
        raise ConfigurationError(f"Missing required component: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to create agent: {e}")


async def create_agent_async(
    settings: Settings,
    env_settings: EnvSettings,
    enable_tools: bool = True
) -> ReflexionLoop:
    """
    Create a fully configured reflexion agent with async initialization.
    
    This version properly initializes async components like tools.
    """
    agent = create_agent(settings, env_settings, enable_tools)
    
    # Initialize tools if enabled
    if enable_tools and settings.tools.enabled:
        tool_registry = get_tool_registry(env_settings)
        await tool_registry.initialize(settings.tools.enabled)
    
    return agent