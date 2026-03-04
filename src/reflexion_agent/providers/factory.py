from typing import Dict, Optional
from string import Template

from reflexion_agent.config import EnvSettings
from reflexion_agent.config.settings import LLMRoleConfig
from reflexion_agent.utils.exceptions import ConfigurationError, ProviderError
from reflexion_agent.observability import get_logger

from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .base import BaseLLMProvider, Message

logger = get_logger(__name__)


# Provider registry for extensibility
_PROVIDER_REGISTRY: Dict[str, type] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def register_provider(name: str, provider_class: type) -> None:
    """
    Register a new provider class.
    
    Args:
        name: Provider name (e.g., "gemini")
        provider_class: Provider class (must subclass BaseLLMProvider)
    """
    if not issubclass(provider_class, BaseLLMProvider):
        raise ValueError(f"Provider class must subclass BaseLLMProvider")
    _PROVIDER_REGISTRY[name] = provider_class
    logger.info("provider_registered", provider=name, class_name=provider_class.__name__)


def build_provider(role_cfg: LLMRoleConfig, env: EnvSettings) -> BaseLLMProvider:
    """
    Build an LLM provider based on configuration.
    
    Args:
        role_cfg: LLM role configuration
        env: Environment settings with API keys
        
    Returns:
        Initialized LLM provider
        
    Raises:
        ConfigurationError: If provider not found or API key missing
    """
    provider_name = role_cfg.provider
    
    # Check if provider is registered
    if provider_name not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ConfigurationError(
            f"Unknown provider: {provider_name}. Available providers: {available}"
        )
    
    # Get API key based on provider
    api_key = None
    if provider_name == "anthropic":
        api_key = env.anthropic_api_key
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set in .env")
    elif provider_name == "openai":
        api_key = env.openai_api_key
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set in .env")
    # Add other providers as needed
    
    try:
        provider_class = _PROVIDER_REGISTRY[provider_name]
        provider = provider_class(
            model=role_cfg.model,
            api_key=api_key,
            # max_retries could be passed from config
        )
        logger.info(
            "provider_initialized",
            provider=provider_name,
            model=role_cfg.model,
            role=role_cfg.__class__.__name__,
        )
        return provider
    except Exception as e:
        raise ProviderError(f"Failed to initialize {provider_name} provider: {e}")


def build_all_providers(llm_config, env: EnvSettings) -> Dict[str, BaseLLMProvider]:
    """
    Build all providers based on LLM configuration.
    
    Args:
        llm_config: LLM configuration object
        env: Environment settings
        
    Returns:
        Dictionary mapping role names to providers
    """
    providers = {}
    
    if hasattr(llm_config, 'actor'):
        providers['actor'] = build_provider(llm_config.actor, env)
    
    if hasattr(llm_config, 'evaluator'):
        providers['evaluator'] = build_provider(llm_config.evaluator, env)
    
    if hasattr(llm_config, 'reflector'):
        providers['reflector'] = build_provider(llm_config.reflector, env)
    
    return providers


# ── PROMPT TEMPLATES ───────────────────────────────────────────────────

# System prompts
ACTOR_SYSTEM = """You are an intelligent agent tasked with solving problems. 
You have access to tools and memory of past experiences. 
Provide clear, accurate, and helpful responses."""

EVALUATOR_SYSTEM = """You are a rigorous evaluator. Score the agent's response on a scale of 0.0 to 1.0.
Consider: correctness, completeness, clarity, and helpfulness.
Provide specific, actionable feedback for improvement."""

REFLECTOR_SYSTEM = """
You are a self-improvement strategist. Analyse why an agent's previous
attempt failed and produce actionable lessons for the next attempt.
Focus on SPECIFIC changes to strategy, not vague advice.
"""

# User prompts with templates
ACTOR_USER = Template("""
## Task
$task

## Available Tools
$tools

## Previous Attempts (if any)
$previous_attempts

## Lessons Learned
$reflections

Please solve this task.
""")

EVALUATOR_USER = Template("""
## Task
$task

## Agent's Response
$response

Please evaluate this response and provide:
1. A score from 0.0 to 1.0
2. Specific strengths
3. Specific weaknesses
4. Suggestions for improvement

Format your response as JSON:
{
    "score": 0.85,
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "suggestions": ["...", "..."]
}
""")

REFLECTOR_USER = Template("""
## Task
$task

## Previous Attempt (iteration $iteration)
$previous_answer

## Evaluator Critique
$critique

## Previous Reflections
$previous_reflections

Write 3-7 bullet points the agent should internalize for its next attempt.
Be specific. Address each weakness directly.
""")


def create_messages(
    system_prompt: str,
    user_template: Template,
    **kwargs
) -> list[Message]:
    """
    Create a list of messages from templates.
    
    Args:
        system_prompt: System prompt
        user_template: Template for user message
        **kwargs: Variables to fill in the template
        
    Returns:
        List of Message objects
    """
    messages = []
    
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    
    user_content = user_template.substitute(**kwargs)
    messages.append(Message(role="user", content=user_content))
    
    return messages
