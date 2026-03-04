"""
LLM provider implementations for Reflexion Agent.
Supports OpenAI, Anthropic, and extensible for other providers.
"""

from reflexion_agent.providers.base import BaseLLMProvider, LLMResponse, Message
from reflexion_agent.providers.anthropic_provider import AnthropicProvider
from reflexion_agent.providers.openai_provider import OpenAIProvider
from reflexion_agent.providers.factory import (
    build_provider,
    build_all_providers,
    register_provider,
    create_messages,
    # System prompts
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    # User templates
    ACTOR_USER,
    EVALUATOR_USER,
    REFLECTOR_USER,
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMResponse",
    "Message",
    
    # Provider implementations
    "AnthropicProvider",
    "OpenAIProvider",
    
    # Factory functions
    "build_provider",
    "build_all_providers",
    "register_provider",
    "create_messages",
    
    # System prompts
    "ACTOR_SYSTEM",
    "EVALUATOR_SYSTEM",
    "REFLECTOR_SYSTEM",
    
    # User templates
    "ACTOR_USER",
    "EVALUATOR_USER",
    "REFLECTOR_USER",
]