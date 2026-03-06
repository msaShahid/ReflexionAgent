"""
Prompt templates for Reflexion Agent.
Centralized management of all prompts used by the agent.
"""

from reflexion_agent.prompts.templates import (
    # Actor prompts
    ACTOR_SYSTEM,
    ACTOR_USER,
    
    # Evaluator prompts
    EVALUATOR_SYSTEM,
    EVALUATOR_USER,
    DEFAULT_EVALUATION_CRITERIA,
    
    # Reflector prompts
    REFLECTOR_SYSTEM,
    REFLECTOR_USER,
    REFLECTOR_SIMPLE_USER,
    
    # Tool prompts
    TOOL_CALLING_SYSTEM,
    TOOL_CALLING_USER,
    
    # Memory prompts
    MEMORY_COMPRESSION_SYSTEM,
    MEMORY_COMPRESSION_USER,
    
    # Formatting helpers
    format_tools_for_prompt,
    format_reflections_for_prompt,
    format_episode_context,
    
    # Template management
    TEMPLATE_REGISTRY,
    get_template,
    render_template,
)

__all__ = [
    # Actor prompts
    "ACTOR_SYSTEM",
    "ACTOR_USER",
    
    # Evaluator prompts
    "EVALUATOR_SYSTEM",
    "EVALUATOR_USER",
    "DEFAULT_EVALUATION_CRITERIA",
    
    # Reflector prompts
    "REFLECTOR_SYSTEM",
    "REFLECTOR_USER",
    "REFLECTOR_SIMPLE_USER",
    
    # Tool prompts
    "TOOL_CALLING_SYSTEM",
    "TOOL_CALLING_USER",
    
    # Memory prompts
    "MEMORY_COMPRESSION_SYSTEM",
    "MEMORY_COMPRESSION_USER",
    
    # Formatting helpers
    "format_tools_for_prompt",
    "format_reflections_for_prompt",
    "format_episode_context",
    
    # Template management
    "TEMPLATE_REGISTRY",
    "get_template",
    "render_template",
]