"""
Prompt templates for Reflexion Agent.
All prompts are stored here for easy customization and versioning.
"""
from string import Template
from typing import Dict, Any, Optional


# ── ACTOR PROMPTS ───────────────────────────────────────────────────────
ACTOR_SYSTEM = """You are an expert AI agent. Your goal is to solve the task given by the user.
You have access to tools and prior reflection notes to guide your approach.
Think step by step. Be thorough and precise.

Guidelines:
- Break down complex problems into steps
- Use available tools when helpful
- Learn from previous mistakes (reflections)
- Provide complete, well-reasoned answers
- If unsure, acknowledge limitations
"""

ACTOR_USER = Template("""
## Task
$task

## Prior Reflections (most relevant lessons from past failures)
$reflections

## Short-Term Memory (context from current episode)
$short_term_memory

## Available Tools
$tools

## Current Iteration
This is attempt #$iteration out of $max_iterations.

Please solve this task. Provide a complete, well-structured answer.
""")


# ── EVALUATOR PROMPTS ───────────────────────────────────────────────────
EVALUATOR_SYSTEM = """You are a strict evaluator. Assess the quality of an AI agent's answer.
Output ONLY a valid JSON object with exactly these fields:
  - score: float between 0.0 and 1.0
  - passed: bool (True if score >= 0.7)
  - critique: str (specific, actionable feedback)
  - strengths: list[str] (what was done well)
  - weaknesses: list[str] (what needs improvement)
  - suggestions: list[str] (how to improve)

No markdown fences. No preamble. JSON only.
"""

EVALUATOR_USER = Template("""
## Original Task
$task

## Agent's Answer
$answer

## Evaluation Criteria
$criteria

## Previous Attempts (if any)
$previous_attempts

Return JSON only.
""")

# Default evaluation criteria if none provided
DEFAULT_EVALUATION_CRITERIA = """
- Correctness: Is the answer factually accurate?
- Completeness: Does it fully address the task?
- Clarity: Is it well-structured and easy to understand?
- Reasoning: Is the logic sound and well-explained?
- Actionability: Does it provide practical value?
"""


# ── REFLECTOR PROMPTS ───────────────────────────────────────────────────
REFLECTOR_SYSTEM = """You are a self-improvement strategist. Analyze why an agent's previous
attempt failed and produce actionable lessons for the next attempt.
Focus on SPECIFIC changes to strategy, not vague advice.

Output format (JSON):
{
    "lessons": ["bullet point 1", "bullet point 2", ...],
    "strategies": ["specific strategy 1", "specific strategy 2", ...],
    "avoid_patterns": ["pattern to avoid 1", ...]
}
"""

REFLECTOR_USER = Template("""
## Task
$task

## Previous Attempt (iteration $iteration)
$previous_answer

## Evaluator Critique
$critique
Score: $score

## Previous Reflections
$previous_reflections

## Task Context
$context

Analyze this failure and generate 3-7 specific, actionable lessons.
Focus on what the agent should do DIFFERENTLY next time.
""")

# Alternative simpler reflector format (for faster models)
REFLECTOR_SIMPLE_USER = Template("""
Task: $task
Previous answer: $previous_answer
Critique: $critique

Write 3-5 bullet points for what to do differently next time.
""")


# ── TOOL PROMPTS ───────────────────────────────────────────────────────
TOOL_CALLING_SYSTEM = """You have access to tools. When you need to use a tool,
respond with a JSON block containing the tool name and arguments.

Format:
{
    "tool": "tool_name",
    "args": {"arg1": "value1", "arg2": "value2"}
}

Available tools:
$tools_description

After getting the tool result, continue with your response.
"""

TOOL_CALLING_USER = Template("""
Task: $task
Current step: $current_step

Do you need to use a tool? If yes, respond with the tool JSON.
If not, continue with your answer.
""")


# ── MEMORY PROMPTS ─────────────────────────────────────────────────────
MEMORY_COMPRESSION_SYSTEM = """You are a memory compression expert.
Summarize the following episode into key insights for future reference.
Focus on:
- What worked well
- What failed
- Important context
- Key decisions
"""

MEMORY_COMPRESSION_USER = Template("""
## Episode Summary
Task: $task
Attempts: $num_attempts
Final score: $final_score

## Full Episode Log
$episode_log

Compress this into a concise reflection (max 200 words) for long-term memory.
""")


# ── FORMATTING HELPERS ─────────────────────────────────────────────────

def format_tools_for_prompt(tools: Dict[str, Any]) -> str:
    """
    Format tools dictionary into a readable string for prompts.
    
    Args:
        tools: Dictionary of tool names to tool objects/descriptions
        
    Returns:
        Formatted tools description
    """
    if not tools:
        return "No tools available."
    
    lines = []
    for name, tool in tools.items():
        description = getattr(tool, 'description', str(tool))
        lines.append(f"- {name}: {description}")
    
    return "\n".join(lines)


def format_reflections_for_prompt(reflections: list[str], max_reflections: int = 5) -> str:
    """
    Format reflection list into a readable string.
    
    Args:
        reflections: List of reflection strings
        max_reflections: Maximum number to include
        
    Returns:
        Formatted reflections
    """
    if not reflections:
        return "No previous reflections."
    
    # Take only the most recent/most relevant
    recent = reflections[-max_reflections:]
    
    lines = []
    for i, reflection in enumerate(recent, 1):
        lines.append(f"{i}. {reflection}")
    
    return "\n".join(lines)


def format_episode_context(episode: Dict[str, Any]) -> str:
    """
    Format episode context for prompts.
    
    Args:
        episode: Episode dictionary with attempts, scores, etc.
        
    Returns:
        Formatted context string
    """
    parts = []
    
    if episode.get('attempts'):
        parts.append(f"Attempts made: {len(episode['attempts'])}")
    
    if episode.get('best_score'):
        parts.append(f"Best score so far: {episode['best_score']:.2f}")
    
    if episode.get('duration'):
        parts.append(f"Time elapsed: {episode['duration']:.1f}s")
    
    return " | ".join(parts) if parts else "No context available."


# Template registry for easy access
TEMPLATE_REGISTRY = {
    'actor': {
        'system': ACTOR_SYSTEM,
        'user': ACTOR_USER,
    },
    'evaluator': {
        'system': EVALUATOR_SYSTEM,
        'user': EVALUATOR_USER,
        'default_criteria': DEFAULT_EVALUATION_CRITERIA,
    },
    'reflector': {
        'system': REFLECTOR_SYSTEM,
        'user': REFLECTOR_USER,
        'simple_user': REFLECTOR_SIMPLE_USER,
    },
    'tool': {
        'system': TOOL_CALLING_SYSTEM,
        'user': TOOL_CALLING_USER,
    },
    'memory': {
        'system': MEMORY_COMPRESSION_SYSTEM,
        'user': MEMORY_COMPRESSION_USER,
    },
}


def get_template(template_type: str, template_name: str = 'user'):
    """
    Get a template by type and name.
    
    Args:
        template_type: 'actor', 'evaluator', 'reflector', etc.
        template_name: 'system', 'user', 'simple_user', etc.
        
    Returns:
        Template object or string
        
    Raises:
        KeyError: If template not found
    """
    return TEMPLATE_REGISTRY[template_type][template_name]


def render_template(template_type: str, template_name: str, **kwargs) -> str:
    """
    Render a template with variables.
    
    Args:
        template_type: Template category
        template_name: Specific template name
        **kwargs: Variables to substitute
        
    Returns:
            Rendered string
    """
    template = get_template(template_type, template_name)
    
    if isinstance(template, Template):
        return template.substitute(**kwargs)
    return template  # System prompts are strings, not Templates
