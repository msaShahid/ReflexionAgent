"""
Agent module for Reflexion Agent.
Implements the Actor-Evaluator-Reflector loop with memory.
"""

from reflexion_agent.agent.models import (
    StopReason,
    ActorOutput,
    EvaluatorOutput,
    ReflectorOutput,
    IterationResult,
    ReflexionResult,
)

from reflexion_agent.agent.actor import Actor
from reflexion_agent.agent.evaluator import Evaluator
from reflexion_agent.agent.reflector import Reflector
from reflexion_agent.agent.reflexion_loop import ReflexionLoop
from reflexion_agent.agent.factory import create_agent, create_agent_async

__all__ = [
    # Models
    "StopReason",
    "ActorOutput",
    "EvaluatorOutput",
    "ReflectorOutput",
    "IterationResult",
    "ReflexionResult",
    
    # Components
    "Actor",
    "Evaluator",
    "Reflector",
    "ReflexionLoop",
    
    # Factory
    "create_agent",
    "create_agent_async",
]