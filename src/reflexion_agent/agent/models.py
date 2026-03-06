from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StopReason(str, Enum):
    """Reasons for stopping the reflexion loop."""
    MAX_ITERATIONS = "max_iterations"
    SCORE_THRESHOLD = "score_threshold"
    NO_IMPROVEMENT = "no_improvement"
    TIMEOUT = "timeout"
    ERROR = "error"


class ActorOutput(BaseModel):
    """Output from the actor."""
    answer: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorOutput(BaseModel):
    """Output from the evaluator."""
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    critique: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReflectorOutput(BaseModel):
    """Output from the reflector."""
    reflection: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IterationResult(BaseModel):
    """Result of a single iteration in the reflexion loop."""
    iteration: int
    answer: str
    score: float
    critique: str
    reflection: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    duration_ms: float


class ReflexionResult(BaseModel):
    """Final result of the reflexion loop."""
    task: str
    final_answer: str
    final_score: float = Field(ge=0.0, le=1.0)
    iterations_used: int
    stop_reason: StopReason
    succeeded: bool
    episode_id: Optional[str] = None
    history: List[IterationResult] = Field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)