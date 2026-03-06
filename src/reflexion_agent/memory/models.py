from datetime import datetime, timezone
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class Episode(BaseModel):
    """One complete Reflexion run."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    final_answer: str
    final_score: float = Field(ge=0.0, le=1.0)
    iterations: int = Field(ge=1)
    succeeded: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @property
    def summary(self) -> str:
        """Get a brief summary of the episode."""
        status = "✅" if self.succeeded else "❌"
        return f"{status} {self.task[:50]}... (score: {self.final_score:.2f})"


class IterationRecord(BaseModel):
    """A single Act → Evaluate → Reflect cycle within an episode."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str
    iteration: int = Field(ge=1)
    answer: str
    score: float = Field(ge=0.0, le=1.0)
    critique: str
    reflection: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used in this iteration."""
        return sum(self.token_usage.values())


class ReflectionMemory(BaseModel):
    """A distilled reflection retrievable for future tasks."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_summary: str  # Short description for embedding/search
    reflection_text: str  # Actual bullet-point lessons
    score_before: float = Field(ge=0.0, le=1.0)
    score_after: float = Field(ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_episode_id: Optional[str] = None
    source_iteration: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def improvement(self) -> float:
        """Calculate improvement from before to after."""
        return self.score_after - self.score_before
    
    @property
    def is_positive(self) -> bool:
        """Check if this reflection led to improvement."""
        return self.improvement > 0


class MemoryStats(BaseModel):
    """Statistics about memory stores."""
    episodic_count: int
    reflection_count: int
    total_iterations: int
    avg_score: float
    success_rate: float
    storage_size_bytes: int