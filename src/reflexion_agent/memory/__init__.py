"""
Memory module for Reflexion Agent.
Provides episodic, reflection, and short-term memory storage.
"""

from reflexion_agent.memory.models import (
    Episode,
    IterationRecord,
    ReflectionMemory,
    MemoryStats,
)

from reflexion_agent.memory.base import (
    BaseMemoryStore,
    BaseVectorStore,
    BaseEpisodicStore,
    BaseReflectionStore,
)

from reflexion_agent.memory.episodic_store import (
    ChromaEpisodicStore,
    InMemoryEpisodicStore,
)

from reflexion_agent.memory.reflection_store import (
    ChromaReflectionStore,
    InMemoryReflectionStore,
)

from reflexion_agent.memory.short_term import ShortTermMemory

from reflexion_agent.memory.factory import (
    build_episodic_store,
    build_reflection_store,
    build_short_term_memory,
    build_all_memories,
    create_test_memories,
)

__all__ = [
    # Models
    "Episode",
    "IterationRecord",
    "ReflectionMemory",
    "MemoryStats",
    
    # Base interfaces
    "BaseMemoryStore",
    "BaseVectorStore",
    "BaseEpisodicStore",
    "BaseReflectionStore",
    
    # Episodic stores
    "ChromaEpisodicStore",
    "InMemoryEpisodicStore",
    
    # Reflection stores
    "ChromaReflectionStore",
    "InMemoryReflectionStore",
    
    # Short-term memory
    "ShortTermMemory",
    
    # Factory functions
    "build_episodic_store",
    "build_reflection_store",
    "build_short_term_memory",
    "build_all_memories",
    "create_test_memories",
]