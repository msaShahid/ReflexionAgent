from typing import Union, Dict, Any
from reflexion_agent.config import MemoryConfig
from reflexion_agent.memory.base import BaseEpisodicStore, BaseReflectionStore
from reflexion_agent.memory.episodic_store import ChromaEpisodicStore, InMemoryEpisodicStore
from reflexion_agent.memory.reflection_store import ChromaReflectionStore, InMemoryReflectionStore
from reflexion_agent.memory.short_term import ShortTermMemory
from reflexion_agent.utils.exceptions import ConfigurationError
from reflexion_agent.observability import get_logger

logger = get_logger(__name__)


def build_episodic_store(config: MemoryConfig) -> BaseEpisodicStore:
    """
    Build episodic memory store from configuration.
    
    Args:
        config: Memory configuration
        
    Returns:
        Configured episodic store
        
    Raises:
        ConfigurationError: If backend not supported
    """
    backend = config.episodic.backend
    
    if backend == "chroma":
        if not config.episodic.persist_directory:
            raise ConfigurationError("persist_directory required for chroma episodic store")
        
        logger.info(
            "building_chroma_episodic_store",
            persist_directory=config.episodic.persist_directory,
            collection=config.episodic.collection_name
        )
        
        return ChromaEpisodicStore(
            persist_directory=config.episodic.persist_directory,
            collection_name=config.episodic.collection_name,
        )
    
    elif backend == "in_memory":
        logger.info("building_in_memory_episodic_store")
        return InMemoryEpisodicStore()
    
    else:
        raise ConfigurationError(f"Unsupported episodic backend: {backend}")


def build_reflection_store(config: MemoryConfig) -> BaseReflectionStore:
    """
    Build reflection memory store from configuration.
    
    Args:
        config: Memory configuration
        
    Returns:
        Configured reflection store
        
    Raises:
        ConfigurationError: If backend not supported
    """
    backend = config.reflection.backend
    
    if backend == "chroma":
        if not config.reflection.persist_directory:
            raise ConfigurationError("persist_directory required for chroma reflection store")
        
        logger.info(
            "building_chroma_reflection_store",
            persist_directory=config.reflection.persist_directory,
            collection=config.reflection.collection_name
        )
        
        return ChromaReflectionStore(
            persist_directory=config.reflection.persist_directory,
            collection_name=config.reflection.collection_name,
        )
    
    elif backend == "in_memory":
        logger.info("building_in_memory_reflection_store")
        return InMemoryReflectionStore()
    
    else:
        raise ConfigurationError(f"Unsupported reflection backend: {backend}")


def build_short_term_memory(config: MemoryConfig) -> ShortTermMemory:
    """
    Build short-term memory from configuration.
    
    Args:
        config: Memory configuration
        
    Returns:
        Configured short-term memory
    """
    max_tokens = config.short_term.get("max_tokens", 8192)
    logger.info("building_short_term_memory", max_tokens=max_tokens)
    return ShortTermMemory(max_tokens=max_tokens)


def build_all_memories(config: MemoryConfig) -> Dict[str, Any]:
    """
    Build all memory components from configuration.
    
    Args:
        config: Memory configuration
        
    Returns:
        Dictionary with episodic, reflection, and short_term memories
    """
    return {
        "episodic": build_episodic_store(config),
        "reflection": build_reflection_store(config),
        "short_term": build_short_term_memory(config),
    }


# Convenience function for quick testing
def create_test_memories():
    """Create in-memory stores for testing."""
    from reflexion_agent.config import MemoryConfig
    
    config = MemoryConfig(
        episodic={"backend": "in_memory", "collection_name": "test_episodes", "similarity_top_k": 5},
        reflection={"backend": "in_memory", "collection_name": "test_reflections", "similarity_top_k": 3},
        short_term={"max_tokens": 4096}
    )
    
    return build_all_memories(config)