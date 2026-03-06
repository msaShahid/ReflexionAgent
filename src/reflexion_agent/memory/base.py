from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generic, TypeVar

from reflexion_agent.memory.models import Episode, IterationRecord, ReflectionMemory
from reflexion_agent.utils.exceptions import MemoryError

T = TypeVar('T')


class BaseMemoryStore(ABC, Generic[T]):
    """Base interface for all memory stores."""
    
    @abstractmethod
    async def add(self, item: T) -> str:
        """Add an item to the store and return its ID."""
        pass
    
    @abstractmethod
    async def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID."""
        pass
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID. Returns True if deleted."""
        pass
    
    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List items with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total number of items."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all items from the store."""
        pass


class BaseVectorStore(ABC):
    """Base interface for vector similarity search stores."""
    
    @abstractmethod
    async def add_item(
        self, 
        item_id: str, 
        text: str, 
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add an item with text for embedding."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar items by text query."""
        pass
    
    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar items by embedding vector."""
        pass
    
    @abstractmethod
    async def delete_item(self, item_id: str) -> bool:
        """Delete an item by ID."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total number of items."""
        pass


class BaseEpisodicStore(BaseMemoryStore[Episode], BaseVectorStore):
    """Episodic memory store with vector search capabilities."""
    
    @abstractmethod
    async def find_similar_tasks(
        self, 
        task: str, 
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Episode]:
        """Find episodes with similar tasks."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the episodic store."""
        pass


class BaseReflectionStore(BaseMemoryStore[ReflectionMemory], BaseVectorStore):
    """Reflection memory store with vector search capabilities."""
    
    @abstractmethod
    async def find_relevant_reflections(
        self,
        task: str,
        top_k: int = 3,
        min_improvement: Optional[float] = None
    ) -> List[ReflectionMemory]:
        """Find reflections relevant to the current task."""
        pass
    
    @abstractmethod
    async def get_effective_reflections(
        self,
        min_improvement: float = 0.0
    ) -> List[ReflectionMemory]:
        """Get reflections that led to improvement."""
        pass