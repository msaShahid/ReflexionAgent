from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings

from reflexion_agent.observability import get_logger, span
from reflexion_agent.memory.models import ReflectionMemory
from reflexion_agent.memory.base import BaseReflectionStore
from reflexion_agent.utils.exceptions import MemoryError

logger = get_logger(__name__)


class ChromaReflectionStore(BaseReflectionStore):
    """ChromaDB implementation of reflection memory store."""
    
    def __init__(
        self, 
        persist_directory: str, 
        collection_name: str,
        embedding_function = None
    ) -> None:
        """
        Initialize Chroma reflection store.
        
        Args:
            persist_directory: Directory to persist data
            collection_name: Name of the collection
            embedding_function: Optional custom embedding function
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            logger.info(
                "reflection_store_initialized",
                persist_directory=persist_directory,
                collection=collection_name,
                existing_items=self._collection.count()
            )
            
        except Exception as e:
            raise MemoryError(f"Failed to initialize Chroma reflection store: {e}")
    
    @span("reflection_store.add")
    async def add(self, reflection: ReflectionMemory) -> str:
        """Add a reflection to the store."""
        try:
            metadata = {
                "score_before": reflection.score_before,
                "score_after": reflection.score_after,
                "improvement": reflection.improvement,
                "is_positive": reflection.is_positive,
                "created_at": reflection.created_at.isoformat(),
                "source_episode_id": reflection.source_episode_id or "",
                "source_iteration": reflection.source_iteration or 0,
                "tags": ",".join(reflection.tags),
            }
            
            self._collection.add(
                ids=[reflection.id],
                documents=[reflection.task_summary],
                metadatas=[metadata],
            )
            
            logger.debug(
                "reflection_added",
                reflection_id=reflection.id,
                improvement=reflection.improvement,
                is_positive=reflection.is_positive
            )
            
            return reflection.id
            
        except Exception as e:
            logger.error("failed_to_add_reflection", error=str(e))
            raise MemoryError(f"Failed to add reflection: {e}")
    
    @span("reflection_store.get")
    async def get(self, reflection_id: str) -> Optional[ReflectionMemory]:
        """Get a reflection by ID."""
        try:
            results = self._collection.get(ids=[reflection_id])
            
            if not results['ids']:
                return None
            
            metadata = results['metadatas'][0]
            task_summary = results['documents'][0]
            
            # We don't store the full reflection_text in Chroma metadata
            # since it could be long. In a real implementation, you might
            # store it in a separate key-value store or in the document field.
            
            return ReflectionMemory(
                id=reflection_id,
                task_summary=task_summary,
                reflection_text="",  # Would need to retrieve from elsewhere
                score_before=metadata['score_before'],
                score_after=metadata['score_after'],
                tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                created_at=datetime.fromisoformat(metadata['created_at']),
                source_episode_id=metadata.get('source_episode_id') or None,
                source_iteration=metadata.get('source_iteration') or None,
            )
            
        except Exception as e:
            logger.error("failed_to_get_reflection", reflection_id=reflection_id, error=str(e))
            return None
    
    @span("reflection_store.delete")
    async def delete(self, reflection_id: str) -> bool:
        """Delete a reflection by ID."""
        try:
            self._collection.delete(ids=[reflection_id])
            return True
        except Exception:
            return False
    
    @span("reflection_store.list")
    async def list(self, limit: int = 100, offset: int = 0) -> List[ReflectionMemory]:
        """List reflections with pagination."""
        try:
            results = self._collection.get()
            
            reflections = []
            for i in range(offset, min(offset + limit, len(results['ids']))):
                reflection = await self.get(results['ids'][i])
                if reflection:
                    reflections.append(reflection)
            
            return reflections
            
        except Exception as e:
            logger.error("failed_to_list_reflections", error=str(e))
            return []
    
    @span("reflection_store.count")
    async def count(self) -> int:
        """Get total number of reflections."""
        try:
            return self._collection.count()
        except Exception:
            return 0
    
    @span("reflection_store.clear")
    async def clear(self) -> None:
        """Clear all reflections."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("reflection_store_cleared", collection=self.collection_name)
        except Exception as e:
            raise MemoryError(f"Failed to clear reflection store: {e}")
    
    @span("reflection_store.add_item")
    async def add_item(
        self, 
        item_id: str, 
        text: str, 
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add an item with text for embedding."""
        try:
            self._collection.add(
                ids=[item_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding] if embedding else None
            )
        except Exception as e:
            raise MemoryError(f"Failed to add reflection item: {e}")
    
    @span("reflection_store.search")
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant reflections by task similarity."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            
            where = None
            if filter_criteria:
                where = filter_criteria
            
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
                where=where
            )
            
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'task_summary': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })
            
            logger.debug(
                "reflection_search_completed",
                query=query[:50],
                results=len(formatted)
            )
            
            return formatted
            
        except Exception as e:
            logger.error("reflection_search_failed", error=str(e))
            return []
    
    @span("reflection_store.search_by_embedding")
    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for reflections by embedding vector."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            
            where = None
            if filter_criteria:
                where = filter_criteria
            
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, count),
                where=where
            )
            
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'task_summary': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })
            
            return formatted
            
        except Exception as e:
            logger.error("reflection_embedding_search_failed", error=str(e))
            return []
    
    @span("reflection_store.find_relevant_reflections")
    async def find_relevant_reflections(
        self,
        task: str,
        top_k: int = 3,
        min_improvement: Optional[float] = None
    ) -> List[ReflectionMemory]:
        """Find reflections relevant to the current task."""
        try:
            filter_criteria = None
            if min_improvement is not None:
                filter_criteria = {"improvement": {"$gte": min_improvement}}
            
            results = await self.search(task, top_k, filter_criteria)
            
            reflections = []
            for result in results:
                # In a real implementation, you'd fetch the full reflection
                # from a separate store using the ID
                reflection = await self.get(result['id'])
                if reflection:
                    reflections.append(reflection)
            
            return reflections
            
        except Exception as e:
            logger.error("failed_to_find_relevant_reflections", error=str(e))
            return []
    
    @span("reflection_store.get_effective_reflections")
    async def get_effective_reflections(
        self,
        min_improvement: float = 0.0
    ) -> List[ReflectionMemory]:
        """Get reflections that led to improvement."""
        try:
            # Get all reflections with improvement >= min_improvement
            results = self._collection.get()
            
            reflections = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata['improvement'] >= min_improvement:
                    reflection = await self.get(results['ids'][i])
                    if reflection:
                        reflections.append(reflection)
            
            return reflections
            
        except Exception as e:
            logger.error("failed_to_get_effective_reflections", error=str(e))
            return []
    
    async def delete_item(self, item_id: str) -> bool:
        """Delete an item by ID."""
        return await self.delete(item_id)


class InMemoryReflectionStore(BaseReflectionStore):
    """In-memory implementation for testing/development."""
    
    def __init__(self):
        self._reflections: Dict[str, ReflectionMemory] = {}
        logger.info("in_memory_reflection_store_initialized")
    
    async def add(self, reflection: ReflectionMemory) -> str:
        self._reflections[reflection.id] = reflection
        return reflection.id
    
    async def get(self, reflection_id: str) -> Optional[ReflectionMemory]:
        return self._reflections.get(reflection_id)
    
    async def delete(self, reflection_id: str) -> bool:
        if reflection_id in self._reflections:
            del self._reflections[reflection_id]
            return True
        return False
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[ReflectionMemory]:
        reflections = list(self._reflections.values())
        return reflections[offset:offset + limit]
    
    async def count(self) -> int:
        return len(self._reflections)
    
    async def clear(self) -> None:
        self._reflections.clear()
    
    async def add_item(self, item_id: str, text: str, metadata: Dict[str, Any], embedding=None) -> None:
        pass
    
    async def search(self, query: str, top_k: int = 5, filter_criteria=None) -> List[Dict[str, Any]]:
        # Simple text search
        results = []
        query_lower = query.lower()
        
        for ref_id, reflection in self._reflections.items():
            if query_lower in reflection.task_summary.lower():
                if filter_criteria:
                    # Apply filters
                    match = True
                    for key, value in filter_criteria.items():
                        if key == 'improvement' and isinstance(value, dict):
                            if '$gte' in value and reflection.improvement < value['$gte']:
                                match = False
                        elif hasattr(reflection, key):
                            if getattr(reflection, key) != value:
                                match = False
                    
                    if not match:
                        continue
                
                results.append({
                    'id': ref_id,
                    'task_summary': reflection.task_summary,
                    'metadata': {
                        'score_before': reflection.score_before,
                        'score_after': reflection.score_after,
                        'improvement': reflection.improvement,
                    }
                })
        
        return results[:top_k]
    
    async def search_by_embedding(self, embedding, top_k=5, filter_criteria=None) -> List[Dict[str, Any]]:
        return []  # Not supported
    
    async def find_relevant_reflections(self, task: str, top_k=3, min_improvement=None) -> List[ReflectionMemory]:
        results = await self.search(task, top_k)
        reflections = []
        for r in results:
            ref = await self.get(r['id'])
            if ref and (min_improvement is None or ref.improvement >= min_improvement):
                reflections.append(ref)
        return reflections
    
    async def get_effective_reflections(self, min_improvement: float = 0.0) -> List[ReflectionMemory]:
        return [r for r in self._reflections.values() if r.improvement >= min_improvement]
    
    async def delete_item(self, item_id: str) -> bool:
        return await self.delete(item_id)
