import json
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings

from reflexion_agent.observability import get_logger, span
from reflexion_agent.memory.models import Episode
from reflexion_agent.memory.base import BaseEpisodicStore
from reflexion_agent.utils.exceptions import MemoryError

logger = get_logger(__name__)


class ChromaEpisodicStore(BaseEpisodicStore):
    """ChromaDB implementation of episodic memory store."""
    
    def __init__(
        self, 
        persist_directory: str, 
        collection_name: str,
        embedding_function = None  # Can add custom embeddings later
    ) -> None:
        """
        Initialize Chroma episodic store.
        
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
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            
            logger.info(
                "episodic_store_initialized",
                persist_directory=persist_directory,
                collection=collection_name,
                existing_items=self._collection.count()
            )
            
        except Exception as e:
            raise MemoryError(f"Failed to initialize Chroma episodic store: {e}")
    
    @span("episodic_store.add")
    async def add(self, episode: Episode) -> str:
        """
        Add an episode to the store.
        
        Args:
            episode: Episode to add
            
        Returns:
            Episode ID
        """
        try:
            # Prepare metadata (convert non-string values)
            metadata = {
                "final_score": episode.final_score,
                "iterations": episode.iterations,
                "succeeded": episode.succeeded,
                "created_at": episode.created_at.isoformat(),
                "final_answer_preview": episode.final_answer[:200],
            }
            
            # Add any additional metadata
            for key, value in episode.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"meta_{key}"] = value
            
            # Add tags if present
            if episode.tags:
                metadata["tags"] = ",".join(episode.tags)
            
            # Add to Chroma
            self._collection.add(
                ids=[episode.id],
                documents=[episode.task],
                metadatas=[metadata],
            )
            
            logger.debug(
                "episode_added",
                episode_id=episode.id,
                task_preview=episode.task[:50],
                score=episode.final_score
            )
            
            return episode.id
            
        except Exception as e:
            logger.error("failed_to_add_episode", error=str(e))
            raise MemoryError(f"Failed to add episode: {e}")
    
    @span("episodic_store.get")
    async def get(self, episode_id: str) -> Optional[Episode]:
        """Get an episode by ID."""
        try:
            results = self._collection.get(ids=[episode_id])
            
            if not results['ids']:
                return None
            
            # Reconstruct Episode from stored data
            metadata = results['metadatas'][0]
            task = results['documents'][0]
            
            # Parse metadata
            return Episode(
                id=episode_id,
                task=task,
                final_answer=metadata.get('final_answer_preview', ''),  # Note: truncated
                final_score=metadata['final_score'],
                iterations=metadata['iterations'],
                succeeded=metadata['succeeded'],
                created_at=datetime.fromisoformat(metadata['created_at']),
                metadata={
                    k.replace('meta_', ''): v 
                    for k, v in metadata.items() 
                    if k.startswith('meta_')
                },
                tags=metadata.get('tags', '').split(',') if metadata.get('tags') else []
            )
            
        except Exception as e:
            logger.error("failed_to_get_episode", episode_id=episode_id, error=str(e))
            raise MemoryError(f"Failed to get episode: {e}")
    
    @span("episodic_store.delete")
    async def delete(self, episode_id: str) -> bool:
        """Delete an episode by ID."""
        try:
            self._collection.delete(ids=[episode_id])
            logger.debug("episode_deleted", episode_id=episode_id)
            return True
        except Exception as e:
            logger.error("failed_to_delete_episode", episode_id=episode_id, error=str(e))
            return False
    
    @span("episodic_store.list")
    async def list(self, limit: int = 100, offset: int = 0) -> List[Episode]:
        """List episodes with pagination."""
        try:
            # Chroma doesn't support offset directly, so we get all and slice
            results = self._collection.get()
            
            episodes = []
            for i in range(offset, min(offset + limit, len(results['ids']))):
                episode = await self.get(results['ids'][i])
                if episode:
                    episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            logger.error("failed_to_list_episodes", error=str(e))
            raise MemoryError(f"Failed to list episodes: {e}")
    
    @span("episodic_store.count")
    async def count(self) -> int:
        """Get total number of episodes."""
        try:
            return self._collection.count()
        except Exception as e:
            logger.error("failed_to_count_episodes", error=str(e))
            return 0
    
    @span("episodic_store.clear")
    async def clear(self) -> None:
        """Clear all episodes."""
        try:
            # Delete and recreate collection
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("episodic_store_cleared", collection=self.collection_name)
        except Exception as e:
            logger.error("failed_to_clear_episodic_store", error=str(e))
            raise MemoryError(f"Failed to clear episodic store: {e}")
    
    @span("episodic_store.add_item")
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
            raise MemoryError(f"Failed to add item: {e}")
    
    @span("episodic_store.search")
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar episodes by task text."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            
            # Convert filter criteria to Chroma where clause
            where = None
            if filter_criteria:
                where = {}
                for key, value in filter_criteria.items():
                    if isinstance(value, (int, float)):
                        # Range queries would need more complex handling
                        where[key] = value
                    else:
                        where[key] = value
            
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
                where=where
            )
            
            # Format results
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'task': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })
            
            logger.debug(
                "episodic_search_completed",
                query=query[:50],
                results=len(formatted)
            )
            
            return formatted
            
        except Exception as e:
            logger.error("episodic_search_failed", error=str(e))
            raise MemoryError(f"Failed to search episodes: {e}")
    
    @span("episodic_store.search_by_embedding")
    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar episodes by embedding vector."""
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
                    'task': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })
            
            return formatted
            
        except Exception as e:
            logger.error("episodic_embedding_search_failed", error=str(e))
            raise MemoryError(f"Failed to search episodes by embedding: {e}")
    
    @span("episodic_store.find_similar_tasks")
    async def find_similar_tasks(
        self, 
        task: str, 
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Episode]:
        """Find episodes with similar tasks."""
        try:
            # Build filter if min_score provided
            filter_criteria = None
            if min_score is not None:
                filter_criteria = {"final_score": {"$gte": min_score}}
            
            results = await self.search(task, top_k, filter_criteria)
            
            episodes = []
            for result in results:
                episode = await self.get(result['id'])
                if episode:
                    episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            logger.error("failed_to_find_similar_tasks", error=str(e))
            return []
    
    @span("episodic_store.get_stats")
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the episodic store."""
        try:
            count = self._collection.count()
            
            # Get all metadata for stats
            if count > 0:
                results = self._collection.get()
                scores = [m['final_score'] for m in results['metadatas']]
                succeeded = [m['succeeded'] for m in results['metadatas']]
                
                avg_score = sum(scores) / len(scores) if scores else 0
                success_rate = sum(succeeded) / len(succeeded) if succeeded else 0
            else:
                avg_score = 0
                success_rate = 0
            
            return {
                "count": count,
                "avg_score": avg_score,
                "success_rate": success_rate,
                "collection": self.collection_name,
                "persist_directory": self.persist_directory,
            }
            
        except Exception as e:
            logger.error("failed_to_get_episodic_stats", error=str(e))
            return {"count": 0, "error": str(e)}
    
    async def delete_item(self, item_id: str) -> bool:
        """Delete an item by ID."""
        return await self.delete(item_id)


class InMemoryEpisodicStore(BaseEpisodicStore):
    """In-memory implementation for testing/development."""
    
    def __init__(self):
        self._episodes: Dict[str, Episode] = {}
        logger.info("in_memory_episodic_store_initialized")
    
    async def add(self, episode: Episode) -> str:
        self._episodes[episode.id] = episode
        logger.debug("episode_added_in_memory", episode_id=episode.id)
        return episode.id
    
    async def get(self, episode_id: str) -> Optional[Episode]:
        return self._episodes.get(episode_id)
    
    async def delete(self, episode_id: str) -> bool:
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            return True
        return False
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[Episode]:
        episodes = list(self._episodes.values())
        return episodes[offset:offset + limit]
    
    async def count(self) -> int:
        return len(self._episodes)
    
    async def clear(self) -> None:
        self._episodes.clear()
        logger.info("in_memory_episodic_store_cleared")
    
    async def add_item(self, item_id: str, text: str, metadata: Dict[str, Any], embedding=None) -> None:
        # In-memory version doesn't do embeddings
        pass
    
    async def search(self, query: str, top_k: int = 5, filter_criteria=None) -> List[Dict[str, Any]]:
        # Simple text search for in-memory
        results = []
        query_lower = query.lower()
        
        for ep_id, episode in self._episodes.items():
            if query_lower in episode.task.lower():
                results.append({
                    'id': ep_id,
                    'task': episode.task,
                    'metadata': {
                        'final_score': episode.final_score,
                        'succeeded': episode.succeeded,
                    }
                })
        
        return results[:top_k]
    
    async def search_by_embedding(self, embedding, top_k=5, filter_criteria=None) -> List[Dict[str, Any]]:
        # Not supported in memory
        return []
    
    async def find_similar_tasks(self, task: str, top_k: int = 5, min_score=None) -> List[Episode]:
        results = await self.search(task, top_k)
        episodes = []
        for r in results:
            ep = await self.get(r['id'])
            if ep and (min_score is None or ep.final_score >= min_score):
                episodes.append(ep)
        return episodes
    
    async def get_stats(self) -> Dict[str, Any]:
        episodes = list(self._episodes.values())
        if episodes:
            avg_score = sum(e.final_score for e in episodes) / len(episodes)
            success_rate = sum(1 for e in episodes if e.succeeded) / len(episodes)
        else:
            avg_score = 0
            success_rate = 0
        
        return {
            "count": len(self._episodes),
            "avg_score": avg_score,
            "success_rate": success_rate,
            "store_type": "in_memory",
        }
    
    async def delete_item(self, item_id: str) -> bool:
        return await self.delete(item_id)