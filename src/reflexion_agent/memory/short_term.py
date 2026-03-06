
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

from reflexion_agent.observability import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in short-term memory."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def format_for_context(self) -> str:
        """Format entry for inclusion in context."""
        prefix = f"[{self.role.upper()}]"
        if self.metadata.get("tool_name"):
            prefix = f"[TOOL:{self.metadata['tool_name']}]"
        return f"{prefix} {self.content}"


@dataclass
class ShortTermMemory:
    """
    Short-term memory with token budgeting and smart eviction.
    
    Maintains conversation context within token limits, evicting oldest
    entries first when budget is exceeded.
    """
    
    max_tokens: int = 8192
    _entries: List[MemoryEntry] = field(default_factory=list)
    _token_count: int = 0
    _preserve_system: bool = True  # Whether to keep system messages during eviction
    
    def add(
        self, 
        role: str, 
        content: str, 
        tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new entry to short-term memory.
        
        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            tokens: Exact token count (if known), otherwise estimated
            metadata: Additional metadata about the entry
        """
        # Estimate tokens if not provided
        if tokens <= 0:
            # Rough estimate: ~4 chars per token for English
            tokens = max(len(content) // 4, 1)
        
        # Create entry
        entry = MemoryEntry(
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {}
        )
        
        self._entries.append(entry)
        self._token_count += tokens
        
        # Evict if needed
        self._evict_if_needed()
        
        logger.debug(
            "memory_entry_added",
            role=role,
            tokens=tokens,
            total_entries=len(self._entries),
            total_tokens=self._token_count
        )
    
    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add multiple messages at once.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        """
        for msg in messages:
            self.add(
                role=msg.get('role', 'user'),
                content=msg.get('content', ''),
                tokens=msg.get('tokens', 0),
                metadata=msg.get('metadata', {})
            )
    
    def _evict_if_needed(self) -> None:
        """Remove oldest entries until under token budget."""
        if self._token_count <= self.max_tokens:
            return
        
        # Track removed entries for logging
        removed_count = 0
        removed_tokens = 0
        
        while self._token_count > self.max_tokens and len(self._entries) > 1:
            # Find first non-system message to evict (if preserving system)
            if self._preserve_system:
                # Look for first non-system entry
                evict_idx = None
                for i, entry in enumerate(self._entries):
                    if entry.role != "system":
                        evict_idx = i
                        break
                
                if evict_idx is None:
                    # Only system messages left - can't evict
                    break
            else:
                # Evict oldest
                evict_idx = 0
            
            # Remove the entry
            removed = self._entries.pop(evict_idx)
            self._token_count -= removed.tokens
            removed_count += 1
            removed_tokens += removed.tokens
        
        if removed_count > 0:
            logger.debug(
                "memory_evicted",
                removed_entries=removed_count,
                removed_tokens=removed_tokens,
                remaining_entries=len(self._entries),
                remaining_tokens=self._token_count
            )
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get formatted context string within token limit.
        
        Args:
            max_tokens: Optional override for max tokens
            
        Returns:
            Formatted context string
        """
        if not self._entries:
            return "(no context)"
        
        limit = max_tokens or self.max_tokens
        
        # If we're under limit, return all entries
        if self._token_count <= limit:
            return "\n".join(entry.format_for_context() for entry in self._entries)
        
        # Otherwise, we need to truncate
        # Start from most recent and work backwards
        entries = []
        tokens_used = 0
        
        for entry in reversed(self._entries):
            if tokens_used + entry.tokens > limit:
                # Can't fit this entry, add truncation notice
                entries.append(MemoryEntry(
                    role="system",
                    content=f"... (context truncated, {len(self._entries) - len(entries)} older messages omitted)",
                    tokens=0
                ))
                break
            
            entries.insert(0, entry)  # Put back in original order
            tokens_used += entry.tokens
        
        return "\n".join(entry.format_for_context() for entry in entries)
    
    def get_messages(self, format: str = "openai") -> List[Dict[str, Any]]:
        """
        Get messages in format suitable for LLM APIs.
        
        Args:
            format: Output format ("openai", "anthropic", or "raw")
            
        Returns:
            List of message dictionaries
        """
        if format == "openai":
            return [
                {"role": e.role, "content": e.content}
                for e in self._entries
            ]
        elif format == "anthropic":
            # Anthropic uses a different format
            messages = []
            for e in self._entries:
                if e.role == "system":
                    # System messages handled separately in Anthropic
                    continue
                messages.append({"role": e.role, "content": e.content})
            return messages
        else:
            return [e.to_dict() for e in self._entries]
    
    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Get the n most recent entries."""
        return self._entries[-n:]
    
    def get_by_role(self, role: str) -> List[MemoryEntry]:
        """Get all entries with a specific role."""
        return [e for e in self._entries if e.role == role]
    
    def search(self, query: str, case_sensitive: bool = False) -> List[MemoryEntry]:
        """
        Search memory entries by content.
        
        Args:
            query: Search query
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching entries
        """
        if not case_sensitive:
            query = query.lower()
            return [
                e for e in self._entries
                if query in e.content.lower()
            ]
        else:
            return [
                e for e in self._entries
                if query in e.content
            ]
    
    def clear(self) -> None:
        """Clear all entries from memory."""
        self._entries.clear()
        self._token_count = 0
        logger.debug("short_term_memory_cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory state."""
        if not self._entries:
            return {
                "total_entries": 0,
                "total_tokens": 0,
                "utilization": 0.0,
                "roles": {}
            }
        
        # Count by role
        roles = {}
        for entry in self._entries:
            roles[entry.role] = roles.get(entry.role, 0) + 1
        
        return {
            "total_entries": len(self._entries),
            "total_tokens": self._token_count,
            "max_tokens": self.max_tokens,
            "utilization": self._token_count / self.max_tokens,
            "roles": roles,
            "oldest_timestamp": min(e.timestamp for e in self._entries).isoformat(),
            "newest_timestamp": max(e.timestamp for e in self._entries).isoformat(),
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save memory contents to a JSON file."""
        data = {
            "max_tokens": self.max_tokens,
            "entries": [e.to_dict() for e in self._entries],
            "total_tokens": self._token_count,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("memory_saved_to_file", filepath=filepath, entries=len(self._entries))
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ShortTermMemory':
        """Load memory contents from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        memory = cls(max_tokens=data.get('max_tokens', 8192))
        for entry_data in data.get('entries', []):
            memory.add(
                role=entry_data['role'],
                content=entry_data['content'],
                tokens=entry_data['tokens'],
                metadata=entry_data.get('metadata', {})
            )
        
        logger.info("memory_loaded_from_file", filepath=filepath, entries=memory.count)
        return memory
    
    @property
    def count(self) -> int:
        """Get number of entries in memory."""
        return len(self._entries)
    
    @property
    def token_count(self) -> int:
        """Get current token count."""
        return self._token_count
    
    @property
    def utilization(self) -> float:
        """Get memory utilization as a percentage."""
        return self._token_count / self.max_tokens if self.max_tokens > 0 else 0
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __getitem__(self, idx: int) -> MemoryEntry:
        return self._entries[idx]
    
    def __iter__(self):
        return iter(self._entries)


# Convenience factory function
def create_short_term_memory(
    max_tokens: int = 8192,
    preserve_system: bool = True
) -> ShortTermMemory:
    """
    Create a new short-term memory instance.
    
    Args:
        max_tokens: Maximum token budget
        preserve_system: Whether to preserve system messages during eviction
        
    Returns:
        Configured ShortTermMemory instance
    """
    return ShortTermMemory(
        max_tokens=max_tokens,
        _preserve_system=preserve_system
    )
