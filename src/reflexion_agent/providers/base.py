from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation with an LLM."""
    role: str  # "system" | "user" | "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format expected by APIs."""
        return {"role": self.role, "content": self.content}


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """Get completion tokens used."""
        return self.usage.get("completion_tokens", 0)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with generated content and usage info
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """
        Count total tokens in a list of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.content)
            # Add overhead for message formatting (approximate)
            total += 4  # ~4 tokens for role and formatting
        return total
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.close()
    
    async def close(self):
        """Close any open resources."""
        # Base implementation does nothing
        pass