from typing import List, Optional, Dict, Any

import anthropic
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

from reflexion_agent.observability import get_logger, span
from reflexion_agent.utils.exceptions import LLMError, ProviderError
from .base import BaseLLMProvider, LLMResponse, Message

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    # Claude models and their approximate context windows
    MODEL_CONTEXT_WINDOWS = {
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2.1": 200000,
        "claude-2.0": 100000,
        "claude-instant": 100000,
    }
    
    def __init__(self, model: str, api_key: str, max_retries: int = 3) -> None:
        """
        Initialize Anthropic provider.
        
        Args:
            model: Model name (e.g., "claude-3-opus-20240229")
            api_key: Anthropic API key
            max_retries: Maximum number of retries on failure
        """
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = None
        self._create_client()
        
        logger.info("anthropic_provider_initialized", model=model)
    
    def _create_client(self) -> None:
        """Create the Anthropic client."""
        try:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        except Exception as e:
            raise ProviderError(f"Failed to create Anthropic client: {e}")
    
    def _get_context_window(self) -> int:
        """Get the context window size for the current model."""
        for model_prefix, window in self.MODEL_CONTEXT_WINDOWS.items():
            if model_prefix in self._model:
                return window
        # Default for unknown models
        return 100000
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError)),
        reraise=True
    )
    async def complete(
        self, 
        messages: List[Message], 
        temperature: float = 0.7, 
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        Send a completion request to Anthropic Claude.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-1 for Anthropic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (stop_sequences, top_p, etc.)
            
        Returns:
            LLMResponse with generated content and usage info
        """
        with span("anthropic.complete", {
            "model": self._model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_messages": len(messages)
        }):
            try:
                # Anthropic API requires system prompt separate from messages array
                system = next((m.content for m in messages if m.role == "system"), "")
                convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
                
                # Prepare request parameters
                request_params = {
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": convo,
                    **kwargs
                }
                
                # Add system prompt if provided
                if system:
                    request_params["system"] = system
                
                # Make the API call
                response = await self._client.messages.create(**request_params)
                
                # Extract content
                content = ""
                if response.content and len(response.content) > 0:
                    content = response.content[0].text
                
                # Log usage
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
                
                logger.info(
                    "anthropic_completion_success",
                    model=self._model,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    finish_reason=response.stop_reason,
                )
                
                return LLMResponse(
                    content=content,
                    model=self._model,
                    usage=usage,
                    finish_reason=response.stop_reason,
                )
                
            except anthropic.RateLimitError as e:
                logger.warning("anthropic_rate_limit", error=str(e))
                raise LLMError(f"Rate limit exceeded: {e}")
            except anthropic.APIStatusError as e:
                logger.error("anthropic_api_error", status_code=e.status_code, error=str(e))
                raise LLMError(f"Anthropic API error ({e.status_code}): {e}")
            except anthropic.APIConnectionError as e:
                logger.error("anthropic_connection_error", error=str(e))
                raise LLMError(f"Connection error: {e}")
            except Exception as e:
                logger.exception("anthropic_unexpected_error", error=str(e))
                raise LLMError(f"Unexpected error: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Anthropic's approximate method.
        
        Note: Anthropic doesn't provide a public tokenizer, so we use
        an approximation (roughly 4 chars per token for English).
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        # More sophisticated approximation based on Claude's tokenizer
        # ~4 characters per token for English text
        return max(1, len(text) // 4)
    
    async def close(self):
        """Close the Anthropic client."""
        if self._client:
            await self._client.close()
            logger.debug("anthropic_client_closed")
    
    @property
    def provider_name(self) -> str: 
        return "anthropic"
    
    @property
    def model_name(self) -> str: 
        return self._model