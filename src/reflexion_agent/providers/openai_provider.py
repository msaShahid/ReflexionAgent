from typing import List, Optional, Dict, Any

import tiktoken
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    # OpenAI models and their context windows
    MODEL_CONTEXT_WINDOWS = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
    }
    
    def __init__(self, model: str, api_key: str, max_retries: int = 3) -> None:
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., "gpt-4o-2024-05-13")
            api_key: OpenAI API key
            max_retries: Maximum number of retries on failure
        """
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = None
        self._enc = None
        self._create_client()
        self._init_tokenizer()
        
        logger.info("openai_provider_initialized", model=model)
    
    def _create_client(self) -> None:
        """Create the OpenAI client."""
        try:
            self._client = AsyncOpenAI(api_key=self._api_key)
        except Exception as e:
            raise ProviderError(f"Failed to create OpenAI client: {e}")
    
    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer for the model."""
        try:
            self._enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            # Fall back to cl100k_base for unknown models
            logger.warning("unknown_model_tokenizer", model=self._model, fallback="cl100k_base")
            self._enc = tiktoken.get_encoding("cl100k_base")
    
    def _get_context_window(self) -> int:
        """Get the context window size for the current model."""
        for model_prefix, window in self.MODEL_CONTEXT_WINDOWS.items():
            if model_prefix in self._model:
                return window
        # Default for unknown models
        return 8192
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
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
        Send a completion request to OpenAI.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (stop, top_p, frequency_penalty, etc.)
            
        Returns:
            LLMResponse with generated content and usage info
        """
        with span("openai.complete", {
            "model": self._model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_messages": len(messages)
        }):
            try:
                # Convert messages to OpenAI format
                oai_msgs = [m.to_dict() for m in messages]
                
                # Prepare request parameters
                request_params = {
                    "model": self._model,
                    "messages": oai_msgs,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
                
                # Make the API call
                response = await self._client.chat.completions.create(**request_params)
                
                # Extract content
                content = response.choices[0].message.content or ""
                
                # Extract usage
                usage_dict = {}
                if response.usage:
                    usage_dict = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                
                finish_reason = response.choices[0].finish_reason
                
                logger.info(
                    "openai_completion_success",
                    model=self._model,
                    prompt_tokens=usage_dict.get("prompt_tokens", 0),
                    completion_tokens=usage_dict.get("completion_tokens", 0),
                    finish_reason=finish_reason,
                )
                
                return LLMResponse(
                    content=content,
                    model=self._model,
                    usage=usage_dict,
                    finish_reason=finish_reason,
                )
                
            except RateLimitError as e:
                logger.warning("openai_rate_limit", error=str(e))
                raise LLMError(f"Rate limit exceeded: {e}")
            except APIError as e:
                logger.error("openai_api_error", status_code=e.status_code, error=str(e))
                raise LLMError(f"OpenAI API error ({e.status_code}): {e}")
            except APIConnectionError as e:
                logger.error("openai_connection_error", error=str(e))
                raise LLMError(f"Connection error: {e}")
            except Exception as e:
                logger.exception("openai_unexpected_error", error=str(e))
                raise LLMError(f"Unexpected error: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self._enc.encode(text))
    
    async def close(self):
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            logger.debug("openai_client_closed")
    
    @property
    def provider_name(self) -> str: 
        return "openai"
    
    @property
    def model_name(self) -> str: 
        return self._model