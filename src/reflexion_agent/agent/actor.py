from typing import Optional, Dict, Any

from reflexion_agent.observability import get_logger, span, bind_context
from reflexion_agent.prompts import ACTOR_SYSTEM, ACTOR_USER, format_tools_for_prompt
from reflexion_agent.providers import BaseLLMProvider, Message
from reflexion_agent.tools import ToolRegistry
from reflexion_agent.utils.exceptions import ActorError
from .models import ActorOutput

logger = get_logger(__name__)


class Actor:
    """Actor component that generates answers to tasks."""
    
    def __init__(
        self, 
        provider: BaseLLMProvider, 
        tool_registry: Optional[ToolRegistry] = None
    ) -> None:
        """
        Initialize the actor.
        
        Args:
            provider: LLM provider for generation
            tool_registry: Optional tool registry for tool use
        """
        self._provider = provider
        self._tools = tool_registry
        
        logger.info(
            "actor_initialized",
            provider=provider.provider_name,
            model=provider.model_name,
            tools_available=len(tool_registry.list_tools()) if tool_registry else 0
        )
    
    @span("actor.act")
    async def act(
        self,
        task: str,
        reflections: str = "",
        short_term_memory: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        iteration: Optional[int] = None,
        **kwargs,
    ) -> ActorOutput:
        """
        Generate an answer for the given task.
        
        Args:
            task: The task to solve
            reflections: Previous reflections/lessons
            short_term_memory: Current conversation context
            temperature: Override default temperature
            max_tokens: Override default max tokens
            iteration: Current iteration number (for logging)
            **kwargs: Additional parameters for the provider
            
        Returns:
            ActorOutput with the generated answer
            
        Raises:
            ActorError: If generation fails
        """
        # Bind context for logging
        bind_context(
            component="actor",
            iteration=iteration,
            task_length=len(task)
        )
        
        try:
            # Format tools for prompt if available
            tools_str = ""
            if self._tools:
                tools_str = format_tools_for_prompt({
                    name: tool.description 
                    for name, tool in self._tools._tools.items()
                })
            
            # Prepare user prompt
            user_prompt = ACTOR_USER.substitute(
                task=task,
                reflections=reflections or "(no prior reflections)",
                short_term_memory=short_term_memory or "(no recent context)",
                tools=tools_str or "No tools available.",
                iteration=iteration or 1,
                max_iterations=kwargs.get('max_iterations', 5)
            )
            
            messages = [
                Message(role="system", content=ACTOR_SYSTEM),
                Message(role="user", content=user_prompt),
            ]
            
            # Log token count estimate
            estimated_tokens = self._provider.count_messages_tokens(messages)
            logger.debug(
                "actor_request",
                estimated_tokens=estimated_tokens,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Call provider
            response = await self._provider.complete(
                messages=messages,
                temperature=temperature or 0.7,
                max_tokens=max_tokens or 4096,
                **kwargs
            )
            
            # Log success
            logger.info(
                "actor_response",
                tokens_used=response.total_tokens,
                finish_reason=response.finish_reason
            )
            
            return ActorOutput(
                answer=response.content,
                token_usage=response.usage,
                metadata={
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                }
            )
            
        except Exception as e:
            logger.exception("actor_failed", error=str(e))
            raise ActorError(f"Actor failed: {e}") from e
        finally:
            # Clear context
            bind_context(component=None, iteration=None)