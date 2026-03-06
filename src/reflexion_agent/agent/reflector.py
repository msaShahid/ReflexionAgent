from typing import Optional, List, Dict, Any

from reflexion_agent.observability import get_logger, span, bind_context
from reflexion_agent.prompts import REFLECTOR_SYSTEM, REFLECTOR_USER, REFLECTOR_SIMPLE_USER
from reflexion_agent.providers import BaseLLMProvider, Message
from reflexion_agent.utils.exceptions import ReflectorError
from .models import ReflectorOutput

logger = get_logger(__name__)


class Reflector:
    """Reflector component that generates lessons from failures."""
    
    def __init__(
        self, 
        provider: BaseLLMProvider,
        use_simple_prompt: bool = False
    ) -> None:
        """
        Initialize the reflector.
        
        Args:
            provider: LLM provider for reflection
            use_simple_prompt: Whether to use simpler prompt for faster models
        """
        self._provider = provider
        self._use_simple_prompt = use_simple_prompt
        
        logger.info(
            "reflector_initialized",
            provider=provider.provider_name,
            model=provider.model_name,
            prompt_type="simple" if use_simple_prompt else "detailed"
        )
    
    @span("reflector.reflect")
    async def reflect(
        self,
        task: str,
        previous_answer: str,
        critique: str,
        score: float,
        iteration: int,
        previous_reflections: Optional[List[str]] = None,
        context: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs,
    ) -> ReflectorOutput:
        """
        Generate reflections/lessons from a failed attempt.
        
        Args:
            task: Original task
            previous_answer: The answer that was evaluated
            critique: Evaluator's critique
            score: Score from evaluator
            iteration: Current iteration number
            previous_reflections: List of previous reflections
            context: Additional context
            temperature: Temperature for generation
            max_tokens: Max tokens for response
            **kwargs: Additional provider parameters
            
        Returns:
            ReflectorOutput with reflection text
            
        Raises:
            ReflectorError: If reflection generation fails
        """
        bind_context(component="reflector", iteration=iteration)
        
        try:
            # Format previous reflections
            prev_reflections_str = ""
            if previous_reflections:
                prev_reflections_str = "\n".join(
                    f"- {r}" for r in previous_reflections[-5:]  # Last 5
                )
            
            # Choose prompt template
            if self._use_simple_prompt:
                user_prompt = REFLECTOR_SIMPLE_USER.substitute(
                    task=task,
                    previous_answer=previous_answer[:1000],  # Truncate for simple prompt
                    critique=critique[:500]
                )
            else:
                user_prompt = REFLECTOR_USER.substitute(
                    task=task,
                    previous_answer=previous_answer,
                    critique=critique,
                    score=score,
                    iteration=iteration,
                    previous_reflections=prev_reflections_str or "(none)",
                    context=context or "No additional context."
                )
            
            messages = [
                Message(role="system", content=REFLECTOR_SYSTEM),
                Message(role="user", content=user_prompt),
            ]
            
            # Call provider
            response = await self._provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract bullet points if present
            reflection_text = response.content
            bullet_points = self._extract_bullet_points(reflection_text)
            
            logger.info(
                "reflector_result",
                iteration=iteration,
                bullet_points=len(bullet_points),
                tokens_used=response.total_tokens
            )
            
            return ReflectorOutput(
                reflection=reflection_text,
                token_usage=response.usage,
                metadata={
                    "model": response.model,
                    "bullet_points": bullet_points,
                    "prompt_type": "simple" if self._use_simple_prompt else "detailed"
                }
            )
            
        except Exception as e:
            logger.exception("reflector_failed", iteration=iteration, error=str(e))
            raise ReflectorError(f"Reflector failed: {e}") from e
        finally:
            bind_context(component=None, iteration=None)
    
    @staticmethod
    def _extract_bullet_points(text: str) -> List[str]:
        """
        Extract bullet points from reflection text.
        
        Args:
            text: Reflection text
            
        Returns:
            List of bullet points
        """
        lines = text.split('\n')
        bullets = []
        
        for line in lines:
            line = line.strip()
            # Check for common bullet markers
            if line and (line.startswith('- ') or line.startswith('• ') or 
                        line.startswith('* ') or re.match(r'^\d+\.', line)):
                # Remove the bullet marker
                clean = re.sub(r'^[-•*\d\.]\s*', '', line)
                bullets.append(clean)
        
        return bullets
    
    async def reflect_batch(
        self,
        failures: List[Dict[str, Any]],
        **kwargs
    ) -> List[ReflectorOutput]:
        """
        Generate reflections for multiple failures.
        
        Args:
            failures: List of failure dicts with task, answer, critique
            **kwargs: Parameters for reflect method
            
        Returns:
            List of reflector outputs
        """
        results = []
        for failure in failures:
            result = await self.reflect(
                task=failure['task'],
                previous_answer=failure['answer'],
                critique=failure['critique'],
                score=failure.get('score', 0.0),
                iteration=failure.get('iteration', 1),
                **kwargs
            )
            results.append(result)
        
        return results