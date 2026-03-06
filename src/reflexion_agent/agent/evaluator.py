import json
import re
from typing import Optional, Dict, Any

from reflexion_agent.observability import get_logger, span, bind_context
from reflexion_agent.prompts import EVALUATOR_SYSTEM, EVALUATOR_USER, DEFAULT_EVALUATION_CRITERIA
from reflexion_agent.providers import BaseLLMProvider, Message
from reflexion_agent.utils.exceptions import EvaluatorError
from .models import EvaluatorOutput

logger = get_logger(__name__)


class Evaluator:
    """Evaluator component that scores answers and provides feedback."""
    
    def __init__(
        self, 
        provider: BaseLLMProvider, 
        stopping_score: float = 0.85
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            provider: LLM provider for evaluation
            stopping_score: Score threshold for success
        """
        self._provider = provider
        self._stopping_score = stopping_score
        
        logger.info(
            "evaluator_initialized",
            provider=provider.provider_name,
            model=provider.model_name,
            stopping_score=stopping_score
        )
    
    @span("evaluator.evaluate")
    async def evaluate(
        self, 
        task: str, 
        answer: str,
        criteria: Optional[str] = None,
        previous_attempts: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs
    ) -> EvaluatorOutput:
        """
        Evaluate an answer to a task.
        
        Args:
            task: Original task
            answer: Answer to evaluate
            criteria: Optional custom evaluation criteria
            previous_attempts: Optional history of previous attempts
            temperature: Temperature for evaluation (low for consistency)
            max_tokens: Max tokens for response
            **kwargs: Additional provider parameters
            
        Returns:
            EvaluatorOutput with score and feedback
            
        Raises:
            EvaluatorError: If evaluation fails
        """
        bind_context(component="evaluator")
        
        try:
            # Prepare user prompt
            user_prompt = EVALUATOR_USER.substitute(
                task=task,
                answer=answer,
                criteria=criteria or DEFAULT_EVALUATION_CRITERIA,
                previous_attempts=previous_attempts or "None"
            )
            
            messages = [
                Message(role="system", content=EVALUATOR_SYSTEM),
                Message(role="user", content=user_prompt),
            ]
            
            # Call provider
            response = await self._provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Parse response
            parsed = self._parse_response(response.content)
            
            # Ensure score is within bounds
            score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
            
            logger.info(
                "evaluator_result",
                score=score,
                passed=score >= self._stopping_score,
                tokens_used=response.total_tokens
            )
            
            return EvaluatorOutput(
                score=score,
                passed=score >= self._stopping_score,
                critique=parsed.get("critique", "No critique provided"),
                strengths=parsed.get("strengths", []),
                weaknesses=parsed.get("weaknesses", []),
                suggestions=parsed.get("suggestions", []),
                token_usage=response.usage,
                metadata={
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                    "parsing_method": "json" if self._is_valid_json(response.content) else "regex"
                }
            )
            
        except Exception as e:
            logger.exception("evaluator_failed", error=str(e))
            raise EvaluatorError(f"Evaluator failed: {e}") from e
        finally:
            bind_context(component=None)
    
    def _is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(self._clean_json_text(text))
            return True
        except:
            return False
    
    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Remove markdown fences and clean up JSON text."""
        # Remove ```json and ``` markers
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        return clean
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse evaluator response, handling various formats.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed dictionary with score, critique, etc.
        """
        # Try to parse as JSON first
        clean_text = self._clean_json_text(text)
        
        try:
            parsed = json.loads(clean_text)
            # Validate required fields
            if "score" not in parsed:
                parsed["score"] = 0.5
            if "critique" not in parsed:
                parsed["critique"] = clean_text
            return parsed
            
        except json.JSONDecodeError:
            # Fallback: extract score using regex
            logger.debug("evaluator_json_parse_failed, using regex fallback")
            
            # Try to find score pattern
            score_match = re.search(r'"score"\s*:\s*([\d.]+)', clean_text)
            score = float(score_match.group(1)) if score_match else 0.5
            
            # Try to find critique
            critique_match = re.search(r'"critique"\s*:\s*"([^"]+)"', clean_text)
            critique = critique_match.group(1) if critique_match else clean_text[:500]
            
            return {
                "score": score,
                "critique": critique,
                "strengths": [],
                "weaknesses": ["Could not parse structured evaluation."],
                "suggestions": ["Please ensure response is valid JSON."]
            }