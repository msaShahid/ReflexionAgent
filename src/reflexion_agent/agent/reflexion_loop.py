import asyncio
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from reflexion_agent.observability import (
    get_logger, span, bind_context, clear_context
)
from reflexion_agent.agent.actor import Actor
from reflexion_agent.agent.evaluator import Evaluator
from reflexion_agent.agent.reflector import Reflector
from reflexion_agent.memory import (
    ShortTermMemory, Episode, ReflectionMemory, IterationRecord
)
from reflexion_agent.memory.base import BaseEpisodicStore, BaseReflectionStore
from reflexion_agent.config import Settings
from reflexion_agent.utils.exceptions import ReflexionError
from .models import (
    ReflexionResult, IterationResult, StopReason, ActorOutput,
    EvaluatorOutput, ReflectorOutput
)

logger = get_logger(__name__)


class ReflexionLoop:
    """
    Main reflexion loop that orchestrates actor, evaluator, and reflector.
    
    The loop:
    1. Actor generates answer
    2. Evaluator scores answer
    3. If score is good enough, stop
    4. If not, reflector generates lessons
    5. Repeat with accumulated reflections
    """
    
    def __init__(
        self,
        actor: Actor,
        evaluator: Evaluator,
        reflector: Reflector,
        episodic_store: BaseEpisodicStore,
        reflection_store: BaseReflectionStore,
        settings: Settings,
    ) -> None:
        """
        Initialize the reflexion loop.
        
        Args:
            actor: Actor component
            evaluator: Evaluator component
            reflector: Reflector component
            episodic_store: Long-term episodic memory
            reflection_store: Long-term reflection memory
            settings: Application settings
        """
        self._actor = actor
        self._evaluator = evaluator
        self._reflector = reflector
        self._episodic_store = episodic_store
        self._reflection_store = reflection_store
        self._cfg = settings.agent
        self._llm_cfg = settings.llm
        
        logger.info(
            "reflexion_loop_initialized",
            max_iterations=self._cfg.max_iterations,
            stopping_score=self._cfg.stopping_score,
            timeout=self._cfg.timeout_seconds
        )
    
    @span("reflexion_loop.run")
    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> ReflexionResult:
        """
        Run the reflexion loop for a task.
        
        Args:
            task: Task to solve
            session_id: Optional session ID for tracking
            config_overrides: Optional overrides for agent config
            
        Returns:
            ReflexionResult with final answer and metadata
            
        Raises:
            ReflexionError: If loop fails catastrophically
        """
        start_time = time.monotonic()
        bind_context(session_id=session_id, task=task[:50])
        
        # Apply config overrides
        max_iterations = self._cfg.max_iterations
        stopping_score = self._cfg.stopping_score
        min_improvement = self._cfg.min_improvement_delta
        timeout = self._cfg.timeout_seconds
        
        if config_overrides:
            max_iterations = config_overrides.get('max_iterations', max_iterations)
            stopping_score = config_overrides.get('stopping_score', stopping_score)
            min_improvement = config_overrides.get('min_improvement_delta', min_improvement)
            timeout = config_overrides.get('timeout_seconds', timeout)
        
        try:
            # Initialize memory for this episode
            short_term = ShortTermMemory(max_tokens=8192)
            short_term.add("system", f"Task: {task}", metadata={"type": "task"})
            
            # Load relevant past reflections
            past_reflections = await self._load_relevant_reflections(task)
            accumulated_reflections = self._format_reflections(past_reflections)
            
            # Track best answer
            best_answer = ""
            best_score = 0.0
            previous_score = -1.0
            
            # History of iterations
            history: List[IterationResult] = []
            
            # Main loop
            stop_reason = StopReason.MAX_ITERATIONS
            
            for iteration in range(1, max_iterations + 1):
                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    stop_reason = StopReason.TIMEOUT
                    logger.warning("loop_timeout", elapsed=elapsed, timeout=timeout)
                    break
                
                iteration_start = time.monotonic()
                bind_context(iteration=iteration)
                
                logger.info(
                    "iteration_start",
                    iteration=iteration,
                    max_iterations=max_iterations,
                    elapsed=elapsed
                )
                
                try:
                    # ── ACT ──────────────────────────────────────────────
                    actor_out = await self._actor.act(
                        task=task,
                        reflections=accumulated_reflections,
                        short_term_memory=short_term.get_context(),
                        temperature=self._llm_cfg.actor.temperature,
                        max_tokens=self._llm_cfg.actor.max_tokens,
                        iteration=iteration,
                        max_iterations=max_iterations
                    )
                    
                    short_term.add(
                        "assistant", 
                        actor_out.answer,
                        tokens=actor_out.token_usage.get('completion_tokens', 0),
                        metadata={"iteration": iteration}
                    )
                    
                    # ── EVALUATE ─────────────────────────────────────────
                    eval_out = await self._evaluator.evaluate(
                        task=task,
                        answer=actor_out.answer,
                        temperature=self._llm_cfg.evaluator.temperature,
                        max_tokens=self._llm_cfg.evaluator.max_tokens,
                        previous_attempts=self._format_previous_attempts(history)
                    )
                    
                    # Update best answer
                    if eval_out.score > best_score:
                        best_score = eval_out.score
                        best_answer = actor_out.answer
                        logger.debug("new_best_answer", score=best_score)
                    
                    # ── CHECK STOPPING CRITERIA ─────────────────────────
                    if eval_out.passed:
                        stop_reason = StopReason.SCORE_THRESHOLD
                        logger.info(
                            "score_threshold_reached",
                            score=eval_out.score,
                            threshold=stopping_score
                        )
                        
                        # Still record this iteration
                        iteration_result = self._create_iteration_result(
                            iteration, actor_out, eval_out, None,
                            time.monotonic() - iteration_start
                        )
                        history.append(iteration_result)
                        break
                    
                    if iteration > 1:
                        improvement = eval_out.score - previous_score
                        if improvement < min_improvement:
                            stop_reason = StopReason.NO_IMPROVEMENT
                            logger.info(
                                "no_improvement",
                                improvement=improvement,
                                min_delta=min_improvement
                            )
                            
                            iteration_result = self._create_iteration_result(
                                iteration, actor_out, eval_out, None,
                                time.monotonic() - iteration_start
                            )
                            history.append(iteration_result)
                            break
                    
                    previous_score = eval_out.score
                    
                    # ── REFLECT ──────────────────────────────────────────
                    reflect_out = await self._reflector.reflect(
                        task=task,
                        previous_answer=actor_out.answer,
                        critique=eval_out.critique,
                        score=eval_out.score,
                        iteration=iteration,
                        previous_reflections=self._extract_reflection_texts(history),
                        context=short_term.get_context(),
                        temperature=self._llm_cfg.reflector.temperature,
                        max_tokens=self._llm_cfg.reflector.max_tokens
                    )
                    
                    # Update accumulated reflections
                    accumulated_reflections = self._update_reflections(
                        accumulated_reflections, 
                        reflect_out.reflection
                    )
                    
                    # Add to short-term memory
                    short_term.add(
                        "system",
                        f"Reflection from iteration {iteration}: {reflect_out.reflection}",
                        metadata={"type": "reflection", "iteration": iteration}
                    )
                    
                    # Record iteration
                    iteration_result = self._create_iteration_result(
                        iteration, actor_out, eval_out, reflect_out,
                        time.monotonic() - iteration_start
                    )
                    history.append(iteration_result)
                    
                except Exception as e:
                    logger.exception(
                        "iteration_failed",
                        iteration=iteration,
                        error=str(e)
                    )
                    # Continue to next iteration? Or break?
                    # For now, break on error
                    stop_reason = StopReason.ERROR
                    break
            
            # ── FINALIZE ────────────────────────────────────────────────
            total_duration = time.monotonic() - start_time
            
            # Determine success
            succeeded = best_score >= stopping_score
            
            # Save to long-term memory
            episode_id = await self._save_episode(
                task=task,
                final_answer=best_answer,
                final_score=best_score,
                iterations=len(history),
                succeeded=succeeded,
                history=history
            )
            
            # Save reflections if we learned something
            if accumulated_reflections and history:
                await self._save_reflections(
                    task=task,
                    reflections=accumulated_reflections,
                    history=history,
                    episode_id=episode_id
                )
            
            # Calculate total tokens
            total_tokens = sum(
                sum(r.token_usage.values()) for r in history
            )
            
            result = ReflexionResult(
                task=task,
                final_answer=best_answer,
                final_score=best_score,
                iterations_used=len(history),
                stop_reason=stop_reason,
                succeeded=succeeded,
                episode_id=episode_id,
                history=history,
                total_tokens=total_tokens,
                total_duration_ms=total_duration * 1000,
                metadata={
                    "session_id": session_id,
                    "config_overrides": config_overrides,
                }
            )
            
            logger.info(
                "reflexion_loop_completed",
                iterations=len(history),
                final_score=best_score,
                succeeded=succeeded,
                stop_reason=stop_reason.value,
                duration=total_duration
            )
            
            return result
            
        except Exception as e:
            logger.exception("reflexion_loop_failed", error=str(e))
            raise ReflexionError(f"Reflexion loop failed: {e}") from e
        finally:
            clear_context()
    
    async def _load_relevant_reflections(self, task: str) -> List[ReflectionMemory]:
        """Load relevant past reflections for this task."""
        try:
            reflections = await self._reflection_store.find_relevant_reflections(
                task=task,
                top_k=3,
                min_improvement=0.0  # Include all
            )
            logger.debug("loaded_reflections", count=len(reflections))
            return reflections
        except Exception as e:
            logger.warning("failed_to_load_reflections", error=str(e))
            return []
    
    def _format_reflections(self, reflections: List[ReflectionMemory]) -> str:
        """Format reflections for inclusion in prompt."""
        if not reflections:
            return ""
        
        lines = ["Previous lessons learned:"]
        for i, ref in enumerate(reflections, 1):
            lines.append(f"\n{i}. {ref.reflection_text}")
        
        return "\n".join(lines)
    
    def _format_previous_attempts(self, history: List[IterationResult]) -> str:
        """Format previous attempts for evaluator."""
        if not history:
            return "None"
        
        lines = []
        for h in history[-3:]:  # Last 3 attempts
            lines.append(f"Iteration {h.iteration} (score: {h.score:.2f}):")
            lines.append(f"  Answer: {h.answer[:100]}...")
            lines.append(f"  Critique: {h.critique[:100]}...")
        
        return "\n".join(lines)
    
    def _extract_reflection_texts(self, history: List[IterationResult]) -> List[str]:
        """Extract reflection texts from history."""
        return [h.reflection for h in history if h.reflection]
    
    def _update_reflections(self, current: str, new_reflection: str) -> str:
        """Update accumulated reflections with new one."""
        if not current:
            return new_reflection
        return f"{current}\n\n{new_reflection}"
    
    def _create_iteration_result(
        self,
        iteration: int,
        actor_out: ActorOutput,
        eval_out: EvaluatorOutput,
        reflect_out: Optional[ReflectorOutput],
        duration_ms: float
    ) -> IterationResult:
        """Create an iteration result from component outputs."""
        return IterationResult(
            iteration=iteration,
            answer=actor_out.answer,
            score=eval_out.score,
            critique=eval_out.critique,
            reflection=reflect_out.reflection if reflect_out else "",
            token_usage={
                **actor_out.token_usage,
                **eval_out.token_usage,
                **(reflect_out.token_usage if reflect_out else {})
            },
            duration_ms=duration_ms * 1000
        )
    
    async def _save_episode(
        self,
        task: str,
        final_answer: str,
        final_score: float,
        iterations: int,
        succeeded: bool,
        history: List[IterationResult]
    ) -> str:
        """Save episode to long-term memory."""
        try:
            episode = Episode(
                task=task,
                final_answer=final_answer,
                final_score=final_score,
                iterations=iterations,
                succeeded=succeeded,
                metadata={
                    "history_length": len(history),
                    "final_score": final_score,
                },
                tags=["success" if succeeded else "failure"]
            )
            
            episode_id = await self._episodic_store.add(episode)
            logger.debug("episode_saved", episode_id=episode_id)
            return episode_id
            
        except Exception as e:
            logger.warning("failed_to_save_episode", error=str(e))
            return ""
    
    async def _save_reflections(
        self,
        task: str,
        reflections: str,
        history: List[IterationResult],
        episode_id: str
    ) -> None:
        """Save reflections to long-term memory."""
        try:
            if not history:
                return
            
            # Create reflection memory
            reflection = ReflectionMemory(
                task_summary=task[:200],
                reflection_text=reflections,
                score_before=history[0].score if history else 0.0,
                score_after=history[-1].score if history else 0.0,
                tags=["learned"],
                source_episode_id=episode_id,
                metadata={
                    "iterations": len(history),
                    "improvement": history[-1].score - history[0].score if len(history) > 1 else 0.0
                }
            )
            
            await self._reflection_store.add(reflection)
            logger.debug("reflections_saved", reflection_id=reflection.id)
            
        except Exception as e:
            logger.warning("failed_to_save_reflections", error=str(e))