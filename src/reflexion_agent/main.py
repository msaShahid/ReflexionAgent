"""
Main entry point for Reflexion Agent.
Supports both CLI and programmatic usage.
"""
import asyncio
import sys
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

from reflexion_agent.agent import ReflexionLoop, ReflexionResult
from reflexion_agent.agent.factory import create_agent_async
from reflexion_agent.config import get_settings, get_config_for_env
from reflexion_agent.observability import (
    configure_logging, 
    configure_tracing,
    get_logger,
    bind_context,
    clear_context
)
from reflexion_agent.utils.exceptions import ReflexionError, ConfigurationError
from reflexion_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)


async def build_agent(
    env: str = "development",
    config_path: Optional[str] = None,
    enable_tools: bool = True
) -> ReflexionLoop:
    """
    Dependency Injection root. Builds the full agent graph.
    
    Args:
        env: Environment name (development, staging, production)
        config_path: Optional path to config file
        enable_tools: Whether to enable tools
        
    Returns:
        Configured ReflexionLoop
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Load configuration
        if config_path:
            # Override config path in env settings
            from reflexion_agent.config import EnvSettings
            env_settings = EnvSettings()
            env_settings.config_path = config_path
            settings, env_settings = get_settings()
        else:
            # Load by environment
            settings, env_settings = get_config_for_env(env)
        
        # 1. Observability FIRST — every module depends on it
        configure_logging(
            level=settings.logging.level,
            fmt=settings.logging.format
        )
        
        if settings.observability.enable_tracing:
            configure_tracing(
                service_name=f"reflexion-agent-{env}",
                exporter=settings.observability.trace_exporter,
                otlp_endpoint=settings.observability.otlp_endpoint,
                config=settings.observability
            )
        
        logger.info(
            "building_agent",
            environment=env,
            config_path=env_settings.get_config_path(),
            agent_name=settings.agent.name
        )
        
        # 2. Create agent with async initialization
        agent = await create_agent_async(
            settings=settings,
            env_settings=env_settings,
            enable_tools=enable_tools
        )
        
        logger.info("agent_built_successfully")
        return agent
        
    except ConfigurationError as e:
        logger.error("configuration_error", error=str(e))
        raise
    except Exception as e:
        logger.exception("unexpected_error_building_agent", error=str(e))
        raise ReflexionError(f"Failed to build agent: {e}")


async def run_task(
    task: str,
    env: str = "development",
    config_path: Optional[str] = None,
    session_id: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> ReflexionResult:
    """
    Run a single task with the agent.
    
    Args:
        task: Task to perform
        env: Environment
        config_path: Optional config file path
        session_id: Optional session ID for tracking
        config_overrides: Optional config overrides
        verbose: Enable verbose output
        
    Returns:
        ReflexionResult with answer and metadata
    """
    bind_context(session_id=session_id, task=task[:50])
    
    try:
        # Build agent
        agent = await build_agent(
            env=env,
            config_path=config_path,
            enable_tools=True
        )
        
        # Run task
        if verbose:
            logger.info("starting_task_execution", task=task)
        
        result = await agent.run(
            task=task,
            session_id=session_id,
            config_overrides=config_overrides
        )
        
        if verbose:
            logger.info(
                "task_completed",
                success=result.succeeded,
                score=result.final_score,
                iterations=result.iterations_used,
                stop_reason=result.stop_reason.value
            )
        
        return result
        
    except ReflexionError as e:
        logger.error("agent_error", error=str(e))
        raise
    except Exception as e:
        logger.exception("unexpected_error", error=str(e))
        raise ReflexionError(f"Unexpected error: {e}")
    finally:
        # Clean up tool registry
        try:
            tool_registry = get_tool_registry()
            await tool_registry.cleanup()
        except:
            pass
        
        clear_context()


def format_output(result: ReflexionResult, format: str = "pretty") -> str:
    """
    Format result for display.
    
    Args:
        result: Reflexion result
        format: Output format (pretty, json, minimal)
        
    Returns:
        Formatted string
    """
    if format == "json":
        import json
        return json.dumps({
            "success": result.succeeded,
            "score": result.final_score,
            "iterations": result.iterations_used,
            "stop_reason": result.stop_reason.value,
            "episode_id": result.episode_id,
            "answer": result.final_answer,
            "total_tokens": result.total_tokens,
            "duration_ms": result.total_duration_ms
        }, indent=2)
    
    elif format == "minimal":
        return result.final_answer
    
    else:  # pretty
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        
        console = Console()
        
        # Create status indicator
        status = "✅ Task Complete" if result.succeeded else "⚠️ Task Incomplete"
        status_color = "green" if result.succeeded else "yellow"
        
        # Create stats table
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Score", f"{result.final_score:.2f}")
        table.add_row("Iterations", str(result.iterations_used))
        table.add_row("Stop Reason", result.stop_reason.value)
        table.add_row("Episode ID", result.episode_id or "N/A")
        table.add_row("Total Tokens", str(result.total_tokens))
        table.add_row("Duration", f"{result.total_duration_ms/1000:.2f}s")
        
        # Build output
        output = []
        output.append("\n")
        output.append(Panel(
            Text(status, style=f"bold {status_color}"),
            border_style=status_color
        ))
        output.append(table)
        output.append(Panel(
            result.final_answer,
            title="Answer",
            border_style="green" if result.succeeded else "yellow"
        ))
        
        # If there were multiple iterations, show brief history
        if len(result.history) > 1:
            history_table = Table(title="Iteration History")
            history_table.add_column("Iter", style="cyan")
            history_table.add_column("Score", style="yellow")
            history_table.add_column("Outcome", style="white")
            
            for i, h in enumerate(result.history, 1):
                outcome = "✓" if h.score >= 0.7 else "✗"
                history_table.add_row(
                    str(i),
                    f"{h.score:.2f}",
                    outcome
                )
            
            output.append(history_table)
        
        return "\n".join(str(o) for o in output)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reflexion Agent - AI agent with episodic + reflection memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  reflexion "What is the capital of France?"
  reflexion --env production "Explain quantum computing"
  reflexion --config custom.yaml --verbose "Calculate 15 * 27"
  reflexion --interactive
        """
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to perform"
    )
    
    parser.add_argument(
        "--env", "-e",
        choices=["development", "staging", "production"],
        default="development",
        help="Environment (default: development)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["pretty", "json", "minimal"],
        default="pretty",
        help="Output format (default: pretty)"
    )
    
    parser.add_argument(
        "--session-id", "-s",
        help="Session ID for tracking"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override max iterations"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        help="Override timeout in seconds"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Reflexion Agent 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        run_interactive(args)
        return
    
    # Handle single task
    if not args.task:
        parser.print_help()
        sys.exit(1)
    
    # Prepare config overrides
    config_overrides = {}
    if args.max_iterations:
        config_overrides["max_iterations"] = args.max_iterations
    if args.timeout:
        config_overrides["timeout_seconds"] = args.timeout
    
    try:
        # Run task
        result = asyncio.run(run_task(
            task=args.task,
            env=args.env,
            config_path=args.config,
            session_id=args.session_id,
            config_overrides=config_overrides if config_overrides else None,
            verbose=args.verbose
        ))
        
        # Output result
        output = format_output(result, format=args.format)
        if args.format == "pretty":
            from rich.console import Console
            Console().print(output)
        else:
            print(output)
        
        # Exit with appropriate code
        sys.exit(0 if result.succeeded else 1)
        
    except ReflexionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_interactive(args: argparse.Namespace) -> None:
    """Run in interactive mode."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import print as rprint
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]Reflexion Agent Interactive Mode[/bold cyan]\n"
        "Type your tasks and get responses. Type 'exit' or 'quit' to stop.",
        border_style="cyan"
    ))
    
    console.print("[dim]Environment:[/dim] " + args.env)
    if args.config:
        console.print("[dim]Config:[/dim] " + args.config)
    console.print()
    
    session_id = args.session_id or f"session_{int(time.time())}"
    
    async def interactive_loop():
        while True:
            try:
                # Get task
                task = Prompt.ask("[bold yellow]Task[/bold yellow]")
                
                if task.lower() in ('exit', 'quit', 'q'):
                    console.print("[cyan]Goodbye![/cyan]")
                    break
                
                if not task.strip():
                    continue
                
                # Show thinking indicator
                with console.status("[bold green]Thinking...[/bold green]"):
                    result = await run_task(
                        task=task,
                        env=args.env,
                        config_path=args.config,
                        session_id=session_id,
                        verbose=args.verbose
                    )
                
                # Show result
                if result.succeeded:
                    console.print(Panel(
                        result.final_answer,
                        title="[green]Answer[/green]",
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        result.final_answer,
                        title="[yellow]Partial Answer[/yellow]",
                        border_style="yellow"
                    ))
                
                # Show stats
                console.print(
                    f"[dim]Score: {result.final_score:.2f} | "
                    f"Iterations: {result.iterations_used} | "
                    f"Tokens: {result.total_tokens}[/dim]"
                )
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[cyan]Goodbye![/cyan]")
                break
            except ReflexionError as e:
                console.print(f"[red]Error: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
                if args.verbose:
                    import traceback
                    console.print(traceback.format_exc())
    
    import time
    asyncio.run(interactive_loop())


# Programmatic API
async def run(
    task: str,
    env: str = "development",
    config_path: Optional[str] = None,
    **kwargs
) -> ReflexionResult:
    """
    Programmatic API for running tasks.
    
    Args:
        task: Task to perform
        env: Environment
        config_path: Optional config file
        **kwargs: Additional arguments passed to run_task
        
    Returns:
        ReflexionResult
    """
    return await run_task(
        task=task,
        env=env,
        config_path=config_path,
        **kwargs
    )


if __name__ == "__main__":
    main()