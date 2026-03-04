import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
import structlog

from reflexion_agent.config import LoggingConfig


def configure_logging(
    level: str = "INFO", 
    fmt: str = "json",
    config: Optional[LoggingConfig] = None,
    log_file: Optional[Path] = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        fmt: Output format ("json" or "pretty")
        config: LoggingConfig object (if provided, overrides level/fmt)
        log_file: Optional file path to write logs to
    """
    # Use config if provided
    if config:
        level = config.level
        fmt = config.format
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Common processors for all configurations
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add call stack processor for debug level
    if level.upper() == "DEBUG":
        shared_processors.append(structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME,
             structlog.processors.CallsiteParameter.LINENO,
             structlog.processors.CallsiteParameter.FUNC_NAME]
        ))
    
    # Configure logging handlers
    handlers = []
    
    # Console handler
    if fmt == "pretty":
        # Human-readable colored output for local development
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=level.upper() == "DEBUG",
            markup=True,
            rich_tracebacks=True,
            tracebacks_word_wrap=False,
        )
        handlers.append(console_handler)
        
        # Configure structlog for pretty output
        structlog.configure(
            processors=shared_processors + [structlog.dev.ConsoleRenderer()],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # JSON for log aggregators: Datadog, CloudWatch, Splunk
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(json_handler)
        
        # Configure structlog for JSON output
        structlog.configure(
            processors=shared_processors + [structlog.processors.JSONRenderer()],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Suppress noisy library logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        Structured logger with bound methods
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get a logger instance bound to this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__).bind(
                class_name=self.__class__.__name__
            )
        return self._logger
    
    def log_debug(self, event: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(event, **kwargs)
    
    def log_info(self, event: str, **kwargs):
        """Log info message with context."""
        self.logger.info(event, **kwargs)
    
    def log_warning(self, event: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(event, **kwargs)
    
    def log_error(self, event: str, **kwargs):
        """Log error message with context."""
        self.logger.error(event, **kwargs)
    
    def log_exception(self, event: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(event, **kwargs)


# Convenience functions for context management
def bind_context(**kwargs) -> None:
    """Bind key-value pairs to the logging context."""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys) -> None:
    """Unbind keys from the logging context."""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all logging context variables."""
    structlog.contextvars.clear_contextvars()
