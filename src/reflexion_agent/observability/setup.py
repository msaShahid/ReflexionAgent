"""
Convenience module to set up all observability at once.
"""
from typing import Optional

from reflexion_agent.config import Settings, ObservabilityConfig, LoggingConfig
from reflexion_agent.observability.logging import configure_logging
from reflexion_agent.observability.tracing import configure_tracing


def setup_observability(
    settings: Optional[Settings] = None,
    service_name: str = "reflexion-agent",
    env: str = "development",
) -> None:
    """
    Set up all observability components (logging, tracing) based on settings.
    
    Args:
        settings: Application settings (if None, uses defaults)
        service_name: Name of the service for tracing
        env: Environment name (development, staging, production)
    """
    # Configure logging
    if settings and settings.logging:
        configure_logging(config=settings.logging)
    else:
        # Default logging configuration
        log_config = LoggingConfig(
            level="INFO" if env == "production" else "DEBUG",
            format="json" if env == "production" else "pretty",
            enable_structlog=True,
        )
        configure_logging(config=log_config)
    
    # Configure tracing
    if settings and settings.observability:
        configure_tracing(
            service_name=service_name,
            config=settings.observability
        )
    else:
        # Default tracing configuration
        trace_config = ObservabilityConfig(
            enable_tracing=env != "production",  # Disable in production by default
            trace_exporter="console" if env == "development" else "none",
        )
        configure_tracing(
            service_name=service_name,
            config=trace_config
        )
