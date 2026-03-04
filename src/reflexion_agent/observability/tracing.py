from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter, 
    SimpleSpanProcessor,
    BatchSpanProcessor
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry import context as otel_context

from reflexion_agent.config import ObservabilityConfig


# Global tracer instance
_tracer = None


def configure_tracing(
    service_name: str = "reflexion-agent",
    exporter: str = "console",
    otlp_endpoint: str = "http://localhost:4317",
    config: Optional[ObservabilityConfig] = None
) -> None:
    """
    Configure OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for tracing
        exporter: Type of exporter ("console", "otlp", or "none")
        otlp_endpoint: OTLP endpoint (for otlp exporter)
        config: ObservabilityConfig object (if provided, overrides other args)
    """
    global _tracer
    
    # Use config if provided
    if config:
        if not config.enable_tracing:
            # Tracing disabled, use no-op tracer
            trace.set_tracer_provider(trace.NoOpTracerProvider())
            _tracer = trace.get_tracer("noop")
            return
        
        exporter = config.trace_exporter
        otlp_endpoint = config.otlp_endpoint or otlp_endpoint
    
    if exporter == "none":
        # Tracing explicitly disabled
        trace.set_tracer_provider(trace.NoOpTracerProvider())
        _tracer = trace.get_tracer("noop")
        return
    
    # Create resource with service name
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",  # Could import from __version__
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
    })
    
    provider = TracerProvider(resource=resource)
    
    # Configure exporter
    if exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            # Production: send spans to Jaeger/Honeycomb/Datadog
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    max_queue_size=2048,
                    schedule_delay_millis=5000,
                )
            )
        except ImportError:
            # Fall back to console if OTLP not available
            print("Warning: OTLP exporter not available, falling back to console")
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        # Development: print spans to console
        provider.add_span_processor(
            SimpleSpanProcessor(
                ConsoleSpanExporter(service_name=service_name)
            )
        )
    
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        # Auto-configure with defaults if not configured
        configure_tracing()
    return _tracer


@contextmanager
def span(
    name: str, 
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True
):
    """
    Wrap any code block in an OTel span with enhanced error handling.
    
    Args:
        name: Name of the span
        attributes: Attributes to add to the span
        record_exception: Whether to record exceptions in the span
        set_status_on_exception: Whether to set span status on exception
        
    Yields:
        The span object
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name) as span:
        try:
            if attributes:
                for k, v in attributes.items():
                    # Convert non-string values to strings
                    if not isinstance(v, (str, bool, int, float)):
                        v = str(v)
                    span.set_attribute(k, v)
            
            yield span
            
        except Exception as e:
            if record_exception:
                span.record_exception(e)
            if set_status_on_exception:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # Add duration if not already set
            if hasattr(span, 'end_time') and span.start_time:
                duration_ms = (span.end_time - span.start_time) / 1_000_000
                span.set_attribute("duration_ms", duration_ms)


def trace_function(name: Optional[str] = None):
    """
    Decorator to trace function calls.
    
    Args:
        name: Optional custom name for the span (defaults to function name)
    
    Example:
        @trace_function()
        async def my_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with span(span_name, {
                "function.module": func.__module__,
                "function.name": func.__name__,
            }):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with span(span_name, {
                "function.module": func.__module__,
                "function.name": func.__name__,
            }):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class TraceMixin:
    """Mixin class to add tracing capability to any class."""
    
    def trace_method(self, name: Optional[str] = None):
        """
        Decorator to trace a method with class context.
        
        Args:
            name: Optional custom name for the span
        """
        def decorator(method):
            @wraps(method)
            async def async_wrapper(*args, **kwargs):
                span_name = name or f"{self.__class__.__name__}.{method.__name__}"
                with span(span_name, {
                    "class.name": self.__class__.__name__,
                    "method.name": method.__name__,
                }):
                    return await method(*args, **kwargs)
            
            @wraps(method)
            def sync_wrapper(*args, **kwargs):
                span_name = name or f"{self.__class__.__name__}.{method.__name__}"
                with span(span_name, {
                    "class.name": self.__class__.__name__,
                    "method.name": method.__name__,
                }):
                    return method(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(method) else sync_wrapper
        
        return decorator


# Utility for measuring execution time
@contextmanager
def measure_time(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Simple context manager to measure execution time without full tracing.
    
    Args:
        name: Name of the operation being measured
        attributes: Additional attributes to log
        
    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger = get_logger(__name__)
        logger.info(f"time_measurement.{name}", duration_ms=duration * 1000, **(attributes or {}))


# Import asyncio at the end to avoid circular imports
import asyncio
from reflexion_agent.observability.logging import get_logger