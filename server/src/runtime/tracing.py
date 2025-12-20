"""Tracing module for agent operations using Langfuse.

This module provides the system's memory of what happened. Agent systems are
non-linear, parallel, and probabilistic — traces are essential for debugging,
evaluation, auditing, and replay.

It provides:
- Span creation
- Context propagation
- Correlation IDs
- Structured metadata
"""

import contextvars
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, Union

from langfuse import Langfuse

from src.config import settings

logger = logging.getLogger(__name__)

# Context variable for trace context propagation
_trace_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("trace_context", default=None)
)


def _convert_to_langfuse_trace_id(trace_id: str) -> str:
    """Convert a trace ID to Langfuse format (32 lowercase hex chars, no dashes).
    
    Langfuse expects trace IDs to be exactly 32 lowercase hexadecimal characters
    without dashes. UUIDs have dashes, so we remove them.
    
    Args:
        trace_id: Trace ID in any format (UUID, hex string, etc.)
        
    Returns:
        Trace ID in Langfuse format (32 lowercase hex chars, no dashes)
    """
    # Remove dashes and convert to lowercase
    cleaned = trace_id.replace("-", "").lower()
    
    # If it's already 32 chars, return it
    if len(cleaned) == 32:
        return cleaned
    
    # If it's a UUID (36 chars with dashes removed = 32), return it
    # Otherwise, pad or truncate to 32 chars (though this shouldn't happen with UUIDs)
    if len(cleaned) > 32:
        return cleaned[:32]
    elif len(cleaned) < 32:
        # Pad with zeros if needed (shouldn't happen with UUIDs)
        return cleaned.ljust(32, "0")
    
    return cleaned


class Tracer:
    """Tracer for agent operations using Langfuse.

    This class manages tracing spans, context propagation, and correlation IDs
    for agent invocations, tool calls, orchestration decisions, retries, failures,
    cancellations, and state transitions.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: bool = True,
    ):
        """Initialize the tracer.

        Args:
            public_key: Langfuse public key. If None, uses settings.langfuse_public_key.
            secret_key: Langfuse secret key. If None, uses settings.langfuse_secret_key.
            base_url: Langfuse base URL. If None, uses settings.langfuse_base_url.
            enabled: Whether tracing is enabled. If False, operations are no-ops.
        """
        self.enabled = enabled and bool(public_key or settings.langfuse_public_key)

        if self.enabled:
            self.client: Optional[Langfuse] = Langfuse(
                public_key=public_key or settings.langfuse_public_key or "",
                secret_key=secret_key or settings.langfuse_secret_key or "",
                base_url=base_url or settings.langfuse_base_url,
            )
        else:
            self.client = None
            logger.warning("Tracing is disabled. Set Langfuse credentials to enable.")

    def create_trace_id(self) -> str:
        """Create a unique trace ID.

        Returns:
            A unique trace ID string (UUID).
        """
        return str(uuid.uuid4())

    def create_correlation_id(self) -> str:
        """Create a unique correlation ID for request tracking.

        Returns:
            A unique correlation ID string (UUID).
        """
        return str(uuid.uuid4())

    def get_trace_context(self) -> Optional[Dict[str, Any]]:
        """Get the current trace context.

        Returns:
            Current trace context dictionary or None.
        """
        return _trace_context.get()

    def set_trace_context(self, context: Dict[str, Any]) -> None:
        """Set the trace context for propagation.

        Args:
            context: Dictionary containing trace context (trace_id, correlation_id, etc.).
        """
        _trace_context.set(context)

    def clear_trace_context(self) -> None:
        """Clear the current trace context."""
        _trace_context.set(None)

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a tracing span context manager.

        Args:
            name: Name of the span.
            trace_id: Optional trace ID. If None, creates a new one.
            parent_observation_id: Optional parent observation ID for nesting.
            metadata: Optional metadata to attach to the span.

        Yields:
            The observation object for adding additional metadata.
        """
        if not self.enabled or self.client is None:
            yield None
            return

        trace_id = trace_id or self.create_trace_id()
        # Langfuse v3 API - use start_as_current_observation() which replaces deprecated start_as_current_span()
        # First, ensure trace context is set if we have a trace_id
        if trace_id:
            # Set trace context for Langfuse to use
            current_context = self.get_trace_context() or {}
            current_context["trace_id"] = trace_id
            self.set_trace_context(current_context)
        
        # Prepare trace_context for Langfuse if we have a trace_id
        trace_context = None
        if trace_id:
            try:
                from langfuse.types import TraceContext
                # Convert UUID to Langfuse format (32 hex chars, no dashes)
                langfuse_trace_id = _convert_to_langfuse_trace_id(trace_id)
                trace_context = TraceContext(trace_id=langfuse_trace_id)
            except (ImportError, AttributeError):
                # Fallback: Langfuse will use its internal context management
                pass
            except Exception as e:
                # Log but don't fail if trace context creation fails
                logger.debug(f"Could not create TraceContext: {e}, using internal context management")
                trace_context = None
        
        try:
            # Use start_as_current_observation() which replaces deprecated start_as_current_span()
            # Parent relationships are handled automatically through Langfuse's current span context
            # This returns a context manager that automatically handles the observation lifecycle
            with self.client.start_as_current_observation(
                name=name,
                as_type="span",
                metadata=metadata or {},
                trace_context=trace_context,
            ) as observation:
                # Set trace context for our internal tracking
                context = {
                    "trace_id": trace_id,
                    "span_name": name,
                    "observation_id": (
                        observation.id if hasattr(observation, "id") else None
                    ),
                }
                if parent_observation_id:
                    context["parent_observation_id"] = parent_observation_id

                self.set_trace_context(context)

                yield observation
        except (AttributeError, TypeError) as e:
            # Fallback if method doesn't exist or has wrong signature
            logger.warning(
                f"Langfuse start_as_current_observation() not available: {e}, tracing disabled for span: {name}"
            )
            yield None
            return
        except Exception as e:
            logger.warning(f"Error in span {name}: {e}")
            yield None
            return

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create an async tracing span context manager.

        Args:
            name: Name of the span.
            trace_id: Optional trace ID. If None, creates a new one.
            parent_observation_id: Optional parent observation ID for nesting.
            metadata: Optional metadata to attach to the span.

        Yields:
            The observation object for adding additional metadata.
        """
        if not self.enabled or self.client is None:
            yield None
            return

        trace_id = trace_id or self.create_trace_id()
        # Langfuse v3 API - use start_as_current_observation() which replaces deprecated start_as_current_span()
        # First, ensure trace context is set if we have a trace_id
        if trace_id:
            # Set trace context for Langfuse to use
            current_context = self.get_trace_context() or {}
            current_context["trace_id"] = trace_id
            self.set_trace_context(current_context)
        
        # Prepare trace_context for Langfuse if we have a trace_id
        trace_context = None
        if trace_id:
            try:
                from langfuse.types import TraceContext
                # Convert UUID to Langfuse format (32 hex chars, no dashes)
                langfuse_trace_id = _convert_to_langfuse_trace_id(trace_id)
                trace_context = TraceContext(trace_id=langfuse_trace_id)
            except (ImportError, AttributeError):
                # Fallback: Langfuse will use its internal context management
                pass
            except Exception as e:
                # Log but don't fail if trace context creation fails
                logger.debug(f"Could not create TraceContext: {e}, using internal context management")
                trace_context = None
        
        try:
            # Use start_as_current_observation() which replaces deprecated start_as_current_span()
            # Parent relationships are handled automatically through Langfuse's current span context
            # This returns a context manager that automatically handles the observation lifecycle
            async with self.client.start_as_current_observation(
                name=name,
                as_type="span",
                metadata=metadata or {},
                trace_context=trace_context,
            ) as observation:
                # Set trace context for our internal tracking
                context = {
                    "trace_id": trace_id,
                    "span_name": name,
                    "observation_id": (
                        observation.id if hasattr(observation, "id") else None
                    ),
                }
                if parent_observation_id:
                    context["parent_observation_id"] = parent_observation_id

                self.set_trace_context(context)

                yield observation
        except (AttributeError, TypeError) as e:
            # Fallback if method doesn't exist or has wrong signature
            logger.warning(
                f"Langfuse start_as_current_observation() not available: {e}, tracing disabled for span: {name}"
            )
            yield None
            return
        except Exception as e:
            logger.warning(f"Error in async span {name}: {e}")
            yield None
            return

    def log_event(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an event to the trace.

        Args:
            name: Name of the event.
            trace_id: Optional trace ID. If None, uses current context or creates new.
            parent_observation_id: Optional parent observation ID.
            metadata: Optional metadata for the event.
        """
        if not self.enabled:
            return

        # Get trace_id from context if not provided
        if trace_id is None:
            context = self.get_trace_context()
            trace_id = context.get("trace_id") if context else None

        trace_id = trace_id or self.create_trace_id()

        if self.client is None:
            return

        # Set trace context so Langfuse can use it
        if trace_id:
            current_context = self.get_trace_context() or {}
            current_context["trace_id"] = trace_id
            self.set_trace_context(current_context)

        try:
            # Langfuse v3 API - use create_event() method
            # trace_id is managed through Langfuse context, not passed as parameter
            self.client.create_event(
                name=name,
                parent_observation_id=parent_observation_id,
                metadata=metadata or {},
            )
        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Langfuse create_event() not available: {e}, event not logged: {name}"
            )
        except Exception as e:
            logger.warning(f"Error logging event {name}: {e}")

    def log_generation(
        self,
        name: str,
        input_data: Any,
        output_data: Any,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a generation (e.g., LLM call) to the trace.

        Args:
            name: Name of the generation.
            input_data: Input data for the generation.
            output_data: Output data from the generation.
            trace_id: Optional trace ID. If None, uses current context or creates new.
            parent_observation_id: Optional parent observation ID.
            metadata: Optional metadata for the generation.
        """
        if not self.enabled:
            return

        # Get trace_id from context if not provided
        if trace_id is None:
            context = self.get_trace_context()
            trace_id = context.get("trace_id") if context else None

        trace_id = trace_id or self.create_trace_id()

        if self.client is None:
            return

        # Set trace context so Langfuse can use it
        if trace_id:
            current_context = self.get_trace_context() or {}
            current_context["trace_id"] = trace_id
            self.set_trace_context(current_context)

        try:
            # Langfuse v3 API - use start_generation() method
            # trace_id is managed through Langfuse context, not passed as parameter
            # Note: start_generation() returns a generation object that needs to be ended
            generation = self.client.start_generation(
                name=name,
                parent_observation_id=parent_observation_id,
                metadata=metadata or {},
            )
            # Set input and output on the generation object
            if hasattr(generation, "update"):
                generation.update(input=input_data, output=output_data)
            # End the generation
            if hasattr(generation, "end"):
                generation.end()
        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Langfuse start_generation() not available: {e}, generation not logged: {name}"
            )
        except Exception as e:
            logger.warning(f"Error logging generation {name}: {e}")

    def update_span_metadata(
        self,
        observation_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Update metadata for an existing span.

        Args:
            observation_id: The observation ID of the span to update.
            metadata: Metadata to add/update.
        """
        if not self.enabled:
            return

        try:
            # Langfuse doesn't have a direct update method, but we can use the context
            # For now, we'll log a warning that this needs to be done during span creation
            logger.debug(
                f"Metadata update requested for observation {observation_id}. "
                "Consider adding metadata during span creation or use span.update() if available."
            )
        except Exception as e:
            logger.warning(f"Error updating span metadata: {e}")

    def flush(self) -> None:
        """Flush pending traces to Langfuse.

        This should be called periodically or at the end of operations
        to ensure all traces are sent.
        """
        if not self.enabled:
            return

        if self.client is None:
            return

        try:
            self.client.flush()
        except Exception as e:
            logger.warning(f"Error flushing traces: {e}")


# Default tracer instance
_default_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the default tracer instance.

    Returns:
        The default Tracer instance.
    """
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the default tracer instance.

    Args:
        tracer: The Tracer instance to use as default.
    """
    global _default_tracer
    _default_tracer = tracer


# Convenience functions for common operations
def create_trace_id() -> str:
    """Create a trace ID using the default tracer.

    Returns:
        A unique trace ID string.
    """
    return get_tracer().create_trace_id()


def create_correlation_id() -> str:
    """Create a correlation ID using the default tracer.

    Returns:
        A unique correlation ID string.
    """
    return get_tracer().create_correlation_id()


def get_trace_context() -> Optional[Dict[str, Any]]:
    """Get the current trace context.

    Returns:
        Current trace context dictionary or None.
    """
    return get_tracer().get_trace_context()


def log_event(
    name: str,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an event using the default tracer.

    Args:
        name: Name of the event.
        trace_id: Optional trace ID.
        metadata: Optional metadata for the event.
    """
    get_tracer().log_event(name=name, trace_id=trace_id, metadata=metadata)
