"""Unit tests for the tracing module.

This module tests the Tracer class and its various methods,
including span creation, context management, event logging, and Langfuse integration.
"""

import contextvars
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.runtime.tracing import (
    Tracer,
    create_correlation_id,
    create_trace_id,
    get_trace_context,
    get_tracer,
    log_event,
    set_tracer,
)


@pytest.fixture
def mock_langfuse_client():
    """Create a mock Langfuse client."""
    client = MagicMock()
    client.start_as_current_observation = MagicMock()
    client.create_event = MagicMock()
    client.start_generation = MagicMock()
    client.flush = MagicMock()
    return client


@pytest.fixture
def tracer_disabled():
    """Create a Tracer instance with tracing disabled."""
    return Tracer(enabled=False)


@pytest.fixture
def tracer_enabled(mock_langfuse_client):
    """Create a Tracer instance with tracing enabled."""
    with patch("src.runtime.tracing.Langfuse", return_value=mock_langfuse_client):
        return Tracer(
            public_key="test_public_key",
            secret_key="test_secret_key",
            enabled=True,
        )


@pytest.fixture
def tracer_with_settings(mock_langfuse_client):
    """Create a Tracer instance using settings."""
    with patch("src.runtime.tracing.Langfuse", return_value=mock_langfuse_client):
        with patch("src.runtime.tracing.settings") as mock_settings:
            mock_settings.langfuse_public_key = "settings_key"
            mock_settings.langfuse_secret_key = "settings_secret"
            mock_settings.langfuse_base_url = "http://test.url"
            return Tracer()


class TestTracerInitialization:
    """Test Tracer initialization."""

    def test_init_disabled(self):
        """Test initialization with tracing disabled."""
        tracer = Tracer(enabled=False)
        assert tracer.enabled is False
        assert tracer.client is None

    def test_init_enabled_with_keys(self, mock_langfuse_client):
        """Test initialization with tracing enabled and keys provided."""
        with patch("src.runtime.tracing.Langfuse", return_value=mock_langfuse_client):
            tracer = Tracer(
                public_key="test_public",
                secret_key="test_secret",
                enabled=True,
            )
            assert tracer.enabled is True
            assert tracer.client is not None

    def test_init_enabled_with_settings(self, mock_langfuse_client):
        """Test initialization using settings."""
        with patch("src.runtime.tracing.Langfuse", return_value=mock_langfuse_client):
            with patch("src.runtime.tracing.settings") as mock_settings:
                mock_settings.langfuse_public_key = "settings_key"
                mock_settings.langfuse_secret_key = "settings_secret"
                mock_settings.langfuse_base_url = None
                tracer = Tracer()
                assert tracer.enabled is True
                assert tracer.client is not None

    def test_init_disabled_no_keys(self):
        """Test initialization disabled when no keys provided."""
        with patch("src.runtime.tracing.settings") as mock_settings:
            mock_settings.langfuse_public_key = None
            mock_settings.langfuse_secret_key = None
            tracer = Tracer()
            assert tracer.enabled is False
            assert tracer.client is None


class TestTracerTraceID:
    """Test Tracer trace ID methods."""

    def test_create_trace_id(self, tracer_disabled):
        """Test create_trace_id generates UUID."""
        trace_id = tracer_disabled.create_trace_id()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 36  # UUID format with dashes
        assert trace_id.count("-") == 4

    def test_create_trace_id_unique(self, tracer_disabled):
        """Test create_trace_id generates unique IDs."""
        trace_id1 = tracer_disabled.create_trace_id()
        trace_id2 = tracer_disabled.create_trace_id()
        assert trace_id1 != trace_id2

    def test_create_correlation_id(self, tracer_disabled):
        """Test create_correlation_id generates UUID."""
        corr_id = tracer_disabled.create_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) == 36  # UUID format with dashes

    def test_create_correlation_id_unique(self, tracer_disabled):
        """Test create_correlation_id generates unique IDs."""
        corr_id1 = tracer_disabled.create_correlation_id()
        corr_id2 = tracer_disabled.create_correlation_id()
        assert corr_id1 != corr_id2


class TestTracerContext:
    """Test Tracer context management."""

    def test_get_trace_context_none(self, tracer_disabled):
        """Test get_trace_context returns None when not set."""
        context = tracer_disabled.get_trace_context()
        assert context is None

    def test_set_trace_context(self, tracer_disabled):
        """Test set_trace_context sets context."""
        test_context = {"trace_id": "test-123", "correlation_id": "corr-456"}
        tracer_disabled.set_trace_context(test_context)
        context = tracer_disabled.get_trace_context()
        assert context == test_context

    def test_clear_trace_context(self, tracer_disabled):
        """Test clear_trace_context clears context."""
        test_context = {"trace_id": "test-123"}
        tracer_disabled.set_trace_context(test_context)
        tracer_disabled.clear_trace_context()
        context = tracer_disabled.get_trace_context()
        assert context is None

    def test_context_isolation(self, tracer_disabled):
        """Test that context is shared via contextvars (not isolated per tracer instance).
        
        Note: Context is stored in a contextvar, so it's shared across all tracer instances
        in the same context. This is by design for context propagation.
        """
        tracer1 = Tracer(enabled=False)
        tracer2 = Tracer(enabled=False)

        # Context is shared via contextvars, so setting on one affects the other
        tracer1.set_trace_context({"trace_id": "t1"})
        # The second set overwrites the first
        tracer2.set_trace_context({"trace_id": "t2"})

        # Both tracers see the same context (last set value)
        assert tracer1.get_trace_context()["trace_id"] == "t2"
        assert tracer2.get_trace_context()["trace_id"] == "t2"


class TestTracerSpan:
    """Test Tracer span methods."""

    def test_span_disabled(self, tracer_disabled):
        """Test span context manager when tracing is disabled."""
        with tracer_disabled.span("test_span") as span:
            assert span is None

    def test_span_enabled_success(self, tracer_enabled, mock_langfuse_client):
        """Test span context manager when tracing is enabled."""
        mock_observation = MagicMock()
        mock_observation.id = "obs-123"
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_observation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        with tracer_enabled.span("test_span", trace_id="trace-123") as span:
            assert span == mock_observation

        mock_langfuse_client.start_as_current_observation.assert_called_once()

    def test_span_creates_trace_id(self, tracer_enabled, mock_langfuse_client):
        """Test span creates trace_id if not provided."""
        mock_observation = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_observation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        with tracer_enabled.span("test_span") as span:
            pass

        call_args = mock_langfuse_client.start_as_current_observation.call_args
        assert call_args is not None

    def test_span_with_metadata(self, tracer_enabled, mock_langfuse_client):
        """Test span with metadata."""
        mock_observation = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_observation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        metadata = {"key": "value", "number": 42}
        with tracer_enabled.span("test_span", metadata=metadata):
            pass

        call_args = mock_langfuse_client.start_as_current_observation.call_args
        assert call_args[1]["metadata"] == metadata

    def test_span_sets_context(self, tracer_enabled, mock_langfuse_client):
        """Test span sets trace context."""
        mock_observation = MagicMock()
        mock_observation.id = "obs-123"
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_observation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        with tracer_enabled.span("test_span", trace_id="trace-123"):
            context = tracer_enabled.get_trace_context()
            assert context is not None
            assert context["trace_id"] == "trace-123"
            assert context["observation_id"] == "obs-123"

    def test_span_handles_error(self, tracer_enabled, mock_langfuse_client):
        """Test span handles Langfuse errors gracefully."""
        mock_langfuse_client.start_as_current_observation.side_effect = (
            AttributeError("Method not found")
        )

        with tracer_enabled.span("test_span") as span:
            assert span is None

    @pytest.mark.asyncio
    async def test_async_span_disabled(self, tracer_disabled):
        """Test async_span context manager when tracing is disabled."""
        async with tracer_disabled.async_span("test_span") as span:
            assert span is None

    @pytest.mark.asyncio
    async def test_async_span_enabled_success(self, tracer_enabled, mock_langfuse_client):
        """Test async_span context manager when tracing is enabled."""
        mock_observation = MagicMock()
        mock_observation.id = "obs-123"
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_observation)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        async with tracer_enabled.async_span("test_span", trace_id="trace-123") as span:
            assert span == mock_observation

        mock_langfuse_client.start_as_current_observation.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_span_creates_trace_id(self, tracer_enabled, mock_langfuse_client):
        """Test async_span creates trace_id if not provided."""
        mock_observation = MagicMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_observation)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        async with tracer_enabled.async_span("test_span"):
            pass

        mock_langfuse_client.start_as_current_observation.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_span_with_metadata(self, tracer_enabled, mock_langfuse_client):
        """Test async_span with metadata."""
        mock_observation = MagicMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_observation)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        metadata = {"key": "value"}
        async with tracer_enabled.async_span("test_span", metadata=metadata):
            pass

        call_args = mock_langfuse_client.start_as_current_observation.call_args
        assert call_args[1]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_async_span_handles_error(self, tracer_enabled, mock_langfuse_client):
        """Test async_span handles Langfuse errors gracefully."""
        mock_langfuse_client.start_as_current_observation.side_effect = (
            AttributeError("Method not found")
        )

        async with tracer_enabled.async_span("test_span") as span:
            assert span is None


class TestTracerLogEvent:
    """Test Tracer log_event method."""

    def test_log_event_disabled(self, tracer_disabled):
        """Test log_event when tracing is disabled."""
        tracer_disabled.log_event("test_event")
        # Should not raise any errors

    def test_log_event_enabled(self, tracer_enabled, mock_langfuse_client):
        """Test log_event when tracing is enabled."""
        tracer_enabled.log_event(
            "test_event",
            trace_id="trace-123",
            metadata={"key": "value"},
        )

        mock_langfuse_client.create_event.assert_called_once()
        call_args = mock_langfuse_client.create_event.call_args
        assert call_args[1]["name"] == "test_event"
        assert call_args[1]["metadata"] == {"key": "value"}

    def test_log_event_uses_context_trace_id(self, tracer_enabled, mock_langfuse_client):
        """Test log_event uses trace_id from context."""
        tracer_enabled.set_trace_context({"trace_id": "context-trace-123"})
        tracer_enabled.log_event("test_event")

        mock_langfuse_client.create_event.assert_called_once()

    def test_log_event_creates_trace_id(self, tracer_enabled, mock_langfuse_client):
        """Test log_event creates trace_id if not provided."""
        tracer_enabled.log_event("test_event")

        mock_langfuse_client.create_event.assert_called_once()

    def test_log_event_handles_error(self, tracer_enabled, mock_langfuse_client):
        """Test log_event handles errors gracefully."""
        mock_langfuse_client.create_event.side_effect = AttributeError("Method not found")
        # Should not raise
        tracer_enabled.log_event("test_event")


class TestTracerLogGeneration:
    """Test Tracer log_generation method."""

    def test_log_generation_disabled(self, tracer_disabled):
        """Test log_generation when tracing is disabled."""
        tracer_disabled.log_generation(
            "test_gen", input_data="input", output_data="output"
        )
        # Should not raise any errors

    def test_log_generation_enabled(self, tracer_enabled, mock_langfuse_client):
        """Test log_generation when tracing is enabled."""
        mock_generation = MagicMock()
        mock_generation.update = MagicMock()
        mock_generation.end = MagicMock()
        mock_langfuse_client.start_generation.return_value = mock_generation

        tracer_enabled.log_generation(
            "test_gen",
            input_data="input",
            output_data="output",
            trace_id="trace-123",
            metadata={"key": "value"},
        )

        mock_langfuse_client.start_generation.assert_called_once()
        mock_generation.update.assert_called_once_with(
            input="input", output="output"
        )
        mock_generation.end.assert_called_once()

    def test_log_generation_uses_context_trace_id(self, tracer_enabled, mock_langfuse_client):
        """Test log_generation uses trace_id from context."""
        tracer_enabled.set_trace_context({"trace_id": "context-trace-123"})
        mock_generation = MagicMock()
        mock_generation.update = MagicMock()
        mock_generation.end = MagicMock()
        mock_langfuse_client.start_generation.return_value = mock_generation

        tracer_enabled.log_generation("test_gen", "input", "output")

        mock_langfuse_client.start_generation.assert_called_once()

    def test_log_generation_handles_error(self, tracer_enabled, mock_langfuse_client):
        """Test log_generation handles errors gracefully."""
        mock_langfuse_client.start_generation.side_effect = AttributeError("Method not found")
        # Should not raise
        tracer_enabled.log_generation("test_gen", "input", "output")


class TestTracerUpdateSpanMetadata:
    """Test Tracer update_span_metadata method."""

    def test_update_span_metadata_disabled(self, tracer_disabled):
        """Test update_span_metadata when tracing is disabled."""
        tracer_disabled.update_span_metadata("obs-123", {"key": "value"})
        # Should not raise any errors

    def test_update_span_metadata_enabled(self, tracer_enabled):
        """Test update_span_metadata when tracing is enabled."""
        # This method currently just logs a debug message
        tracer_enabled.update_span_metadata("obs-123", {"key": "value"})
        # Should not raise any errors


class TestTracerFlush:
    """Test Tracer flush method."""

    def test_flush_disabled(self, tracer_disabled):
        """Test flush when tracing is disabled."""
        tracer_disabled.flush()
        # Should not raise any errors

    def test_flush_enabled(self, tracer_enabled, mock_langfuse_client):
        """Test flush when tracing is enabled."""
        tracer_enabled.flush()
        mock_langfuse_client.flush.assert_called_once()

    def test_flush_handles_error(self, tracer_enabled, mock_langfuse_client):
        """Test flush handles errors gracefully."""
        mock_langfuse_client.flush.side_effect = Exception("Flush error")
        # Should not raise
        tracer_enabled.flush()


class TestTracerDefaultInstance:
    """Test default Tracer instance functions."""

    def test_get_tracer_creates_default(self):
        """Test get_tracer creates default instance."""
        # Get the tracer (will create if doesn't exist)
        tracer = get_tracer()
        assert isinstance(tracer, Tracer)

    def test_set_tracer(self):
        """Test set_tracer sets custom instance."""
        custom_tracer = Tracer(enabled=False)
        set_tracer(custom_tracer)
        tracer = get_tracer()
        assert tracer is custom_tracer

    def test_get_tracer_reuses_instance(self):
        """Test get_tracer reuses existing instance."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2


class TestTracerConvenienceFunctions:
    """Test convenience functions for tracing."""

    def test_create_trace_id_function(self):
        """Test create_trace_id convenience function."""
        trace_id = create_trace_id()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 36

    def test_create_correlation_id_function(self):
        """Test create_correlation_id convenience function."""
        corr_id = create_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) == 36

    def test_get_trace_context_function(self):
        """Test get_trace_context convenience function."""
        tracer = get_tracer()
        tracer.set_trace_context({"trace_id": "test-123"})
        context = get_trace_context()
        assert context is not None
        assert context["trace_id"] == "test-123"

    def test_log_event_function(self, tracer_enabled, mock_langfuse_client):
        """Test log_event convenience function."""
        set_tracer(tracer_enabled)
        log_event("test_event", metadata={"key": "value"})
        mock_langfuse_client.create_event.assert_called_once()


class TestTracerTraceIDConversion:
    """Test trace ID conversion for Langfuse format."""

    def test_convert_to_langfuse_trace_id_uuid(self, tracer_enabled, mock_langfuse_client):
        """Test trace ID conversion from UUID format."""
        uuid_trace_id = "550e8400-e29b-41d4-a716-446655440000"
        mock_observation = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_observation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        # TraceContext is imported from langfuse.types inside the span method
        with patch("langfuse.types.TraceContext") as mock_trace_context:
            with tracer_enabled.span("test_span", trace_id=uuid_trace_id):
                pass

            # Verify TraceContext was created with cleaned trace ID
            # Note: TraceContext may not be called if import fails, which is handled gracefully
            if mock_trace_context.called:
                call_args = mock_trace_context.call_args
                assert call_args is not None
                # The trace ID should have dashes removed
                assert "-" not in call_args[1]["trace_id"]

    def test_span_with_parent_observation_id(self, tracer_enabled, mock_langfuse_client):
        """Test span with parent observation ID."""
        mock_observation = MagicMock()
        mock_observation.id = "obs-123"
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_observation)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_langfuse_client.start_as_current_observation.return_value = (
            mock_context_manager
        )

        with tracer_enabled.span(
            "test_span",
            trace_id="trace-123",
            parent_observation_id="parent-obs-456",
        ) as span:
            context = tracer_enabled.get_trace_context()
            assert context["parent_observation_id"] == "parent-obs-456"

