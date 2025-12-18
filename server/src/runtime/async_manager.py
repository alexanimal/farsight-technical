"""Async manager for concurrent execution.

This module provides the AsyncManager class that serves as the "traffic controller"
for concurrent execution. It manages concurrency limits, fan-out/fan-in patterns,
cancellation propagation, and group-level timeouts.

This avoids scattered asyncio.gather() calls, uncontrolled task spawning, and
impossible-to-debug cancellation behavior.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncManager:
    """Manager for structured concurrent execution.

    The async manager is responsible for:
    - Concurrency limits
    - Fan-out / fan-in patterns
    - Cancellation propagation
    - Group-level timeouts

    It enables structured, predictable concurrency across the system.
    """

    def __init__(
        self,
        max_concurrency: Optional[int] = None,
        default_timeout: Optional[float] = None,
    ):
        """Initialize the async manager.

        Args:
            max_concurrency: Maximum number of concurrent operations.
                If None, no limit is enforced.
            default_timeout: Default timeout in seconds for operation groups.
                If None, no timeout is enforced by default.
        """
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self._semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )

    async def execute_parallel(
        self,
        operations: List[Callable[[], Awaitable[T]]],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
        cancel_on_error: bool = False,
    ) -> List[T]:
        """Execute multiple operations in parallel with concurrency control.

        This is a fan-out/fan-in pattern that respects concurrency limits
        and handles cancellation propagation.

        Args:
            operations: List of async callables (no arguments) to execute.
            timeout: Timeout in seconds for the entire group. If None, uses default_timeout.
            return_exceptions: If True, exceptions are returned as results.
                If False, first exception is raised.
            cancel_on_error: If True, cancel remaining operations on first error.

        Returns:
            List of results in the same order as operations.

        Raises:
            TimeoutError: If the group execution exceeds the timeout.
            asyncio.CancelledError: If execution is cancelled.
        """
        timeout = timeout or self.default_timeout

        if not operations:
            return []

        # Create tasks with concurrency control
        tasks: List[asyncio.Task[T]] = []
        for operation in operations:
            task = asyncio.create_task(self._execute_with_semaphore(operation))
            tasks.append(task)

        try:
            # Wait for all tasks with optional timeout
            if timeout is not None:
                results = await asyncio.wait_for(
                    self._gather_with_cancellation(tasks, cancel_on_error),
                    timeout=timeout,
                )
            else:
                results = await self._gather_with_cancellation(tasks, cancel_on_error)

            # Handle exceptions based on return_exceptions flag
            if return_exceptions:
                return results
            else:
                # Check for exceptions and raise the first one
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Cancel remaining tasks if requested
                        if cancel_on_error:
                            for task in tasks[i + 1 :]:
                                if not task.done():
                                    task.cancel()
                        raise result
                return results

        except asyncio.TimeoutError:
            # Cancel all remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            raise TimeoutError(
                f"Parallel execution timed out after {timeout}s"
            ) from None

        except asyncio.CancelledError:
            # Cancel all remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def execute_batch(
        self,
        items: List[T],
        operation: Callable[[T], Awaitable[Any]],
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[Any]:
        """Execute an operation on a batch of items with concurrency control.

        This is useful for processing collections with controlled parallelism.

        Args:
            items: List of items to process.
            operation: Async callable that takes an item and returns a result.
            batch_size: Maximum number of concurrent operations. If None, uses max_concurrency.
            timeout: Timeout in seconds for the entire batch. If None, uses default_timeout.
            return_exceptions: If True, exceptions are returned as results.

        Returns:
            List of results in the same order as items.

        Raises:
            TimeoutError: If batch execution exceeds the timeout.
        """
        if not items:
            return []

        batch_size = batch_size or self.max_concurrency

        # Create operations list
        operations = [lambda item=item: operation(item) for item in items]

        # If batch_size is specified, process in batches
        if batch_size is not None and batch_size > 0:
            results = []
            for i in range(0, len(operations), batch_size):
                batch = operations[i : i + batch_size]
                batch_results = await self.execute_parallel(
                    batch, timeout=timeout, return_exceptions=return_exceptions
                )
                results.extend(batch_results)
            return results
        else:
            return await self.execute_parallel(
                operations, timeout=timeout, return_exceptions=return_exceptions
            )

    async def execute_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ) -> T:
        """Execute an operation with retry logic.

        Args:
            operation: Async callable to execute.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay in seconds between retries.
            timeout: Timeout in seconds for each attempt. If None, uses default_timeout.
            retry_on: Optional callable that takes an exception and returns True
                if the exception should trigger a retry.

        Returns:
            Result from the operation.

        Raises:
            Exception: The last exception if all retries fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                if timeout is not None:
                    return await asyncio.wait_for(operation(), timeout=timeout)
                else:
                    return await operation()

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if attempt < max_retries:
                    if retry_on is not None and not retry_on(e):
                        # Don't retry for this exception type
                        raise

                    # Wait before retrying
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)

                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying..."
                    )
                else:
                    # No more retries
                    logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                    raise

        # This should never be reached, but satisfy type checker
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unexpected error in execute_with_retry")

    async def _execute_with_semaphore(
        self, operation: Callable[[], Awaitable[T]]
    ) -> T:
        """Execute an operation with semaphore-based concurrency control.

        Args:
            operation: Async callable to execute.

        Returns:
            Result from the operation.
        """
        if self._semaphore is not None:
            async with self._semaphore:
                return await operation()
        else:
            return await operation()

    async def _gather_with_cancellation(
        self,
        tasks: List[asyncio.Task[T]],
        cancel_on_error: bool = False,
    ) -> List[T]:
        """Gather task results with proper cancellation handling.

        Args:
            tasks: List of tasks to gather.
            cancel_on_error: If True, cancel remaining tasks on first error.

        Returns:
            List of results from tasks.

        Raises:
            Exception: First exception encountered if cancel_on_error is True.
        """
        if not tasks:
            return []

        # Use asyncio.gather with return_exceptions to handle errors gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if cancel_on_error:
                    # Cancel remaining tasks
                    for task in tasks[i + 1 :]:
                        if not task.done():
                            task.cancel()
                    # Wait for cancellation
                    await asyncio.gather(*tasks[i + 1 :], return_exceptions=True)
                raise result

        return results


# Default async manager instance
_default_async_manager: Optional[AsyncManager] = None


def get_async_manager() -> AsyncManager:
    """Get the default async manager instance.

    Returns:
        The default AsyncManager instance.
    """
    global _default_async_manager
    if _default_async_manager is None:
        _default_async_manager = AsyncManager()
    return _default_async_manager


def set_async_manager(manager: AsyncManager) -> None:
    """Set the default async manager instance.

    Args:
        manager: The AsyncManager instance to use as default.
    """
    global _default_async_manager
    _default_async_manager = manager

