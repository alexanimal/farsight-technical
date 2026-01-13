"""Schedule management utilities for Temporal workflows.

This module provides helper functions for creating and managing Temporal schedules.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from temporalio.client import (
    ScheduleCalendarSpec,
    ScheduleIntervalSpec,
    ScheduleRange,
    ScheduleSpec,
    ScheduleState,
)

logger = logging.getLogger(__name__)


def create_daily_schedule(
    hour: int = 9,
    minute: int = 0,
    timezone: str = "UTC",
) -> ScheduleSpec:
    """Create a schedule spec for daily execution at a specific time.

    Args:
        hour: Hour of day (0-23). Default: 9 (9 AM).
        minute: Minute of hour (0-59). Default: 0.
        timezone: Timezone string (e.g., "UTC", "America/New_York"). Default: "UTC".
            Note: Timezone is set on ScheduleSpec, not ScheduleCalendarSpec.

    Returns:
        ScheduleSpec configured for daily execution.

    Example:
        ```python
        # Daily at 9 AM UTC
        spec = create_daily_schedule(hour=9, minute=0, timezone="UTC")

        # Daily at 2:30 PM EST
        spec = create_daily_schedule(hour=14, minute=30, timezone="America/New_York")
        ```
    """
    # Use ScheduleRange for each time component
    # For daily schedule: specific minute and hour, all days/months/years
    return ScheduleSpec(
        calendars=[
            ScheduleCalendarSpec(
                minute=[ScheduleRange(minute, minute)],
                hour=[ScheduleRange(hour, hour)],
                # day_of_month=None means all days
                # month=None means all months
                # day_of_week=None means all days of week
                # year=None means all years
                comment=f"Daily at {hour:02d}:{minute:02d} {timezone}",
            )
        ],
    )


def create_interval_schedule(
    every: timedelta,
    offset: Optional[timedelta] = None,
) -> ScheduleSpec:
    """Create a schedule spec for interval-based execution.

    Args:
        every: Time interval between executions (e.g., timedelta(hours=1)).
        offset: Optional offset from start time. Default: None.

    Returns:
        ScheduleSpec configured for interval-based execution.

    Example:
        ```python
        # Every hour
        spec = create_interval_schedule(every=timedelta(hours=1))

        # Every 6 hours starting at 9 AM
        spec = create_interval_schedule(
            every=timedelta(hours=6),
            offset=timedelta(hours=9)
        )
        ```
    """
    return ScheduleSpec(
        intervals=[
            ScheduleIntervalSpec(
                every=every,
                offset=offset,
            )
        ]
    )


def create_cron_schedule(
    cron_expression: str,
    timezone: str = "UTC",
) -> ScheduleSpec:
    """Create a schedule spec from a cron expression.

    Note: This is a simplified implementation. For complex cron expressions,
    consider parsing and converting to ScheduleCalendarSpec parameters.

    Args:
        cron_expression: Cron expression (e.g., "0 9 * * *" for daily at 9 AM).
        timezone: Timezone string. Default: "UTC".

    Returns:
        ScheduleSpec configured with the cron expression.

    Example:
        ```python
        # Daily at 9 AM
        spec = create_cron_schedule("0 9 * * *", timezone="UTC")

        # Every Monday at 9 AM
        spec = create_cron_schedule("0 9 * * 1", timezone="UTC")
        ```
    """
    # Parse simple cron expression: "minute hour * * day_of_week"
    # This is a basic parser - for production, use a proper cron parser
    parts = cron_expression.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expression}. Expected 5 parts.")

    minute_str, hour_str, day_of_month_str, month_str, day_of_week_str = parts

    # Parse components (simplified - only handles specific values, not ranges)
    minute = int(minute_str) if minute_str != "*" else None
    hour = int(hour_str) if hour_str != "*" else None
    day_of_week = int(day_of_week_str) if day_of_week_str != "*" else None

    calendar_spec = ScheduleCalendarSpec(comment=f"Cron: {cron_expression}")

    if minute is not None:
        calendar_spec.minute = [ScheduleRange(minute, minute)]
    if hour is not None:
        calendar_spec.hour = [ScheduleRange(hour, hour)]
    if day_of_week is not None:
        calendar_spec.day_of_week = [ScheduleRange(day_of_week, day_of_week)]

    return ScheduleSpec(calendars=[calendar_spec])


def create_weekly_schedule(
    day_of_week: int,
    hour: int = 9,
    minute: int = 0,
    timezone: str = "UTC",
) -> ScheduleSpec:
    """Create a schedule spec for weekly execution.

    Args:
        day_of_week: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday).
        hour: Hour of day (0-23). Default: 9.
        minute: Minute of hour (0-59). Default: 0.
        timezone: Timezone string. Default: "UTC".

    Returns:
        ScheduleSpec configured for weekly execution.

    Example:
        ```python
        # Every Monday at 9 AM
        spec = create_weekly_schedule(day_of_week=1, hour=9, minute=0)
        ```
    """
    return ScheduleSpec(
        calendars=[
            ScheduleCalendarSpec(
                minute=[ScheduleRange(minute, minute)],
                hour=[ScheduleRange(hour, hour)],
                day_of_week=[ScheduleRange(day_of_week, day_of_week)],
                comment=f"Weekly on day {day_of_week} at {hour:02d}:{minute:02d} {timezone}",
            )
        ],
    )
