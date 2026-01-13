"""Initialize Temporal schedules for production use.

This script creates and manages Temporal schedules. It can be run during
deployment/initialization to set up scheduled workflows.

Usage:
    python -m src.temporal.initialize_schedules
    # Or as a module:
    from src.temporal.initialize_schedules import initialize_schedules
    await initialize_schedules()
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.temporal.client import DEFAULT_TASK_QUEUE, TemporalClient, get_client
from src.temporal.schedules import create_daily_schedule

logger = logging.getLogger(__name__)

# Schedule configuration
DAILY_SECTOR_ANALYSIS_SCHEDULE_ID = "daily-sector-analysis"
DEFAULT_SECTOR_NAME = os.getenv("SCHEDULED_SECTOR_NAME", "Technology")
DEFAULT_SCHEDULE_HOUR = int(os.getenv("SCHEDULE_HOUR", "9"))
DEFAULT_SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "0"))
DEFAULT_SCHEDULE_TIMEZONE = os.getenv("SCHEDULE_TIMEZONE", "UTC")


async def create_daily_sector_analysis_schedule(
    sector_name: str = DEFAULT_SECTOR_NAME,
    hour: int = DEFAULT_SCHEDULE_HOUR,
    minute: int = DEFAULT_SCHEDULE_MINUTE,
    timezone: str = DEFAULT_SCHEDULE_TIMEZONE,
    task_queue: Optional[str] = None,
    temporal_client: Optional[TemporalClient] = None,
) -> str:
    """Create or update the daily sector analysis schedule.

    This schedule runs the sector_analysis pipeline daily at the specified time.

    Args:
        sector_name: Name of the sector to analyze daily.
        hour: Hour of day (0-23). Default: 9 (9 AM).
        minute: Minute of hour (0-59). Default: 0.
        timezone: Timezone string. Default: "UTC".
        task_queue: Optional task queue name. Uses default if not provided.
        temporal_client: Optional TemporalClient instance. Creates new if not provided.

    Returns:
        The schedule ID.

    Raises:
        Exception: If schedule creation fails.
    """
    # Use provided client or create new one
    if temporal_client is None:
        client = await get_client()
    else:
        client = temporal_client

    # Ensure client is connected
    if not client._client_initialized:
        await client.connect()

    # Calculate time period (last 12 months)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)

    # Build pipeline context
    context = {
        "sector_name": sector_name,
        "time_period_start": start_date.isoformat() + "Z",
        "time_period_end": end_date.isoformat() + "Z",
        "granularity": "quarterly",
    }

    # Build workflow args for pipeline
    workflow_args = [
        "sector_analysis",  # pipeline_type
        context,  # context
        None,  # pipeline_config
    ]

    # Create schedule spec
    schedule_spec = create_daily_schedule(hour=hour, minute=minute, timezone=timezone)

    try:
        # Check if schedule already exists
        try:
            existing_schedule = await client.get_schedule(DAILY_SECTOR_ANALYSIS_SCHEDULE_ID)
            logger.info(f"Schedule {DAILY_SECTOR_ANALYSIS_SCHEDULE_ID} already exists, updating...")

            # Update schedule
            await client.update_schedule(
                DAILY_SECTOR_ANALYSIS_SCHEDULE_ID,
                schedule_spec=schedule_spec,
                enabled=True,
            )

            logger.info(
                f"Updated schedule {DAILY_SECTOR_ANALYSIS_SCHEDULE_ID} "
                f"(daily at {hour:02d}:{minute:02d} {timezone})"
            )
            return DAILY_SECTOR_ANALYSIS_SCHEDULE_ID

        except RuntimeError:
            # Schedule doesn't exist, create it
            logger.info(
                f"Creating schedule {DAILY_SECTOR_ANALYSIS_SCHEDULE_ID} "
                f"(daily at {hour:02d}:{minute:02d} {timezone})"
            )

            handle = await client.create_schedule(
                schedule_id=DAILY_SECTOR_ANALYSIS_SCHEDULE_ID,
                workflow_type="pipeline",
                workflow_args=workflow_args,
                schedule_spec=schedule_spec,
                task_queue=task_queue,
                workflow_id_template=f"{DAILY_SECTOR_ANALYSIS_SCHEDULE_ID}-{{timestamp}}",
                enabled=True,
            )

            logger.info(
                f"Successfully created schedule {DAILY_SECTOR_ANALYSIS_SCHEDULE_ID} "
                f"for sector: {sector_name}"
            )
            return DAILY_SECTOR_ANALYSIS_SCHEDULE_ID

    except Exception as e:
        logger.error(
            f"Failed to create/update schedule {DAILY_SECTOR_ANALYSIS_SCHEDULE_ID}: {e}",
            exc_info=True,
        )
        raise


async def initialize_schedules(
    sector_name: str = DEFAULT_SECTOR_NAME,
    hour: int = DEFAULT_SCHEDULE_HOUR,
    minute: int = DEFAULT_SCHEDULE_MINUTE,
    timezone: str = DEFAULT_SCHEDULE_TIMEZONE,
) -> Dict[str, str]:
    """Initialize all Temporal schedules.

    This function creates all required schedules for the system.

    Args:
        sector_name: Name of the sector for daily analysis.
        hour: Hour of day for daily schedule.
        minute: Minute of hour for daily schedule.
        timezone: Timezone for schedules.

    Returns:
        Dictionary mapping schedule IDs to status messages.

    Raises:
        Exception: If schedule initialization fails.
    """
    results = {}

    try:
        # Create daily sector analysis schedule
        schedule_id = await create_daily_sector_analysis_schedule(
            sector_name=sector_name,
            hour=hour,
            minute=minute,
            timezone=timezone,
        )
        results[schedule_id] = "created"

        logger.info(f"Successfully initialized {len(results)} schedule(s)")
        return results

    except Exception as e:
        logger.error(f"Failed to initialize schedules: {e}", exc_info=True)
        raise


async def main() -> None:
    """Main entrypoint for running as a script."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        results = await initialize_schedules()
        print(f"Successfully initialized {len(results)} schedule(s):")
        for schedule_id, status in results.items():
            print(f"  - {schedule_id}: {status}")
    except Exception as e:
        logger.error(f"Schedule initialization failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
