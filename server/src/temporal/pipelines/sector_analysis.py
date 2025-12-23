"""Sector analysis pipeline.

This pipeline performs a standardized analysis of funding trends within a sector.

Pipeline steps:
1. Search for organizations in the sector (semantic search)
2. Aggregate funding trends over time
3. Calculate funding velocity metrics
4. Identify funding patterns
5. Generate final analysis using sector_trends agent
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from src.temporal.pipelines.base import PipelineBase, PipelineConfig, PipelineStep

logger = logging.getLogger(__name__)


class SectorAnalysisPipeline(PipelineBase):
    """Pipeline for analyzing funding trends within a sector.

    This pipeline executes a fixed sequence of steps to analyze sector funding:
    - Organization discovery via semantic search
    - Funding trend aggregation
    - Velocity calculation
    - Pattern identification
    - Final analysis generation
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the sector analysis pipeline."""
        if config is None:
            config = PipelineConfig(
                name="sector_analysis",
                description="Analyze funding trends and patterns within a sector",
                version="1.0.0",
                metadata={
                    "category": "financial_analysis",
                    "output_type": "sector_trends_report",
                },
            )
        super().__init__(config)

    def build_steps(
        self,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[PipelineStep]:
        """Build the steps for sector analysis pipeline.

        Args:
            context: Pipeline context with:
                - sector_name: Name of the sector to analyze
                - time_period_start: Start date (ISO format)
                - time_period_end: End date (ISO format)
                - granularity: Time granularity (monthly/quarterly/yearly)
                - min_funding_amount: Optional minimum funding filter
            config: Optional configuration overrides.

        Returns:
            List of PipelineStep objects in execution order.
        """
        # Extract parameters from context
        sector_name = context.get("sector_name", "")
        time_period_start = context.get("time_period_start", "")
        time_period_end = context.get("time_period_end", "")
        granularity = context.get("granularity", "quarterly")
        min_funding_amount = context.get("min_funding_amount")

        # Merge config overrides if provided
        if config:
            granularity = config.get("granularity", granularity)
            min_funding_amount = config.get("min_funding_amount", min_funding_amount)

        steps = []

        # Step 1: Semantic search for organizations in sector
        steps.append(
            PipelineStep(
                step_id="step_1_search_orgs",
                step_type="tool",
                name="semantic_search_organizations",
                parameters={
                    "text": sector_name,
                    "total_funding_usd_min": min_funding_amount,
                    "top_k": 100,
                },
            )
        )

        # Step 2: Aggregate funding trends
        steps.append(
            PipelineStep(
                step_id="step_2_aggregate_trends",
                step_type="tool",
                name="aggregate_funding_trends",
                parameters={
                    # org_uuids will be populated from step_1 result
                    "time_period_start": time_period_start,
                    "time_period_end": time_period_end,
                    "granularity": granularity,
                    "min_funding_amount": min_funding_amount,
                },
            )
        )

        # Step 3: Calculate funding velocity
        steps.append(
            PipelineStep(
                step_id="step_3_calculate_velocity",
                step_type="tool",
                name="calculate_funding_velocity",
                parameters={
                    # trend_data will be populated from step_2 result
                },
            )
        )

        # Step 4: Identify funding patterns
        steps.append(
            PipelineStep(
                step_id="step_4_identify_patterns",
                step_type="tool",
                name="identify_funding_patterns",
                parameters={
                    # trend_data will be populated from step_2 result
                    "granularity": granularity,
                    "anomaly_threshold": 2.0,
                    "detect_seasonality": True,
                },
            )
        )

        # Step 5: Generate final analysis using sector_trends agent
        steps.append(
            PipelineStep(
                step_id="step_5_generate_analysis",
                step_type="agent",
                name="sector_trends",
                parameters={
                    # Agent will receive accumulated pipeline_data
                    "query": f"Analyze funding trends for {sector_name}",
                },
            )
        )

        return steps

    def resolve_step_parameters(
        self,
        step: PipelineStep,
        pipeline_data: Dict[str, Any],
        step_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve step parameters with sector-specific logic.

        Args:
            step: The PipelineStep to resolve parameters for.
            pipeline_data: Accumulated data from all previous steps.
            step_results: Dictionary mapping step_id to step results.

        Returns:
            Resolved parameters dictionary.
        """
        # Call base implementation first
        resolved = super().resolve_step_parameters(step, pipeline_data, step_results)

        # Sector-specific parameter resolution
        # Step 2 needs org_uuids from step 1
        if step.step_id == "step_2_aggregate_trends":
            if "org_uuids" not in resolved:
                # Try to get organizations from step_1 result
                step_1_result = step_results.get("step_1_search_orgs")
                # semantic_search_organizations returns a list directly, not a dict
                if isinstance(step_1_result, list):
                    organizations = step_1_result
                elif isinstance(step_1_result, dict):
                    organizations = step_1_result.get("organizations", [])
                else:
                    organizations = []

                if organizations:
                    resolved["org_uuids"] = [
                        org.get("org_uuid") if isinstance(org, dict) else str(org)
                        for org in organizations
                        if org is not None
                    ]
                    logger.debug(
                        f"Resolved {len(resolved['org_uuids'])} org_uuids from step_1 result"
                    )

        # Step 3 and 4 need trend_data from step 2
        if step.step_id in ("step_3_calculate_velocity", "step_4_identify_patterns"):
            if "trend_data" not in resolved:
                step_2_result = step_results.get("step_2_aggregate_trends", {})
                if not step_2_result:
                    # Log warning if step 2 result is missing
                    logger.warning(
                        f"Step {step.step_id} requires trend_data from step_2_aggregate_trends, "
                        f"but step_2 result not found in step_results. "
                        f"Available step_ids: {list(step_results.keys())}"
                    )
                trend_data = step_2_result.get("trend_data", [])
                if trend_data:
                    resolved["trend_data"] = trend_data
                else:
                    logger.warning(
                        f"Step {step.step_id} requires trend_data, but step_2 result "
                        f"does not contain 'trend_data' field. Step_2 result keys: {list(step_2_result.keys())}"
                    )

        return resolved

    def build_final_result(
        self,
        step_results: Dict[str, Dict[str, Any]],
        pipeline_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the final consolidated result for sector analysis.

        Args:
            step_results: Dictionary mapping step_id to step results.
            pipeline_data: Accumulated data from all steps.
            context: Original pipeline context.

        Returns:
            Final consolidated result dictionary.
        """
        # Get organizations count from step 1
        step_1_result = step_results.get("step_1_search_orgs")
        # semantic_search_organizations returns a list directly, not a dict
        if isinstance(step_1_result, list):
            organizations = step_1_result
        elif isinstance(step_1_result, dict):
            organizations = step_1_result.get("organizations", [])
        else:
            organizations = []
        organizations_count = len(organizations) if isinstance(organizations, list) else 0

        return {
            "pipeline_type": "sector_analysis",
            "pipeline_name": self.name,
            "pipeline_version": self.config.version,
            "sector_name": context.get("sector_name"),
            "time_period": {
                "start": context.get("time_period_start"),
                "end": context.get("time_period_end"),
            },
            "organizations_found": organizations_count,
            "trend_analysis": step_results.get("step_2_aggregate_trends", {}),
            "velocity_metrics": step_results.get("step_3_calculate_velocity", {}),
            "patterns": step_results.get("step_4_identify_patterns", {}),
            "final_analysis": step_results.get("step_5_generate_analysis", {}),
            "metadata": {
                "steps_executed": len(step_results),
                "granularity": context.get("granularity", "quarterly"),
            },
        }
