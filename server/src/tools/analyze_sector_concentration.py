"""Tool for analyzing sector concentration in a portfolio.

This tool categorizes portfolio companies by sector using their existing
categories and category_groups data from the database and calculates
sector concentration metrics including the Herfindahl index.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from src.contracts.tool_io import ToolMetadata, ToolOutput, ToolParameterSchema, create_tool_output
from src.models.acquisitions import AcquisitionModel
from src.models.organizations import OrganizationModel

logger = logging.getLogger(__name__)


def get_tool_metadata() -> ToolMetadata:
    """Get the ToolMetadata for the analyze_sector_concentration tool.

    Returns:
        ToolMetadata object describing this tool's capabilities and parameters.
    """
    return ToolMetadata(
        name="analyze_sector_concentration",
        description="Analyze sector concentration in a portfolio by categorizing companies using their existing categories and category_groups data from the database, then calculating concentration metrics including the Herfindahl index.",
        version="2.0.0",
        parameters=[
            ToolParameterSchema(
                name="portfolio_companies",
                type="array",
                description="List of portfolio companies from find_investor_portfolio. Each item should have 'org_uuid' and optionally 'name', 'total_invested_usd'. Required.",
                required=True,
            ),
            ToolParameterSchema(
                name="sectors",
                type="array",
                description="List of sector names to categorize companies into (e.g., ['AI', 'fintech', 'healthcare']). Companies are matched to sectors based on their categories/category_groups. If not provided, automatically uses all unique categories/category_groups from the portfolio companies. Optional.",
                required=False,
            ),
            ToolParameterSchema(
                name="use_category_groups",
                type="boolean",
                description="If true, use category_groups for matching. If false, use categories. Default: true (uses category_groups).",
                required=False,
                default=True,
            ),
            ToolParameterSchema(
                name="case_sensitive",
                type="boolean",
                description="Whether sector matching should be case-sensitive. Default: false (case-insensitive matching).",
                required=False,
                default=False,
            ),
        ],
        returns={
            "type": "object",
            "description": "Sector concentration analysis results",
            "properties": {
                "sectors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sector": {"type": "string"},
                            "company_count": {"type": "integer"},
                            "percentage": {"type": "number"},
                            "total_investment_usd": {
                                "type": "integer",
                                "nullable": True,
                            },
                            "exit_rate_pct": {"type": "number", "nullable": True},
                        },
                    },
                },
                "concentration_index": {
                    "type": "number",
                    "description": "Herfindahl index (0-1) measuring concentration. Higher values indicate more concentration.",
                },
                "uncategorized_companies": {
                    "type": "integer",
                    "description": "Number of companies that couldn't be categorized into any sector",
                },
            },
        },
        cost_per_call=None,  # Database queries and computation, minimal cost
        estimated_latency_ms=500.0,  # Database queries for portfolio companies
        timeout_seconds=60.0,
        side_effects=False,  # Read-only operation
        idempotent=True,  # Safe to retry
        tags=["analysis", "portfolio", "sector", "concentration", "read-only"],
    )


def _calculate_herfindahl_index(sector_percentages: List[float]) -> float:
    """Calculate Herfindahl index for sector concentration.

    Args:
        sector_percentages: List of percentages (0-100) for each sector.

    Returns:
        Herfindahl index (0-1). Higher values indicate more concentration.
    """
    if not sector_percentages:
        return 0.0

    # Convert percentages to proportions (0-1)
    proportions = [p / 100.0 for p in sector_percentages]

    # Herfindahl index = sum of squared proportions
    herfindahl = sum(p * p for p in proportions)

    return round(herfindahl, 4)


def _match_category_to_sector(category: str, sector: str, case_sensitive: bool = False) -> bool:
    """Check if a category matches a sector name.

    Args:
        category: Category or category_group from organization.
        sector: Sector name to match against.
        case_sensitive: Whether matching should be case-sensitive.

    Returns:
        True if category matches sector, False otherwise.
    """
    if not category or not sector:
        return False

    if not case_sensitive:
        category = category.lower()
        sector = sector.lower()

    # Exact match
    if category == sector:
        return True

    # Check if sector keyword appears in category
    # e.g., "AI" matches "Artificial Intelligence", "fintech" matches "Financial Technology"
    sector_keywords = sector.split()
    for keyword in sector_keywords:
        if keyword in category:
            return True

    # Check if category appears in sector
    # e.g., "healthcare" in "healthcare startups"
    if category in sector:
        return True

    return False


@observe(as_type="tool")
async def analyze_sector_concentration(
    portfolio_companies: List[Dict[str, Any]],
    sectors: Optional[List[str]] = None,
    use_category_groups: bool = True,
    case_sensitive: bool = False,
) -> ToolOutput:
    """Analyze sector concentration in a portfolio.

    This tool:
    1. Looks up each portfolio company in the database
    2. Uses their existing categories or category_groups to match to sectors
    3. Categorizes each company into matching sectors
    4. Calculates sector distribution and concentration metrics
    5. Calculates exit rates per sector

    Args:
        portfolio_companies: List of portfolio company dictionaries from find_investor_portfolio.
            Each should have 'org_uuid' and optionally 'name', 'total_invested_usd'.
        sectors: List of sector names to categorize into. Companies are matched based on
            their categories/category_groups. If not provided, uses defaults.
        use_category_groups: If true, use category_groups for matching. If false, use categories.
        case_sensitive: Whether sector matching should be case-sensitive.

    Returns:
        ToolOutput object containing:
        - success: Whether the analysis succeeded
        - result: Dictionary with:
            - sectors: List of sector breakdowns with counts and percentages
            - concentration_index: Herfindahl index (0-1)
            - uncategorized_companies: Count of companies not assigned to any sector
        - error: Error message (if failed)
        - execution_time_ms: Time taken to execute
        - metadata: Additional metadata about the execution

    Example:
        ```python
        # Analyze sector concentration
        portfolio = [
            {"org_uuid": "...", "name": "Company A", "total_invested_usd": 1000000}
        ]
        result = await analyze_sector_concentration(
            portfolio_companies=portfolio,
            sectors=["AI", "fintech", "healthcare"]
        )
        ```
    """
    start_time = time.time()
    try:
        # Validate inputs
        if not portfolio_companies:
            raise ValueError("portfolio_companies cannot be empty")

        # If sectors not provided, we'll auto-generate from company categories after fetching them
        use_auto_sectors = sectors is None

        logger.info(
            f"Analyzing sector concentration for {len(portfolio_companies)} companies"
            + (
                f" across {len(sectors)} sectors"
                if sectors
                else " (sectors will be auto-generated)"
            )
        )

        # Extract org UUIDs
        org_uuids = [
            UUID(company.get("org_uuid"))
            for company in portfolio_companies
            if company.get("org_uuid")
        ]

        if not org_uuids:
            raise ValueError("No valid org_uuids found in portfolio_companies")

        # Create org lookup for quick access
        org_lookup = {
            UUID(company.get("org_uuid")): company
            for company in portfolio_companies
            if company.get("org_uuid")
        }

        # Look up each portfolio company in the database to get their categories
        org_model = OrganizationModel()
        await org_model.initialize()

        logger.info(f"Looking up {len(org_uuids)} portfolio companies in database")

        # Fetch organizations in parallel batches
        async def fetch_org(uuid: UUID):
            """Fetch a single organization by UUID."""
            try:
                orgs = await org_model.get(org_uuid=uuid)
                return orgs[0] if orgs else None
            except Exception as e:
                logger.warning(f"Failed to fetch organization {uuid}: {e}")
                return None

        # Query organizations in parallel batches
        BATCH_SIZE = 50
        all_orgs = []
        for i in range(0, len(org_uuids), BATCH_SIZE):
            batch = org_uuids[i : i + BATCH_SIZE]
            batch_results = await asyncio.gather(*[fetch_org(uuid) for uuid in batch])
            all_orgs.extend([org for org in batch_results if org is not None])

        logger.info(f"Retrieved {len(all_orgs)} organizations from database")

        # Collect all unique categories/category_groups if auto-generating sectors
        if use_auto_sectors:
            all_categories = set()
            for org in all_orgs:
                if use_category_groups and org.category_groups:
                    all_categories.update(org.category_groups)
                elif not use_category_groups and org.categories:
                    all_categories.update(org.categories)
            sectors = sorted(list(all_categories))  # Sort for consistency
            logger.info(f"Auto-generated {len(sectors)} sectors from company categories")

        # Match companies to sectors based on their categories/category_groups
        # Each company gets assigned to ONE primary sector (exclusive assignment)
        sector_assignments: Dict[str, List[UUID]] = defaultdict(list)
        company_to_sector: Dict[UUID, Optional[str]] = {}

        for org in all_orgs:
            org_uuid = org.org_uuid

            # Get categories or category_groups based on preference
            categories_to_check = []
            if use_category_groups and org.category_groups:
                categories_to_check = org.category_groups
            elif not use_category_groups and org.categories:
                categories_to_check = org.categories

            if not categories_to_check:
                company_to_sector[org_uuid] = None
                logger.debug(f"Organization {org_uuid} has no categories/category_groups")
                continue

            # Find best matching sector
            # Strategy: If sectors were provided by user, match categories to those sectors
            # If sectors are auto-generated from categories, use exact category match
            best_sector = None

            if use_auto_sectors:
                # Auto-generated sectors: use first category that exists in sectors list
                if sectors is not None:
                    for category in categories_to_check:
                        if category in sectors:
                            best_sector = category
                            break
            else:
                # User-provided sectors: match categories to sectors
                # Prefer exact matches, then keyword matches
                exact_match = None
                keyword_match = None

                if sectors is not None:
                    for category in categories_to_check:
                        for sector in sectors:
                            if _match_category_to_sector(category, sector, case_sensitive):
                                # Check if exact match
                                if (
                                    category.lower() == sector.lower()
                                    if not case_sensitive
                                    else category == sector
                                ):
                                    exact_match = sector
                                    break
                                # Otherwise take first keyword match
                                elif keyword_match is None:
                                    keyword_match = sector

                best_sector = exact_match or keyword_match

            # Assign company to best matching sector (exclusive)
            if best_sector:
                company_to_sector[org_uuid] = best_sector
                sector_assignments[best_sector].append(org_uuid)
                logger.debug(
                    f"Organization {org.name} ({org_uuid}) assigned to sector: {best_sector}"
                )
            else:
                company_to_sector[org_uuid] = None

        categorized_count = sum(1 for sector in company_to_sector.values() if sector is not None)
        logger.info(f"Matched {categorized_count}/{len(org_uuids)} companies to sectors")

        # Get exit data for calculating exit rates per sector
        acquisition_model = AcquisitionModel()
        await acquisition_model.initialize()

        # Query acquisitions for all portfolio companies
        # Note: AcquisitionModel doesn't support batch queries, so we query all and filter
        # Order by announce date descending to get most recent acquisitions first
        all_acquisitions = await acquisition_model.get(
            order_by="acquisition_announce_date",
            order_direction="desc",
        )

        # Filter to only acquisitions of portfolio companies
        org_uuids_set = set(org_uuids)
        portfolio_acquisitions = [
            acq
            for acq in all_acquisitions
            if acq.acquiree_uuid and acq.acquiree_uuid in org_uuids_set
        ]

        # Create exit lookup
        exited_companies = {
            acq.acquiree_uuid for acq in portfolio_acquisitions if acq.acquiree_uuid
        }

        # Calculate sector metrics (only for sectors that have companies)
        sector_data = []
        total_companies = len(portfolio_companies)
        uncategorized_count = sum(1 for sector in company_to_sector.values() if sector is None)

        # Only include sectors that have at least one company assigned
        sectors_with_companies = [s for s in (sectors or []) if s in sector_assignments]

        for sector in sectors_with_companies:
            company_uuids = sector_assignments.get(sector, [])
            company_count = len(company_uuids)

            # Calculate total investment for this sector
            total_investment = sum(
                org_lookup[uuid].get("total_invested_usd") or 0 for uuid in company_uuids
            )

            # Calculate exit rate for this sector
            exited_in_sector = sum(1 for uuid in company_uuids if uuid in exited_companies)
            exit_rate_pct = (exited_in_sector / company_count * 100) if company_count > 0 else None

            percentage = (company_count / total_companies * 100) if total_companies > 0 else 0.0

            sector_data.append(
                {
                    "sector": sector,
                    "company_count": company_count,
                    "percentage": round(percentage, 1),
                    "total_investment_usd": (total_investment if total_investment > 0 else None),
                    "exit_rate_pct": (
                        round(exit_rate_pct, 1) if exit_rate_pct is not None else None
                    ),
                }
            )

        # Sort by company count (descending)
        def get_company_count(x: Dict[str, Any]) -> int:
            count = x.get("company_count", 0)
            if isinstance(count, int):
                return count
            elif isinstance(count, (float, str)):
                return int(count)
            return 0

        sector_data.sort(key=get_company_count, reverse=True)

        # Calculate Herfindahl index
        def get_percentage(s: Dict[str, Any]) -> float:
            pct = s.get("percentage", 0.0)
            if isinstance(pct, (int, float)):
                return float(pct)
            return 0.0

        sector_percentages = [get_percentage(s) for s in sector_data]
        concentration_index = _calculate_herfindahl_index(sector_percentages)

        # Build result
        result = {
            "sectors": sector_data,
            "concentration_index": concentration_index,
            "uncategorized_companies": uncategorized_count,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "num_companies": total_companies,
            "num_sectors": len(sectors) if sectors is not None else 0,
            "num_categorized": total_companies - uncategorized_count,
            "use_category_groups": use_category_groups,
            "case_sensitive": case_sensitive,
        }

        logger.debug(
            f"Analyzed sector concentration for {total_companies} companies "
            f"across {len(sector_data)} sectors in {execution_time_ms:.2f}ms"
        )

        # Return ToolOutput with successful result
        return create_tool_output(
            tool_name="analyze_sector_concentration",
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to analyze sector concentration: {e}"
        logger.error(error_msg, exc_info=True)

        # Return ToolOutput with error information
        return create_tool_output(
            tool_name="analyze_sector_concentration",
            success=False,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            metadata={"exception_type": type(e).__name__},
        )
