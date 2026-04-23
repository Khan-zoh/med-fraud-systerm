"""
Dagster asset definitions for the CMS FWA ingestion pipeline.

Each data source is modeled as a pair of assets:
  1. A "download" asset (produces a raw file on disk)
  2. A "load" asset (reads the file into DuckDB)

This asset-based approach (vs. Airflow-style task DAGs) gives us:
  - Automatic lineage tracking
  - Materialization metadata (row counts, file sizes)
  - Built-in re-execution of failed assets without re-running everything
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    Definitions,
    MaterializeResult,
    MetadataValue,
    asset,
)
from loguru import logger

from cms_fwa.config import settings
from cms_fwa.ingestion.downloaders import (
    download_leie_data,
    download_partb_data,
)
from cms_fwa.ingestion.loaders import (
    load_leie,
    load_nppes_from_api_fallback,
    load_partb,
)
from cms_fwa.ingestion.validators import validate_leie, validate_nppes, validate_partb
from cms_fwa.utils.db import get_connection
from cms_fwa.utils.logging import setup_logging


# ==========================================================================
# Part B assets
# ==========================================================================

@asset(
    group_name="ingestion",
    description="Download CMS Part B billing data (Provider & Service level)",
)
def partb_raw_file(context: AssetExecutionContext) -> MaterializeResult:
    """Download Part B data from data.cms.gov API."""
    setup_logging()
    output_path = download_partb_data()
    file_size = output_path.stat().st_size

    return MaterializeResult(
        metadata={
            "file_path": MetadataValue.path(str(output_path)),
            "file_size_mb": MetadataValue.float(file_size / (1024 * 1024)),
            "year": MetadataValue.int(settings.cms_data_year),
            "state": MetadataValue.text(settings.cms_data_state),
        }
    )


@asset(
    group_name="ingestion",
    deps=[AssetKey("partb_raw_file")],
    description="Load Part B data into DuckDB raw schema and validate",
)
def partb_raw_table(context: AssetExecutionContext) -> MaterializeResult:
    """Load Part B JSONL into DuckDB and run quality checks."""
    setup_logging()
    raw_dir = settings.raw_data_dir
    state = settings.cms_data_state.lower()
    year = settings.cms_data_year
    jsonl_path = raw_dir / f"partb_{state}_{year}.jsonl"

    row_count = load_partb(jsonl_path)
    validation = validate_partb()

    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(row_count),
            "validation_passed": MetadataValue.bool(validation.success),
            "checks_passed": MetadataValue.text(
                f"{validation.successful_expectations}/{validation.total_expectations}"
            ),
        }
    )


# ==========================================================================
# LEIE assets
# ==========================================================================

@asset(
    group_name="ingestion",
    description="Download OIG LEIE exclusions list",
)
def leie_raw_file(context: AssetExecutionContext) -> MaterializeResult:
    """Download LEIE exclusions CSV from OIG."""
    setup_logging()
    output_path = download_leie_data()
    file_size = output_path.stat().st_size

    return MaterializeResult(
        metadata={
            "file_path": MetadataValue.path(str(output_path)),
            "file_size_mb": MetadataValue.float(file_size / (1024 * 1024)),
        }
    )


@asset(
    group_name="ingestion",
    deps=[AssetKey("leie_raw_file")],
    description="Load LEIE data into DuckDB raw schema and validate",
)
def leie_raw_table(context: AssetExecutionContext) -> MaterializeResult:
    """Load LEIE CSV into DuckDB and run quality checks."""
    setup_logging()
    csv_path = settings.raw_data_dir / "leie_updated.csv"

    row_count = load_leie(csv_path)
    validation = validate_leie()

    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(row_count),
            "validation_passed": MetadataValue.bool(validation.success),
        }
    )


# ==========================================================================
# NPPES assets
# ==========================================================================

@asset(
    group_name="ingestion",
    deps=[AssetKey("partb_raw_table")],
    description="Load NPPES provider data via API for NPIs found in Part B",
)
def nppes_raw_table(context: AssetExecutionContext) -> MaterializeResult:
    """Query NPPES API for providers in our Part B dataset."""
    setup_logging()

    # Get unique NPIs from Part B
    with get_connection() as conn:
        cols = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'raw' AND table_name = 'partb_by_provider_service'"
            ).fetchall()
        ]
        npi_col = next(c for c in cols if c.lower() == "rndrng_npi")
        npis = [
            row[0]
            for row in conn.execute(
                f'SELECT DISTINCT "{npi_col}" FROM raw.partb_by_provider_service'
            ).fetchall()
        ]

    context.log.info(f"Found {len(npis):,} unique NPIs")

    # Limit for dev speed
    if len(npis) > 5000:
        context.log.warning(f"Limiting to 5,000 NPIs for dev (of {len(npis):,})")
        npis = npis[:5000]

    row_count = load_nppes_from_api_fallback(npis)
    validation = validate_nppes()

    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(row_count),
            "npis_queried": MetadataValue.int(len(npis)),
            "validation_passed": MetadataValue.bool(validation.success),
        }
    )
