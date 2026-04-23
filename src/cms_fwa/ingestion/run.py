"""
CLI entry point for the ingestion pipeline.

Run with: python -m cms_fwa.ingestion.run
Or:       make ingest

Orchestrates: download -> load -> validate for all three data sources.
Designed to be idempotent — safe to re-run after interruptions.
"""

import sys

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
from cms_fwa.ingestion.validators import validate_all
from cms_fwa.utils.db import get_connection
from cms_fwa.utils.logging import setup_logging


def run_ingestion() -> bool:
    """Execute the full ingestion pipeline.

    Returns:
        True if all validations passed, False otherwise.
    """
    setup_logging()
    states = settings.cms_data_states
    logger.info("=" * 60)
    logger.info("CMS FWA Detection — Data Ingestion Pipeline")
    logger.info(f"  Year:   {settings.cms_data_year}")
    logger.info(f"  States: {', '.join(states)} ({len(states)} states)")
    logger.info(f"  DB:     {settings.duckdb_abs_path}")
    logger.info("=" * 60)

    # Step 1: Download raw files (one Part B file per state)
    logger.info(f"Step 1/3: Downloading raw data files for {len(states)} state(s)")
    partb_files = []
    for state in states:
        logger.info(f"--- Downloading {state} ---")
        partb_files.append(download_partb_data(state=state))
    leie_file = download_leie_data()

    # Step 2: Load into DuckDB (append all state files into the same raw table)
    logger.info("Step 2/3: Loading data into DuckDB")
    partb_rows = 0
    for i, pf in enumerate(partb_files):
        # First file creates the table; subsequent files append
        append = i > 0
        partb_rows += load_partb(pf, append=append)
    leie_rows = load_leie(leie_file)

    # For NPPES, use the API fallback approach to avoid the 7GB download.
    # This queries only the NPIs found in our Part B data.
    logger.info("Loading NPPES data via API (for NPIs in Part B)...")
    with get_connection() as conn:
        npi_col = _find_npi_column(conn)
        npis = [
            row[0]
            for row in conn.execute(
                f'SELECT DISTINCT "{npi_col}" FROM raw.partb_by_provider_service'
            ).fetchall()
        ]
    logger.info(f"Found {len(npis):,} unique NPIs in Part B data")

    NPPES_API_CAP = 15000  # Raised for multi-state ingestion
    if len(npis) > NPPES_API_CAP:
        logger.warning(
            f"Large NPI count ({len(npis):,}). API lookup will be slow. "
            "Consider downloading the full NPPES dissemination file instead "
            "(update sources.py NPPES_DOWNLOAD_URL and use load_nppes)."
        )
        logger.info(f"Limiting to first {NPPES_API_CAP:,} NPIs for dev speed")
        npis = npis[:NPPES_API_CAP]

    nppes_rows = load_nppes_from_api_fallback(npis)

    logger.info(
        f"Loaded: Part B={partb_rows:,}, LEIE={leie_rows:,}, NPPES={nppes_rows:,}"
    )

    # Step 3: Validate
    logger.info("Step 3/3: Running data quality validation")
    results = validate_all()

    all_passed = all(r.success for r in results)
    if not all_passed:
        logger.error("Data quality validation FAILED:")
        for r in results:
            if not r.success:
                for detail in r.failure_details:
                    logger.error(f"  {r.table_name}: {detail}")

    return all_passed


def _find_npi_column(conn) -> str:
    """Find the NPI column name in the Part B table (case-insensitive)."""
    cols = [
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'raw' AND table_name = 'partb_by_provider_service'"
        ).fetchall()
    ]
    for col in cols:
        if col.lower() == "rndrng_npi":
            return col
    raise RuntimeError(f"NPI column not found. Available columns: {cols}")


def main() -> None:
    """CLI entry point."""
    success = run_ingestion()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
