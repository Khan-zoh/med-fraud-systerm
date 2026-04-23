"""
Load downloaded raw files into DuckDB's `raw` schema.

Each loader is idempotent: it drops and recreates the target table.
DuckDB's built-in CSV/JSON readers handle parsing efficiently without
loading entire files into Python memory.
"""

from pathlib import Path

import duckdb
from loguru import logger

from cms_fwa.ingestion.sources import (
    NPPES_COLUMN_RENAMES,
    NPPES_COLUMNS,
    RAW_TABLES,
)
from cms_fwa.utils.db import ensure_schemas, get_connection


def load_partb(jsonl_path: Path, append: bool = False) -> int:
    """Load Part B JSONL data into DuckDB.

    Args:
        jsonl_path: Path to the newline-delimited JSON file.
        append: If True, append to the existing table (for multi-state loads).
                If False (default), drop and recreate the table.

    Returns:
        Number of rows loaded from THIS file.
    """
    logger.info(f"Loading Part B data from {jsonl_path} (append={append})")

    with get_connection() as conn:
        ensure_schemas(conn)
        table = RAW_TABLES.partb

        if not append:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"""
                CREATE TABLE {table} AS
                SELECT * FROM read_json_auto(
                    '{jsonl_path}',
                    format = 'newline_delimited',
                    maximum_object_size = 33554432
                )
            """)
        else:
            conn.execute(f"""
                INSERT INTO {table}
                SELECT * FROM read_json_auto(
                    '{jsonl_path}',
                    format = 'newline_delimited',
                    maximum_object_size = 33554432
                )
            """)

        # Count how many rows came from this specific file
        file_name = Path(jsonl_path).name
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM read_json_auto('{jsonl_path}', "
            f"format='newline_delimited', maximum_object_size=33554432)"
        ).fetchone()[0]

    logger.info(f"Loaded {row_count:,} rows from {file_name} into {table}")
    return row_count


def load_leie(csv_path: Path) -> int:
    """Load LEIE exclusions CSV into DuckDB.

    Args:
        csv_path: Path to the LEIE CSV file.

    Returns:
        Number of rows loaded.
    """
    logger.info(f"Loading LEIE data from {csv_path}")

    with get_connection() as conn:
        ensure_schemas(conn)
        table = RAW_TABLES.leie

        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"""
            CREATE TABLE {table} AS
            SELECT * FROM read_csv_auto(
                '{csv_path}',
                header = true,
                ignore_errors = true,
                all_varchar = true
            )
        """)

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info(f"Loaded {row_count:,} rows into {table}")
    return row_count


def load_nppes(zip_path: Path) -> int:
    """Load NPPES data from the dissemination ZIP into DuckDB.

    Uses DuckDB's ability to read CSVs directly from ZIP files.
    Only selects the columns we need (NPPES has 300+ columns) and
    renames them to clean snake_case names.

    Args:
        zip_path: Path to the NPPES dissemination ZIP file.

    Returns:
        Number of rows loaded.
    """
    logger.info(f"Loading NPPES data from {zip_path}")

    # Build the SELECT clause with column renames
    select_cols = ", ".join(
        f'"{orig}" AS {renamed}' for orig, renamed in NPPES_COLUMN_RENAMES.items()
    )

    with get_connection() as conn:
        ensure_schemas(conn)
        table = RAW_TABLES.nppes

        conn.execute(f"DROP TABLE IF EXISTS {table}")

        # DuckDB can read CSVs inside ZIP files. The NPPES zip contains
        # a file matching npidata_pfile_*.csv — we use a glob pattern.
        conn.execute(f"""
            CREATE TABLE {table} AS
            SELECT {select_cols}
            FROM read_csv_auto(
                '{zip_path}/*.csv',
                header = true,
                ignore_errors = true,
                all_varchar = true,
                filename = true
            )
            WHERE filename LIKE '%npidata_pfile%'
        """)

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info(f"Loaded {row_count:,} rows into {table}")
    return row_count


def load_nppes_from_api_fallback(npi_list: list[str]) -> int:
    """Fallback: load NPPES data via the public NPI Registry API.

    Use this if downloading the full dissemination file is impractical.
    Queries the NPPES API in batches for NPIs found in Part B data.

    Args:
        npi_list: List of NPI strings to look up.

    Returns:
        Number of rows loaded.
    """
    import httpx

    logger.info(f"Loading NPPES data via API for {len(npi_list):,} NPIs")

    api_url = "https://npiregistry.cms.hhs.gov/api/"
    records: list[dict] = []

    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        for i, npi in enumerate(npi_list):
            if i > 0 and i % 100 == 0:
                logger.info(f"  ... queried {i:,}/{len(npi_list):,} NPIs")

            resp = client.get(api_url, params={"number": npi, "version": "2.1"})
            if resp.status_code != 200:
                continue

            data = resp.json()
            results = data.get("results", [])
            if not results:
                continue

            r = results[0]
            basic = r.get("basic", {})
            addresses = r.get("addresses", [{}])
            practice = next(
                (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                addresses[0] if addresses else {},
            )
            taxonomies = r.get("taxonomies", [{}])

            records.append({
                "npi": str(r.get("number", "")),
                "entity_type_code": str(r.get("enumeration_type", "")),
                "last_name": basic.get("last_name", ""),
                "first_name": basic.get("first_name", ""),
                "organization_name": basic.get("organization_name", ""),
                "practice_state": practice.get("state", ""),
                "practice_zip": practice.get("postal_code", ""),
                "taxonomy_code": taxonomies[0].get("code", "") if taxonomies else "",
                "enumeration_date": basic.get("enumeration_date", ""),
                "gender_code": basic.get("gender", ""),
            })

    if not records:
        logger.warning("No NPPES records retrieved from API")
        return 0

    # Load into DuckDB via pandas (small enough for in-memory)
    import pandas as pd

    df = pd.DataFrame(records)

    with get_connection() as conn:
        ensure_schemas(conn)
        table = RAW_TABLES.nppes
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info(f"Loaded {row_count:,} rows into {table} (via API)")
    return row_count
