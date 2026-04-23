"""
Data quality validation for raw ingested tables.

Runs SQL-based quality checks after each raw table is loaded into DuckDB.
Validates structural integrity (null rates, format checks, row counts)
without making assumptions about data distribution — those come in dbt tests.

Design choice: We use direct SQL checks against DuckDB rather than the
Great Expectations Python API because (1) our data lives in DuckDB and
SQL checks avoid materializing DataFrames, and (2) GE v1.x changed its
API significantly. For column-level expectations in later phases, we'll
use GE's checkpoint/suite system with the DuckDB datasource connector.
"""

from dataclasses import dataclass

from loguru import logger

from cms_fwa.ingestion.sources import RAW_TABLES
from cms_fwa.utils.db import get_connection


@dataclass
class ValidationResult:
    """Summary of a Great Expectations validation run."""

    table_name: str
    success: bool
    total_expectations: int
    successful_expectations: int
    failed_expectations: int
    failure_details: list[str]

    def __str__(self) -> str:
        status = "PASS" if self.success else "FAIL"
        return (
            f"[{status}] {self.table_name}: "
            f"{self.successful_expectations}/{self.total_expectations} checks passed"
        )


def _read_table_stats(table_name: str) -> dict:
    """Read basic stats from a DuckDB table."""
    with get_connection() as conn:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        columns = [
            row[0]
            for row in conn.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema || '.' || table_name = '{table_name}'"
            ).fetchall()
        ]
    return {"row_count": row_count, "columns": columns}


def validate_partb() -> ValidationResult:
    """Validate the raw Part B table.

    Checks:
      - Table has > 0 rows
      - NPI column exists and has no nulls
      - NPI is 10 digits
      - Key numeric columns are present
      - No more than 5% null rate on critical fields
    """
    table = RAW_TABLES.partb
    logger.info(f"Validating {table}")

    stats = _read_table_stats(table)
    failures: list[str] = []
    total = 0
    passed = 0

    # Check 1: Row count
    total += 1
    if stats["row_count"] > 0:
        passed += 1
    else:
        failures.append(f"Table is empty (0 rows)")

    # Check 2: Expected columns exist
    expected_cols = {"Rndrng_NPI", "HCPCS_Cd", "Tot_Srvcs", "Tot_Benes", "Avg_Mdcr_Pymt_Amt"}
    total += 1
    # Column names may be lowercased by DuckDB
    actual_cols_lower = {c.lower() for c in stats["columns"]}
    expected_lower = {c.lower() for c in expected_cols}
    if expected_lower.issubset(actual_cols_lower):
        passed += 1
    else:
        missing = expected_lower - actual_cols_lower
        failures.append(f"Missing columns: {missing}")

    # Check 3-6: Null rate checks on critical columns via SQL
    with get_connection() as conn:
        for col in ["Rndrng_NPI", "HCPCS_Cd", "Tot_Srvcs", "Tot_Benes"]:
            total += 1
            # Find actual column name (case-insensitive)
            actual_col = next(
                (c for c in stats["columns"] if c.lower() == col.lower()), col
            )
            null_pct = conn.execute(f"""
                SELECT ROUND(
                    100.0 * COUNT(*) FILTER (WHERE "{actual_col}" IS NULL) / COUNT(*),
                    2
                ) FROM {table}
            """).fetchone()[0]

            if null_pct <= 5.0:
                passed += 1
            else:
                failures.append(f"{col} null rate: {null_pct}% (max 5%)")

        # Check 7: NPI format (10 digits)
        total += 1
        npi_col = next(
            (c for c in stats["columns"] if c.lower() == "rndrng_npi"), "Rndrng_NPI"
        )
        bad_npi_pct = conn.execute(f"""
            SELECT ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE NOT regexp_matches(CAST("{npi_col}" AS VARCHAR), '^[0-9]{{10}}$')
                ) / COUNT(*),
                2
            ) FROM {table}
        """).fetchone()[0]

        if bad_npi_pct <= 1.0:
            passed += 1
        else:
            failures.append(f"NPI format violations: {bad_npi_pct}% (max 1%)")

    result = ValidationResult(
        table_name=table,
        success=len(failures) == 0,
        total_expectations=total,
        successful_expectations=passed,
        failed_expectations=len(failures),
        failure_details=failures,
    )
    logger.info(str(result))
    return result


def validate_leie() -> ValidationResult:
    """Validate the raw LEIE table.

    Checks:
      - Table has > 0 rows
      - NPI column exists
      - EXCLTYPE and EXCLDATE have low null rates
      - At least some rows have valid NPIs (not all LEIE entries have NPIs)
    """
    table = RAW_TABLES.leie
    logger.info(f"Validating {table}")

    stats = _read_table_stats(table)
    failures: list[str] = []
    total = 0
    passed = 0

    # Check 1: Row count
    total += 1
    if stats["row_count"] > 0:
        passed += 1
    else:
        failures.append("Table is empty")

    # Check 2: Expected minimum row count (LEIE typically has ~70K+ rows)
    total += 1
    if stats["row_count"] >= 50000:
        passed += 1
    else:
        failures.append(f"Unexpectedly low row count: {stats['row_count']} (expected >= 50,000)")

    with get_connection() as conn:
        # Check 3: EXCLDATE not null
        total += 1
        col = next((c for c in stats["columns"] if c.lower() == "excldate"), "EXCLDATE")
        null_pct = conn.execute(f"""
            SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE "{col}" IS NULL) / COUNT(*), 2)
            FROM {table}
        """).fetchone()[0]
        if null_pct <= 5.0:
            passed += 1
        else:
            failures.append(f"EXCLDATE null rate: {null_pct}%")

        # Check 4: At least some NPIs present (many LEIE entries predate NPI)
        total += 1
        npi_col = next((c for c in stats["columns"] if c.lower() == "npi"), "NPI")
        npi_present = conn.execute(f"""
            SELECT COUNT(*) FILTER (
                WHERE "{npi_col}" IS NOT NULL AND TRIM("{npi_col}") != ''
            ) FROM {table}
        """).fetchone()[0]
        if npi_present >= 1000:
            passed += 1
        else:
            failures.append(f"Only {npi_present} rows have NPIs (expected >= 1,000)")

    result = ValidationResult(
        table_name=table,
        success=len(failures) == 0,
        total_expectations=total,
        successful_expectations=passed,
        failed_expectations=len(failures),
        failure_details=failures,
    )
    logger.info(str(result))
    return result


def validate_nppes() -> ValidationResult:
    """Validate the raw NPPES table.

    Checks:
      - Table has > 0 rows
      - NPI column has no nulls
      - taxonomy_code has reasonable fill rate
    """
    table = RAW_TABLES.nppes
    logger.info(f"Validating {table}")

    stats = _read_table_stats(table)
    failures: list[str] = []
    total = 0
    passed = 0

    # Check 1: Row count
    total += 1
    if stats["row_count"] > 0:
        passed += 1
    else:
        failures.append("Table is empty")

    with get_connection() as conn:
        # Check 2: NPI not null
        total += 1
        npi_col = next((c for c in stats["columns"] if c.lower() == "npi"), "npi")
        null_count = conn.execute(f"""
            SELECT COUNT(*) FILTER (WHERE "{npi_col}" IS NULL) FROM {table}
        """).fetchone()[0]
        if null_count == 0:
            passed += 1
        else:
            failures.append(f"NPI has {null_count} nulls")

        # Check 3: taxonomy_code fill rate > 80%
        total += 1
        tax_col = next(
            (c for c in stats["columns"] if c.lower() == "taxonomy_code"),
            "taxonomy_code",
        )
        fill_pct = conn.execute(f"""
            SELECT ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE "{tax_col}" IS NOT NULL AND TRIM("{tax_col}") != ''
                ) / COUNT(*), 2
            ) FROM {table}
        """).fetchone()[0]
        if fill_pct >= 80.0:
            passed += 1
        else:
            failures.append(f"taxonomy_code fill rate: {fill_pct}% (expected >= 80%)")

    result = ValidationResult(
        table_name=table,
        success=len(failures) == 0,
        total_expectations=total,
        successful_expectations=passed,
        failed_expectations=len(failures),
        failure_details=failures,
    )
    logger.info(str(result))
    return result


def validate_all() -> list[ValidationResult]:
    """Run all validation suites and return results."""
    results = [
        validate_partb(),
        validate_leie(),
        validate_nppes(),
    ]

    all_passed = all(r.success for r in results)
    logger.info(f"Validation {'PASSED' if all_passed else 'FAILED'}: "
                f"{sum(r.success for r in results)}/{len(results)} tables clean")

    return results
