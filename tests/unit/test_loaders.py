"""Tests for DuckDB loaders using a temporary in-memory database."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from cms_fwa.ingestion.loaders import load_leie, load_partb
from cms_fwa.ingestion.sources import RAW_TABLES


@pytest.fixture
def tmp_duckdb(tmp_path):
    """Patch settings to use a temp DuckDB file and yield its path."""
    db_path = tmp_path / "test.duckdb"
    with patch("cms_fwa.ingestion.loaders.get_connection") as mock_conn:
        conn = duckdb.connect(str(db_path))
        # Create raw schema
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw")

        # Make the context manager yield our test connection
        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield conn

        mock_conn.side_effect = _fake_conn
        yield conn, tmp_path
        conn.close()


def test_load_partb_creates_table(tmp_duckdb) -> None:
    """Loading a JSONL file should create the Part B raw table."""
    conn, tmp_path = tmp_duckdb

    # Create a small test JSONL file
    jsonl_path = tmp_path / "test_partb.jsonl"
    records = [
        {
            "Rndrng_NPI": "1234567890",
            "HCPCS_Cd": "99213",
            "Tot_Srvcs": "100",
            "Tot_Benes": "50",
            "Avg_Mdcr_Pymt_Amt": "75.00",
            "Rndrng_Prvdr_State_Abrvtn": "TX",
        },
        {
            "Rndrng_NPI": "0987654321",
            "HCPCS_Cd": "99214",
            "Tot_Srvcs": "200",
            "Tot_Benes": "80",
            "Avg_Mdcr_Pymt_Amt": "95.00",
            "Rndrng_Prvdr_State_Abrvtn": "TX",
        },
    ]
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Patch ensure_schemas to be a no-op (schema already created)
    with patch("cms_fwa.ingestion.loaders.ensure_schemas"):
        row_count = load_partb(jsonl_path)

    assert row_count == 2

    # Verify data is queryable
    result = conn.execute(f"SELECT COUNT(*) FROM {RAW_TABLES.partb}").fetchone()
    assert result[0] == 2


def test_load_leie_creates_table(tmp_duckdb) -> None:
    """Loading a CSV file should create the LEIE raw table."""
    conn, tmp_path = tmp_duckdb

    # Create a small test CSV
    csv_path = tmp_path / "test_leie.csv"
    csv_path.write_text(
        "LASTNAME,FIRSTNAME,NPI,EXCLTYPE,EXCLDATE\n"
        "DOE,JOHN,1234567890,1128a(1),20200115\n"
        "SMITH,JANE,0987654321,1128a(2),20210301\n"
    )

    with patch("cms_fwa.ingestion.loaders.ensure_schemas"):
        row_count = load_leie(csv_path)

    assert row_count == 2

    result = conn.execute(f"SELECT COUNT(*) FROM {RAW_TABLES.leie}").fetchone()
    assert result[0] == 2
