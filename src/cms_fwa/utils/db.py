"""DuckDB connection management."""

from contextlib import contextmanager
from typing import Generator

import duckdb

from cms_fwa.config import settings


@contextmanager
def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield a DuckDB connection, ensuring it is closed on exit.

    Uses the database path from settings. Creates the file and parent
    directories if they don't exist yet.
    """
    db_path = settings.duckdb_abs_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def ensure_schemas(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the standard DuckDB schemas if they don't already exist."""
    for schema in ("raw", "staging", "intermediate", "marts"):
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
