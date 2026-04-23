# ADR-002: DuckDB for Local Analytical Storage

## Status
Accepted

## Context
We need a database that supports analytical SQL queries over millions of rows on a laptop with 16GB RAM. Options considered: SQLite, PostgreSQL, DuckDB, Parquet files with pandas.

## Decision
Use **DuckDB** as the single storage layer for raw, staging, intermediate, and marts data.

## Rationale

| Factor | DuckDB | PostgreSQL | SQLite | Parquet + pandas |
|--------|--------|-----------|--------|-----------------|
| **Install** | `pip install duckdb` | Separate server process | Built-in | Files only |
| **Analytical perf** | Columnar, vectorized — built for OLAP | Row-oriented, needs tuning for analytics | Slow on aggregations | Fast reads, no SQL joins |
| **SQL dialect** | PostgreSQL-compatible | PostgreSQL | Limited | None (pandas API) |
| **dbt support** | `dbt-duckdb` adapter | `dbt-postgres` | No | No |
| **Memory** | Lazy evaluation, streams large files | Server memory | Loads to memory | Full DataFrame in RAM |
| **Migration path** | Schema maps 1:1 to Postgres/Snowflake | Is Postgres | N/A | Requires rewrite |

Key advantage: DuckDB can `read_csv_auto` and `read_json_auto` directly from files (including inside ZIP archives) without loading into Python memory first. This is critical for the 7GB NPPES dissemination file.

The schema design (raw → staging → intermediate → marts) uses DuckDB schemas that map directly to PostgreSQL schemas, so migrating to a production database is a schema copy, not a rewrite.

## Consequences
- Single-file database (`.duckdb`) — easy to share, back up, or delete and regenerate
- No concurrent write access (fine for a pipeline that runs sequentially)
- Production deployment would swap to PostgreSQL or Snowflake via dbt profile change — all SQL stays the same
