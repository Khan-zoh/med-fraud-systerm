# ADR-003: dbt for SQL Transformations

## Status
Accepted

## Context
The transformation layer needs to clean raw data, join sources, compute aggregations, and produce ML-ready feature tables. Options: plain Python/pandas, plain SQL scripts, dbt, or custom PySpark.

## Decision
Use **dbt-duckdb** for all SQL transformations (staging → intermediate → marts).

## Rationale

1. **Version-controlled SQL with testing**: Every model is a `.sql` file in git. Every column can have `not_null`, `unique`, and `accepted_values` tests that run with `dbt test`. This gives us 34 automated data quality checks across 8 models.

2. **Lineage graph**: dbt automatically builds a DAG from `{{ ref() }}` and `{{ source() }}` calls. Reviewers can see exactly how raw data flows to the ML feature table without reading code.

3. **Industry standard**: dbt is the most widely adopted transformation framework in analytics engineering. Using it signals professional data engineering practice in a portfolio.

4. **Separation of concerns**: SQL handles set-based operations (joins, aggregations, z-scores) cleanly. Python handles operations that need scipy/numpy (JSD, entropy, Mahalanobis). Neither is forced into the other's domain.

5. **Reusable macros**: `safe_zscore` and `ratio_vs_benchmark` macros capture IE concepts as reusable SQL patterns.

## Alternatives considered

- **Plain pandas**: Would work but produces opaque `.py` files instead of readable SQL. No built-in testing framework. Harder for a SQL-fluent reviewer to audit.
- **Plain SQL scripts**: No dependency management, no testing, no lineage.
- **PySpark**: Overkill for laptop-scale data. Adds operational complexity.

## Consequences
- Developers need to know both SQL (dbt) and Python (features, models)
- The dbt project has its own `profiles.yml` pointing to the same DuckDB file
- `dbt run` and `dbt test` are separate commands (wrapped in `make transform` and `make dbt-test`)
