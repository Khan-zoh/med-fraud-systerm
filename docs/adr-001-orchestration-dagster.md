# ADR-001: Dagster over Airflow for Pipeline Orchestration

## Status
Accepted

## Context
We need a workflow orchestrator to manage the data ingestion, transformation, and model training pipeline. The two leading open-source options are Apache Airflow and Dagster.

This project runs on a developer laptop and will be evaluated by senior engineers reviewing a GitHub repo, so local development experience and code clarity matter as much as production scalability.

## Decision
Use **Dagster** with its asset-based programming model.

## Rationale

| Factor | Dagster | Airflow |
|--------|---------|---------|
| **Programming model** | Asset-centric (declare *what* to produce) | Task-centric (declare *what to do*) |
| **Local dev** | `dagster dev` — instant UI, no Docker needed | Requires Docker Compose or standalone scheduler + webserver |
| **Testability** | Assets are plain Python functions, directly unit-testable | DAG testing requires mocking the execution context |
| **Lineage** | Automatic from asset dependencies | Manual via XCom or external catalog |
| **Materialization metadata** | Built-in (row counts, file sizes attached to each asset run) | Requires custom XCom pushes |

The asset model also maps naturally to our pipeline: each data source is an asset (download file → load to DuckDB → validate), and dbt models can be wrapped as Dagster assets for unified lineage.

## Consequences
- Engineers familiar with Airflow will need to learn the Dagster asset API (small surface area)
- Production deployment would use Dagster Cloud or a Kubernetes-based Dagster deployment instead of `dagster dev`
- The `dagster_project/` directory was renamed from `dagster/` to avoid shadowing the installed package on Windows
