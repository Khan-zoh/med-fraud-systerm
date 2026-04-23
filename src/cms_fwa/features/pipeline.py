"""
Feature Pipeline — combines all feature modules into one ML-ready table.

Orchestrates the computation of all Python-based features and merges them
with the SQL-computed features from the dbt mart. The output is saved
back to DuckDB as `marts.provider_features_full` — the final input to
the modeling layer.

Feature groups:
  1. SQL-based (from dbt): SPC z-scores, peer ratios, upcoding, percentile flags
  2. Jensen-Shannon divergence: procedure mix distance from specialty median
  3. Geographic entropy: patient zip-code dispersion
  4. Billing velocity: throughput utilization vs. theoretical limits
  5. Mahalanobis distance: multivariate outlier within specialty
"""

import pandas as pd
from loguru import logger

from cms_fwa.features.base import load_claims, load_hcpcs_mix, load_mart_features
from cms_fwa.features.divergence import compute_jsd_features
from cms_fwa.features.geographic import compute_geographic_entropy
from cms_fwa.features.mahalanobis import compute_mahalanobis_distance
from cms_fwa.features.velocity import compute_velocity_features
from cms_fwa.utils.db import ensure_schemas, get_connection
from cms_fwa.utils.logging import setup_logging


# All numeric feature columns used for ML (excluding identifiers and labels)
ML_FEATURE_COLUMNS: list[str] = [
    # --- SQL-based (dbt mart) ---
    "total_services",
    "total_beneficiaries",
    "total_medicare_payments",
    "total_submitted_charges",
    "unique_hcpcs_codes",
    "services_per_beneficiary",
    "facility_service_ratio",
    "drug_service_ratio",
    "wavg_submitted_charge",
    "wavg_medicare_payment",
    "zscore_total_services",
    "zscore_total_payments",
    "zscore_total_beneficiaries",
    "zscore_services_per_bene",
    "zscore_unique_hcpcs",
    "zscore_avg_charge",
    "ratio_services_vs_median",
    "ratio_payments_vs_median",
    "ratio_svc_per_bene_vs_median",
    "beneficiary_day_intensity",
    "high_complexity_ratio",
    "upcoding_ratio_vs_peers",
    # --- Python-computed ---
    "jsd_vs_specialty",
    "zip_entropy",
    "zip_entropy_normalized",
    "unique_zip_count",
    "daily_services",
    "daily_beneficiaries",
    "services_per_bene_per_day",
    "throughput_utilization",
    "mahalanobis_distance",
    "mahalanobis_pvalue",
]

LABEL_COLUMN = "is_excluded"

ID_COLUMNS = [
    "npi",
    "provider_last_name",
    "provider_first_name",
    "provider_specialty",
    "entity_type",
    "provider_state",
    "provider_zip5",
]


def run_feature_pipeline() -> pd.DataFrame:
    """Execute the full feature pipeline and return the combined table.

    Returns:
        DataFrame with one row per provider, all features, and the label.
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    # Load base data
    logger.info("Loading base data from DuckDB...")
    mart_features = load_mart_features()
    hcpcs_mix = load_hcpcs_mix()
    claims = load_claims()

    # Compute Python features
    logger.info("Computing Jensen-Shannon divergence...")
    jsd_df = compute_jsd_features(hcpcs_mix, mart_features)

    logger.info("Computing geographic entropy...")
    geo_df = compute_geographic_entropy(claims)

    logger.info("Computing billing velocity...")
    velocity_df = compute_velocity_features(claims)

    logger.info("Computing Mahalanobis distance...")
    mahal_df = compute_mahalanobis_distance(mart_features)

    # Merge all features onto the base mart
    logger.info("Merging all features...")
    combined = mart_features.copy()
    combined["npi"] = combined["npi"].astype(str)

    for feature_df in [jsd_df, geo_df, velocity_df, mahal_df]:
        feature_df["npi"] = feature_df["npi"].astype(str)
        combined = combined.merge(feature_df, on="npi", how="left")

    # Log feature completeness
    feature_cols = [c for c in ML_FEATURE_COLUMNS if c in combined.columns]
    missing_cols = [c for c in ML_FEATURE_COLUMNS if c not in combined.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")

    null_rates = combined[feature_cols].isnull().mean()
    high_null = null_rates[null_rates > 0.1]
    if len(high_null) > 0:
        logger.warning(f"Features with >10% nulls:\n{high_null}")

    logger.info(
        f"Combined feature table: {len(combined):,} providers × "
        f"{len(feature_cols)} features"
    )

    # Save to DuckDB
    _save_to_duckdb(combined)

    return combined


def _save_to_duckdb(df: pd.DataFrame) -> None:
    """Save the combined feature table to DuckDB marts schema."""
    with get_connection() as conn:
        ensure_schemas(conn)
        table_name = "main_marts.provider_features_full"
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"Saved {row_count:,} rows to {table_name}")


if __name__ == "__main__":
    run_feature_pipeline()
