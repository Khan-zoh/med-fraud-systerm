"""
Base utilities for feature engineering.

Provides the DuckDB-to-DataFrame bridge and common helper functions
used across all feature modules.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from cms_fwa.utils.db import get_connection


def load_mart_features() -> pd.DataFrame:
    """Load the dbt mart_provider_features table into a DataFrame.

    This is the starting point for Python-based feature engineering.
    The dbt mart already contains SQL-computed features (z-scores,
    ratios, percentile flags). Python adds features that need
    scipy/numpy (divergences, entropy, Mahalanobis).
    """
    with get_connection() as conn:
        df = conn.execute("SELECT * FROM main_marts.mart_provider_features").fetchdf()
    logger.info(f"Loaded mart_provider_features: {len(df):,} providers")
    return df


def load_hcpcs_mix() -> pd.DataFrame:
    """Load the provider × HCPCS mix table for divergence calculations."""
    with get_connection() as conn:
        df = conn.execute(
            "SELECT * FROM main_intermediate.int_provider_hcpcs_mix"
        ).fetchdf()
    logger.info(f"Loaded HCPCS mix: {len(df):,} rows")
    return df


def load_claims() -> pd.DataFrame:
    """Load staged claims for features that need line-level detail."""
    with get_connection() as conn:
        df = conn.execute("SELECT * FROM main_staging.stg_partb_claims").fetchdf()
    logger.info(f"Loaded staged claims: {len(df):,} rows")
    return df


def safe_log(x: np.ndarray) -> np.ndarray:
    """Compute log with zero-safe handling (0 * log(0) = 0 by convention)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 0, x * np.log2(x), 0.0)
    return result
