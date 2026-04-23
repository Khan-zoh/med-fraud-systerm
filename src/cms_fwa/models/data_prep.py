"""
Data preparation for ML modeling.

Handles the bridge between the feature pipeline output and model inputs:
  - Train/test splitting (stratified, time-aware when possible)
  - Feature imputation and scaling
  - Class imbalance reporting

Design note: We use a stratified split on the exclusion label to preserve
the class ratio in both sets. In production with multi-year data, you'd
train on year T and test on T+1 (temporal split). With single-year data,
stratified random split is the best we can do.
"""

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from cms_fwa.config import settings
from cms_fwa.features.pipeline import ML_FEATURE_COLUMNS, ID_COLUMNS, LABEL_COLUMN
from cms_fwa.utils.db import get_connection


class ModelDataset(NamedTuple):
    """Container for train/test split data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    # Full dataframes with IDs for evaluation
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def load_feature_table() -> pd.DataFrame:
    """Load the full feature table from DuckDB."""
    with get_connection() as conn:
        # Try the Python-enriched table first, fall back to dbt mart
        try:
            df = conn.execute(
                "SELECT * FROM main_marts.provider_features_full"
            ).fetchdf()
            logger.info(f"Loaded provider_features_full: {len(df):,} rows")
        except Exception:
            df = conn.execute(
                "SELECT * FROM main_marts.mart_provider_features"
            ).fetchdf()
            logger.info(f"Loaded mart_provider_features (dbt only): {len(df):,} rows")
    return df


def prepare_dataset(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelDataset:
    """Prepare train/test datasets for modeling.

    Args:
        df: Feature table. If None, loads from DuckDB.
        test_size: Fraction of data for test set.
        random_state: Random seed for reproducibility.

    Returns:
        ModelDataset with train/test splits.
    """
    if df is None:
        df = load_feature_table()

    # Identify available feature columns
    available_features = [c for c in ML_FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in ML_FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features: {missing_features}")

    logger.info(f"Using {len(available_features)} features for modeling")

    # Extract features and label
    X = df[available_features].copy()
    y = df[LABEL_COLUMN].astype(bool)

    # Report class balance
    n_excluded = y.sum()
    n_total = len(y)
    logger.info(
        f"Class balance: {n_excluded:,} excluded ({100*n_excluded/n_total:.2f}%) "
        f"/ {n_total - n_excluded:,} not excluded"
    )

    # Impute missing values with column median
    null_counts = X.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        logger.info(f"Imputing nulls in {len(cols_with_nulls)} columns (median strategy)")
        X = X.fillna(X.median())

    # Replace any remaining infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split: train={len(X_train):,} ({y_train.sum()} excluded), "
        f"test={len(X_test):,} ({y_test.sum()} excluded)"
    )

    return ModelDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=available_features,
        train_df=df.iloc[train_idx].reset_index(drop=True),
        test_df=df.iloc[test_idx].reset_index(drop=True),
    )


def save_artifact(obj: object, name: str) -> Path:
    """Save a model artifact to the models directory."""
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved artifact: {path}")
    return path


def load_artifact(name: str) -> object:
    """Load a model artifact from the models directory."""
    path = settings.models_dir / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
