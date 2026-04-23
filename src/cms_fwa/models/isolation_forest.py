"""
Layer 1: Isolation Forest (Unsupervised Anomaly Detection)

Identifies providers with unusual billing patterns without using labels.
This is our first line of defense — it catches novel fraud patterns that
the supervised model might miss because they don't match known exclusions.

Why Isolation Forest over autoencoders:
  - More interpretable (tree-based, compatible with SHAP)
  - Faster to train on tabular data
  - No hyperparameter-sensitive architecture choices
  - Well-suited to the feature space size (~30 features)

The anomaly score is calibrated to [0, 1] where 1 = most anomalous.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from cms_fwa.models.data_prep import ModelDataset, save_artifact


def train_isolation_forest(
    dataset: ModelDataset,
    contamination: float = 0.05,
    n_estimators: int = 300,
    random_state: int = 42,
) -> dict:
    """Train an Isolation Forest on the feature set.

    Note: Isolation Forest is unsupervised — we train on ALL data (no labels).
    The contamination parameter is our prior estimate of the anomaly rate.
    5% is conservative for Medicare FWA.

    Args:
        dataset: Prepared model dataset.
        contamination: Expected fraction of anomalies.
        n_estimators: Number of trees.
        random_state: Random seed.

    Returns:
        Dict with model, scaler, and score arrays.
    """
    logger.info("Training Isolation Forest (Layer 1 — Unsupervised)")

    # Scale features (Isolation Forest benefits from standardization
    # when features have very different scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(dataset.X_train)
    X_test_scaled = scaler.transform(dataset.X_test)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train_scaled)

    # Raw scores: more negative = more anomalous
    train_raw_scores = model.decision_function(X_train_scaled)
    test_raw_scores = model.decision_function(X_test_scaled)

    # Calibrate to [0, 1] where 1 = most anomalous
    # Use min-max on combined scores for consistent scaling
    all_raw = np.concatenate([train_raw_scores, test_raw_scores])
    min_score, max_score = all_raw.min(), all_raw.max()

    def calibrate(raw: np.ndarray) -> np.ndarray:
        # Invert: raw is negative for anomalies, we want high = anomalous
        normalized = (max_score - raw) / (max_score - min_score + 1e-10)
        return np.clip(normalized, 0, 1)

    train_scores = calibrate(train_raw_scores)
    test_scores = calibrate(test_raw_scores)

    # Report
    logger.info(
        f"Isolation Forest: {n_estimators} trees, contamination={contamination}"
    )
    logger.info(
        f"  Train anomaly scores: mean={train_scores.mean():.4f}, "
        f"p95={np.percentile(train_scores, 95):.4f}, "
        f"max={train_scores.max():.4f}"
    )
    logger.info(
        f"  Test anomaly scores:  mean={test_scores.mean():.4f}, "
        f"p95={np.percentile(test_scores, 95):.4f}, "
        f"max={test_scores.max():.4f}"
    )

    # Save artifacts
    save_artifact(model, "isolation_forest_model")
    save_artifact(scaler, "isolation_forest_scaler")

    return {
        "model": model,
        "scaler": scaler,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }
