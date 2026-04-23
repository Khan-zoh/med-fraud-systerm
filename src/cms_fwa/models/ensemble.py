"""
Ensemble Risk Scoring

Combines the three model layers into a single risk score (0-100):
  - Layer 1 (Isolation Forest): unsupervised anomaly score [0, 1]
  - Layer 2 (XGBoost): supervised fraud probability [0, 1]
  - Layer 3 (Graph): network-based anomaly score [0, 1]

The ensemble uses a weighted average with weights reflecting our
confidence in each signal:
  - XGBoost gets the highest weight when labels are available
  - Isolation Forest provides the safety net for novel patterns
  - Graph features detect coordinated schemes

The weights can be tuned based on precision@k evaluation.
"""

import numpy as np
import pandas as pd
from loguru import logger

from cms_fwa.models.data_prep import save_artifact


# Default ensemble weights — tunable based on evaluation
DEFAULT_WEIGHTS = {
    "isolation_forest": 0.25,
    "xgboost": 0.50,
    "graph": 0.25,
}


def compute_ensemble_score(
    if_scores: np.ndarray,
    xgb_probas: np.ndarray,
    graph_scores: np.ndarray,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Combine three model scores into a single risk score (0-100).

    Args:
        if_scores: Isolation Forest anomaly scores [0, 1].
        xgb_probas: XGBoost fraud probabilities [0, 1].
        graph_scores: Graph anomaly scores [0, 1].
        weights: Dict of model -> weight. Must sum to 1.

    Returns:
        Array of risk scores in [0, 100].
    """
    weights = weights or DEFAULT_WEIGHTS

    # Validate weights
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
        weights = {k: v / total_weight for k, v in weights.items()}

    # Handle case where XGBoost couldn't train (no labels)
    if xgb_probas.max() == 0 and xgb_probas.min() == 0:
        logger.warning("XGBoost scores are all zero — redistributing weight")
        weights = {
            "isolation_forest": 0.6,
            "xgboost": 0.0,
            "graph": 0.4,
        }

    combined = (
        weights["isolation_forest"] * if_scores
        + weights["xgboost"] * xgb_probas
        + weights["graph"] * graph_scores
    )

    # Scale to 0-100
    risk_scores = combined * 100

    # Clip to valid range
    risk_scores = np.clip(risk_scores, 0, 100)

    return risk_scores


def build_risk_table(
    test_df: pd.DataFrame,
    risk_scores: np.ndarray,
    if_scores: np.ndarray,
    xgb_probas: np.ndarray,
    graph_scores: np.ndarray,
) -> pd.DataFrame:
    """Build the final risk assessment table.

    Combines provider identifiers, all model scores, and the final
    risk score into a single table suitable for the dashboard.

    Args:
        test_df: Test set DataFrame with provider info.
        risk_scores: Ensemble risk scores (0-100).
        if_scores: Isolation Forest scores.
        xgb_probas: XGBoost probabilities.
        graph_scores: Graph anomaly scores.

    Returns:
        DataFrame sorted by risk score (highest first).
    """
    risk_table = test_df[
        ["npi", "provider_last_name", "provider_first_name",
         "provider_specialty", "provider_state", "is_excluded"]
    ].copy()

    risk_table["risk_score"] = risk_scores
    risk_table["if_anomaly_score"] = if_scores
    risk_table["xgb_fraud_prob"] = xgb_probas
    risk_table["graph_anomaly_score"] = graph_scores

    # Risk tier labels — percentile-based so every tier is populated regardless
    # of the absolute score distribution. Normalized ensemble scores cluster
    # tightly (~20-40), so fixed 0/20/40/60/80/100 bins leave "Low" empty.
    # These percentiles mirror how real FWA triage teams allocate attention:
    #   Low      = bottom 40%   (no action)
    #   Moderate = 40-80%       (monitor)
    #   Elevated = 80-95%       (analyst review queue)
    #   High     = 95-99%       (priority review)
    #   Critical = top 1%       (immediate investigation)
    percentile_edges = np.quantile(
        risk_table["risk_score"], [0.0, 0.40, 0.80, 0.95, 0.99, 1.0]
    )
    # Guard against duplicate edges if the score distribution is highly
    # concentrated (nudge each edge strictly above the previous one).
    for i in range(1, len(percentile_edges)):
        if percentile_edges[i] <= percentile_edges[i - 1]:
            percentile_edges[i] = percentile_edges[i - 1] + 1e-9

    risk_table["risk_tier"] = pd.cut(
        risk_table["risk_score"],
        bins=percentile_edges,
        labels=["Low", "Moderate", "Elevated", "High", "Critical"],
        include_lowest=True,
    )

    risk_table = risk_table.sort_values("risk_score", ascending=False)

    logger.info(f"Risk table: {len(risk_table):,} providers")
    logger.info(
        f"Percentile tier edges: {[round(float(e), 2) for e in percentile_edges]}"
    )
    logger.info(f"Risk tier distribution:\n{risk_table['risk_tier'].value_counts().to_string()}")

    save_artifact(risk_table, "risk_table")
    return risk_table
