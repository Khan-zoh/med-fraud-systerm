"""
Model Evaluation

Metrics chosen for the FWA detection context:
  - Precision@k: investigators can only review N cases per week —
    what fraction of the top-k flagged providers are truly excluded?
  - PR-AUC: better than ROC-AUC for imbalanced classes (most providers
    are NOT fraudulent, so ROC-AUC inflates from easy true negatives)
  - Per-specialty breakdown: model performance varies by specialty
    because billing patterns differ fundamentally

False-positive cost analysis:
  - Each investigation costs ~$X in analyst time
  - False positives waste investigator time and erode trust
  - We report the estimated investigation yield at various k
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from cms_fwa.models.data_prep import save_artifact


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute precision at top-k ranked providers.

    Args:
        y_true: Binary labels (1 = excluded).
        scores: Model scores (higher = more suspicious).
        k: Number of top providers to evaluate.

    Returns:
        Fraction of top-k that are truly excluded.
    """
    if k > len(scores):
        k = len(scores)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k) if k > 0 else 0.0


def evaluate_model(
    y_true: pd.Series,
    risk_scores: np.ndarray,
    test_df: pd.DataFrame,
    model_name: str = "ensemble",
) -> dict:
    """Run full evaluation suite.

    Args:
        y_true: Binary labels.
        risk_scores: Model risk scores (0-100).
        test_df: Test set with provider_specialty for breakdown.
        model_name: Name for logging.

    Returns:
        Dict of evaluation metrics.
    """
    y_true_arr = y_true.values.astype(int)
    n_positive = y_true_arr.sum()

    logger.info(f"Evaluating {model_name} on {len(y_true_arr):,} providers "
                f"({n_positive:,} excluded)")

    metrics: dict = {"model": model_name, "n_test": len(y_true_arr), "n_positive": int(n_positive)}

    if n_positive == 0:
        logger.warning("No positive labels in test set — limited evaluation possible")
        metrics["pr_auc"] = None
        metrics["precision_at_k"] = {}
        return metrics

    # PR-AUC
    pr_auc = average_precision_score(y_true_arr, risk_scores)
    metrics["pr_auc"] = pr_auc
    logger.info(f"  PR-AUC: {pr_auc:.4f}")

    # Precision@k for various k values
    k_values = [10, 25, 50, 100, 200, 500]
    precision_at_k_results = {}
    for k in k_values:
        if k > len(risk_scores):
            continue
        p_at_k = precision_at_k(y_true_arr, risk_scores, k)
        precision_at_k_results[k] = p_at_k
        logger.info(f"  Precision@{k}: {p_at_k:.4f} "
                     f"({int(p_at_k * k)}/{k} true positives)")
    metrics["precision_at_k"] = precision_at_k_results

    # Per-specialty breakdown
    if "provider_specialty" in test_df.columns:
        specialty_metrics = _per_specialty_evaluation(
            y_true_arr, risk_scores, test_df
        )
        metrics["per_specialty"] = specialty_metrics

    # False positive analysis
    threshold_50 = np.percentile(risk_scores, 95)  # top 5%
    y_pred_50 = (risk_scores >= threshold_50).astype(int)
    fp = ((y_pred_50 == 1) & (y_true_arr == 0)).sum()
    tp = ((y_pred_50 == 1) & (y_true_arr == 1)).sum()
    flagged = y_pred_50.sum()
    metrics["top_5pct_analysis"] = {
        "threshold": float(threshold_50),
        "flagged": int(flagged),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "precision": float(tp / flagged) if flagged > 0 else 0,
    }
    logger.info(
        f"  Top 5% (score >= {threshold_50:.1f}): "
        f"{flagged} flagged, {tp} TP, {fp} FP, "
        f"precision={tp/max(flagged,1):.4f}"
    )

    save_artifact(metrics, "evaluation_metrics")
    return metrics


def _per_specialty_evaluation(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    test_df: pd.DataFrame,
) -> list[dict]:
    """Evaluate model performance per specialty.

    Only reports specialties with at least 1 positive case and
    >= 20 total providers (meaningful evaluation).
    """
    results = []
    for specialty, idx in test_df.groupby("provider_specialty").groups.items():
        idx_arr = idx.values
        y_spec = y_true[idx_arr]
        scores_spec = risk_scores[idx_arr]

        if len(y_spec) < 20 or y_spec.sum() == 0:
            continue

        pr_auc = average_precision_score(y_spec, scores_spec)
        p_at_10 = precision_at_k(y_spec, scores_spec, min(10, len(y_spec)))

        results.append({
            "specialty": str(specialty),
            "n_providers": len(y_spec),
            "n_excluded": int(y_spec.sum()),
            "exclusion_rate": float(y_spec.mean()),
            "pr_auc": float(pr_auc),
            "precision_at_10": float(p_at_10),
        })

    results.sort(key=lambda x: x["pr_auc"], reverse=True)

    if results:
        logger.info(f"  Per-specialty PR-AUC (top 5):")
        for r in results[:5]:
            logger.info(
                f"    {r['specialty']}: PR-AUC={r['pr_auc']:.4f} "
                f"({r['n_excluded']}/{r['n_providers']} excluded)"
            )

    return results
