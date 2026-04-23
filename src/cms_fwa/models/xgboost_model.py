"""
Layer 2: XGBoost Supervised Classifier

Learns from LEIE exclusion labels to predict which providers are likely
to engage in FWA. Handles the severe class imbalance (~1-2% positive rate)
through scale_pos_weight and careful evaluation metrics.

Why XGBoost over other classifiers:
  - Excellent with tabular data and mixed feature types
  - Built-in handling of missing values
  - Native feature importance + SHAP compatibility
  - Regularization prevents overfitting on small positive class
  - scale_pos_weight handles imbalance without synthetic oversampling

We prefer scale_pos_weight over SMOTE because:
  - SMOTE creates synthetic samples that may not represent real fraud patterns
  - scale_pos_weight adjusts the loss function directly (cleaner)
  - Less risk of data leakage in cross-validation
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from cms_fwa.models.data_prep import ModelDataset, save_artifact


def train_xgboost(
    dataset: ModelDataset,
    random_state: int = 42,
) -> dict:
    """Train an XGBoost classifier with class imbalance handling.

    Args:
        dataset: Prepared model dataset with labels.
        random_state: Random seed.

    Returns:
        Dict with model, predictions, and probabilities.
    """
    logger.info("Training XGBoost Classifier (Layer 2 — Supervised)")

    # Compute class weight ratio for imbalanced learning
    n_pos = dataset.y_train.sum()
    n_neg = len(dataset.y_train) - n_pos

    if n_pos == 0:
        logger.warning("No positive examples in training set! Skipping XGBoost.")
        return {
            "model": None,
            "train_probas": np.zeros(len(dataset.X_train)),
            "test_probas": np.zeros(len(dataset.X_test)),
        }

    scale_pos_weight = n_neg / n_pos
    logger.info(f"Class ratio: {n_neg}:{n_pos} (scale_pos_weight={scale_pos_weight:.1f})")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=1.0,     # L2 regularization
        random_state=random_state,
        eval_metric="aucpr",  # PR-AUC is the right metric for imbalanced data
        early_stopping_rounds=20,
        verbosity=0,
    )

    model.fit(
        dataset.X_train,
        dataset.y_train,
        eval_set=[(dataset.X_test, dataset.y_test)],
        verbose=False,
    )

    # Predict probabilities
    train_probas = model.predict_proba(dataset.X_train)[:, 1]
    test_probas = model.predict_proba(dataset.X_test)[:, 1]

    # Evaluate
    if dataset.y_test.sum() > 0:
        pr_auc = average_precision_score(dataset.y_test, test_probas)
        logger.info(f"  Test PR-AUC: {pr_auc:.4f}")

        # Precision at various recall levels
        precision, recall, thresholds = precision_recall_curve(
            dataset.y_test, test_probas
        )
        for target_recall in [0.5, 0.7, 0.9]:
            idx = np.argmin(np.abs(recall - target_recall))
            logger.info(
                f"  Precision@recall={target_recall:.0%}: "
                f"{precision[idx]:.4f} (threshold={thresholds[min(idx, len(thresholds)-1)]:.4f})"
            )
    else:
        logger.warning("No positive examples in test set — cannot compute PR-AUC")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=dataset.feature_names
    ).sort_values(ascending=False)
    logger.info(f"  Top 10 features:\n{importance.head(10).to_string()}")

    save_artifact(model, "xgboost_model")

    return {
        "model": model,
        "train_probas": train_probas,
        "test_probas": test_probas,
        "feature_importance": importance,
    }
