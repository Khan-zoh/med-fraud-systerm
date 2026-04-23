"""
Training Orchestrator

Run with: python -m cms_fwa.models.train
Or:       make train

Executes the full modeling pipeline:
  1. Load and prepare data
  2. Train Layer 1 (Isolation Forest)
  3. Train Layer 2 (XGBoost)
  4. Train Layer 3 (Graph analysis)
  5. Compute ensemble risk scores
  6. Evaluate
  7. Generate SHAP explanations
  8. Save all artifacts
"""

import sys

import numpy as np
import pandas as pd
from loguru import logger

from cms_fwa.models.data_prep import ModelDataset, prepare_dataset, save_artifact
from cms_fwa.models.ensemble import build_risk_table, compute_ensemble_score
from cms_fwa.models.evaluation import evaluate_model
from cms_fwa.models.explainability import compute_shap_values
from cms_fwa.models.graph_model import train_graph_model
from cms_fwa.models.isolation_forest import train_isolation_forest
from cms_fwa.models.xgboost_model import train_xgboost
from cms_fwa.utils.logging import setup_logging


def run_training() -> dict:
    """Execute the full training pipeline.

    Returns:
        Dict with all model results and evaluation metrics.
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("CMS FWA Detection — Model Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Prepare data
    logger.info("Step 1/7: Preparing dataset")
    dataset = prepare_dataset()

    # Step 2: Layer 1 — Isolation Forest
    logger.info("\nStep 2/7: Training Isolation Forest")
    if_results = train_isolation_forest(dataset)

    # Step 3: Layer 2 — XGBoost
    logger.info("\nStep 3/7: Training XGBoost")
    xgb_results = train_xgboost(dataset)

    # Step 4: Layer 3 — Graph Analysis
    logger.info("\nStep 4/7: Building Graph Model")
    try:
        graph_results = train_graph_model()

        # Align graph scores with test set
        graph_features = graph_results["features"]
        test_npis = dataset.test_df["npi"].astype(str)
        graph_features["npi"] = graph_features["npi"].astype(str)

        graph_test_scores = test_npis.map(
            graph_features.set_index("npi")["graph_anomaly_score"]
        ).fillna(0.0).values
    except Exception as e:
        logger.warning(f"Graph model failed: {e}. Using zero scores.")
        graph_test_scores = np.zeros(len(dataset.X_test))
        graph_results = {"features": pd.DataFrame(), "scores": np.array([])}

    # Step 5: Ensemble
    logger.info("\nStep 5/7: Computing ensemble risk scores")
    risk_scores = compute_ensemble_score(
        if_scores=if_results["test_scores"],
        xgb_probas=xgb_results["test_probas"],
        graph_scores=graph_test_scores,
    )

    risk_table = build_risk_table(
        test_df=dataset.test_df,
        risk_scores=risk_scores,
        if_scores=if_results["test_scores"],
        xgb_probas=xgb_results["test_probas"],
        graph_scores=graph_test_scores,
    )

    # Step 6: Evaluate
    logger.info("\nStep 6/7: Evaluating model performance")
    metrics = evaluate_model(
        y_true=dataset.y_test,
        risk_scores=risk_scores,
        test_df=dataset.test_df,
    )

    # Step 7: SHAP explanations
    logger.info("\nStep 7/7: Computing SHAP explanations")
    shap_values = compute_shap_values(
        model=xgb_results["model"],
        X=dataset.X_test,
        feature_names=dataset.feature_names,
    )

    # Save the full results
    save_artifact({
        "feature_names": dataset.feature_names,
        "n_train": len(dataset.X_train),
        "n_test": len(dataset.X_test),
        "metrics": metrics,
    }, "training_summary")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete — Summary")
    logger.info("=" * 60)
    logger.info(f"  Providers scored: {len(risk_table):,}")
    logger.info(f"  Risk tier distribution:")
    for tier, count in risk_table["risk_tier"].value_counts().items():
        logger.info(f"    {tier}: {count:,}")
    if metrics.get("pr_auc") is not None:
        logger.info(f"  Ensemble PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"  Top 10 highest risk providers:")
    for _, row in risk_table.head(10).iterrows():
        excluded = " [EXCLUDED]" if row["is_excluded"] else ""
        logger.info(
            f"    NPI {row['npi']}: score={row['risk_score']:.1f} "
            f"({row['risk_tier']}){excluded}"
        )
    logger.info(f"\nArtifacts saved to: {settings.models_dir}")

    return {
        "dataset": dataset,
        "if_results": if_results,
        "xgb_results": xgb_results,
        "graph_results": graph_results,
        "risk_table": risk_table,
        "metrics": metrics,
        "shap_values": shap_values,
    }


# Need settings import for the summary
from cms_fwa.config import settings


def main() -> None:
    """CLI entry point."""
    results = run_training()
    has_labels = results["metrics"].get("pr_auc") is not None
    if has_labels:
        logger.info(f"\nFinal PR-AUC: {results['metrics']['pr_auc']:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
