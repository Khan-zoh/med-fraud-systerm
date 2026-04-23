"""
SHAP Explainability

Generates human-readable explanations for why a provider was flagged.
Essential for government/regulated use — investigators need to understand
the model's reasoning before acting on a flag.

We compute SHAP values for the XGBoost model (Layer 2) and translate
feature contributions into plain English. The Isolation Forest and
graph scores are reported as-is since they're already interpretable.

Language guidelines (per CMS/OIG conventions):
  - "anomalous billing pattern" not "fraud"
  - "flagged for review" not "guilty"
  - "elevated risk score" not "suspicious"
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from loguru import logger

from cms_fwa.models.data_prep import save_artifact


# Plain-English descriptions for each feature
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "total_services": "Total number of services billed",
    "total_beneficiaries": "Total unique Medicare beneficiaries served",
    "total_medicare_payments": "Total Medicare payments received",
    "total_submitted_charges": "Total charges submitted",
    "unique_hcpcs_codes": "Number of distinct procedure codes billed",
    "services_per_beneficiary": "Average services per beneficiary",
    "facility_service_ratio": "Fraction of services in facility setting",
    "drug_service_ratio": "Fraction of services that are drug-related",
    "wavg_submitted_charge": "Average submitted charge per service",
    "wavg_medicare_payment": "Average Medicare payment per service",
    "zscore_total_services": "Billing volume vs. specialty peers (z-score)",
    "zscore_total_payments": "Payment volume vs. specialty peers (z-score)",
    "zscore_total_beneficiaries": "Patient volume vs. specialty peers (z-score)",
    "zscore_services_per_bene": "Services per patient vs. specialty peers (z-score)",
    "zscore_unique_hcpcs": "Procedure diversity vs. specialty peers (z-score)",
    "zscore_avg_charge": "Average charge vs. specialty peers (z-score)",
    "ratio_services_vs_median": "Billing volume relative to specialty median",
    "ratio_payments_vs_median": "Payments relative to specialty median",
    "ratio_svc_per_bene_vs_median": "Services per patient relative to specialty median",
    "beneficiary_day_intensity": "Services per beneficiary per encounter day",
    "high_complexity_ratio": "Fraction of high-complexity procedure codes",
    "upcoding_ratio_vs_peers": "High-complexity ratio relative to specialty peers",
    "jsd_vs_specialty": "Procedure mix divergence from specialty norm",
    "zip_entropy": "Geographic spread of patients (entropy)",
    "zip_entropy_normalized": "Normalized geographic dispersion (0=local, 1=dispersed)",
    "unique_zip_count": "Number of distinct patient zip codes",
    "daily_services": "Estimated services per working day",
    "daily_beneficiaries": "Estimated patients per working day",
    "services_per_bene_per_day": "Services per beneficiary per encounter day",
    "throughput_utilization": "Daily service rate vs. theoretical capacity",
    "mahalanobis_distance": "Multivariate distance from specialty peer group center",
    "mahalanobis_pvalue": "Statistical significance of multivariate outlier status",
}


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 1000,
) -> shap.Explanation:
    """Compute SHAP values for the XGBoost model.

    Args:
        model: Trained XGBoost model.
        X: Feature matrix.
        feature_names: Column names.
        max_samples: Max samples to explain (for speed).

    Returns:
        SHAP Explanation object.
    """
    if model is None:
        logger.warning("No XGBoost model available — cannot compute SHAP values")
        return None

    logger.info(f"Computing SHAP values for {min(len(X), max_samples):,} providers...")

    # Use TreeExplainer (exact, fast for tree models)
    explainer = shap.TreeExplainer(model)

    # Subsample for speed if needed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    shap_values = explainer(X_sample)

    save_artifact(explainer, "shap_explainer")
    logger.info("SHAP values computed successfully")

    return shap_values


def explain_provider(
    npi: str,
    risk_table: pd.DataFrame,
    shap_values: shap.Explanation | None,
    X: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> dict:
    """Generate a human-readable explanation for a single provider.

    Args:
        npi: Provider NPI to explain.
        risk_table: Risk assessment table.
        shap_values: SHAP values (may be None if XGBoost wasn't trained).
        X: Feature matrix aligned with risk_table.
        feature_names: Feature column names.
        top_k: Number of top contributing features to show.

    Returns:
        Dict with explanation components.
    """
    # Find provider in risk table
    provider_mask = risk_table["npi"].astype(str) == str(npi)
    if not provider_mask.any():
        return {"error": f"NPI {npi} not found in risk table"}

    provider = risk_table[provider_mask].iloc[0]

    explanation = {
        "npi": str(npi),
        "provider_name": f"{provider.get('provider_first_name', '')} {provider.get('provider_last_name', '')}".strip(),
        "specialty": provider.get("provider_specialty", "Unknown"),
        "state": provider.get("provider_state", ""),
        "risk_score": float(provider["risk_score"]),
        "risk_tier": str(provider["risk_tier"]),
        "model_scores": {
            "isolation_forest": float(provider["if_anomaly_score"]),
            "xgboost": float(provider["xgb_fraud_prob"]),
            "graph": float(provider["graph_anomaly_score"]),
        },
        "top_contributors": [],
        "narrative": "",
    }

    # SHAP-based feature contributions
    if shap_values is not None:
        provider_idx = provider_mask[provider_mask].index[0]
        # Find this index in the SHAP values (may be subsampled)
        if provider_idx < len(shap_values):
            sv = shap_values[provider_idx]
            feature_contribs = pd.Series(
                sv.values, index=feature_names[:len(sv.values)]
            ).abs().sort_values(ascending=False)

            for feat_name in feature_contribs.head(top_k).index:
                description = FEATURE_DESCRIPTIONS.get(feat_name, feat_name)
                feat_idx = feature_names.index(feat_name) if feat_name in feature_names else -1
                feat_value = float(X.iloc[provider_idx][feat_name]) if feat_idx >= 0 and feat_name in X.columns else None
                shap_value = float(feature_contribs[feat_name])

                explanation["top_contributors"].append({
                    "feature": feat_name,
                    "description": description,
                    "value": feat_value,
                    "shap_impact": shap_value,
                    "direction": "increases risk" if sv.values[feature_names.index(feat_name)] > 0 else "decreases risk",
                })

    # Generate narrative
    explanation["narrative"] = _generate_narrative(explanation)

    return explanation


def _generate_narrative(explanation: dict) -> str:
    """Generate a plain-English narrative for the explanation."""
    parts = [
        f"Provider {explanation['npi']} ({explanation['specialty']}, "
        f"{explanation['state']}) has a risk score of "
        f"{explanation['risk_score']:.0f}/100 ({explanation['risk_tier']})."
    ]

    if explanation["top_contributors"]:
        parts.append("\nKey factors contributing to this assessment:")
        for i, contrib in enumerate(explanation["top_contributors"], 1):
            value_str = f" (value: {contrib['value']:.2f})" if contrib["value"] is not None else ""
            parts.append(
                f"  {i}. {contrib['description']}{value_str} — "
                f"{contrib['direction']}"
            )

    parts.append(
        "\nNote: This is a statistical flag for review, not an accusation. "
        "Anomalous billing patterns may have legitimate explanations."
    )

    return "\n".join(parts)
