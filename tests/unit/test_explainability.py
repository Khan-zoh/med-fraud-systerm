"""Tests for SHAP explainability module."""

from cms_fwa.models.explainability import FEATURE_DESCRIPTIONS, _generate_narrative


def test_feature_descriptions_cover_ml_columns() -> None:
    """All ML feature columns should have plain-English descriptions."""
    from cms_fwa.features.pipeline import ML_FEATURE_COLUMNS

    missing = [f for f in ML_FEATURE_COLUMNS if f not in FEATURE_DESCRIPTIONS]
    assert missing == [], f"Missing descriptions for: {missing}"


def test_narrative_generation() -> None:
    """Narrative should include key information."""
    explanation = {
        "npi": "1234567890",
        "provider_name": "John Doe",
        "specialty": "Cardiology",
        "state": "TX",
        "risk_score": 85.0,
        "risk_tier": "Critical",
        "model_scores": {
            "isolation_forest": 0.8,
            "xgboost": 0.9,
            "graph": 0.7,
        },
        "top_contributors": [
            {
                "feature": "zscore_total_services",
                "description": "Billing volume vs. specialty peers (z-score)",
                "value": 4.5,
                "shap_impact": 0.3,
                "direction": "increases risk",
            }
        ],
    }
    narrative = _generate_narrative(explanation)
    assert "1234567890" in narrative
    assert "85" in narrative
    assert "Critical" in narrative
    assert "review" in narrative.lower()
    assert "accusation" in narrative.lower() or "not an accusation" in narrative.lower()
