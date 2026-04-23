"""Tests for ensemble risk scoring."""

import numpy as np
import pytest

from cms_fwa.models.ensemble import compute_ensemble_score


def test_all_zero_inputs() -> None:
    """All-zero inputs should produce all-zero risk scores."""
    scores = compute_ensemble_score(
        if_scores=np.zeros(10),
        xgb_probas=np.zeros(10),
        graph_scores=np.zeros(10),
    )
    assert np.allclose(scores, 0.0)


def test_all_max_inputs() -> None:
    """All-1.0 inputs should produce risk score of 100."""
    scores = compute_ensemble_score(
        if_scores=np.ones(10),
        xgb_probas=np.ones(10),
        graph_scores=np.ones(10),
    )
    assert np.allclose(scores, 100.0)


def test_output_range() -> None:
    """Risk scores should always be in [0, 100]."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        scores = compute_ensemble_score(
            if_scores=rng.uniform(0, 1, 100),
            xgb_probas=rng.uniform(0, 1, 100),
            graph_scores=rng.uniform(0, 1, 100),
        )
        assert scores.min() >= 0
        assert scores.max() <= 100


def test_xgb_weight_dominates() -> None:
    """With default weights, XGBoost should contribute most to the score."""
    # XGBoost high, others low
    scores_xgb_high = compute_ensemble_score(
        if_scores=np.array([0.1]),
        xgb_probas=np.array([0.9]),
        graph_scores=np.array([0.0]),
    )
    # XGBoost low, others high
    scores_xgb_low = compute_ensemble_score(
        if_scores=np.array([0.9]),
        xgb_probas=np.array([0.1]),
        graph_scores=np.array([0.0]),
    )
    # XGBoost has 50% weight vs IF 25%, so high XGBoost should give higher total
    assert scores_xgb_high[0] > scores_xgb_low[0]


def test_custom_weights() -> None:
    """Custom weights should be respected."""
    # Use non-zero xgb_probas to avoid the redistribution trigger
    scores = compute_ensemble_score(
        if_scores=np.array([1.0]),
        xgb_probas=np.array([0.5]),
        graph_scores=np.array([0.0]),
        weights={"isolation_forest": 1.0, "xgboost": 0.0, "graph": 0.0},
    )
    assert scores[0] == pytest.approx(100.0)


def test_no_xgb_redistributes_weight() -> None:
    """When XGBoost scores are all zero, weight should redistribute."""
    scores = compute_ensemble_score(
        if_scores=np.array([0.5]),
        xgb_probas=np.array([0.0]),
        graph_scores=np.array([0.5]),
    )
    # With redistribution: 0.6*0.5 + 0.0*0.0 + 0.4*0.5 = 0.50 -> 50
    assert scores[0] == pytest.approx(50.0)
