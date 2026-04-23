"""Tests for model evaluation metrics."""

import numpy as np

from cms_fwa.models.evaluation import precision_at_k


def test_precision_at_k_perfect() -> None:
    """Perfect ranking: all positives at the top."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.05, 0.01])
    assert precision_at_k(y_true, scores, k=3) == 1.0


def test_precision_at_k_worst() -> None:
    """Worst ranking: all positives at the bottom."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.05, 0.01])
    assert precision_at_k(y_true, scores, k=3) == 0.0


def test_precision_at_k_mixed() -> None:
    """Mixed ranking: 2 of top 4 are positive."""
    y_true = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y_true, scores, k=4) == 0.5


def test_precision_at_k_larger_than_n() -> None:
    """k larger than array size should be clamped."""
    y_true = np.array([1, 0, 1])
    scores = np.array([0.9, 0.5, 0.1])
    # k=100 but only 3 items, precision = 2/3
    result = precision_at_k(y_true, scores, k=100)
    assert abs(result - 2/3) < 1e-10
