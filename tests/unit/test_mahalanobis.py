"""Tests for Mahalanobis distance feature."""

import numpy as np
import pandas as pd
import pytest

from cms_fwa.features.mahalanobis import MAHALANOBIS_FEATURES, compute_mahalanobis_distance


def _make_synthetic_providers(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create synthetic provider feature data for testing."""
    rng = np.random.default_rng(seed)

    data = {
        "npi": [f"NPI{i:04d}" for i in range(n)],
        "provider_specialty": ["Cardiology"] * n,
    }
    for feat in MAHALANOBIS_FEATURES:
        data[feat] = rng.normal(100, 20, size=n)

    # Make the last provider an extreme outlier
    for feat in MAHALANOBIS_FEATURES:
        data[feat][-1] = 500.0

    return pd.DataFrame(data)


def test_outlier_has_highest_distance() -> None:
    """The extreme outlier should have the largest Mahalanobis distance."""
    df = _make_synthetic_providers(n=30)
    result = compute_mahalanobis_distance(df, min_peer_count=10)

    outlier_dist = result.loc[result["npi"] == "NPI0029", "mahalanobis_distance"].iloc[0]
    max_dist = result["mahalanobis_distance"].max()
    assert outlier_dist == max_dist


def test_small_group_gets_nan() -> None:
    """Specialties with fewer than min_peer_count should get NaN."""
    df = _make_synthetic_providers(n=5)
    result = compute_mahalanobis_distance(df, min_peer_count=10)

    assert result["mahalanobis_distance"].isna().all()


def test_distances_are_non_negative() -> None:
    """All Mahalanobis distances should be >= 0."""
    df = _make_synthetic_providers(n=50)
    result = compute_mahalanobis_distance(df, min_peer_count=10)

    valid = result["mahalanobis_distance"].dropna()
    assert (valid >= 0).all()


def test_pvalue_range() -> None:
    """P-values should be in [0, 1]."""
    df = _make_synthetic_providers(n=50)
    result = compute_mahalanobis_distance(df, min_peer_count=10)

    valid = result["mahalanobis_pvalue"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()
