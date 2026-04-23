"""Tests for Jensen-Shannon divergence feature."""

import numpy as np
import pandas as pd
import pytest

from cms_fwa.features.divergence import (
    _jsd,
    compute_jsd_features,
    compute_specialty_median_distribution,
)


def test_jsd_identical_distributions() -> None:
    """JSD of identical distributions should be 0."""
    p = np.array([0.5, 0.3, 0.2])
    assert _jsd(p, p) == pytest.approx(0.0, abs=1e-10)


def test_jsd_completely_different() -> None:
    """JSD of non-overlapping distributions should be 1.0."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert _jsd(p, q) == pytest.approx(1.0, abs=1e-10)


def test_jsd_symmetric() -> None:
    """JSD should be symmetric: JSD(P||Q) == JSD(Q||P)."""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.4, 0.3])
    assert _jsd(p, q) == pytest.approx(_jsd(q, p), abs=1e-10)


def test_jsd_bounded() -> None:
    """JSD should always be in [0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        jsd = _jsd(p, q)
        assert 0.0 <= jsd <= 1.0 + 1e-10


def test_compute_jsd_features_basic() -> None:
    """Smoke test: compute JSD for a small synthetic dataset."""
    hcpcs_mix = pd.DataFrame({
        "npi": ["A", "A", "B", "B", "C", "C"],
        "hcpcs_code": ["99213", "99214", "99213", "99214", "99213", "99214"],
        "provider_type": ["Cardiology"] * 6,
        "hcpcs_fraction": [0.6, 0.4, 0.5, 0.5, 0.9, 0.1],
    })
    provider_features = pd.DataFrame({
        "npi": ["A", "B", "C"],
        "provider_specialty": ["Cardiology"] * 3,
    })

    result = compute_jsd_features(hcpcs_mix, provider_features)

    assert len(result) == 3
    assert "jsd_vs_specialty" in result.columns
    # Provider C (0.9/0.1) should be most divergent from median
    c_jsd = result.loc[result["npi"] == "C", "jsd_vs_specialty"].iloc[0]
    b_jsd = result.loc[result["npi"] == "B", "jsd_vs_specialty"].iloc[0]
    assert c_jsd > b_jsd
