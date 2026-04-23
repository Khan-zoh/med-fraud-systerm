"""Tests for geographic dispersion entropy feature."""

import numpy as np
import pandas as pd

from cms_fwa.features.geographic import compute_geographic_entropy


def test_single_zip_zero_entropy() -> None:
    """A provider with all patients in one zip should have zero entropy."""
    claims = pd.DataFrame({
        "npi": ["A", "A"],
        "provider_zip5": ["75001", "75001"],
        "total_beneficiaries": [50, 30],
    })
    result = compute_geographic_entropy(claims)
    assert len(result) == 1
    assert result.iloc[0]["zip_entropy"] == 0.0
    assert result.iloc[0]["zip_entropy_normalized"] == 0.0
    assert result.iloc[0]["unique_zip_count"] == 1


def test_uniform_distribution_max_entropy() -> None:
    """Equal patients across N zips should give normalized entropy of 1.0."""
    claims = pd.DataFrame({
        "npi": ["A"] * 4,
        "provider_zip5": ["75001", "75002", "75003", "75004"],
        "total_beneficiaries": [25, 25, 25, 25],
    })
    result = compute_geographic_entropy(claims)
    assert result.iloc[0]["zip_entropy_normalized"] == 1.0
    assert result.iloc[0]["unique_zip_count"] == 4
    # Max entropy for 4 categories = log2(4) = 2.0
    assert abs(result.iloc[0]["zip_entropy"] - 2.0) < 1e-10


def test_multiple_providers() -> None:
    """Should compute entropy independently for each provider."""
    claims = pd.DataFrame({
        "npi": ["A", "A", "B", "B", "B"],
        "provider_zip5": ["75001", "75001", "75001", "75002", "75003"],
        "total_beneficiaries": [50, 30, 10, 10, 10],
    })
    result = compute_geographic_entropy(claims)
    assert len(result) == 2

    a_entropy = result.loc[result["npi"] == "A", "zip_entropy"].iloc[0]
    b_entropy = result.loc[result["npi"] == "B", "zip_entropy"].iloc[0]
    # Provider B has 3 zips, A has 1 — B should have higher entropy
    assert b_entropy > a_entropy
