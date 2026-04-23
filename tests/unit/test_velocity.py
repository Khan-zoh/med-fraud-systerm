"""Tests for billing velocity / throughput analysis feature."""

import numpy as np
import pandas as pd

from cms_fwa.features.velocity import compute_velocity_features


def _make_claims(
    npi: str = "A",
    total_services: float = 5000,
    total_benes: int = 200,
    total_bene_day: float = 300,
    place: str = "O",
) -> pd.DataFrame:
    return pd.DataFrame({
        "npi": [npi],
        "total_services": [total_services],
        "total_beneficiaries": [total_benes],
        "total_beneficiary_day_services": [total_bene_day],
        "place_of_service": [place],
    })


def test_normal_provider_below_limit() -> None:
    """A provider with 5000 services/year should be well below throughput."""
    claims = _make_claims(total_services=5000)
    result = compute_velocity_features(claims)

    assert len(result) == 1
    assert result.iloc[0]["exceeds_throughput_limit"] == False
    assert result.iloc[0]["throughput_utilization"] < 1.0


def test_extreme_provider_exceeds_limit() -> None:
    """A provider with 15000 office services/year exceeds 40/day × 250 days."""
    claims = _make_claims(total_services=15000)
    result = compute_velocity_features(claims)

    # 15000 / 250 = 60 per day, limit is 40
    assert result.iloc[0]["exceeds_throughput_limit"] == True
    assert result.iloc[0]["throughput_utilization"] > 1.0


def test_daily_services_calculation() -> None:
    """Daily services should be total / working_days."""
    claims = _make_claims(total_services=2500)
    result = compute_velocity_features(claims, working_days=250)

    assert result.iloc[0]["daily_services"] == 10.0


def test_multiple_providers() -> None:
    """Should compute independently for each provider."""
    claims = pd.concat([
        _make_claims("A", total_services=1000),
        _make_claims("B", total_services=12000),
    ])
    result = compute_velocity_features(claims)

    assert len(result) == 2
    a_util = result.loc[result["npi"] == "A", "throughput_utilization"].iloc[0]
    b_util = result.loc[result["npi"] == "B", "throughput_utilization"].iloc[0]
    assert b_util > a_util
