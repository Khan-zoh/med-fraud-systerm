"""
Feature: Billing Velocity / Throughput Analysis

Flags providers whose billing rates exceed theoretical throughput limits.
If a provider bills more services per day than is physically possible
given standard appointment durations, that's a strong anomaly signal.

IE Context: This is direct application of queueing theory and throughput
analysis from industrial engineering. Every service process has a maximum
throughput determined by:
  - Service time per patient/procedure
  - Available hours per day
  - Number of parallel servers (the provider + staff)

A solo practitioner billing 200 office visits in a single day is
exceeding their theoretical capacity — either they have unusual staffing
or the billing is suspicious.

Assumptions (conservative, configurable):
  - Working days per year: 250 (5 days/week, minus holidays)
  - Max services per day (office): 40 (one every 12 minutes for 8 hours)
  - Max services per day (facility): 25 (procedures take longer)
"""

import numpy as np
import pandas as pd
from loguru import logger


# Throughput limits by place of service
# These are generous upper bounds — most providers bill far fewer
WORKING_DAYS_PER_YEAR = 250
MAX_OFFICE_SERVICES_PER_DAY = 40
MAX_FACILITY_SERVICES_PER_DAY = 25


def compute_velocity_features(
    claims: pd.DataFrame,
    working_days: int = WORKING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute billing velocity metrics for each provider.

    Args:
        claims: Staged claims with [npi, total_services, total_beneficiaries,
               total_beneficiary_day_services, place_of_service].
        working_days: Assumed working days per year.

    Returns:
        DataFrame with columns [npi, daily_services, daily_beneficiaries,
        services_per_bene_per_day, exceeds_throughput_limit,
        throughput_utilization].
    """
    # Pre-compute facility/office service columns before groupby
    df = claims.copy()
    df["facility_services"] = df["total_services"].where(
        df["place_of_service"] == "F", 0
    )
    df["office_services"] = df["total_services"].where(
        df["place_of_service"] == "O", 0
    )

    # Aggregate to provider level
    provider_agg = df.groupby("npi").agg(
        total_services=("total_services", "sum"),
        total_beneficiaries=("total_beneficiaries", "sum"),
        total_bene_day_services=("total_beneficiary_day_services", "sum"),
        facility_services=("facility_services", "sum"),
        office_services=("office_services", "sum"),
    ).reset_index()

    # Compute daily rates (annualized)
    provider_agg["daily_services"] = (
        provider_agg["total_services"] / working_days
    )
    provider_agg["daily_beneficiaries"] = (
        provider_agg["total_beneficiaries"] / working_days
    )

    # Services per beneficiary per encounter day
    provider_agg["services_per_bene_per_day"] = (
        provider_agg["total_bene_day_services"]
        / provider_agg["total_beneficiaries"].replace(0, np.nan)
    )

    # Throughput analysis: compare daily service rate to theoretical max
    # Use weighted max based on facility vs. office mix
    total_svc = provider_agg["total_services"].replace(0, np.nan)
    facility_frac = provider_agg["facility_services"] / total_svc
    office_frac = provider_agg["office_services"] / total_svc

    # Blended throughput limit
    provider_agg["theoretical_max_daily"] = (
        facility_frac.fillna(0) * MAX_FACILITY_SERVICES_PER_DAY
        + office_frac.fillna(0) * MAX_OFFICE_SERVICES_PER_DAY
    )
    # Floor at a minimum to avoid division by zero
    provider_agg["theoretical_max_daily"] = provider_agg[
        "theoretical_max_daily"
    ].clip(lower=1.0)

    # Throughput utilization: how close to theoretical max?
    # > 1.0 means exceeding theoretical capacity
    provider_agg["throughput_utilization"] = (
        provider_agg["daily_services"] / provider_agg["theoretical_max_daily"]
    )

    provider_agg["exceeds_throughput_limit"] = (
        provider_agg["throughput_utilization"] > 1.0
    )

    result = provider_agg[
        [
            "npi",
            "daily_services",
            "daily_beneficiaries",
            "services_per_bene_per_day",
            "throughput_utilization",
            "exceeds_throughput_limit",
        ]
    ].copy()

    exceed_count = result["exceeds_throughput_limit"].sum()
    logger.info(
        f"Computed velocity features for {len(result):,} providers | "
        f"{exceed_count:,} ({100*exceed_count/max(len(result),1):.1f}%) exceed "
        f"throughput limits"
    )
    return result
