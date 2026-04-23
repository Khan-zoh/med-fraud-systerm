"""
Feature: Jensen-Shannon Divergence (Peer Benchmarking)

Measures how different a provider's procedure mix is from their specialty
peer group median. Unlike simple ratios, JSD captures the *shape* of the
entire billing distribution.

IE Context: This is analogous to comparing a manufacturing process's output
distribution against the specification — not just the mean, but whether
the whole distribution of work products looks normal for that process type.

JSD Properties:
  - Bounded [0, 1] (using log base 2)
  - Symmetric: JSD(P || Q) = JSD(Q || P)
  - 0 means identical distributions
  - 1 means completely different distributions
  - A provider with JSD > 0.3 bills very differently from their peers
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_specialty_median_distribution(hcpcs_mix: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute the median HCPCS distribution for each specialty.

    For each specialty, we find the median fraction of services for each
    HCPCS code across all providers in that specialty. This represents
    the "typical" procedure mix for that specialty.

    Args:
        hcpcs_mix: DataFrame with columns [npi, hcpcs_code, provider_type,
                   hcpcs_fraction].

    Returns:
        Dict mapping specialty -> Series indexed by hcpcs_code with median fractions.
    """
    specialty_medians: dict[str, pd.Series] = {}

    for specialty, group in hcpcs_mix.groupby("provider_type"):
        # Pivot to provider × hcpcs_code matrix, fill missing codes with 0
        pivot = group.pivot_table(
            index="npi",
            columns="hcpcs_code",
            values="hcpcs_fraction",
            fill_value=0.0,
        )
        # Median across providers for each HCPCS code
        median_dist = pivot.median(axis=0)
        # Re-normalize to sum to 1 (medians may not sum to 1)
        total = median_dist.sum()
        if total > 0:
            median_dist = median_dist / total
        specialty_medians[str(specialty)] = median_dist

    logger.info(f"Computed median distributions for {len(specialty_medians)} specialties")
    return specialty_medians


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence between two distributions.

    Args:
        p: Provider's HCPCS distribution (sums to 1).
        q: Specialty median distribution (sums to 1).

    Returns:
        JSD value in [0, 1].
    """
    # Align and handle zero entries
    m = 0.5 * (p + q)

    # KL divergence components (with 0*log(0) = 0 convention)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1)), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1)), 0.0)

    return float(0.5 * np.sum(kl_pm) + 0.5 * np.sum(kl_qm))


def compute_jsd_features(
    hcpcs_mix: pd.DataFrame,
    provider_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute JSD for every provider vs. their specialty median.

    Args:
        hcpcs_mix: Provider × HCPCS code mix with fractions.
        provider_features: Provider feature table (needs npi, provider_specialty).

    Returns:
        DataFrame with columns [npi, jsd_vs_specialty].
    """
    specialty_medians = compute_specialty_median_distribution(hcpcs_mix)

    results: list[dict] = []

    # Group providers by specialty for efficient batch computation
    for specialty, group in hcpcs_mix.groupby("provider_type"):
        specialty_str = str(specialty)
        if specialty_str not in specialty_medians:
            continue

        median_dist = specialty_medians[specialty_str]

        # Get each provider's distribution
        for npi, provider_group in group.groupby("npi"):
            provider_dist = provider_group.set_index("hcpcs_code")["hcpcs_fraction"]

            # Align both distributions to the same set of HCPCS codes
            all_codes = median_dist.index.union(provider_dist.index)
            p = provider_dist.reindex(all_codes, fill_value=0.0).values
            q = median_dist.reindex(all_codes, fill_value=0.0).values

            # Normalize
            p_sum = p.sum()
            q_sum = q.sum()
            if p_sum > 0:
                p = p / p_sum
            if q_sum > 0:
                q = q / q_sum

            jsd = _jsd(p, q)
            results.append({"npi": str(npi), "jsd_vs_specialty": jsd})

    df = pd.DataFrame(results)
    logger.info(
        f"Computed JSD for {len(df):,} providers | "
        f"mean={df['jsd_vs_specialty'].mean():.4f}, "
        f"max={df['jsd_vs_specialty'].max():.4f}"
    )
    return df
