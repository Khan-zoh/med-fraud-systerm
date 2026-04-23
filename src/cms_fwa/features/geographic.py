"""
Feature: Geographic Dispersion (Patient Zip-Code Entropy)

Measures how geographically spread out a provider's patients are.
A provider with high patient zip-code entropy serves patients from
many different zip codes — which could indicate:
  - Legitimate: large referral practice or hospital
  - Suspicious: patient brokering, ambulance chasing, phantom billing

IE Context: In facility layout and logistics, geographic dispersion
is a standard metric for service area analysis. High entropy in a
provider's patient catchment area, especially relative to peers in
the same specialty and location, warrants investigation.

Shannon Entropy Properties:
  - 0 when all patients come from one zip code
  - log2(N) when patients are uniformly distributed across N zip codes
  - We normalize to [0, 1] by dividing by log2(N)
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_geographic_entropy(claims: pd.DataFrame) -> pd.DataFrame:
    """Compute patient zip-code entropy for each provider.

    Uses the provider's patient zip code distribution (from Part B data,
    where provider_zip5 approximates the patient's area). In real CMS
    claims, we'd use beneficiary zip — here we use the provider's
    reported zip as a proxy available in public data.

    Args:
        claims: Staged claims with [npi, provider_zip5, total_beneficiaries].

    Returns:
        DataFrame with columns [npi, zip_entropy, zip_entropy_normalized,
        unique_zip_count].
    """
    # Aggregate beneficiaries per provider × zip code
    zip_dist = (
        claims.groupby(["npi", "provider_zip5"])["total_beneficiaries"]
        .sum()
        .reset_index()
    )

    results: list[dict] = []

    for npi, group in zip_dist.groupby("npi"):
        counts = group["total_beneficiaries"].values.astype(float)
        total = counts.sum()

        if total <= 0:
            results.append({
                "npi": str(npi),
                "zip_entropy": 0.0,
                "zip_entropy_normalized": 0.0,
                "unique_zip_count": 0,
            })
            continue

        # Convert to probability distribution
        probs = counts / total

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(np.where(probs > 0, probs, 1)))

        # Normalized entropy (0 = single zip, 1 = uniform across all zips)
        n_zips = len(counts)
        max_entropy = np.log2(n_zips) if n_zips > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        results.append({
            "npi": str(npi),
            "zip_entropy": float(entropy),
            "zip_entropy_normalized": float(normalized),
            "unique_zip_count": n_zips,
        })

    df = pd.DataFrame(results)
    logger.info(
        f"Computed geographic entropy for {len(df):,} providers | "
        f"mean_entropy={df['zip_entropy'].mean():.3f}, "
        f"mean_unique_zips={df['unique_zip_count'].mean():.1f}"
    )
    return df
