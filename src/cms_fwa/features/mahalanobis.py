"""
Feature: Mahalanobis Distance (Specialty-Conditional Multivariate Outlier)

Measures how far a provider is from their specialty peer group centroid
in multivariate feature space, accounting for correlations between features.

Unlike z-scores (which flag single-feature outliers), Mahalanobis distance
captures providers who are outliers across *combinations* of features — e.g.,
a provider might have individually plausible volume and charges, but the
combination is anomalous.

IE Context: Mahalanobis distance is standard in multivariate SPC (Statistical
Process Control). In manufacturing, a product might pass each individual
dimension check but fail the multivariate profile. The same applies here —
a provider's billing "profile" must be evaluated holistically.

Properties:
  - Follows a chi-squared distribution with p degrees of freedom
  - Values > chi2_ppf(0.99, p) are extreme outliers
  - Accounts for feature correlations (unlike Euclidean distance)
"""

import numpy as np
import pandas as pd
from loguru import logger


# Features used for Mahalanobis distance calculation.
# These are the numeric features from the dbt mart that characterize
# a provider's billing profile.
MAHALANOBIS_FEATURES: list[str] = [
    "total_services",
    "total_beneficiaries",
    "total_medicare_payments",
    "unique_hcpcs_codes",
    "services_per_beneficiary",
    "wavg_submitted_charge",
    "wavg_medicare_payment",
    "facility_service_ratio",
    "drug_service_ratio",
]


def compute_mahalanobis_distance(
    provider_features: pd.DataFrame,
    features: list[str] | None = None,
    min_peer_count: int = 20,
) -> pd.DataFrame:
    """Compute Mahalanobis distance for each provider within their specialty.

    Args:
        provider_features: Provider feature table with provider_specialty
                          and the numeric features.
        features: List of feature column names to use. Defaults to
                 MAHALANOBIS_FEATURES.
        min_peer_count: Minimum providers in a specialty to compute
                       Mahalanobis (need enough for a stable covariance matrix).

    Returns:
        DataFrame with columns [npi, mahalanobis_distance, mahalanobis_pvalue].
    """
    features = features or MAHALANOBIS_FEATURES
    results: list[dict] = []

    for specialty, group in provider_features.groupby("provider_specialty"):
        if len(group) < min_peer_count:
            # Not enough peers — assign NaN
            for npi in group["npi"]:
                results.append({
                    "npi": str(npi),
                    "mahalanobis_distance": np.nan,
                    "mahalanobis_pvalue": np.nan,
                })
            continue

        # Extract feature matrix, fill NaN with column median
        X = group[features].copy()
        for col in features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.fillna(X.median())

        # Compute mean and covariance
        mean = X.mean().values
        cov = X.cov().values

        # Regularize covariance to avoid singular matrix
        # Add small ridge to diagonal (Tikhonov regularization)
        cov = cov + np.eye(len(features)) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            cov_inv = np.linalg.pinv(cov)

        # Compute Mahalanobis distance for each provider
        npis = group["npi"].values
        X_values = X.values

        for i in range(len(X_values)):
            diff = X_values[i] - mean
            # D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
            d_squared = float(diff @ cov_inv @ diff)
            d = np.sqrt(max(d_squared, 0))

            # P-value under chi-squared distribution
            from scipy import stats

            pvalue = 1.0 - stats.chi2.cdf(d_squared, df=len(features))

            results.append({
                "npi": str(npis[i]),
                "mahalanobis_distance": float(d),
                "mahalanobis_pvalue": float(pvalue),
            })

    df = pd.DataFrame(results)
    valid = df["mahalanobis_distance"].notna()
    logger.info(
        f"Computed Mahalanobis distance for {valid.sum():,} providers "
        f"({(~valid).sum():,} skipped due to small peer groups) | "
        f"mean={df.loc[valid, 'mahalanobis_distance'].mean():.2f}"
    )
    return df
