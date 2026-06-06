"""Sanity check: examine the top-flagged providers and a random middle-risk
sample to confirm the model is ranking *something* sensible (outliers in
feature space), even if those outliers don't overlap heavily with OIG labels.
"""
from __future__ import annotations

import pickle

import duckdb
import pandas as pd


def main() -> None:
    with open("data/models/risk_table.pkl", "rb") as f:
        rt: pd.DataFrame = pickle.load(f)
    rt["npi"] = rt["npi"].astype(str)

    con = duckdb.connect("data/cms_fwa.duckdb", read_only=True)
    feats = con.execute("""
        select cast(npi as varchar) as npi,
               provider_specialty,
               total_medicare_payments,
               total_services,
               total_beneficiaries,
               services_per_beneficiary,
               zscore_total_services,
               zscore_total_payments,
               high_complexity_ratio,
               upcoding_ratio_vs_peers
        from main_marts.mart_provider_features
    """).df()
    con.close()
    # Drop duplicated cols from rt to avoid _x/_y collisions
    drop_cols = [c for c in feats.columns if c != "npi" and c in rt.columns]
    rt2 = rt.drop(columns=drop_cols)
    df = rt2.merge(feats, on="npi", how="left")

    cols = ["npi", "provider_specialty", "risk_score", "is_excluded",
            "total_medicare_payments", "total_services", "services_per_beneficiary",
            "zscore_total_services", "zscore_total_payments",
            "upcoding_ratio_vs_peers"]

    print("=== TOP 25 FLAGGED PROVIDERS ===")
    top = df.sort_values("risk_score", ascending=False).head(25)
    print(top[cols].to_string(index=False))

    print("\n=== TOP 25 SUMMARY ===")
    print(f"Median payments:           ${top['total_medicare_payments'].median():,.0f}")
    print(f"Median services:           {top['total_services'].median():,.0f}")
    print(f"Median services/bene:      {top['services_per_beneficiary'].median():.1f}")
    print(f"Median z(services):        {top['zscore_total_services'].median():.2f}")
    print(f"Median upcoding ratio:     {top['upcoding_ratio_vs_peers'].median():.2f}")
    print(f"Excluded in top 25:        {top['is_excluded'].sum()}")
    print(f"Specialties (top 5):")
    print(top['provider_specialty'].value_counts().head().to_string())

    print("\n=== POPULATION COMPARISON ===")
    print(f"Median payments (all):     ${df['total_medicare_payments'].median():,.0f}")
    print(f"Median services (all):     {df['total_services'].median():,.0f}")
    print(f"Median services/bene (all):{df['services_per_beneficiary'].median():.1f}")

    print("\n=== RANDOM 10 MIDDLE-RISK (40th-60th percentile) ===")
    p40, p60 = df["risk_score"].quantile([0.4, 0.6])
    mid = df[(df["risk_score"] >= p40) & (df["risk_score"] <= p60)].sample(10, random_state=7)
    print(mid[cols].to_string(index=False))


if __name__ == "__main__":
    main()
