"""Per-tier and per-feature diagnostics for the current ensemble.

Helps answer: do the labels we added in Phase A actually look anomalous in
feature space, or are they administratively excluded people who bill normally?
"""
from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def main() -> None:
    with open("data/models/risk_table.pkl", "rb") as f:
        rt: pd.DataFrame = pickle.load(f)
    rt["npi"] = rt["npi"].astype(str)

    # Join match-tier info from the labels table
    con = duckdb.connect("data/cms_fwa.duckdb", read_only=True)
    labels = con.execute("""
        select cast(npi as varchar) as npi, match_tier
        from main.provider_exclusion_labels
    """).df()
    con.close()

    rt = rt.merge(labels, on="npi", how="left")

    y_full = rt["is_excluded"].astype(int).values
    score_full = rt["risk_score"].values

    def block(name, y, s):
        if y.sum() == 0:
            print(f"\n{name}: no positives")
            return
        print(f"\n=== {name} ===")
        print(f"  n={len(y):,}, positives={int(y.sum())}, base rate={y.mean():.5f}")
        try:
            print(f"  ROC-AUC: {roc_auc_score(y, s):.4f}")
            print(f"  PR-AUC:  {average_precision_score(y, s):.5f}")
        except Exception as e:
            print(f"  AUC err: {e}")
        for pct in (0.01, 0.05, 0.10, 0.20):
            k = max(1, int(pct * len(y)))
            top = np.argsort(s)[-k:]
            p = y[top].mean()
            rec = y[top].sum() / max(1, y.sum())
            lift = p / y.mean() if y.mean() else float("nan")
            print(f"  top-{pct*100:>4.1f}% (n={k:>5d}): precision={p:.4f}, recall={rec:.2%}, lift={lift:.1f}x")

    block("OVERALL (all tiers)", y_full, score_full)

    # Per-tier eval: keep negatives + only this tier's positives
    for tier in (1, 2, 3):
        m = ((rt["match_tier"] == tier) | (~rt["is_excluded"]))
        sub = rt[m]
        y = sub["is_excluded"].astype(int).values
        s = sub["risk_score"].values
        block(f"Tier {tier} positives only (vs all negatives)", y, s)

    # T1 + T2 only (high-confidence)
    m = (rt["match_tier"].isin([1, 2])) | (~rt["is_excluded"])
    sub = rt[m]
    block("T1+T2 only (high-confidence labels)",
          sub["is_excluded"].astype(int).values, sub["risk_score"].values)

    # Feature-space sanity: are positives actually anomalous?
    print("\n=== Feature distribution: excluded vs not ===")
    for col in ["total_medicare_payments", "zscore_total_services",
                "services_per_beneficiary", "high_complexity_ratio"]:
        if col not in rt.columns:
            continue
        pos = rt.loc[rt["is_excluded"], col].dropna()
        neg = rt.loc[~rt["is_excluded"], col].dropna()
        print(f"  {col}:")
        print(f"    excluded:    mean={pos.mean():>12.3f}, median={pos.median():>12.3f}, p95={pos.quantile(.95):>12.3f}")
        print(f"    not excluded:mean={neg.mean():>12.3f}, median={neg.median():>12.3f}, p95={neg.quantile(.95):>12.3f}")

    # Specialty distribution of positives
    print("\n=== Top specialties among labeled positives ===")
    print(rt[rt["is_excluded"]]["provider_specialty"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
