"""Programmatic audit of LEIE-to-provider match candidates.

Generates two artifacts:
  - data/processed/leie_audit_sample.csv  (sampled per tier, with FP flags, for
    optional human eyeballing)
  - data/processed/leie_audit_report.json (estimated FP rate per tier, decision)

Heuristic FP flags (applied to T3/T4 only — T1/T2 are exact-key matches):
  - cross_gender:   first-name pair is a known gendered variant (DANIEL/DANIELLE,
                    PAUL/PAULA, ROBERT/ROBERTA, etc.)
  - both_low:       last-name ratio < 90 AND first-name ratio < 90
  - common_last:    very common last name (SMITH/JONES/JOHNSON/...) combined with
                    any first-name mismatch
  - last_too_short: normalized last name <= 3 chars (high collision risk)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import pandas as pd

PROCESSED = "data/processed"
DOCS_DIR = "docs"
RNG_SEED = 7

# (male_form, female_form) — first-name pairs that look identical to a fuzzy
# matcher but are almost certainly different people
GENDERED_PAIRS = {
    frozenset({"DANIEL", "DANIELLE"}),
    frozenset({"PAUL", "PAULA"}),
    frozenset({"ROBERT", "ROBERTA"}),
    frozenset({"JOHN", "JOHANNA"}),
    frozenset({"GLENN", "GLENNA"}),
    frozenset({"PATRICK", "PATRICIA"}),
    frozenset({"MARTIN", "MARTINA"}),
    frozenset({"JOSEPH", "JOSEPHINE"}),
    frozenset({"WILLIAM", "WILMA"}),
    frozenset({"CARL", "CARLA"}),
    frozenset({"FRANCIS", "FRANCES"}),
    frozenset({"JAMES", "JAMIE"}),
    frozenset({"GEORGE", "GEORGIA"}),
    frozenset({"DENNIS", "DENISE"}),
}

COMMON_LAST_NAMES = {
    "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER",
    "DAVIS", "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ",
    "WILSON", "ANDERSON", "THOMAS", "TAYLOR", "MOORE", "JACKSON", "MARTIN",
    "LEE", "PEREZ", "THOMPSON", "WHITE", "HARRIS", "SANCHEZ", "CLARK",
    "RAMIREZ", "LEWIS", "ROBINSON", "WALKER", "YOUNG", "ALLEN", "KING",
    "WRIGHT", "SCOTT", "TORRES", "NGUYEN",
}


def flag_row(row) -> dict:
    """Apply heuristic FP flags. Returns dict of flag_name -> bool."""
    flags = {
        "cross_gender": False,
        "both_low": False,
        "common_last": False,
        "last_too_short": False,
    }
    if row["match_tier"] not in (3, 4):
        return flags

    last_l = (row.get("leie_last_norm") or "").upper().strip()
    last_m = (row.get("mart_last_norm") or "").upper().strip()
    first_l = (row.get("leie_first_norm") or "").upper().strip()
    first_m = (row.get("mart_first_norm") or "").upper().strip()

    # Gendered pair check (T3 only — T4 is business)
    if row["match_tier"] == 3 and first_l and first_m:
        if frozenset({first_l, first_m}) in GENDERED_PAIRS:
            flags["cross_gender"] = True

    # Per-component scores (saved by tier3_fuzzy_individual)
    last_score = row.get("last_score")
    first_score = row.get("first_score")
    if pd.notna(last_score) and pd.notna(first_score):
        if last_score < 90 and first_score < 90:
            flags["both_low"] = True

    # Common last name + ANY first-name mismatch
    if last_l in COMMON_LAST_NAMES and first_l != first_m:
        flags["common_last"] = True

    # Very short last name (individuals only; meaningless for businesses)
    if row["match_tier"] == 3 and len(last_l) <= 3:
        flags["last_too_short"] = True

    return flags


def audit() -> None:
    audited = pd.read_parquet(f"{PROCESSED}/leie_match_candidates.parquet")
    final = pd.read_parquet(f"{PROCESSED}/leie_final_labels.parquet")

    # Filter to rows that survived conflict resolution
    kept = audited[audited["keep_for_training"]].copy()

    # Apply flags
    flag_records = kept.apply(flag_row, axis=1)
    flag_df = pd.DataFrame(flag_records.tolist(), index=kept.index)
    kept = pd.concat([kept, flag_df], axis=1)
    kept["any_flag"] = kept[list(flag_df.columns)].any(axis=1)

    # Per-tier FP estimate (suspect flag = proxy for "would human reject")
    report = {"per_tier": {}, "totals": {}}
    for tier in sorted(kept["match_tier"].unique()):
        sub = kept[kept["match_tier"] == tier]
        n = len(sub)
        n_flagged = int(sub["any_flag"].sum()) if "any_flag" in sub else 0
        report["per_tier"][int(tier)] = {
            "n": int(n),
            "n_flagged_fp": n_flagged,
            "estimated_fp_rate": round(n_flagged / n, 4) if n else 0.0,
        }
    report["totals"] = {
        "n_kept": int(len(kept)),
        "n_flagged": int(kept["any_flag"].sum()),
        "estimated_fp_rate_overall": round(kept["any_flag"].mean(), 4),
    }

    # Per-tier sample for CSV (30 T2, 30 T3, 10 T4, all T1) — biased toward flagged
    samples = []
    rng = pd.Series(range(len(kept))).sample(frac=1.0, random_state=RNG_SEED).index
    for tier, n_sample in [(1, 30), (2, 30), (3, 30), (4, 10)]:
        sub = kept[kept["match_tier"] == tier]
        if sub.empty:
            continue
        # Bias: prefer flagged rows so audit catches issues
        flagged = sub[sub.get("any_flag", False)]
        clean = sub[~sub.get("any_flag", False)]
        n_flag = min(len(flagged), n_sample // 2)
        n_clean = min(len(clean), n_sample - n_flag)
        picks = pd.concat([
            flagged.sample(n=n_flag, random_state=RNG_SEED) if n_flag else flagged.iloc[0:0],
            clean.sample(n=n_clean, random_state=RNG_SEED) if n_clean else clean.iloc[0:0],
        ])
        samples.append(picks)
    audit_sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    cols_out = [
        "match_tier", "similarity_score", "last_score", "first_score",
        "leie_last_norm", "leie_first_norm", "leie_specialty", "leie_state_norm",
        "mart_last_norm", "mart_first_norm", "mart_specialty", "mart_state_norm",
        "leie_busname_norm",
        "leie_row_n_candidates", "provider_n_leie_rows",
        "cross_gender", "both_low", "common_last", "last_too_short", "any_flag",
        "leie_excldate", "leie_excltype", "npi",
    ]
    cols_present = [c for c in cols_out if c in audit_sample.columns]
    audit_sample[cols_present].to_csv(f"{PROCESSED}/leie_audit_sample.csv", index=False)

    # Decision: if any tier > 15% flagged FP, recommend tightening
    decisions = []
    for tier_id, stats in report["per_tier"].items():
        if tier_id in (3, 4) and stats["estimated_fp_rate"] > 0.15:
            decisions.append(
                f"Tier {tier_id} estimated FP rate {stats['estimated_fp_rate']:.1%} "
                f"exceeds 15% — recommend tightening."
            )
    if not decisions:
        decisions.append("All tiers within tolerance; no tightening recommended.")
    report["decisions"] = decisions

    with open(f"{PROCESSED}/leie_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Also dump a CSV of all flagged rows (for any tier) for easy inspection
    flagged_only = kept[kept["any_flag"]]
    flagged_only[cols_present].to_csv(f"{PROCESSED}/leie_audit_flagged.csv", index=False)

    # Console print
    print("=== LEIE Audit Report ===")
    print(json.dumps(report, indent=2))
    print(f"\nWrote audit sample: {PROCESSED}/leie_audit_sample.csv ({len(audit_sample)} rows)")
    print(f"Wrote flagged rows:  {PROCESSED}/leie_audit_flagged.csv ({len(flagged_only)} rows)")


if __name__ == "__main__":
    audit()
