"""LEIE-to-provider matcher with tiered match strategy.

Tier 1: NPI exact (LEIE NPI != sentinel '0000000000')
Tier 2: (last, first, state) exact after normalization
Tier 3: (last + first) fuzzy (rapidfuzz token_set_ratio >= threshold) AND state exact
Tier 4: business-name fuzzy for organization providers (entity_type='2')

Outputs a candidate-match DataFrame; callers audit and materialize the final label
set downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process

SENTINEL_NPI = "0000000000"

# First-name pairs that look fuzzy-similar but are almost certainly different
# people (gendered counterparts). Hard-dropped from T3.
CROSS_GENDER_PAIRS = {
    frozenset({"DANIEL", "DANIELLE"}),
    frozenset({"PAUL", "PAULA"}),
    frozenset({"ROBERT", "ROBERTA"}),
    frozenset({"PATRICK", "PATRICIA"}),
    frozenset({"MARTIN", "MARTINA"}),
    frozenset({"JOSEPH", "JOSEPHINE"}),
    frozenset({"CARL", "CARLA"}),
    frozenset({"FRANCIS", "FRANCES"}),
    frozenset({"GEORGE", "GEORGIA"}),
    frozenset({"DENNIS", "DENISE"}),
    frozenset({"GLENN", "GLENNA"}),
    frozenset({"WILLIAM", "WILMA"}),
}

# LEIE SPECIALTY values that almost never bill Medicare Part B.
# Pre-filter drops these to reduce false-positive risk in fuzzy matching.
NON_PARTB_SPECIALTIES = {
    "NURSE/NURSES AIDE",
    "NURSING PROFESSION",
    "HEALTH CARE AIDE",
    "PERSONAL CARE PROVID",
    "PERSONAL CARE PROVIDER",
    "OWNER/OPERATOR",
    "BUSINESS MANAGER",
    "RECRUITER/CAPPER",
    "TECHNICIAN",
    "PHARMACY TECHNICIAN",
    "EMPLOYEE",
    "CLERICAL",
    "DRIVER",
    "BILLING SERVICE",
    "TRANSPORTATION",
    "HOME HEALTH AGENCY",  # org-level; dropped from individual stream
    "DME - GENERAL",
    "DME - OXYGEN",
    "DME COMPANY",
    "PHARMACY",
    "CLINIC",
    "OTHER BUSINESS",
}

# Suffix tokens to strip from last names
_SUFFIX_TOKENS = {"JR", "SR", "II", "III", "IV", "V", "MD", "DO", "DDS", "PHD", "RN",
                  "LPN", "NP", "PA", "MBBS", "DPM", "OD"}


def normalize_name(s) -> str:
    """Uppercase, strip punctuation except apostrophes/hyphens, drop suffix tokens,
    collapse whitespace. Accepts None, NaN, or strings."""
    if s is None or (isinstance(s, float)) or not isinstance(s, str):
        return ""
    s = s.upper()
    # Replace punctuation with space, preserve apostrophes and hyphens
    s = re.sub(r"[^\w'\- ]", " ", s)
    # Tokenize, drop suffix tokens from end
    tokens = s.split()
    while tokens and tokens[-1].strip(".") in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens).strip()


def normalize_state(s) -> str:
    if s is None or not isinstance(s, str):
        return ""
    return s.strip().upper()[:2]


@dataclass
class MatchConfig:
    fuzzy_threshold_t3: int = 92
    fuzzy_threshold_t3_last: int = 88   # per-component floor for last name
    fuzzy_threshold_t3_first: int = 85  # per-component floor for first name
    fuzzy_threshold_t4: int = 90
    states: tuple[str, ...] = ("TX", "AZ", "CO", "OK", "NV", "NM")


def load_leie(con: duckdb.DuckDBPyConnection, states: tuple[str, ...]) -> pd.DataFrame:
    placeholders = ", ".join(f"'{s}'" for s in states)
    q = f"""
        select
            LASTNAME    as leie_last,
            FIRSTNAME   as leie_first,
            MIDNAME     as leie_mid,
            BUSNAME     as leie_busname,
            SPECIALTY   as leie_specialty,
            GENERAL     as leie_general,
            NPI         as leie_npi,
            STATE       as leie_state,
            EXCLTYPE    as leie_excltype,
            EXCLDATE    as leie_excldate
        from raw.leie_exclusions
        where STATE in ({placeholders})
    """
    df = con.execute(q).df()
    df["leie_last_norm"] = df["leie_last"].map(normalize_name)
    df["leie_first_norm"] = df["leie_first"].map(normalize_name)
    df["leie_busname_norm"] = df["leie_busname"].map(normalize_name)
    df["leie_state_norm"] = df["leie_state"].map(normalize_state)
    return df


def load_providers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = """
        select
            npi,
            provider_last_name as mart_last,
            provider_first_name as mart_first,
            provider_specialty as mart_specialty,
            provider_state as mart_state,
            entity_type as mart_entity_type
        from main_marts.mart_provider_features
    """
    df = con.execute(q).df()
    df["mart_last_norm"] = df["mart_last"].map(normalize_name)
    df["mart_first_norm"] = df["mart_first"].map(normalize_name)
    df["mart_state_norm"] = df["mart_state"].map(normalize_state)
    return df


def split_leie_streams(leie: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (individuals, businesses) after specialty pre-filter."""
    sp_upper = leie["leie_specialty"].fillna("").str.upper().str.strip()
    drop_mask = sp_upper.isin(NON_PARTB_SPECIALTIES)

    indiv = leie[
        leie["leie_last_norm"].astype(bool)
        & leie["leie_first_norm"].astype(bool)
        & ~drop_mask
    ].copy()

    biz = leie[leie["leie_busname_norm"].astype(bool) & ~drop_mask].copy()

    return indiv, biz


def tier1_npi(leie: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    """Direct NPI match. Highest confidence."""
    mask = (leie["leie_npi"].notna()) & (leie["leie_npi"] != SENTINEL_NPI) & (leie["leie_npi"].str.strip() != "")
    npi_leie = leie[mask].copy()
    npi_leie["leie_npi"] = npi_leie["leie_npi"].astype(str).str.strip()
    providers = providers.copy()
    providers["npi"] = providers["npi"].astype(str)
    merged = npi_leie.merge(providers, left_on="leie_npi", right_on="npi", how="inner")
    merged["match_tier"] = 1
    merged["similarity_score"] = 100.0
    return merged


def tier2_exact(leie: pd.DataFrame, providers: pd.DataFrame, already_matched_npis: set) -> pd.DataFrame:
    """Exact (last, first, state) match after normalization."""
    sub_leie = leie[
        leie["leie_last_norm"].astype(bool)
        & leie["leie_first_norm"].astype(bool)
        & leie["leie_state_norm"].astype(bool)
    ]
    sub_prov = providers[
        providers["mart_last_norm"].astype(bool)
        & providers["mart_first_norm"].astype(bool)
        & providers["mart_state_norm"].astype(bool)
        & ~providers["npi"].astype(str).isin(already_matched_npis)
    ]
    merged = sub_leie.merge(
        sub_prov,
        left_on=["leie_last_norm", "leie_first_norm", "leie_state_norm"],
        right_on=["mart_last_norm", "mart_first_norm", "mart_state_norm"],
        how="inner",
    )
    merged["match_tier"] = 2
    merged["similarity_score"] = 100.0
    return merged


def tier3_fuzzy_individual(
    leie: pd.DataFrame,
    providers: pd.DataFrame,
    already_matched_npis: set,
    threshold: int,
    threshold_last: int = 88,
    threshold_first: int = 85,
) -> pd.DataFrame:
    """Fuzzy match on full name within same state.

    Requires three conditions:
      1. combined token_set_ratio(LAST FIRST, LAST FIRST) >= threshold
      2. ratio(last, last) >= threshold_last
      3. ratio(first, first) >= threshold_first

    Also skips provider rows where last == first (NPPES data-quality artifacts
    like 'DAVID DAVID' that otherwise pathologically attract matches).
    """
    out_rows: list[dict] = []
    leie = leie[
        leie["leie_last_norm"].astype(bool)
        & leie["leie_first_norm"].astype(bool)
        & leie["leie_state_norm"].astype(bool)
    ].copy()
    providers = providers[
        providers["mart_last_norm"].astype(bool)
        & providers["mart_first_norm"].astype(bool)
        & providers["mart_state_norm"].astype(bool)
        & ~providers["npi"].astype(str).isin(already_matched_npis)
        & (providers["mart_last_norm"] != providers["mart_first_norm"])  # drop DAVID/DAVID etc.
    ].copy()

    leie["leie_full"] = leie["leie_last_norm"] + " " + leie["leie_first_norm"]
    providers["mart_full"] = providers["mart_last_norm"] + " " + providers["mart_first_norm"]

    for state, prov_state in providers.groupby("mart_state_norm"):
        leie_state_df = leie[leie["leie_state_norm"] == state]
        if leie_state_df.empty:
            continue
        prov_names = prov_state["mart_full"].tolist()
        prov_records = prov_state.to_dict("records")
        prov_name_set = set(prov_names)

        for _, leie_row in leie_state_df.iterrows():
            target = leie_row["leie_full"]
            if target in prov_name_set:
                continue  # tier-2 territory
            matches = process.extract(
                target,
                prov_names,
                scorer=fuzz.token_set_ratio,
                limit=5,
                score_cutoff=threshold,
            )
            if not matches:
                continue
            for matched_name, score, idx in matches:
                pr = prov_records[idx]
                last_score = fuzz.ratio(leie_row["leie_last_norm"], pr["mart_last_norm"])
                first_score = fuzz.ratio(leie_row["leie_first_norm"], pr["mart_first_norm"])
                if last_score < threshold_last or first_score < threshold_first:
                    continue
                # Hard-drop cross-gender pairs
                if frozenset({leie_row["leie_first_norm"], pr["mart_first_norm"]}) in CROSS_GENDER_PAIRS:
                    continue
                # First-name first-letter must match (catches DANH/ANH, SANDRA/CASANDRA,
                # etc. while preserving legit variants like JOE/JOSE, LOUIS/LUIS)
                if leie_row["leie_first_norm"][:1] != pr["mart_first_norm"][:1]:
                    continue
                out_rows.append({
                    **leie_row.to_dict(),
                    **pr,
                    "match_tier": 3,
                    "similarity_score": float(score),
                    "last_score": float(last_score),
                    "first_score": float(first_score),
                })

    return pd.DataFrame(out_rows)


def tier4_business(
    leie_biz: pd.DataFrame,
    providers: pd.DataFrame,
    already_matched_npis: set,
    threshold: int,
) -> pd.DataFrame:
    """Fuzzy match LEIE business names against organization NPPES providers."""
    org_prov = providers[
        (providers["mart_entity_type"] == "O")
        & ~providers["npi"].astype(str).isin(already_matched_npis)
    ].copy()
    if org_prov.empty or leie_biz.empty:
        return pd.DataFrame()
    # Provider org name not in mart -- the mart uses last_name field for orgs. Use mart_last_norm as org name.
    org_prov["org_name_norm"] = org_prov["mart_last_norm"]
    org_prov = org_prov[org_prov["org_name_norm"].astype(bool)]

    out_rows: list[dict] = []
    for state, prov_state in org_prov.groupby("mart_state_norm"):
        leie_state_df = leie_biz[leie_biz["leie_state_norm"] == state]
        if leie_state_df.empty:
            continue
        names = prov_state["org_name_norm"].tolist()
        recs = prov_state.to_dict("records")
        for _, leie_row in leie_state_df.iterrows():
            target = leie_row["leie_busname_norm"]
            matches = process.extract(target, names, scorer=fuzz.token_set_ratio,
                                       limit=3, score_cutoff=threshold)
            for matched_name, score, idx in matches:
                pr = recs[idx]
                out_rows.append({**leie_row.to_dict(), **pr,
                                 "match_tier": 4, "similarity_score": float(score)})
    return pd.DataFrame(out_rows)


def run_match(con: duckdb.DuckDBPyConnection, cfg: MatchConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MatchConfig()

    leie_all = load_leie(con, cfg.states)
    providers = load_providers(con)

    leie_indiv, leie_biz = split_leie_streams(leie_all)

    # Tier 1 — NPI direct, on full LEIE (no specialty filter, NPI is high-confidence)
    t1 = tier1_npi(leie_all, providers)
    matched_npis: set = set(t1["npi"].astype(str).tolist()) if not t1.empty else set()

    # Tier 2 — exact name+state on individuals (post-filter)
    t2 = tier2_exact(leie_indiv, providers, matched_npis)
    matched_npis |= set(t2["npi"].astype(str).tolist())

    # Tier 3 — fuzzy on individuals
    t3 = tier3_fuzzy_individual(
        leie_indiv, providers, matched_npis,
        cfg.fuzzy_threshold_t3,
        cfg.fuzzy_threshold_t3_last,
        cfg.fuzzy_threshold_t3_first,
    )

    # Tier 4 — businesses: DISABLED after audit (only ~4 candidates with high FP rate;
    # not worth the methodological complexity at this label volume).
    t4 = pd.DataFrame()

    all_matches = pd.concat([t1, t2, t3, t4], ignore_index=True, sort=False)

    if all_matches.empty:
        return all_matches

    # Ambiguity bookkeeping
    all_matches["leie_row_key"] = (
        all_matches["leie_last_norm"].fillna("")
        + "|" + all_matches["leie_first_norm"].fillna("")
        + "|" + all_matches["leie_busname_norm"].fillna("")
        + "|" + all_matches["leie_state_norm"].fillna("")
        + "|" + all_matches["leie_excldate"].fillna("")
    )
    # How many providers does this LEIE row match? (ambiguity on LEIE side)
    all_matches["leie_row_n_candidates"] = (
        all_matches.groupby("leie_row_key")["npi"].transform("count")
    )
    # How many LEIE rows match this provider? (multi-exclusion on provider side, not ambiguity)
    all_matches["provider_n_leie_rows"] = (
        all_matches.groupby(all_matches["npi"].astype(str))["leie_row_key"].transform("nunique")
    )

    return all_matches


def resolve_conflicts(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply conflict-resolution rules and collapse to a final per-provider label set.

    Returns
    -------
    audited : pd.DataFrame
        All candidate rows with an added boolean column `keep_for_training`.
        Useful for audit/diagnostics.
    final_labels : pd.DataFrame
        One row per NPI, ready to materialize into the mart. Picks earliest
        exclusion date when a provider has multiple LEIE matches.
    """
    if matches.empty:
        return matches, matches

    audited = matches.copy()
    audited["npi"] = audited["npi"].astype(str)

    # Tier-aware ambiguity rules
    #   T1/T2: always keep (exact key match — no ambiguity to resolve)
    #   T3:   drop if a single LEIE row matched >3 providers (too ambiguous)
    #   T4:   drop if a single LEIE row matched >2 org providers (orgs are noisier)
    keep = pd.Series(True, index=audited.index)
    t3_mask = audited["match_tier"] == 3
    t4_mask = audited["match_tier"] == 4
    keep &= ~(t3_mask & (audited["leie_row_n_candidates"] > 3))
    keep &= ~(t4_mask & (audited["leie_row_n_candidates"] > 2))
    audited["keep_for_training"] = keep

    # Build final per-NPI label set from kept rows
    kept = audited[audited["keep_for_training"]].copy()
    if kept.empty:
        return audited, kept

    # Parse exclusion date (LEIE EXCLDATE is YYYYMMDD string)
    kept["excldate_int"] = pd.to_numeric(kept["leie_excldate"], errors="coerce")

    # For each NPI, pick the earliest-excluded LEIE row (most conservative — captures
    # earliest known exclusion). Ties broken by tier (lower tier = higher confidence).
    kept_sorted = kept.sort_values(
        ["npi", "excldate_int", "match_tier"], ascending=[True, True, True]
    )
    final_labels = kept_sorted.drop_duplicates(subset=["npi"], keep="first").copy()

    final_labels = final_labels[[
        "npi", "match_tier", "similarity_score", "leie_excldate", "leie_excltype",
        "leie_last_norm", "leie_first_norm", "leie_busname_norm",
        "leie_state_norm", "leie_specialty",
        "mart_last_norm", "mart_first_norm", "mart_specialty", "mart_state_norm",
        "leie_row_n_candidates", "provider_n_leie_rows",
    ]].rename(columns={
        "leie_excldate": "exclusion_date",
        "leie_excltype": "exclusion_type",
        "leie_state_norm": "leie_state",
        "leie_specialty": "leie_specialty",
        "mart_state_norm": "mart_state",
    })

    return audited, final_labels


def write_outputs(audited: pd.DataFrame, final_labels: pd.DataFrame, out_dir: str = "data/processed") -> None:
    """Persist matcher outputs for downstream audit and mart materialization."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    audited.to_parquet(f"{out_dir}/leie_match_candidates.parquet", index=False)
    final_labels.to_parquet(f"{out_dir}/leie_final_labels.parquet", index=False)


def materialize_to_duckdb(final_labels: pd.DataFrame, db_path: str = "data/cms_fwa.duckdb",
                          schema: str = "main", table: str = "provider_exclusion_labels") -> None:
    """Write the final label set into DuckDB so dbt can reference it as a source.

    Schema: one row per NPI with provenance fields.
    """
    con = duckdb.connect(db_path)
    try:
        con.execute(f"create schema if not exists {schema}")
        # Stable schema for dbt reference
        df = final_labels.copy()
        df["npi"] = df["npi"].astype(str)
        df["match_tier"] = df["match_tier"].astype("int32")
        df["similarity_score"] = df["similarity_score"].astype("float64")
        df["is_excluded"] = True  # this table contains only positive labels
        # Cast exclusion_date (YYYYMMDD str) to a proper date when possible
        df["exclusion_date_parsed"] = pd.to_datetime(
            df["exclusion_date"], format="%Y%m%d", errors="coerce"
        )
        cols = [
            "npi", "is_excluded", "match_tier", "similarity_score",
            "exclusion_date", "exclusion_date_parsed", "exclusion_type",
            "leie_last_norm", "leie_first_norm", "leie_state", "leie_specialty",
            "mart_specialty", "leie_row_n_candidates", "provider_n_leie_rows",
        ]
        df = df[[c for c in cols if c in df.columns]]
        con.register("labels_df", df)
        con.execute(f"drop table if exists {schema}.{table}")
        con.execute(f"create table {schema}.{table} as select * from labels_df")
        n = con.execute(f"select count(*) from {schema}.{table}").fetchone()[0]
        print(f"Materialized {n} rows -> {schema}.{table}")
    finally:
        con.close()


if __name__ == "__main__":
    con = duckdb.connect("data/cms_fwa.duckdb", read_only=True)
    matches = run_match(con)
    con.close()
    audited, final_labels = resolve_conflicts(matches)
    write_outputs(audited, final_labels)
    materialize_to_duckdb(final_labels)

    print(f"Total candidate matches: {len(matches)}")
    print("\nCandidates by tier:")
    print(matches["match_tier"].value_counts().sort_index().to_string())

    print(f"\nKept for training: {audited['keep_for_training'].sum()} / {len(audited)}")
    print("\nKept by tier:")
    print(audited[audited['keep_for_training']]['match_tier'].value_counts().sort_index().to_string())

    print(f"\nFinal unique providers (one row per NPI): {len(final_labels)}")
    print("\nFinal labels by tier:")
    print(final_labels['match_tier'].value_counts().sort_index().to_string())

    # Diagnostics
    n_dropped = (~audited['keep_for_training']).sum()
    print(f"\nDropped as too-ambiguous (leie_row_n_candidates > tier threshold): {n_dropped}")
    if n_dropped:
        print("Dropped row tier distribution:")
        print(audited[~audited['keep_for_training']]['match_tier'].value_counts().to_string())

    n_multi = (final_labels['provider_n_leie_rows'] > 1).sum()
    print(f"\nProviders matching >1 LEIE row (multi-exclusion collapsed to earliest): {n_multi}")

    print("\nFinal-labels tier-3 sample:")
    cols = ["npi", "leie_last_norm", "leie_first_norm", "leie_specialty",
            "mart_last_norm", "mart_first_norm", "mart_specialty",
            "similarity_score", "exclusion_date", "leie_row_n_candidates"]
    print(final_labels[final_labels["match_tier"] == 3][cols].head(15).to_string())
