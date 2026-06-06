# LEIE Labeling Methodology

## Problem

The original ingestion pipeline matched OIG LEIE exclusion records to Part B
providers via NPI alone. LEIE has ~83K total rows but only ~8.5K carry a real
NPI (the rest carry the sentinel `0000000000`), and most LEIE-excluded
individuals never bill Medicare Part B in our 6-state window. This produced
**26 labeled positives across 150K providers** — far too few for a supervised
classifier to be evaluated meaningfully (~1:5,800 imbalance with a 6-positive
test set, where any metric is dominated by sampling noise).

## Approach: tiered fuzzy matching

`src/cms_fwa/ingestion/leie_matcher.py` implements a four-tier match strategy.
Tiers are applied in priority order; a provider matched at an earlier tier is
excluded from later tiers.

| Tier | Logic | Confidence | Final count |
|------|-------|-----------|-------------|
| T1 | LEIE.NPI == Mart.NPI (real NPIs only, sentinel excluded) | ~100% | 26 |
| T2 | (last, first, state) exact match after normalization | ~95–100% | 213 |
| T3 | Fuzzy `token_set_ratio ≥ 92` on full name + state match, plus per-component floors (last ≥ 88, first ≥ 85), first-letter match on first name, and cross-gender hard-drop list | ~80–85% | 47 |
| T4 | Business-name fuzzy match | n/a — disabled | 0 |

**Total: 286 unique labeled providers (11× the original 26).**

## Pre-filters

Before fuzzy matching, LEIE rows are split into individual and business
streams. The individual stream additionally drops LEIE rows whose SPECIALTY
field is a non-Part-B role (NURSE/NURSES AIDE, OWNER/OPERATOR,
PHARMACY TECHNICIAN, HEALTH CARE AIDE, etc. — see
`NON_PARTB_SPECIALTIES` in the matcher). This drops 7.2K nurses' aides and
similar roles from the matching pool, reducing FP risk in fuzzy matching.

## Name normalization

- Uppercase
- Strip punctuation except apostrophes and hyphens (preserved internally)
- Drop trailing suffix tokens (`JR`, `SR`, `II`, `III`, `IV`, `V`, `MD`, `DO`,
  `DDS`, `PHD`, `RN`, `LPN`, `NP`, `PA`, `MBBS`, `DPM`, `OD`)
- Collapse whitespace

## Tier-3 guard rails

After early experimentation showed `token_set_ratio` was vulnerable to
pathological short-name matches (`LE/LE`, `DAVID/DAVID`, `JAMES/JAMES`) and
gendered first-name pairs (`DANIEL/DANIELLE`), the following additional rules
were layered in:

1. Skip provider rows where `last_name == first_name` (NPPES data-quality
   artifacts).
2. Require per-component ratios: `fuzz.ratio(last, last) ≥ 88` AND
   `fuzz.ratio(first, first) ≥ 85`.
3. Hard-drop pairs in a curated cross-gender list
   (`DANIEL/DANIELLE`, `PAUL/PAULA`, `ROBERT/ROBERTA`, `PATRICK/PATRICIA`, etc.).
4. Require same first letter on first name (catches `DANH/ANH`,
   `SANDRA/CASANDRA` while preserving legitimate Anglo/Spanish/transliteration
   variants like `JOE/JOSE`, `LOUIS/LUIS`, `RICHARD/RICARDO`).

## Tier-4 status

Tier 4 (business-name fuzzy matching) yielded only 4 candidates with high FP
rate (e.g., `MED AMERICA DIAGNOSTICS INC` vs `TX DIAGNOSTICS INC`). It is
disabled until the modeling work justifies the complexity. Drop is in
`run_match()`; re-enable by removing the `t4 = pd.DataFrame()` line.

## Conflict resolution

Two ambiguity counts are tracked on every candidate:
- `leie_row_n_candidates` — providers a single LEIE row matched to
- `provider_n_leie_rows` — distinct LEIE rows that matched a single provider

Rules:
- T1/T2 candidates: always kept (exact-key match, no ambiguity).
- T3: drop if `leie_row_n_candidates > 3`.
- T4: drop if `leie_row_n_candidates > 2` (n/a — T4 disabled).
- Final per-NPI dedup: when one provider matches multiple LEIE rows, sort by
  (excldate asc, tier asc) and keep the earliest record. 5 providers were
  collapsed this way; the earliest exclusion date is used downstream.

## Audit & estimated error rate

A programmatic audit (`src/cms_fwa/ingestion/leie_audit.py`) flags suspect
matches via four heuristics: `cross_gender`, `both_low` (both per-component
ratios < 90), `common_last` (Smith/Jones/Garcia/... combined with any
first-name mismatch), and `last_too_short` (≤ 3 chars).

Audit output:

| Tier | n | Flagged | Estimated FP (heuristic) |
|------|---|---------|--------------------------|
| T1 | 26 | 0 | 0% |
| T2 | 213 | 0 | 0% |
| T3 | 47 | 14 | 30% |
| **Overall** | **286** | **14** | **~5%** |

The 30% T3 flag rate is **conservative** — the `common_last` heuristic
over-flags legitimate Spanish/English/Anglo spelling variants
(`RAMIREZ RICHARD/RICARDO`, `GARCIA LOUIS/LUIS`,
`RODRIGUEZ CHRISTINE/CHRISTINA`, `WHITE JEFFREY/JEFFERY`). Manual eyeball
review of the 14 flagged rows suggests **actual T3 FP rate of ~15–20%**
(≈ 7–10 false positives among 47 T3 matches). Overall actual FP rate across
all 286 labels is therefore **~3%**.

Audit artifacts:
- `data/processed/leie_audit_sample.csv` — stratified sample (≥ 30 per tier),
  biased toward flagged rows, for optional manual review.
- `data/processed/leie_audit_flagged.csv` — every flagged row in full.
- `data/processed/leie_audit_report.json` — machine-readable summary.

## Known limitations

- LEIE STATE is exclusion-record state, not necessarily current practice
  state. Tier-3 requires state match, which is conservative; some legitimate
  matches across state moves are missed.
- LEIE SPECIALTY is role at time of exclusion, often non-clinical
  (billing/payee/admin roles). Specialty mismatch is not used as a kill signal
  because real Part B providers can have been excluded for non-clinical work.
- Time-window mismatch: a provider excluded *after* the Part B billing year is
  still labeled positive. This is intentional — pre-exclusion billing patterns
  are exactly what the model should detect.
- The audit heuristic is conservative; absolute FP rates above are estimates,
  not guarantees.

## Reproducibility

```bash
python -m cms_fwa.ingestion.leie_matcher  # writes parquet to data/processed/
python -m cms_fwa.ingestion.leie_audit    # writes audit CSVs + report JSON
```

Outputs are deterministic given the input DuckDB state and rapidfuzz version
pinned in `pyproject.toml`.
