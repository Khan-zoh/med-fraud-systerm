# Detecting Anomalous Billing in Medicare: An Industrial Engineering Approach to FWA

## The Problem

Medicare processes over 1 billion claims annually. Within that volume, Fraud, Waste, and Abuse (FWA) costs the federal government an estimated $60-90 billion per year. The Centers for Medicare & Medicaid Services (CMS) and the Office of Inspector General (OIG) maintain programs to detect and investigate anomalous billing, but the scale of the data demands automated screening.

This project builds an end-to-end system that ingests real public CMS data, engineers features grounded in industrial engineering principles, trains a layered ML model, and serves risk scores through an API and investigator dashboard. It is a portfolio demonstration, not a production system, but every component is built to production standards.

## What Makes This Different: Industrial Engineering Meets Healthcare Analytics

Most FWA detection tutorials apply generic anomaly detection to billing data. This project applies **domain-specific IE principles** that give the features operational meaning:

### Statistical Process Control (SPC)
In manufacturing, SPC uses control charts to detect when a process drifts out of specification. We apply the same logic: each provider specialty is a "process," and each provider's billing volume is a measurement. A provider whose total services fall beyond 3 standard deviations from their specialty mean is "out of control" — statistically anomalous within their peer group.

This isn't just a z-score. It's a z-score *computed against the right baseline*. A cardiologist billing 10,000 services may be normal; a podiatrist billing 10,000 services is an extreme outlier. The specialty peer group is the control chart.

### Peer Benchmarking via Jensen-Shannon Divergence
Beyond volume, we ask: does this provider's *mix of procedures* look normal? A provider might bill a reasonable total volume but concentrate on high-reimbursement codes that peers rarely use. Jensen-Shannon divergence measures the distance between a provider's procedure distribution and the specialty median distribution. Unlike simple ratios, JSD captures the shape of the entire billing pattern.

### Queueing Theory / Throughput Analysis
Every service process has a theoretical maximum throughput. A solo office-based physician working 8 hours with 12-minute appointments can see at most ~40 patients per day, ~10,000 per year. Billing 20,000 office visits in a year is physically impossible for a single provider without unusual staffing. We compute each provider's implied daily throughput and flag those exceeding theoretical capacity — a direct application of queueing theory to billing data.

### Geographic Dispersion (Entropy)
Patient brokering schemes involve recruiting patients from wide geographic areas. We measure each provider's patient zip-code entropy: low entropy means a local practice (normal), high entropy means patients traveling from many different areas (warrants investigation for certain specialties). This borrows from facility location analysis in IE — the service area footprint should match the practice type.

### Multivariate Outlier Detection (Mahalanobis Distance)
A provider might pass every single-feature check but be anomalous across the *combination* of features. Mahalanobis distance is standard in multivariate SPC — it measures how far a data point is from its group centroid in a way that accounts for feature correlations. A provider with individually plausible volume, charges, and beneficiary count might still have an anomalous combination of all three.

## Data Sources and Their Limitations

All data is public and free:

| Source | What It Provides | Key Limitation |
|--------|-----------------|----------------|
| CMS Part B by Provider & Service | Billing volumes, charges, payments per provider per HCPCS code | Aggregated annually — no individual claim dates |
| OIG LEIE | Excluded providers (our labels) | Exclusion != fraud. Many exclusions are for license issues, not billing fraud. Many fraudulent providers were never excluded. |
| NPPES NPI Registry | Provider specialty, location | Self-reported taxonomy codes; not all providers maintain current records |

### The Label Problem
The most important caveat in this project: **LEIE exclusion is an imperfect proxy for fraud.**

LEIE exclusions cover offenses ranging from patient abuse to license revocation to billing fraud. Conversely, many providers who engage in FWA are never formally excluded — they may receive fines, voluntary refunds, or simply go undetected. Our supervised model learns from this noisy, incomplete label. It will:
- Miss fraud patterns that don't resemble past exclusions
- Potentially flag providers for non-fraud exclusion reasons

This is why the unsupervised layer (Isolation Forest) is essential — it catches anomalies regardless of labels. And it's why every output says "flagged for review" rather than "fraudulent."

## Model Architecture

We use a three-layer ensemble because no single model handles all the challenges:

```
Layer 1: Isolation Forest (25% weight)
  → Catches novel patterns without label dependency
  → 300 trees, 5% contamination prior

Layer 2: XGBoost (50% weight)
  → Learns from known exclusions
  → scale_pos_weight handles ~50:1 class imbalance
  → Early stopping on PR-AUC

Layer 3: Graph Analysis (25% weight)
  → Provider-procedure bipartite graph
  → PageRank + Louvain community detection
  → Detects coordinated billing patterns

Combined → Risk Score (0-100) + SHAP Explanations
```

### Why scale_pos_weight over SMOTE?
SMOTE generates synthetic minority samples by interpolating between existing positive examples. In our context, this means creating synthetic "excluded providers" that are blends of real exclusion cases. These synthetic providers may not represent realistic fraud patterns. `scale_pos_weight` instead adjusts the loss function to penalize false negatives more heavily — achieving the same effect without fabricating data.

### Why PR-AUC over ROC-AUC?
With ~1-2% positive rate, a model predicting all-negative achieves 98%+ accuracy and a misleadingly high ROC-AUC (because the massive number of true negatives keeps the false positive rate low). PR-AUC directly measures what investigators care about: when the model says "review this provider," how often is it right?

## Honest Caveats

1. **Single-year, single-state**: The dev pipeline processes Texas 2022 data. Real FWA detection uses multi-year, multi-state data to capture year-over-year changes and cross-state schemes.

2. **No temporal split**: With one year of data, we use stratified random splitting instead of the ideal train-on-year-T, test-on-year-T+1 approach. This means our evaluation likely overestimates real-world performance.

3. **Label coverage**: LEIE exclusions are a small, biased subset of actual FWA. The supervised model's recall ceiling is bounded by label quality.

4. **No claims-level features**: The public Part B data is aggregated to provider × HCPCS code level. Individual claim-level features (claim dates, beneficiary demographics, diagnosis codes) would significantly improve detection but aren't publicly available.

5. **Geographic entropy limitation**: We use provider zip code as a proxy for patient geography since beneficiary zip codes aren't in the public data. This is a weaker signal than true patient-origin analysis.

## False Positive Cost Analysis

Every false positive consumes investigator time. If a team can review 100 cases per week and our precision@100 is 0.15, then:
- 15 of those 100 cases lead to actionable findings
- 85 are investigated and closed with no action
- At an estimated $500 per investigation, that's $42,500/week in wasted effort on false positives

Improving precision@100 from 0.15 to 0.25 saves $5,000/week. This is the practical framing for model improvement — not abstract accuracy numbers, but investigator hours and dollars.

## Technical Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Orchestration | Dagster | Asset-based model fits data lineage; excellent local dev |
| Storage | DuckDB | Columnar analytics on a laptop; SQL dialect maps to Postgres |
| Transforms | dbt-duckdb | Version-controlled SQL with built-in testing; 34 data quality checks |
| ML | scikit-learn + XGBoost + NetworkX | Right tools for tabular data at this scale |
| Explainability | SHAP | Required for government/regulated use |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Dashboard | Streamlit | Rapid prototyping for investigator workflows |

Full architecture decision records are in the `docs/` directory.

## What I'd Do Differently in Production

1. **Multi-year temporal splits**: Train on 2020-2021, validate on 2022, deploy for 2023. Monitor for concept drift quarterly.
2. **Claims-level features**: Access to individual claims (via CMS DUA) would enable temporal billing pattern analysis, beneficiary-level features, and diagnosis-procedure consistency checks.
3. **Active learning loop**: The dashboard's "Review Log" captures investigator decisions. These should feed back into model retraining — the model improves as investigators confirm or dismiss flags.
4. **Monitoring and drift detection**: Track feature distributions and model scores over time. Alert when the score distribution shifts (may indicate a new fraud scheme or a data quality issue).
5. **Role-based access**: Investigators should only see providers in their assigned region/specialty. Audit logging for every score lookup.

## Resume Bullet Points

- **Built end-to-end Medicare FWA detection system** processing real CMS Part B billing data (100K+ provider records) through ingestion, transformation, ML modeling, and serving layers using Dagster, dbt, DuckDB, XGBoost, and FastAPI
- **Engineered 32 domain-specific features** grounded in industrial engineering principles (statistical process control z-scores, Jensen-Shannon divergence for peer benchmarking, queueing-theory throughput analysis, Mahalanobis multivariate outlier detection)
- **Designed layered ML architecture** combining unsupervised anomaly detection (Isolation Forest), supervised classification (XGBoost with class imbalance handling), and graph-based network analysis (bipartite community detection) into an ensemble risk score with SHAP explainability
- **Implemented investigator-facing tooling** including a FastAPI scoring API and Streamlit dashboard with NPI search, peer benchmark visualization, and review decision logging for active learning feedback
- **Applied analytics engineering best practices** with dbt transformation layer (8 models, 34 automated tests), comprehensive pytest suite (51 unit tests), Docker packaging, and GitHub Actions CI/CD
