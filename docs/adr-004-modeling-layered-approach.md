# ADR-004: Layered ML Architecture (Unsupervised + Supervised + Graph)

## Status
Accepted

## Context
We need a modeling approach for FWA detection that handles:
- Severe class imbalance (~1-2% positive rate from LEIE labels)
- Label noise (LEIE exclusion != fraud; some fraud is never excluded)
- Novel fraud patterns not represented in historical exclusions
- Coordinated fraud rings across multiple providers
- Explainability requirements for government/regulatory use

## Decision
Use a **three-layer ensemble**:
1. **Isolation Forest** (unsupervised, 25% weight) — catches novel anomalies
2. **XGBoost** (supervised, 50% weight) — learns from LEIE exclusion patterns
3. **Graph analysis** (network, 25% weight) — detects coordinated schemes

Combined into a single risk score (0-100) with SHAP explanations.

## Rationale

### Why not a single model?
A single supervised classifier would only flag patterns similar to past LEIE exclusions. But:
- LEIE captures a fraction of actual FWA (label coverage gap)
- Novel fraud schemes won't match historical patterns
- Coordinated rings are invisible to per-provider features

### Why these specific models?

**Isolation Forest** over autoencoders:
- More interpretable (tree-based, SHAP-compatible)
- Faster on tabular data (~30 features)
- No architecture hyperparameter tuning needed

**XGBoost** over logistic regression or neural networks:
- Handles mixed feature types and missing values natively
- `scale_pos_weight` handles class imbalance without SMOTE
- Feature importance is built-in
- State of the art for tabular classification

**Graph/NetworkX** over GNN (PyTorch Geometric):
- NetworkX features (PageRank, community detection) are sufficient at our scale (~50K providers)
- GNN would add complexity without proportional benefit at this data size
- NetworkX features are fully interpretable

### Why these weights (25/50/25)?
- XGBoost gets the highest weight because when labels are available, supervised learning is the strongest signal
- The weights are configurable and can be tuned by optimizing precision@k on a validation set
- When XGBoost can't train (no labels), weights automatically redistribute to 60/0/40

## Consequences
- Three models to maintain and monitor (vs. one)
- Graph model requires the HCPCS mix table (intermediate layer dependency)
- Ensemble weights are a hyperparameter that should be tuned per deployment
- SHAP explanations come from XGBoost only; IF and graph scores are reported directly
