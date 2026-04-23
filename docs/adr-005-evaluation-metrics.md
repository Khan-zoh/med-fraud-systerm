# ADR-005: PR-AUC and Precision@k over ROC-AUC and Accuracy

## Status
Accepted

## Context
Choosing the right evaluation metric is critical for FWA detection because the class distribution is extremely imbalanced (~1-2% excluded providers). The wrong metric creates a false sense of model quality.

## Decision
Use **PR-AUC** (Precision-Recall Area Under Curve) as the primary model metric and **Precision@k** as the operational metric. Do not report ROC-AUC or accuracy as primary metrics.

## Rationale

### Why not accuracy?
With 1-2% positive rate, a model that predicts "not fraud" for everyone achieves 98%+ accuracy. Accuracy is meaningless here.

### Why not ROC-AUC?
ROC-AUC measures the tradeoff between true positive rate and false positive rate. With massive class imbalance, the false positive rate stays low even with many false positives (because the denominator — true negatives — is huge). This inflates ROC-AUC.

Example: a model that flags 1,000 providers (500 true positives, 500 false positives) has:
- FPR = 500 / 49,000 = 1% (looks great on ROC)
- Precision = 500 / 1,000 = 50% (tells the real story)

### Why PR-AUC?
PR-AUC directly measures the precision-recall tradeoff. It penalizes models that generate many false positives, which is exactly what we care about — investigators have limited capacity.

### Why Precision@k?
The real-world constraint: an investigation team can review N cases per week. Precision@k answers: "Of the top k flagged providers, how many are truly problematic?" This is the metric that determines ROI of the system.

We report precision@k for k = 10, 25, 50, 100, 200, 500 to match different team capacities.

### Per-specialty breakdown
Model performance varies by specialty because billing patterns differ fundamentally. A cardiology-specific PR-AUC might be 0.6 while dermatology is 0.2. Reporting the aggregate alone hides these differences. Per-specialty metrics help investigators calibrate their trust in the scores.

## Consequences
- PR-AUC is harder to explain to non-technical stakeholders than accuracy
- Precision@k requires choosing k values (we provide a range)
- Per-specialty evaluation requires sufficient positive labels per specialty (we require >= 20 providers and >= 1 positive)
- We still compute and save ROC-AUC for completeness but don't feature it in reporting
