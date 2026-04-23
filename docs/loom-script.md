# Loom Video Script — 3-Minute Project Walkthrough

Target audience: Senior data engineer or hiring manager reviewing your portfolio.
Goal: Show the system works end-to-end and highlight what makes it interesting.

---

## [0:00 - 0:20] Hook — The Problem

> "Medicare processes over a billion claims a year. Somewhere between 60 and 90 billion dollars of that is estimated to be fraud, waste, or abuse. This project builds a detection system that flags providers with anomalous billing patterns for investigator review."

**Screen:** Show the README architecture diagram (scroll slowly).

---

## [0:20 - 0:50] Data Pipeline

> "I ingest three public CMS data sources — Part B billing data, the OIG exclusion list, and the NPI registry. The pipeline is orchestrated with Dagster, and the data flows through a dbt transformation layer into DuckDB."

**Screen:** Show the dbt model lineage (terminal output of `dbt docs generate && dbt docs serve`, or just the project structure).

> "Each dbt model has schema tests — 34 checks across 8 models for things like null NPIs, accepted values for specialty codes, and unique provider records."

---

## [0:50 - 1:30] Feature Engineering — The IE Edge

> "This is where the industrial engineering background comes in. I'm not just computing generic statistics — each feature maps to an IE principle."

**Screen:** Open `mart_provider_features.sql` and scroll through the feature sections.

> "Statistical process control z-scores — same logic as a manufacturing control chart, but the 'process' is a provider's specialty group. Jensen-Shannon divergence measures how different a provider's procedure mix is from their peers — not just volume, but the shape of their billing distribution."

> "Billing velocity uses queueing theory. A solo practitioner billing 200 office visits in a day is exceeding theoretical throughput — physically impossible without something unusual going on."

> "And Mahalanobis distance catches providers who look normal on any single metric but are outliers across the combination."

---

## [1:30 - 2:10] Model Architecture

> "The model has three layers. First, an Isolation Forest for unsupervised anomaly detection — this catches novel patterns the labeled data doesn't cover. Second, XGBoost trained on OIG exclusion labels, handling 50-to-1 class imbalance with scale_pos_weight. Third, a graph model that builds a provider-procedure network and uses community detection to find coordinated billing patterns."

**Screen:** Show the ensemble diagram from the README.

> "These combine into a risk score from 0 to 100. Every score comes with SHAP explanations in plain English — required for government use."

---

## [2:10 - 2:40] Live Demo

> "Let me show the investigator dashboard."

**Screen:** Open Streamlit dashboard.

1. Show the Overview page (KPI cards, risk distribution chart)
2. Navigate to Provider Detail, search an NPI
3. Show the risk score, model layer scores, and billing profile vs. peers table
4. Point out the disclaimer at the bottom

> "Investigators can search by NPI, browse top-risk providers, and log their review decisions. Those decisions feed back into model improvement."

---

## [2:40 - 3:00] Close — Honest Caveats

> "Two important caveats I want to be transparent about. First, LEIE exclusion is not the same as fraud — it's a noisy label, and the writeup goes deep on that distinction. Second, this uses single-year, single-state public data. Real CMS systems use multi-year claims-level data with temporal validation."

> "The full code, architecture decision records, and a detailed writeup are in the GitHub repo. Thanks for watching."

**Screen:** Show the repo URL.

---

## Recording Tips

- Use a clean terminal with a dark theme and large font (16pt+)
- Have the Streamlit dashboard and terminal pre-loaded in split screen
- Practice once before recording — 3 minutes is tight
- Speak at a measured pace — you have exactly enough time if you don't rush
- Don't apologize for limitations — state them confidently as design awareness
