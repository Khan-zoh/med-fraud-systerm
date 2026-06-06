"""
Medicare FWA Detection — Investigator Dashboard (Streamlit Cloud build)

Self-contained version of the dashboard that ships to Streamlit Community Cloud.
Reads precomputed parquet/json artifacts; has no DuckDB / dbt / model dependency
at runtime.

Source repo: github.com/[YOUR_USERNAME]/medicare-fwa-dashboard
Project repo: github.com/[YOUR_USERNAME]/medicare-fwa-detection
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS = Path(__file__).parent / "artifacts"
FEEDBACK_FILE = Path(tempfile.gettempdir()) / "fwa_review_feedback.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medicare FWA Detection — Investigator Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_risk_table() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "risk_table.parquet")


@st.cache_data
def load_evaluation_metrics() -> dict:
    path = ARTIFACTS / "evaluation_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Risk tier helper (the original was computed during training; regenerate here
# from risk_score so the parquet stays slim)
# ---------------------------------------------------------------------------
def add_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    if "risk_tier" in df.columns:
        return df
    bins = [-1, 20, 40, 60, 80, 101]
    labels = ["Low", "Moderate", "Elevated", "High", "Critical"]
    df = df.copy()
    df["risk_tier"] = pd.cut(df["risk_score"], bins=bins, labels=labels, right=True)
    df["risk_tier"] = df["risk_tier"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.title("Filters")

    if st.sidebar.button("🔄 Reset all filters"):
        for k in ["tier_filter", "specialty_filter", "state_filter"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    tiers = ["All"] + sorted([t for t in df["risk_tier"].dropna().unique().tolist() if t != "nan"])
    selected_tier = st.sidebar.selectbox("Risk Tier", tiers, key="tier_filter")

    specialties = ["All"] + sorted(df["provider_specialty"].dropna().unique().tolist())
    selected_specialty = st.sidebar.selectbox("Specialty", specialties, key="specialty_filter")

    states = ["All"] + sorted(df["provider_state"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states, key="state_filter")

    min_score, max_score = st.sidebar.slider(
        "Risk Score Range", 0.0, 100.0, (0.0, 100.0), 1.0,
    )

    show_excluded = st.sidebar.checkbox("Show only OIG-excluded providers", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Note:** Risk scores identify anomalous billing patterns "
        "for review. They are not accusations of fraud."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Portfolio demo — synthetic review log resets when the Streamlit Cloud "
        "container restarts. [Source on GitHub ↗](https://github.com/)"
    )

    return {
        "tier": selected_tier,
        "specialty": selected_specialty,
        "state": selected_state,
        "min_score": min_score,
        "max_score": max_score,
        "show_excluded": show_excluded,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters["tier"] != "All":
        out = out[out["risk_tier"] == filters["tier"]]
    if filters["specialty"] != "All":
        out = out[out["provider_specialty"] == filters["specialty"]]
    if filters["state"] != "All":
        out = out[out["provider_state"] == filters["state"]]
    if filters["show_excluded"]:
        out = out[out["is_excluded"] == True]  # noqa: E712
    out = out[(out["risk_score"] >= filters["min_score"]) & (out["risk_score"] <= filters["max_score"])]
    return out


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_overview(df: pd.DataFrame, metrics: dict) -> None:
    st.title("Medicare FWA Detection Dashboard")
    st.markdown(
        "Investigator workflow for reviewing providers flagged for anomalous "
        "billing patterns. Built on 1.27M Part B claims across 150K providers "
        "from 3 federal data sources (CMS, OIG LEIE, NPPES)."
    )

    with st.expander("ℹ️ What am I looking at?", expanded=False):
        st.markdown("""
        **Risk Score (0-100):** Composite anomaly score combining three ML models.
        Higher = billing pattern is more unusual relative to the provider's specialty peers.

        **Risk Tiers:**
        - 🟢 **Low (0-20):** Normal billing. Don't review.
        - 🟡 **Moderate (20-40):** Slightly unusual. Review if capacity allows.
        - 🟠 **Elevated (40-60):** Worth investigating.
        - 🔴 **High (60-80):** Strong anomaly signal. Prioritize.
        - ⛔ **Critical (80-100):** Extreme outlier. Review immediately.

        **PR-AUC:** Precision-Recall Area Under Curve. Measures how well the model
        ranks excluded providers ahead of non-excluded ones. Higher is better.

        **Important:** A high risk score ≠ fraud. It means this provider's billing
        pattern is unusual enough to warrant human review.
        """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Providers (test set)", f"{len(df):,}")
    with col2:
        st.metric("Excluded (LEIE)", f"{int(df['is_excluded'].sum()):,}")
    with col3:
        high_risk = len(df[df["risk_score"] >= 60])
        pct = 100 * high_risk / len(df) if len(df) > 0 else 0
        st.metric("High/Critical Risk", f"{high_risk:,}", f"{pct:.1f}% of total")
    with col4:
        pr_auc = metrics.get("pr_auc") if metrics else None
        st.metric("Model PR-AUC", f"{pr_auc:.3f}" if pr_auc else "N/A")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk Tier Distribution")
        tier_counts = df["risk_tier"].value_counts()
        st.bar_chart(tier_counts)
    with right:
        st.subheader("Risk Score Distribution")
        if len(df) > 0:
            import altair as alt
            scores = df["risk_score"].clip(0, 100)
            counts, edges = np.histogram(scores, bins=20, range=(0, 100))
            hist_df = pd.DataFrame({
                "Risk Score": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))],
                "Providers": counts,
                "order": range(len(counts)),
            })
            chart = alt.Chart(hist_df).mark_bar().encode(
                x=alt.X("Risk Score:N", sort=alt.SortField("order"), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Providers:Q"),
                tooltip=["Risk Score", "Providers"],
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data to display.")

    st.markdown("---")
    st.subheader("Which Specialties Have the Highest Risk?")
    st.caption("Specialties with the most anomalous billing patterns on average. Useful for targeting investigative resources.")

    if len(df) > 0 and "provider_specialty" in df.columns:
        p95 = df["risk_score"].quantile(0.95)
        p99 = df["risk_score"].quantile(0.99)
        stats = (
            df.groupby("provider_specialty")
            .agg(
                providers=("npi", "count"),
                avg_risk=("risk_score", "mean"),
                max_risk=("risk_score", "max"),
                top_5pct=("risk_score", lambda x: (x >= p95).sum()),
                top_1pct=("risk_score", lambda x: (x >= p99).sum()),
                excluded=("is_excluded", "sum"),
            )
            .reset_index()
        )
        stats = stats[stats["providers"] >= 20].copy()
        stats["% in top 5%"] = (100 * stats["top_5pct"] / stats["providers"]).round(1)
        stats["avg_risk"] = stats["avg_risk"].round(1)
        stats["max_risk"] = stats["max_risk"].round(1)
        stats = stats.sort_values("avg_risk", ascending=False).head(15).rename(columns={
            "provider_specialty": "Specialty",
            "providers": "Providers",
            "avg_risk": "Avg Risk",
            "max_risk": "Max Risk",
            "top_5pct": "In Top 5%",
            "top_1pct": "In Top 1%",
            "excluded": "LEIE",
        })[["Specialty", "Providers", "Avg Risk", "Max Risk", "In Top 5%", "In Top 1%", "% in top 5%", "LEIE"]]
        st.caption(f"Top 5% cutoff = **{p95:.1f}** risk score. Top 1% cutoff = **{p99:.1f}**. Uses dataset-relative percentiles.")
        st.dataframe(stats, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for specialty breakdown.")

    st.markdown("---")
    st.subheader("Which Model Layer Is Flagging These Providers?")
    st.caption("How each of the 3 ML layers contributes to the top-100 risk scores. Validates the ensemble isn't dominated by one model.")

    top_100 = df.head(100)
    if len(top_100) > 0:
        import altair as alt
        layer_means = pd.DataFrame({
            "Model Layer": ["Isolation Forest", "XGBoost", "Graph"],
            "Avg Score": [
                top_100["if_anomaly_score"].mean() if "if_anomaly_score" in top_100 else 0,
                top_100["xgb_fraud_prob"].mean() if "xgb_fraud_prob" in top_100 else 0,
                top_100["graph_anomaly_score"].mean() if "graph_anomaly_score" in top_100 else 0,
            ],
            "Weight": [0.25, 0.50, 0.25],
        })
        layer_means["Contribution"] = (layer_means["Avg Score"] * layer_means["Weight"]).round(3)
        layer_means["Avg Score"] = layer_means["Avg Score"].round(3)

        a, b = st.columns([1, 2])
        with a:
            st.dataframe(layer_means, use_container_width=True, hide_index=True)
        with b:
            chart = alt.Chart(layer_means).mark_bar().encode(
                x=alt.X("Model Layer:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Contribution:Q", title="Contribution to Risk Score"),
                color=alt.Color("Model Layer:N", legend=None),
                tooltip=["Model Layer", "Avg Score", "Weight", "Contribution"],
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 20 Highest-Risk Providers")
    top_20 = df.head(20)[[
        "npi", "provider_last_name", "provider_first_name",
        "provider_specialty", "provider_state", "risk_score", "risk_tier", "is_excluded",
    ]].copy()
    top_20["risk_score"] = top_20["risk_score"].round(1)
    st.dataframe(top_20, use_container_width=True, hide_index=True)


def render_provider_detail(df: pd.DataFrame) -> None:
    st.title("Provider Detail")

    npi_input = st.text_input("Search by NPI", placeholder="Enter 10-digit NPI...", max_chars=10)

    if not npi_input:
        st.info("Enter an NPI above, or select a provider from the table below.")
        display_df = df.head(100)[["npi", "provider_last_name", "provider_specialty", "risk_score", "risk_tier"]].copy()
        display_df["risk_score"] = display_df["risk_score"].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        return

    mask = df["npi"].astype(str) == npi_input
    if not mask.any():
        st.error(f"NPI {npi_input} not found in the scored test set.")
        return

    provider = df[mask].iloc[0]

    first = provider.get("provider_first_name", "") or ""
    last = provider.get("provider_last_name", "") or ""
    name = f"{first} {last}".strip() or "Unknown"
    st.header(f"{name} (NPI: {provider['npi']})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{provider['risk_score']:.1f}/100")
    col2.metric("Risk Tier", str(provider["risk_tier"]))
    col3.metric("Specialty", str(provider.get("provider_specialty", "Unknown")))
    col4.metric("LEIE Excluded", "Yes" if provider.get("is_excluded") else "No")

    st.markdown("---")
    st.subheader("Why Was This Provider Flagged?")
    narrative = []
    rs = provider.get("risk_score", 0)

    zscore_cols = [c for c in provider.index if c.startswith("zscore_")]
    findings = []
    for col in zscore_cols:
        z = provider.get(col)
        if pd.notna(z) and abs(float(z)) >= 2.0:
            metric_name = col.replace("zscore_", "").replace("_", " ")
            findings.append((abs(float(z)), metric_name, float(z)))
    findings.sort(reverse=True)

    if rs >= 80:
        narrative.append(f"🔴 **Critical risk ({rs:.0f}/100).** This provider is an extreme outlier and warrants immediate review.")
    elif rs >= 60:
        narrative.append(f"🟠 **High risk ({rs:.0f}/100).** Strong anomaly signal — prioritize for investigation.")
    elif rs >= 40:
        narrative.append(f"🟡 **Elevated risk ({rs:.0f}/100).** Worth reviewing when capacity allows.")
    else:
        narrative.append(f"🟢 **Low-to-moderate risk ({rs:.0f}/100).** No strong anomaly signals detected.")

    if findings:
        narrative.append("**Key anomalies (vs. specialty peers):**")
        for _, metric, z in findings[:5]:
            direction = "above" if z > 0 else "below"
            narrative.append(f"- {metric.title()} is **{abs(z):.1f} standard deviations {direction}** the specialty mean.")

    throughput = provider.get("throughput_utilization")
    if throughput is not None and pd.notna(throughput) and float(throughput) > 1.0:
        narrative.append(f"- ⚠️ **Throughput utilization: {float(throughput)*100:.0f}%** — exceeds theoretical daily capacity for this specialty.")
    jsd = provider.get("jsd_vs_specialty")
    if jsd is not None and pd.notna(jsd) and float(jsd) > 0.7:
        narrative.append(f"- Procedure mix diverges sharply from specialty peers (JSD = {float(jsd):.2f}).")
    mahal_p = provider.get("mahalanobis_pvalue")
    if mahal_p is not None and pd.notna(mahal_p) and float(mahal_p) < 0.01:
        narrative.append(f"- Multivariate outlier (Mahalanobis p-value = {float(mahal_p):.4f}).")

    st.markdown("\n\n".join(narrative))

    st.markdown("---")
    st.subheader("Model Layer Scores")
    a, b, c = st.columns(3)
    a.metric("Isolation Forest (Unsupervised)", f"{provider.get('if_anomaly_score', 0):.3f}",
             help="Anomaly score from unsupervised model. Higher = more unusual.")
    b.metric("XGBoost (Supervised)", f"{provider.get('xgb_fraud_prob', 0):.3f}",
             help="Probability from supervised model trained on LEIE exclusions.")
    c.metric("Graph (Network)", f"{provider.get('graph_anomaly_score', 0):.3f}",
             help="Anomaly score from provider-procedure network analysis.")

    st.markdown("---")
    st.subheader("Billing Profile vs. Specialty Peers")
    rows = []
    pairs = [
        ("total_services", "Total Services"),
        ("total_beneficiaries", "Total Beneficiaries"),
        ("total_medicare_payments", "Total Medicare Payments ($)"),
        ("services_per_beneficiary", "Services per Beneficiary"),
        ("unique_hcpcs_codes", "Unique Procedure Codes"),
        ("wavg_submitted_charge", "Avg Submitted Charge ($)"),
        ("high_complexity_ratio", "High-Complexity Code Ratio"),
    ]
    for col, label in pairs:
        val = provider.get(col)
        z = provider.get(f"zscore_{col}")
        if val is not None and pd.notna(val):
            rows.append({
                "Metric": label,
                "Value": f"{float(val):,.2f}",
                "Z-Score vs Peers": f"{float(z):.2f}" if z is not None and pd.notna(z) else "—",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Peer-comparison features unavailable for this provider.")

    st.markdown("---")
    st.warning(
        "**Disclaimer:** This risk assessment identifies anomalous billing patterns "
        "for investigative review. It is not evidence of fraud, waste, or abuse. "
        "All flagged providers should be evaluated through standard review procedures."
    )


def render_feedback(df: pd.DataFrame) -> None:
    st.title("Review Decisions Log")
    st.markdown("Log review decisions here. This feedback can be used to improve the model in future iterations (active learning).")
    st.info(
        "**Portfolio demo note:** decisions are logged to ephemeral storage on "
        "Streamlit Cloud and reset when the container restarts. In production, "
        "this would write to a database table."
    )

    npi_input = st.text_input("NPI Reviewed", placeholder="Enter NPI...")
    decision = st.selectbox(
        "Decision",
        ["Select...", "Referred for Investigation", "No Action — Legitimate Pattern",
         "No Action — Insufficient Evidence", "Needs More Data"],
    )
    notes = st.text_area("Notes", placeholder="Optional notes about this review...")

    if st.button("Log Decision"):
        if not npi_input or decision == "Select...":
            st.error("Please enter an NPI and select a decision.")
        else:
            import csv
            row = {
                "timestamp": datetime.now().isoformat(),
                "npi": npi_input,
                "decision": decision,
                "notes": notes,
            }
            file_exists = FEEDBACK_FILE.exists()
            with open(FEEDBACK_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            st.success(f"Decision logged for NPI {npi_input}: {decision}")

    if FEEDBACK_FILE.exists():
        st.markdown("---")
        feedback_df = pd.read_csv(FEEDBACK_FILE)

        st.subheader("Investigator ROI")
        total = len(feedback_df)
        referred = (feedback_df["decision"] == "Referred for Investigation").sum()
        no_action = feedback_df["decision"].str.startswith("No Action").sum()
        precision = (100 * referred / total) if total > 0 else 0
        a, b, c, d = st.columns(4)
        a.metric("Total Reviewed", f"{total:,}")
        b.metric("Referred", f"{referred:,}")
        c.metric("No Action", f"{no_action:,}")
        d.metric("Referral Rate", f"{precision:.1f}%")
        avg_case_value = 75000
        est_savings = referred * avg_case_value
        st.caption(
            f"Assuming ~${avg_case_value:,} average recovery per referral: "
            f"estimated value of referrals = **${est_savings:,}**. "
            f"(Illustrative — real recovery depends on case outcomes.)"
        )

        st.markdown("---")
        st.subheader("Previous Decisions")
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    risk_table = load_risk_table()
    risk_table = add_risk_tier(risk_table)
    metrics = load_evaluation_metrics()

    filters = render_sidebar(risk_table)
    filtered_df = apply_filters(risk_table, filters)

    page = st.sidebar.radio("Navigation", ["Overview", "Provider Detail", "Review Log"])
    if page == "Overview":
        render_overview(filtered_df, metrics)
    elif page == "Provider Detail":
        render_provider_detail(filtered_df)
    elif page == "Review Log":
        render_feedback(filtered_df)


if __name__ == "__main__":
    main()
else:
    main()
