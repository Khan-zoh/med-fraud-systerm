"""
Streamlit Investigator Dashboard

An interactive interface for FWA investigators to:
  1. Search providers by NPI
  2. Browse top-risk providers with filters
  3. Drill into individual provider explanations
  4. View peer benchmark comparisons visually
  5. Log review decisions to a feedback table

Run with: streamlit run src/cms_fwa/serving/dashboard.py
Or:       make dashboard
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from cms_fwa.config import settings


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CMS FWA Detection — Investigator Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_risk_table() -> pd.DataFrame | None:
    """Load the risk table from model artifacts."""
    path = settings.models_dir / "risk_table.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_evaluation_metrics() -> dict | None:
    """Load evaluation metrics."""
    path = settings.models_dir / "evaluation_metrics.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> dict:
    """Render sidebar filters and return filter state."""
    st.sidebar.title("Filters")

    # Reset button — clears all filters in one click
    if st.sidebar.button("🔄 Reset all filters"):
        for k in ["tier_filter", "specialty_filter", "state_filter"]:
            if k in st.session_state:
                del st.session_state[k] 
        st.rerun()

    # Risk tier filter
    tiers = ["All"] + sorted(df["risk_tier"].dropna().unique().tolist())
    selected_tier = st.sidebar.selectbox("Risk Tier", tiers, key="tier_filter")

    # Specialty filter
    specialties = ["All"] + sorted(df["provider_specialty"].dropna().unique().tolist())
    selected_specialty = st.sidebar.selectbox("Specialty", specialties, key="specialty_filter")

    # State filter
    states = ["All"] + sorted(df["provider_state"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states, key="state_filter")

    # Score range
    min_score, max_score = st.sidebar.slider(
        "Risk Score Range",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
    )

    # Excluded only toggle
    show_excluded = st.sidebar.checkbox("Show only excluded providers", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Note:** Risk scores identify anomalous billing patterns "
        "for review. They are not accusations of fraud."
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
    """Apply sidebar filters to the risk table."""
    filtered = df.copy()

    if filters["tier"] != "All":
        filtered = filtered[filtered["risk_tier"] == filters["tier"]]
    if filters["specialty"] != "All":
        filtered = filtered[filtered["provider_specialty"] == filters["specialty"]]
    if filters["state"] != "All":
        filtered = filtered[filtered["provider_state"] == filters["state"]]
    if filters["show_excluded"]:
        filtered = filtered[filtered["is_excluded"] == True]

    filtered = filtered[
        (filtered["risk_score"] >= filters["min_score"])
        & (filtered["risk_score"] <= filters["max_score"])
    ]

    return filtered


# ---------------------------------------------------------------------------
# Main pages
# ---------------------------------------------------------------------------
def render_overview(df: pd.DataFrame, metrics: dict | None) -> None:
    """Render the overview / summary page."""
    st.title("Medicare FWA Detection Dashboard")
    st.markdown("Investigator workflow for reviewing providers flagged for anomalous billing patterns.")

    # What you're looking at — explainer
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

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Providers", f"{len(df):,}")
    with col2:
        st.metric("Excluded (LEIE)", f"{df['is_excluded'].sum():,}")
    with col3:
        high_risk = len(df[df["risk_score"] >= 60])
        pct = 100 * high_risk / len(df) if len(df) > 0 else 0
        st.metric("High/Critical Risk", f"{high_risk:,}", f"{pct:.1f}% of total")
    with col4:
        if metrics and metrics.get("pr_auc"):
            st.metric("Model PR-AUC", f"{metrics['pr_auc']:.3f}")
        else:
            st.metric("Model PR-AUC", "N/A")

    st.markdown("---")

    # Risk distribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Tier Distribution")
        tier_counts = df["risk_tier"].value_counts()
        st.bar_chart(tier_counts)

    with col_right:
        st.subheader("Risk Score Distribution")
        if len(df) > 0:
            import numpy as np
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

    # Specialty-level insights
    st.subheader("Which Specialties Have the Highest Risk?")
    st.caption("Specialties with the most anomalous billing patterns on average. Useful for targeting investigative resources.")

    if len(df) > 0 and "provider_specialty" in df.columns:
        # Use top-5% threshold (dataset-relative) instead of fixed 60/80 cutoffs
        # This stays meaningful regardless of how the scores are distributed
        p95_threshold = df["risk_score"].quantile(0.95)
        p99_threshold = df["risk_score"].quantile(0.99)

        specialty_stats = (
            df.groupby("provider_specialty")
            .agg(
                providers=("npi", "count"),
                avg_risk=("risk_score", "mean"),
                max_risk=("risk_score", "max"),
                top_5pct=("risk_score", lambda x: (x >= p95_threshold).sum()),
                top_1pct=("risk_score", lambda x: (x >= p99_threshold).sum()),
                excluded=("is_excluded", "sum"),
            )
            .reset_index()
        )
        specialty_stats = specialty_stats[specialty_stats["providers"] >= 20]
        specialty_stats["% in top 5%"] = (
            100 * specialty_stats["top_5pct"] / specialty_stats["providers"]
        ).round(1)
        specialty_stats["avg_risk"] = specialty_stats["avg_risk"].round(1)
        specialty_stats["max_risk"] = specialty_stats["max_risk"].round(1)
        specialty_stats = specialty_stats.sort_values("avg_risk", ascending=False).head(15)
        specialty_stats = specialty_stats.rename(columns={
            "provider_specialty": "Specialty",
            "providers": "Providers",
            "avg_risk": "Avg Risk",
            "max_risk": "Max Risk",
            "top_5pct": "In Top 5%",
            "top_1pct": "In Top 1%",
            "excluded": "LEIE",
        })
        specialty_stats = specialty_stats[[
            "Specialty", "Providers", "Avg Risk", "Max Risk",
            "In Top 5%", "In Top 1%", "% in top 5%", "LEIE"
        ]]
        st.caption(
            f"Top 5% cutoff = **{p95_threshold:.1f}** risk score. "
            f"Top 1% cutoff = **{p99_threshold:.1f}**. "
            "Uses dataset-relative percentiles so these stay meaningful."
        )
        st.dataframe(specialty_stats, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for specialty breakdown.")

    st.markdown("---")

    # Model layer contribution for high-risk providers
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

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(layer_means, use_container_width=True, hide_index=True)
        with col_b:
            chart = alt.Chart(layer_means).mark_bar().encode(
                x=alt.X("Model Layer:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Contribution:Q", title="Contribution to Risk Score"),
                color=alt.Color("Model Layer:N", legend=None),
                tooltip=["Model Layer", "Avg Score", "Weight", "Contribution"],
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # Top risk table
    st.subheader("Top 20 Highest Risk Providers")
    top_20 = df.head(20)[
        ["npi", "provider_last_name", "provider_first_name",
         "provider_specialty", "provider_state", "risk_score",
         "risk_tier", "is_excluded"]
    ].copy()
    top_20["risk_score"] = top_20["risk_score"].round(1)
    st.dataframe(top_20, use_container_width=True, hide_index=True)


def render_provider_detail(df: pd.DataFrame) -> None:
    """Render the provider drill-down page."""
    st.title("Provider Detail")

    # NPI search
    npi_input = st.text_input(
        "Search by NPI",
        placeholder="Enter 10-digit NPI...",
        max_chars=10,
    )

    if not npi_input:
        st.info("Enter an NPI above, or select a provider from the table below.")

        # Show a selectable table
        display_df = df.head(100)[
            ["npi", "provider_last_name", "provider_specialty",
             "risk_score", "risk_tier"]
        ].copy()
        display_df["risk_score"] = display_df["risk_score"].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        return

    # Find provider
    mask = df["npi"].astype(str) == npi_input
    if not mask.any():
        st.error(f"NPI {npi_input} not found in the scored dataset.")
        return

    provider = df[mask].iloc[0]

    # Header
    first = provider.get("provider_first_name", "") or ""
    last = provider.get("provider_last_name", "") or ""
    name = f"{first} {last}".strip() or "Unknown"

    st.header(f"{name} (NPI: {provider['npi']})")

    # Risk summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Score", f"{provider['risk_score']:.1f}/100")
    with col2:
        st.metric("Risk Tier", str(provider["risk_tier"]))
    with col3:
        st.metric("Specialty", str(provider.get("provider_specialty", "Unknown")))
    with col4:
        excluded_label = "Yes" if provider.get("is_excluded") else "No"
        st.metric("LEIE Excluded", excluded_label)

    st.markdown("---")

    # Plain-English narrative
    st.subheader("Why Was This Provider Flagged?")
    narrative_parts = []
    risk_score_val = provider.get("risk_score", 0)

    # Identify the most extreme z-scores
    zscore_cols = [c for c in provider.index if c.startswith("zscore_")]
    zscore_findings = []
    for col in zscore_cols:
        z = provider.get(col)
        if pd.notna(z) and abs(float(z)) >= 2.0:
            metric_name = col.replace("zscore_", "").replace("_", " ")
            zscore_findings.append((abs(float(z)), metric_name, float(z)))
    zscore_findings.sort(reverse=True)

    if risk_score_val >= 80:
        narrative_parts.append(f"🔴 **Critical risk ({risk_score_val:.0f}/100).** This provider is an extreme outlier and warrants immediate review.")
    elif risk_score_val >= 60:
        narrative_parts.append(f"🟠 **High risk ({risk_score_val:.0f}/100).** Strong anomaly signal — prioritize for investigation.")
    elif risk_score_val >= 40:
        narrative_parts.append(f"🟡 **Elevated risk ({risk_score_val:.0f}/100).** Worth reviewing when capacity allows.")
    else:
        narrative_parts.append(f"🟢 **Low-to-moderate risk ({risk_score_val:.0f}/100).** No strong anomaly signals detected.")

    if zscore_findings:
        narrative_parts.append("**Key anomalies (vs. specialty peers):**")
        for _, metric, z in zscore_findings[:5]:
            direction = "above" if z > 0 else "below"
            narrative_parts.append(f"- {metric.title()} is **{abs(z):.1f} standard deviations {direction}** the specialty mean.")

    throughput = provider.get("throughput_utilization")
    if throughput is not None and pd.notna(throughput) and float(throughput) > 1.0:
        narrative_parts.append(f"- ⚠️ **Throughput utilization: {float(throughput)*100:.0f}%** — exceeds theoretical daily capacity for this specialty.")

    jsd = provider.get("jsd_vs_specialty")
    if jsd is not None and pd.notna(jsd) and float(jsd) > 0.7:
        narrative_parts.append(f"- Procedure mix diverges sharply from specialty peers (JSD = {float(jsd):.2f}). Billing pattern is unusual for this specialty.")

    mahal_p = provider.get("mahalanobis_pvalue")
    if mahal_p is not None and pd.notna(mahal_p) and float(mahal_p) < 0.01:
        narrative_parts.append(f"- Multivariate outlier: the *combination* of this provider's features is highly unusual (Mahalanobis p-value = {float(mahal_p):.4f}).")

    st.markdown("\n\n".join(narrative_parts))

    st.markdown("---")

    # Model layer scores
    st.subheader("Model Layer Scores")
    layer_col1, layer_col2, layer_col3 = st.columns(3)
    with layer_col1:
        score = provider.get("if_anomaly_score", 0)
        st.metric(
            "Isolation Forest (Unsupervised)",
            f"{score:.3f}",
            help="Anomaly score from unsupervised model. Higher = more unusual.",
        )
    with layer_col2:
        score = provider.get("xgb_fraud_prob", 0)
        st.metric(
            "XGBoost (Supervised)",
            f"{score:.3f}",
            help="Probability from supervised model trained on LEIE exclusions.",
        )
    with layer_col3:
        score = provider.get("graph_anomaly_score", 0)
        st.metric(
            "Graph (Network)",
            f"{score:.3f}",
            help="Anomaly score from provider-procedure network analysis.",
        )

    st.markdown("---")

    # Key metrics (peer comparison)
    st.subheader("Billing Profile vs. Specialty Peers")

    metrics_data = []
    metric_pairs = [
        ("total_services", "Total Services"),
        ("total_beneficiaries", "Total Beneficiaries"),
        ("total_medicare_payments", "Total Medicare Payments ($)"),
        ("services_per_beneficiary", "Services per Beneficiary"),
        ("unique_hcpcs_codes", "Unique Procedure Codes"),
        ("wavg_submitted_charge", "Avg Submitted Charge ($)"),
        ("high_complexity_ratio", "High-Complexity Code Ratio"),
    ]

    for col_name, display_name in metric_pairs:
        val = provider.get(col_name)
        zscore_col = f"zscore_{col_name}" if f"zscore_{col_name}" in provider.index else None
        zscore = provider.get(zscore_col) if zscore_col else None

        if val is not None:
            metrics_data.append({
                "Metric": display_name,
                "Value": f"{float(val):,.2f}" if val else "N/A",
                "Z-Score vs Peers": f"{float(zscore):.2f}" if zscore and pd.notna(zscore) else "—",
            })

    if metrics_data:
        st.dataframe(
            pd.DataFrame(metrics_data),
            use_container_width=True,
            hide_index=True,
        )

    # Disclaimer
    st.markdown("---")
    st.warning(
        "**Disclaimer:** This risk assessment identifies anomalous billing patterns "
        "for investigative review. It is not evidence of fraud, waste, or abuse. "
        "All flagged providers should be evaluated through standard review procedures."
    )


def render_feedback(df: pd.DataFrame) -> None:
    """Render the review feedback logging page."""
    st.title("Review Decisions Log")
    st.markdown(
        "Log your review decisions here. This feedback can be used to "
        "improve the model in future iterations (active learning)."
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
            # In production, this would write to a database table.
            # For the portfolio demo, we log to a CSV.
            feedback_dir = settings.processed_data_dir
            feedback_dir.mkdir(parents=True, exist_ok=True)
            feedback_file = feedback_dir / "review_feedback.csv"

            import csv
            from datetime import datetime

            row = {
                "timestamp": datetime.now().isoformat(),
                "npi": npi_input,
                "decision": decision,
                "notes": notes,
            }

            file_exists = feedback_file.exists()
            with open(feedback_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            st.success(f"Decision logged for NPI {npi_input}: {decision}")

    # Show existing feedback + ROI metrics
    feedback_file = settings.processed_data_dir / "review_feedback.csv"
    if feedback_file.exists():
        st.markdown("---")
        feedback_df = pd.read_csv(feedback_file)

        # ROI summary
        st.subheader("Investigator ROI")
        total = len(feedback_df)
        referred = (feedback_df["decision"] == "Referred for Investigation").sum()
        no_action = feedback_df["decision"].str.startswith("No Action").sum()
        precision = (100 * referred / total) if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviewed", f"{total:,}")
        col2.metric("Referred", f"{referred:,}")
        col3.metric("No Action", f"{no_action:,}")
        col4.metric("Referral Rate", f"{precision:.1f}%")

        # Estimated savings (crude proxy)
        avg_case_value = 75000  # rough average Medicare fraud referral amount
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
# App entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main Streamlit app."""
    risk_table = load_risk_table()

    if risk_table is None:
        st.error(
            "No model artifacts found. Run `make train` to generate the risk table, "
            "then restart the dashboard."
        )
        st.stop()

    metrics = load_evaluation_metrics()

    # Apply filters
    filters = render_sidebar(risk_table)
    filtered_df = apply_filters(risk_table, filters)

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Provider Detail", "Review Log"],
    )

    if page == "Overview":
        render_overview(filtered_df, metrics)
    elif page == "Provider Detail":
        render_provider_detail(filtered_df)
    elif page == "Review Log":
        render_feedback(filtered_df)


if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run`, __name__ is "__main__" in the script context
    # but Streamlit also imports the module — this handles both cases
    main()
