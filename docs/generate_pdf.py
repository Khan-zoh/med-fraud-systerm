from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import os

OUTPUT = "docs/Medicare_FWA_Detection_Work_Sample.pdf"

NAVY  = colors.HexColor("#0A2342")
BLUE  = colors.HexColor("#1565C0")
LIGHT = colors.HexColor("#E3F2FD")
GREY  = colors.HexColor("#546E7A")
WHITE = colors.white
BLACK = colors.HexColor("#212121")

doc = SimpleDocTemplate(
    OUTPUT, pagesize=letter,
    leftMargin=0.75*inch, rightMargin=0.75*inch,
    topMargin=0.75*inch,  bottomMargin=0.75*inch,
    title="Medicare FWA Detection System - Work Sample",
    author="Zohair Khan",
)

base = getSampleStyleSheet()

def mkstyle(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

ST = {
    "h1":     mkstyle("h1",    "Heading1", fontSize=16, textColor=NAVY,
                      spaceAfter=4, spaceBefore=14, fontName="Helvetica-Bold"),
    "body":   mkstyle("body",  "Normal",   fontSize=9.5, textColor=BLACK,
                      spaceAfter=5, leading=14, alignment=TA_JUSTIFY),
    "bullet": mkstyle("bullet","Normal",   fontSize=9.5, textColor=BLACK,
                      spaceAfter=3, leading=13, leftIndent=12),
    "title":  mkstyle("title", "Normal",   fontSize=24, textColor=WHITE,
                      alignment=TA_CENTER, fontName="Helvetica-Bold",
                      spaceAfter=4, leading=30),
    "sub":    mkstyle("sub",   "Normal",   fontSize=12, textColor=LIGHT,
                      alignment=TA_CENTER, spaceAfter=3),
    "meta":   mkstyle("meta",  "Normal",   fontSize=9,  textColor=LIGHT,
                      alignment=TA_CENTER),
    "cap":    mkstyle("cap",   "Normal",   fontSize=8.5, textColor=GREY,
                      alignment=TA_CENTER, spaceAfter=6, leading=12),
    "kpi_v":  mkstyle("kpi_v", "Normal",   fontSize=18, textColor=NAVY,
                      alignment=TA_CENTER, fontName="Helvetica-Bold", leading=22),
    "kpi_l":  mkstyle("kpi_l", "Normal",   fontSize=7.5, textColor=GREY,
                      alignment=TA_CENTER),
    "th":     mkstyle("th",    "Normal",   fontSize=9, textColor=WHITE,
                      fontName="Helvetica-Bold", leading=12),
    "td":     mkstyle("td",    "Normal",   fontSize=9, textColor=BLACK,
                      leading=12, spaceAfter=0),
    "footer": mkstyle("footer","Normal",   fontSize=8, textColor=GREY,
                      alignment=TA_CENTER),
}

def hr(color=BLUE, thick=0.8):
    return HRFlowable(width="100%", thickness=thick, color=color,
                      spaceAfter=7, spaceBefore=2)

def p(text, style="td"):
    return Paragraph(text, ST[style])

def make_table(rows, col_widths, row_heights=None):
    """rows is list of lists of Paragraph objects (or strings auto-wrapped)."""
    processed = []
    for i, row in enumerate(rows):
        prow = []
        for cell in row:
            if isinstance(cell, str):
                prow.append(p(cell, "th" if i == 0 else "td"))
            else:
                prow.append(cell)
        processed.append(prow)

    t = Table(processed, colWidths=col_widths, rowHeights=row_heights,
              repeatRows=1)
    ts = [
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 7),
        ("RIGHTPADDING",   (0,0), (-1,-1), 7),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#CFD8DC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, colors.HexColor("#F5F7FA")]),
        # Header row
        ("BACKGROUND",     (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",      (0,0), (-1,0), WHITE),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
    ]
    t.setStyle(TableStyle(ts))
    return t

W = 7.0  # total content width in inches

story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
cover_inner = [
    [p("Medicare Part B", "title")],
    [p("Fraud, Waste &amp; Abuse Detection System", "title")],
    [Spacer(1, 8)],
    [p("Work Sample  |  Portfolio Project", "sub")],
    [Spacer(1, 4)],
    [p("Zohair Khan  |  khan.zoh25@gmail.com", "sub")],
    [p("Industrial Engineering  |  Data Engineering Certificate", "meta")],
    [Spacer(1, 6)],
]
cover = Table([[row[0]] for row in cover_inner], colWidths=[W*inch])
cover.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), NAVY),
    ("TOPPADDING",    (0,0), (-1,-1), 14),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ("LEFTPADDING",   (0,0), (-1,-1), 20),
    ("RIGHTPADDING",  (0,0), (-1,-1), 20),
]))
story.append(cover)
story.append(Spacer(1, 14))

# ── KPI bar ───────────────────────────────────────────────────────────────────
kpis = [
    ("1.27M",   "Part B\nRecords"),
    ("$13.95B", "Medicare\nPayments"),
    ("150K",    "Providers\nScored"),
    ("6",       "States"),
    ("51",      "Risk\nFeatures"),
    ("3",       "Model\nLayers"),
]
n = len(kpis)
cw = (W / n) * inch
kpi_cells = []
for val, label in kpis:
    cell = Table(
        [[p(val, "kpi_v")], [p(label, "kpi_l")]],
        colWidths=[cw - 4]
    )
    cell.setStyle(TableStyle([
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("LEFTPADDING",   (0,0), (-1,-1), 2),
        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
    ]))
    kpi_cells.append(cell)

kpi_bar = Table([kpi_cells], colWidths=[cw]*n)
kpi_bar.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), LIGHT),
    ("BOX",           (0,0), (-1,-1), 0.8, BLUE),
    ("INNERGRID",     (0,0), (-1,-1), 0.4, colors.HexColor("#BBDEFB")),
    ("TOPPADDING",    (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN",         (0,0), (-1,-1), "CENTRE"),
]))
story.append(kpi_bar)
story.append(Spacer(1, 14))

# ── 1. Problem ────────────────────────────────────────────────────────────────
story.append(p("1.  The Problem", "h1"))
story.append(hr())
story.append(p(
    "Medicare processes over <b>1 billion claims annually</b>. Within that volume, Fraud, "
    "Waste, and Abuse (FWA) costs the federal government an estimated <b>$60-90 billion per "
    "year</b>. CMS and OIG maintain detection programs, but the scale of the data demands "
    "automated screening tools that triage providers before human investigators are engaged.",
    "body"))
story.append(p(
    "This project builds an end-to-end FWA detection system using real public CMS data, "
    "domain-specific features grounded in <b>industrial engineering principles</b>, a "
    "three-layer ML ensemble, and an investigator dashboard. Every component is built to "
    "production standards: data-quality gates, versioned transforms, and a 51-test pytest suite.",
    "body"))

# ── 2. IE Angle ───────────────────────────────────────────────────────────────
story.append(Spacer(1, 4))
story.append(p("2.  Industrial Engineering Principles Applied to Healthcare Fraud", "h1"))
story.append(hr())
story.append(p(
    "Most FWA tools apply generic anomaly detection. This project applies IE principles "
    "that give each feature operational meaning a domain expert can defend:", "body"))
story.append(Spacer(1, 4))

c1, c2, c3 = 1.6*inch, 2.15*inch, 3.25*inch
ie_rows = [
    ["IE Principle", "Feature", "What It Detects"],
    ["Statistical Process\nControl (SPC)",
     "Billing volume z-scores\nvs. specialty peer group",
     "Providers statistically out-of-control\nvs. peers in same specialty"],
    ["Peer Benchmarking\n(Jensen-Shannon Div.)",
     "Procedure-mix distribution\nvs. specialty median",
     "Unusual concentration in\nhigh-reimbursement HCPCS codes"],
    ["Queueing Theory /\nThroughput Analysis",
     "Services-per-day vs.\ntheoretical capacity limit",
     "Physically impossible billing rates\nfor a single provider"],
    ["Geographic Entropy",
     "Patient zip-code\ndispersion index",
     "Patient brokering — recruiting from\nabnormally wide service areas"],
    ["Mahalanobis Distance",
     "Multivariate outlier within\nspecialty peer group",
     "Anomalous combinations of billing\ndimensions not caught by single features"],
]
story.append(make_table(ie_rows, [c1, c2, c3]))

# ── 3. Data ───────────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(p("3.  Data Sources", "h1"))
story.append(hr())
story.append(p(
    "All data is <b>publicly available and free</b> — no PHI, no PII, no credentials required. "
    "Three federal sources are joined on NPI (National Provider Identifier):", "body"))
story.append(Spacer(1, 4))

d1, d2, d3 = 1.6*inch, 3.2*inch, 2.2*inch
data_rows = [
    ["Source", "Contents", "Scale"],
    ["CMS Part B\nProvider and Service\n(data.cms.gov)",
     "Billing volumes, avg charges, avg Medicare payments per provider per HCPCS code. "
     "Years: 2021, 2022, 2023.",
     "1.27M service lines\n150K unique providers\n4,542 HCPCS codes\n102 specialties\n6 states"],
    ["OIG LEIE Exclusions\n(oig.hhs.gov)",
     "Providers excluded from Medicare and Medicaid participation. "
     "Used as noisy fraud labels for supervised model training.",
     "83,001 exclusion records\n29 matched to the\n6-state population"],
    ["NPPES NPI Registry\n(download.cms.gov)",
     "Provider specialty taxonomy, practice location, and entity type. "
     "Enriches provider identity fields.",
     "15,000 records sampled\nfor 6-state coverage"],
]
story.append(make_table(data_rows, [d1, d2, d3]))
story.append(Spacer(1, 6))
story.append(p(
    "<b>Label caveat:</b> LEIE exclusions cover license revocations, controlled-substance "
    "offenses, and other non-billing misconduct in addition to Medicare billing fraud. "
    "The supervised layer learns from this noisy signal; the unsupervised Isolation Forest "
    "catches anomalies regardless of labels.", "body"))

# ── 4. Architecture ───────────────────────────────────────────────────────────
story.append(Spacer(1, 6))
story.append(p("4.  System Architecture", "h1"))
story.append(hr())

a1, a2, a3 = 1.1*inch, 1.8*inch, 4.1*inch
arch_rows = [
    ["Layer", "Technology", "Role"],
    ["Ingestion",
     "Python +\nGreat Expectations",
     "Download CMS API, LEIE, NPPES to DuckDB raw schema with automated data-quality gates"],
    ["Transformation",
     "dbt-duckdb\n(8 models)",
     "staging to intermediate to marts; version-controlled SQL with built-in data tests and lineage"],
    ["Feature Eng.",
     "Python / pandas\n(51 features)",
     "SPC z-scores, Jensen-Shannon divergence, throughput flags, Mahalanobis distance, geographic entropy, upcoding ratios"],
    ["Model Layer 1",
     "Isolation Forest\n(300 trees, 5%)",
     "Unsupervised anomaly detection; catches novel patterns without label dependency"],
    ["Model Layer 2",
     "XGBoost\nscale_pos_weight\n1:5,187",
     "Supervised classifier trained on LEIE exclusion labels; handles extreme class imbalance without synthetic oversampling"],
    ["Model Layer 3",
     "NetworkX\nbipartite graph",
     "Provider-HCPCS network; PageRank, degree anomaly, and dominant-procedure community detection"],
    ["Ensemble",
     "Weighted blend\n(25 / 50 / 25)",
     "0-100 composite risk score; percentile-based triage tiers: Low / Moderate / Elevated / High / Critical"],
    ["Explainability",
     "SHAP\nTreeExplainer",
     "Per-provider feature contributions; plain-English finding narratives displayed on the investigator dashboard"],
    ["Serving",
     "FastAPI +\nStreamlit",
     "REST scoring endpoint with OpenAPI docs; investigator dashboard with ROI metrics, filters, and drill-down"],
    ["Quality / CI",
     "pytest (51 tests)\nGitHub Actions",
     "Unit and integration tests, ruff lint; 100% pass rate on CI"],
]
story.append(make_table(arch_rows, [a1, a2, a3]))

# ── 5. Dashboard ──────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(p("5.  Investigator Dashboard", "h1"))
story.append(hr())
story.append(p(
    "The Streamlit dashboard is the primary interface for analysts reviewing flagged providers. "
    "It surfaces risk tiers, model layer contribution breakdowns, SHAP-based finding narratives, "
    "and investigator ROI metrics on the Review Log tab.", "body"))
story.append(Spacer(1, 8))

img1_path = "docs/screenshots/dashboard_overview.png"
if os.path.exists(img1_path):
    story.append(Image(img1_path, width=W*inch, height=3.7*inch, kind="proportional"))
    story.append(p(
        "Fig 1. Overview page: 30K+ providers scored across 5 percentile-based risk tiers "
        "(Low to Critical), risk score histogram, and headline metrics.", "cap"))

story.append(Spacer(1, 10))

img2_path = "docs/screenshots/dashboard_providers.png"
if os.path.exists(img2_path):
    story.append(Image(img2_path, width=W*inch, height=3.7*inch, kind="proportional"))
    story.append(p(
        "Fig 2. Model layer contribution breakdown (Isolation Forest, XGBoost, Graph) "
        "and Top-20 highest-risk providers ranked by composite score.", "cap"))

# ── 6. Results ────────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(p("6.  Key Results", "h1"))
story.append(hr())
story.append(p(
    "The system functions as a <b>peer-relative triage tool</b>. The top-25 risk-scored "
    "providers show substantially elevated billing activity versus the overall population:",
    "body"))
story.append(Spacer(1, 4))

r1, r2, r3 = 2.8*inch, 2.0*inch, 2.2*inch
results_rows = [
    ["Metric", "Top-25 Flagged Providers", "vs. Population"],
    ["Median Medicare payments",      "$1.02M",  "40.7x population median"],
    ["Median services rendered",      "4,966",   "11.9x population median"],
    ["Median services / beneficiary", "5.0",     "3.6x population median"],
    ["LEIE class imbalance",          "--",      "1 : 5,187"],
    ["Model PR-AUC",                  "0.001",   "Baseline: 1/5,187 = 0.0002"],
]
story.append(make_table(results_rows, [r1, r2, r3]))
story.append(Spacer(1, 8))
story.append(p(
    "<b>Honest framing:</b> PR-AUC is intentionally low. With only 29 LEIE-matched providers "
    "out of 150K, meaningful supervised lift is limited by label quality, not model design. "
    "The system's practical value is surfacing extreme statistical outliers for investigator "
    "review — the same function CMS fraud-analytics teams perform in practice. "
    "A production deployment would use CMS payment-suspension data and DOJ strike-force "
    "enforcement labels, which are not publicly available.", "body"))

# ── 7. Stack ──────────────────────────────────────────────────────────────────
story.append(Spacer(1, 8))
story.append(p("7.  Technical Stack", "h1"))
story.append(hr())

s1, s2, s3 = 1.2*inch, 1.6*inch, 4.2*inch
stack_rows = [
    ["Component",      "Choice",                    "Decision Rationale"],
    ["Storage",        "DuckDB",
     "Columnar analytics on laptop; SQL dialect maps directly to Postgres/Snowflake in production"],
    ["Transforms",     "dbt-duckdb",
     "Version-controlled SQL with built-in testing and data lineage; analytics engineering standard"],
    ["ML",             "scikit-learn\n+ XGBoost",
     "Production-grade tabular ML; widely understood in government analytics contexts"],
    ["Graph",          "NetworkX",
     "Bipartite provider-HCPCS network; PageRank and community features without GPU requirement"],
    ["Explainability", "SHAP\nTreeExplainer",
     "Required for government/regulated use; investigators need to understand why a provider was flagged"],
    ["API",            "FastAPI",
     "Async, auto-generated OpenAPI docs, Pydantic validation; easily containerised with Docker"],
    ["Dashboard",      "Streamlit",
     "Rapid investigator-workflow prototyping; interactive filters, drill-down, and review logging"],
    ["Quality",        "pytest + ruff\n+ Great Exp.",
     "51 unit tests, lint CI via GitHub Actions, data contracts on every ingestion run"],
]
story.append(make_table(stack_rows, [s1, s2, s3]))

# ── 8. Future Work ────────────────────────────────────────────────────────────
story.append(Spacer(1, 10))
story.append(p("8.  What I Would Do Differently in Production", "h1"))
story.append(hr())
fw = [
    "<b>Multi-year temporal splits:</b> Train on 2020-2021, validate on 2022, deploy for 2023. "
    "Monitor for concept drift quarterly.",
    "<b>Better labels:</b> Replace LEIE with CMS payment suspensions and DOJ healthcare-fraud "
    "strike-force takedowns — billing-fraud-specific labels rather than all exclusion types.",
    "<b>Claims-level features:</b> Access to individual claims via CMS DUA would enable "
    "temporal billing-pattern analysis, beneficiary-level features, and diagnosis-procedure "
    "consistency checks.",
    "<b>Active learning loop:</b> The dashboard Review Log captures investigator decisions. "
    "These should feed back into model retraining as investigators confirm or dismiss flags.",
    "<b>Role-based access and audit logging:</b> Investigators should only see providers in "
    "their assigned region and specialty, with every score lookup logged for compliance.",
]
for item in fw:
    story.append(p("  •  " + item, "bullet"))

# ── Footer ─────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 18))
story.append(HRFlowable(width="100%", thickness=0.5, color=GREY, spaceAfter=5))
story.append(p(
    "Zohair Khan  |  khan.zoh25@gmail.com  |  github.com/Khan-zoh/med-fraud-systerm",
    "footer"))

doc.build(story)
print("PDF saved to:", OUTPUT)
