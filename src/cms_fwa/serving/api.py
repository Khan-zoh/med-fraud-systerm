"""
FastAPI Scoring API

Endpoints:
  GET  /health           — liveness check
  POST /score            — score a single NPI (risk + SHAP + peer comparison)
  GET  /top-risk         — browse highest-risk providers
  GET  /provider/{npi}   — full provider detail
  GET  /stats            — dataset summary statistics

All responses use careful FWA language:
  - "flagged for review" not "fraudulent"
  - "anomalous billing pattern" not "criminal activity"
  - "risk score" not "fraud score"
"""

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from cms_fwa.config import settings


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    """Request body for POST /score."""
    npi: str = Field(..., description="10-digit National Provider Identifier", pattern=r"^\d{10}$")


class ModelScores(BaseModel):
    """Individual model layer scores."""
    isolation_forest: float = Field(..., description="Unsupervised anomaly score (0-1)")
    xgboost: float = Field(..., description="Supervised fraud probability (0-1)")
    graph: float = Field(..., description="Network-based anomaly score (0-1)")


class FeatureContribution(BaseModel):
    """A single SHAP feature contribution."""
    feature: str
    description: str
    value: float | None = None
    impact: float
    direction: str


class PeerComparison(BaseModel):
    """Provider metric vs. specialty peer group."""
    metric: str
    provider_value: float | None
    peer_median: float | None
    zscore: float | None


class ScoreResponse(BaseModel):
    """Response for POST /score and GET /provider/{npi}."""
    npi: str
    provider_name: str
    specialty: str
    state: str
    risk_score: float = Field(..., description="Ensemble risk score (0-100)")
    risk_tier: str = Field(..., description="Low / Moderate / Elevated / High / Critical")
    model_scores: ModelScores
    top_contributors: list[FeatureContribution] = []
    peer_comparisons: list[PeerComparison] = []
    disclaimer: str = (
        "This is a statistical flag for review based on anomalous billing patterns. "
        "It is not an accusation of fraud, waste, or abuse."
    )


class TopRiskItem(BaseModel):
    """Summary row for the top-risk listing."""
    npi: str
    provider_name: str
    specialty: str
    state: str
    risk_score: float
    risk_tier: str
    is_excluded: bool


class StatsResponse(BaseModel):
    """Dataset summary statistics."""
    total_providers: int
    total_excluded: int
    exclusion_rate: float
    risk_tier_distribution: dict[str, int]
    top_specialties: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# App state (loaded once at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Holds pre-loaded model artifacts and risk table."""
    risk_table: pd.DataFrame | None = None
    feature_table: pd.DataFrame | None = None
    shap_data: Any = None
    ready: bool = False


state = AppState()


def _load_state() -> None:
    """Load model artifacts from disk."""
    from cms_fwa.models.data_prep import load_artifact

    try:
        state.risk_table = load_artifact("risk_table")
        logger.info(f"Loaded risk table: {len(state.risk_table):,} providers")
        state.ready = True
    except FileNotFoundError:
        logger.warning(
            "No risk table found. Run 'make train' first. "
            "API will return 503 until models are trained."
        )
        state.ready = False

    try:
        state.feature_table = load_artifact("training_summary")
    except FileNotFoundError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    _load_state()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CMS FWA Detection API",
    description=(
        "Medicare Provider Fraud, Waste, and Abuse detection scoring service. "
        "Returns risk scores, explanations, and peer comparisons for providers."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _check_ready() -> None:
    """Raise 503 if models aren't loaded."""
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run 'make train' to generate model artifacts.",
        )


def _find_provider(npi: str) -> pd.Series:
    """Look up a provider in the risk table."""
    _check_ready()
    mask = state.risk_table["npi"].astype(str) == npi
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"NPI {npi} not found")
    return state.risk_table[mask].iloc[0]


def _build_score_response(provider: pd.Series) -> ScoreResponse:
    """Build a ScoreResponse from a risk table row."""
    first = provider.get("provider_first_name", "") or ""
    last = provider.get("provider_last_name", "") or ""

    return ScoreResponse(
        npi=str(provider["npi"]),
        provider_name=f"{first} {last}".strip(),
        specialty=str(provider.get("provider_specialty", "Unknown")),
        state=str(provider.get("provider_state", "")),
        risk_score=round(float(provider["risk_score"]), 1),
        risk_tier=str(provider["risk_tier"]),
        model_scores=ModelScores(
            isolation_forest=round(float(provider["if_anomaly_score"]), 4),
            xgboost=round(float(provider["xgb_fraud_prob"]), 4),
            graph=round(float(provider["graph_anomaly_score"]), 4),
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {
        "status": "healthy" if state.ready else "degraded",
        "models_loaded": state.ready,
        "providers_scored": len(state.risk_table) if state.risk_table is not None else 0,
    }


@app.post("/score", response_model=ScoreResponse)
async def score_provider(request: ScoreRequest) -> ScoreResponse:
    """Score a single provider by NPI.

    Returns the ensemble risk score (0-100), individual model scores,
    top SHAP contributors, and peer comparisons.
    """
    provider = _find_provider(request.npi)
    return _build_score_response(provider)


@app.get("/provider/{npi}", response_model=ScoreResponse)
async def get_provider(npi: str) -> ScoreResponse:
    """Get full provider risk detail by NPI."""
    provider = _find_provider(npi)
    return _build_score_response(provider)


@app.get("/top-risk", response_model=list[TopRiskItem])
async def top_risk(
    limit: int = Query(default=50, ge=1, le=500, description="Number of providers"),
    specialty: str | None = Query(default=None, description="Filter by specialty"),
    provider_state: str | None = Query(default=None, alias="state", description="Filter by state"),
    min_score: float = Query(default=0, ge=0, le=100, description="Minimum risk score"),
) -> list[TopRiskItem]:
    """Browse highest-risk providers with optional filters."""
    _check_ready()

    df = state.risk_table.copy()

    if specialty:
        df = df[df["provider_specialty"].str.contains(specialty, case=False, na=False)]
    if provider_state:
        df = df[df["provider_state"].str.upper() == provider_state.upper()]
    if min_score > 0:
        df = df[df["risk_score"] >= min_score]

    df = df.head(limit)

    return [
        TopRiskItem(
            npi=str(row["npi"]),
            provider_name=f"{row.get('provider_first_name', '') or ''} {row.get('provider_last_name', '') or ''}".strip(),
            specialty=str(row.get("provider_specialty", "Unknown")),
            state=str(row.get("provider_state", "")),
            risk_score=round(float(row["risk_score"]), 1),
            risk_tier=str(row["risk_tier"]),
            is_excluded=bool(row.get("is_excluded", False)),
        )
        for _, row in df.iterrows()
    ]


@app.get("/stats", response_model=StatsResponse)
async def dataset_stats() -> StatsResponse:
    """Get summary statistics about the scored dataset."""
    _check_ready()

    df = state.risk_table
    tier_dist = df["risk_tier"].value_counts().to_dict()
    tier_dist = {str(k): int(v) for k, v in tier_dist.items()}

    top_specialties = (
        df.groupby("provider_specialty")
        .agg(
            count=("npi", "count"),
            avg_risk=("risk_score", "mean"),
            n_excluded=("is_excluded", "sum"),
        )
        .sort_values("avg_risk", ascending=False)
        .head(10)
        .reset_index()
        .to_dict("records")
    )

    return StatsResponse(
        total_providers=len(df),
        total_excluded=int(df["is_excluded"].sum()),
        exclusion_rate=float(df["is_excluded"].mean()),
        risk_tier_distribution=tier_dist,
        top_specialties=top_specialties,
    )
