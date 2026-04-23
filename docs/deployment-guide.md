# Deployment Guide

This guide covers deploying the FastAPI scoring API and Streamlit dashboard for a live demo. Two options: Railway (simplest) or Fly.io (more control).

## Prerequisites

Before deploying, you need trained model artifacts. Run locally:

```bash
cp .env.example .env
make install
make ingest        # ~10 min (downloads CMS data)
make transform     # ~1 min (dbt models)
python -m cms_fwa.features.pipeline  # ~2 min (Python features)
make train         # ~3 min (ML models)
```

This produces `data/models/*.pkl` files that the API and dashboard need.

---

## Option A: Railway (Recommended for Portfolio Demos)

Railway offers a generous free tier and deploys from GitHub with zero config.

### 1. Push to GitHub

```bash
git init
git add -A
git commit -m "Initial commit — Medicare FWA detection system"
git remote add origin https://github.com/YOUR_USERNAME/medicare-fwa-detection.git
git push -u origin main
```

### 2. Deploy the API

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `medicare-fwa-detection` repo
4. Railway auto-detects the Python project. Override the settings:
   - **Build command:** `pip install -e .`
   - **Start command:** `uvicorn cms_fwa.serving.api:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `CMS_FWA_ENV=production`
   - `PORT=8000`
6. Click **Deploy**

The API will be available at `https://your-project.up.railway.app/docs` (auto-generated Swagger UI).

### 3. Deploy the Dashboard

Add a second service in the same Railway project:
   - **Start command:** `streamlit run src/cms_fwa/serving/dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - Set the same environment variables

### 4. Include Model Artifacts

For the demo, commit the trained model artifacts (they're small — a few MB):

```bash
# Temporarily un-ignore the models directory
git add -f data/models/*.pkl
git commit -m "Add trained model artifacts for demo"
git push
```

> **Note:** In production, artifacts would live in cloud storage (S3/GCS), not in git.

---

## Option B: Fly.io

Fly.io gives more control and offers a free tier with 3 shared-CPU VMs.

### 1. Install flyctl

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Deploy the API

```bash
cd docker/
fly launch --name cms-fwa-api --dockerfile Dockerfile.api --region dfw
fly deploy
```

### 3. Deploy the Dashboard

```bash
fly launch --name cms-fwa-dashboard --dockerfile Dockerfile.dashboard --region dfw
fly deploy
```

---

## Option C: Docker Compose (Local Demo)

For presenting locally or on a VM:

```bash
make docker-up
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

To stop:
```bash
make docker-down
```

---

## Verifying the Deployment

After deploying, verify:

```bash
# Health check
curl https://your-api-url/health

# Score a provider
curl -X POST https://your-api-url/score \
  -H "Content-Type: application/json" \
  -d '{"npi": "1234567890"}'

# Browse top risk
curl https://your-api-url/top-risk?limit=10
```

The Streamlit dashboard should show the Overview page with risk distribution charts.

---

## Cost Estimates

| Platform | Free Tier | Cost Beyond Free |
|----------|-----------|-----------------|
| Railway | $5/month credit (enough for demo) | $0.000463/min per service |
| Fly.io | 3 shared-CPU VMs, 160GB outbound | $0.0000022/s per machine |
| Docker (local) | Free | Your electricity |

For a portfolio demo that gets occasional traffic, both Railway and Fly.io free tiers are sufficient.
