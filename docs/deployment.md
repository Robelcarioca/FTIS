# FTIS Deployment Guide

## Local Production Run

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/evaluate_model.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
$env:BACKEND_URL="http://localhost:8000"
streamlit run dashboard/app.py
```

## Docker Compose

```bash
copy .env.example .env
docker compose up --build
```

Services:

- API: `$env:BACKEND_URL` or `http://localhost:8000`
- API docs: `$env:BACKEND_URL/docs`
- Dashboard: http://localhost:8501

## Deployment Notes

- Mount `models/ftis_model.pkl` as a release artifact for API and dashboard services.
- Mount `data/cache/` if repeated route weather queries should survive container restarts.
- Set `BACKEND_URL` for dashboard deployments, for example `https://ftis-api.example.com`.
- Set `FTIS_REQUIRE_AUTH=true` and rotate `FTIS_JWT_SECRET` before exposing the API publicly.
- Keep `FTIS_RATE_LIMIT_PER_MINUTE` conservative for public demos.
- Keep ingestion jobs separate from inference services in production.
- Store OpenSky credentials as environment variables or managed secrets.
- Schedule feature generation and retraining with Airflow, Prefect, GitHub Actions, or a managed ML workflow.
- Add object storage for raw and processed data before scaling beyond the MVP.

## Cloud Deployment Checklist

- Build and push the Docker image from the `FTIS` directory.
- Provide the model artifact through a mounted volume, object-storage sync, or image-baked release artifact.
- Configure secrets through the cloud runtime, not source control.
- Add HTTPS and authentication at the gateway or load balancer.
- Configure health checks against `/health` and readiness checks against `/system/status`.
- Route dashboard traffic to port `8501` and API traffic to port `8000`.
