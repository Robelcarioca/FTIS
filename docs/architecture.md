# FTIS Architecture

```mermaid
flowchart LR
    A["OpenSky aircraft states"] --> B["Flight ingestion"]
    C["Open-Meteo atmospheric data"] --> D["Weather service"]
    N["NOAA api.weather.gov"] --> D
    W["AviationWeather.gov METAR/PIREP"] --> D
    D --> DC["Local weather cache"]
    B --> E["Merged training dataset"]
    D --> E
    E --> F["Feature engineering and FTI"]
    F --> G["Model training pipeline"]
    G --> H["Model artifact and registry"]
    H --> I["FastAPI v1 services"]
    D --> I
    R["Route simulation engine"] --> I
    R --> J["Streamlit operations center"]
    D --> R
    H --> J
    M["Explainability and monitoring"] --> I
    M --> J
    F --> K["EDA and evaluation reports"]
```

## Layers

- Data ingestion: `scripts/fetch_flights.py`, `scripts/fetch_weather.py`
- Live weather: `ftis/weather/`
- Route analysis: `ftis/routes/`
- Dataset preparation: `scripts/preprocess_data.py`
- Feature engineering: `scripts/feature_engineering.py`, `ftis/features.py`
- Modeling: `scripts/train_model.py`, `scripts/evaluate_model.py`, `ftis/modeling.py`
- Explainability and monitoring: `ftis/explain_model.py`, `ftis/model_monitoring.py`
- Inference/API: `api/main.py`, `ftis/inference.py`
- Decision support UI: `dashboard/app.py`
- Deployment: `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`
