# FTIS Benchmark Report

## Current Benchmark Scope

The baseline pipeline evaluates candidate classifiers on the engineered FTIS feature dataset using weighted F1 as the selection metric. Candidate models include:

- Dummy baseline
- Random Forest
- XGBoost, when installed

## Metrics

The generated model artifact stores:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1
- Macro F1
- Confusion matrix
- Classification report

## Calibration and Monitoring

`ftis/model_monitoring.py` adds:

- Expected calibration error
- Brier score
- Multiclass log loss
- Calibration curve generation
- Population Stability Index drift reports

## Remaining Benchmark Gaps

- Validate against pilot reports, EDR, AMDAR, or radar-derived turbulence truth.
- Add temporal backtesting by route corridor and season.
- Benchmark live-provider latency and cache hit rate.
- Compare route-level decisions against operational reroute baselines.
