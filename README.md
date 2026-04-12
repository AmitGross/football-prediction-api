# Football Prediction API

A machine-learning API that predicts WC 2026 match scores and probabilities, deployed on Render. Automatically retrains after each real match result and pushes updated predictions to Supabase.

## What it does

- Trains a RandomForest + XGBoost ensemble on ~965 historical international matches
- Predicts scorelines and win/draw/loss probabilities for all WC 2026 group stage matches
- Retrains automatically after each real result (walk-forward learning)
- Writes predictions to Supabase → frontend displays them live

## Architecture

```
GitHub Actions (every 10 min, Jun 11 – Jul 19)
    └── /fetch-scores
            ├── Calls football-data.org API
            ├── Writes finished scores to Supabase (matches table)
            └── If new finished match → triggers /retrain
                        ├── Loads historical matches.csv
                        ├── Fetches finished WC 2026 results from Supabase
                        ├── Trains RF + XGBoost ensemble (68 features)
                        └── Writes predictions to Supabase (ml_predictions)
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model status and feature count |
| `/retrain` | POST | Retrains model + updates all predictions |
| `/fetch-scores` | POST | Fetches live scores, triggers retrain if needed |
| `/predict-all` | POST | Re-runs predictions without retraining |

All write endpoints require `x-secret` header matching `RETRAIN_SECRET`.

## Model

- **Algorithm**: `AveragingEnsemble` (RandomForestRegressor + XGBRegressor) for goal prediction, XGBClassifier for W/D/L
- **Features**: 68 total — Elo ratings, form (last 2 + last 5), H2H, FIFA rankings, neighbourhood stats, stage features
- **Training data**: ~965 historical international matches + all finished WC 2026 results from Supabase
- **Version**: v1.6
- **WC 2022 benchmark (frozen)**: Accuracy 48.4% · RPS 0.2122 · RMSE 1.381
- **WC 2022 benchmark (retrain)**: Accuracy 54.7% · RPS 0.2081 · RMSE 1.349

## Environment variables (set on Render)

| Variable | Description |
|----------|-------------|
| `RETRAIN_SECRET` | Shared secret for all write endpoints |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service-role key |
| `FOOTBALL_DATA_API_KEY` | football-data.org API key |

## Local setup

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Manual operations

```powershell
# Retrain model and update predictions
Invoke-RestMethod -Uri "https://<render-url>/retrain" -Method Post -Headers @{"x-secret"="<secret>"}

# Fetch latest scores (and retrain if any finished)
Invoke-RestMethod -Uri "https://<render-url>/fetch-scores" -Method Post -Headers @{"x-secret"="<secret>"}
```

## Key dates

- **June 10, 2026** — Update FIFA rankings before tournament starts
- **June 11, 2026** — Tournament starts, GitHub Actions cron begins firing
- **July 19, 2026** — Final, cron stops
