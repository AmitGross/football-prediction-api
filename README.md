# Football Prediction API

FastAPI service deployed on Render that powers the live ML predictions for [Against All Odds](https://againstallodds.app) — a WC 2026 prediction game.

Automatically retrains after every real match result and writes updated predictions to Supabase, where the Next.js frontend reads them in real time.

**Model v1.6 · 68 features · Best result: 54.7% outcome accuracy (WC 2022 walk-forward) · Predicted WC 2026 champion: France**

---

## Production URLs

| Service | URL |
|---------|-----|
| Render API | `https://football-prediction-api-zbdj.onrender.com` |
| Frontend | `https://againstallodds.app` |
| Supabase | `https://jstepfodbhecmrvmwwsi.supabase.co` |

CD: Render auto-deploys on every push to `master`. Build step runs `python train.py` to produce a fresh `model.pkl`.

---

## Architecture

```
GitHub Actions cron (every 10 min, Jun 11 – Jul 19 2026)
    └── POST /fetch-scores
            ├── Calls football-data.org API
            ├── Writes finished scores to Supabase → matches table
            └── If any match just finished → triggers /retrain
                        ├── Loads data/matches.csv (965 historical matches)
                        ├── Fetches all finished WC 2026 results from Supabase
                        ├── Trains RF + XGBoost ensemble (68 features)
                        ├── Uploads model.pkl to Supabase Storage
                        └── POST /predict-all → writes to ml_predictions table
```

---

## Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Model status, version, feature count |
| `/retrain` | POST | `x-secret` | Full retrain + upload + predict-all |
| `/fetch-scores` | POST | `x-secret` | Fetch live scores; auto-retrains if match finished |
| `/predict-all` | POST | `x-secret` | Re-run predictions without retraining |

All write endpoints require `x-secret` header = `RETRAIN_SECRET` env var.

---

## Manual operations (PowerShell)

```powershell
# Check model health
Invoke-RestMethod -Uri "https://football-prediction-api-zbdj.onrender.com/health"

# Retrain model and update all predictions
Invoke-RestMethod -Uri "https://football-prediction-api-zbdj.onrender.com/retrain" `
  -Method POST -Headers @{"x-secret"="wc2026-retrain-secret"} -TimeoutSec 180

# Update predictions only (no retrain)
Invoke-RestMethod -Uri "https://football-prediction-api-zbdj.onrender.com/predict-all" `
  -Method POST -Headers @{"x-secret"="wc2026-retrain-secret"} -TimeoutSec 120

# Manually trigger score fetch
Invoke-RestMethod -Uri "https://football-prediction-api-zbdj.onrender.com/fetch-scores" `
  -Method POST -Headers @{"x-secret"="wc2026-retrain-secret"}
```

Note: Render free tier cold-starts in ~30s. `/retrain` takes ~2–3 min. Use `-TimeoutSec 180`.

---

## Model

- **Algorithms**: `AveragingEnsemble` (RandomForestRegressor + XGBRegressor) predicting (λ_A, λ_B); XGBClassifier for W/D/L
- **Probabilities**: Predicted λ → Poisson score grid → P(win)/P(draw)/P(loss), isotonic-calibrated
- **Training data**: `data/matches.csv` (965 historical matches) + finished WC 2026 results from Supabase
- **Version**: v1.6 · `random_state=42` · `n_jobs=-1`

### Benchmarks

| Tournament | Mode | Accuracy | RPS | RMSE |
|------------|------|----------|-----|------|
| WC 2022 | Frozen | 48.4% | 0.2122 | 1.381 |
| WC 2022 | **Retrain** | **54.7%** | **0.2081** | **1.349** |
| WC 2018 | Frozen | 35.9% | 0.2523 | 1.287 |
| WC 2018 | Retrain | 39.1% | 0.2536 | 1.256 |

### 68 features (v1.6)

| Group | Count | Features |
|-------|-------|----------|
| Elo | 3 | elo_A, elo_B, elo_diff |
| Form-5 | 14 | wins/draws/losses/goals/weighted per team × 2 |
| Form-2 | 14 | same as Form-5 but last 2 matches |
| H2H | 3 | h2h_wins_A, h2h_wins_B, h2h_draws |
| FIFA Rankings | 3 | fifa_A, fifa_B, fifa_diff |
| Rest & Match Count | 4 | rest_days_A/B, matches_played_A/B |
| Neighbourhood Basic | 9 | avg_opp_elo, avg_opp_scored, avg_opp_conceded + diffs |
| Neighbourhood Perf | 12 | weighted_opp_elo, win_rate_vs_top_teams, avg_goal_diff_vs_opp, weighted_goal_diff_by_opp + diffs |
| Stage & Volatility | 6 | is_knockout, round_number, games_in_tournament_A/B, goal_diff_std_A/B |

---

## File map

| File | Purpose |
|------|---------|
| `app.py` | FastAPI app — all endpoints |
| `train.py` | `retrain_and_upload()` — trains models, uploads `model.pkl` to Supabase Storage |
| `features.py` | Feature engineering — Elo, form, H2H, FIFA rankings, neighbourhood, stage features |
| `predict.py` | `predict_match()` — single match prediction |
| `fetch_scores.py` | Polls football-data.org, writes finished scores to Supabase `matches` table |
| `model_store.py` | Upload/download `model.pkl` to/from Supabase Storage |
| `supabase_client.py` | Supabase client singleton |
| `ensemble.py` | `AveragingEnsemble` (RF + XGBoost) |
| `poisson.py` | Score grid + result probabilities |
| `best_params.json` | Hyperparameters (from Optuna tuning in the research repo) |
| `data/matches.csv` | Historical training data (~965 international matches) |
| `data/fifa_rankings_2026.csv` | FIFA rankings April 1, 2026 — 213 teams |

---

## Supabase tables used

| Table | Written by | Read by |
|-------|-----------|---------|
| `matches` | `fetch_scores.py` (finished scores) | `/retrain` (training data), frontend (group standings) |
| `ml_predictions` | `/predict-all` | Frontend (AI predictions strip) |

---

## Environment variables (Render)

| Variable | Description |
|----------|-------------|
| `RETRAIN_SECRET` | Shared secret for all write endpoints |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service-role key |
| `FOOTBALL_DATA_API_KEY` | football-data.org API key |

---

## Local setup

```bash
pip install -r requirements.txt
# Set env vars in .env or export them, then:
uvicorn app:app --reload
```

---

## Key dates

| Date | Action |
|------|--------|
| **June 10, 2026** | Update `data/fifa_rankings_2026.csv` with new official rankings → push → Render retrains on deploy → call `/retrain` to refresh Supabase predictions |
| **June 11, 2026** | Tournament starts — GitHub Actions cron begins firing every 10 min |
| **July 19, 2026** | Final — cron stops |

---

## Planned features

### "Predict by model" (not yet built)

Users can opt in to "predict by model" per match. After each game finishes and the model retrains + `/predict-all` updates `ml_predictions`, the system automatically overwrites that user's `score_predictions` for **unplayed** matches with the latest model scores.

**Implementation:**
1. Add `predict_by_model boolean default false` to `score_predictions` table in Supabase
2. After `/predict-all` writes to `ml_predictions`, run an additional step in `app.py`:
   - Query `score_predictions` where `predict_by_model = true` and match `status != 'finished'`
   - Join with `ml_predictions` for the new scores
   - Bulk upsert into `score_predictions` per opted-in user
3. Never update already-scored/finished matches

---

## Related repos

| Repo | Purpose |
|------|---------|
| `AmitGross/Against-All-Odds` | Next.js frontend (Vercel) |
| `AmitGross/football-prediction` | Research repo — training, evaluation, simulation, ablation |
