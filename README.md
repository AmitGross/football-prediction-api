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

## WC 2026 pre-tournament predictions (model v1.6, April 2026)

Predicted champion: **🏆 France** (beats Spain 3-1 in the Final)

### France's path
| Round | Match | Score |
|-------|-------|-------|
| R32 | France vs Austria | 3-0 |
| R16 | France vs Argentina | 3-1 |
| QF | France vs England | 1-1 (on pens) |
| SF | France vs Uruguay | 2-1 |
| **Final** | **France vs Spain** | **3-1** |

### Group stage — predicted scores (key matches)

| Group | Team A | Score | Team B | P(A win) | P(Draw) | P(B win) |
|-------|--------|-------|--------|----------|---------|---------|
| A | Mexico | 3-0 | South Africa | 76.1% | 18.8% | 5.1% |
| A | South Korea | 2-0 | Czech Republic | 60.9% | 27.9% | 11.2% |
| B | Canada | 2-1 | Bosnia | 66.2% | 22.2% | 11.7% |
| B | Qatar | 1-1 | Switzerland | 20.2% | 35.9% | 43.9% |
| C | Brazil | 1-1 | Morocco | 42.8% | 32.1% | 25.2% |
| C | Haiti | 0-1 | Scotland | 15.9% | 34.6% | 49.5% |
| D | United States | 2-1 | Paraguay | 57.3% | 29.0% | 13.7% |
| E | Germany | 3-0 | Curacao | 84.3% | 14.0% | 1.7% |
| F | Netherlands | 1-1 | Japan | 39.1% | 32.4% | 28.5% |
| G | Belgium | 2-1 | Egypt | 52.3% | 27.0% | 20.7% |
| H | Spain | 3-0 | Cape Verde | 78.0% | 16.4% | 5.6% |
| I | France | 2-1 | Senegal | 42.0% | 28.8% | 29.2% |
| J | Argentina | 2-1 | Algeria | 67.0% | 22.3% | 10.7% |
| K | Portugal | 2-0 | DR Congo | 60.8% | 26.8% | 12.4% |
| L | England | 2-1 | Croatia | 42.9% | 32.3% | 24.8% |

### Full knockout bracket prediction

```
R32:   Mexico → Switzerland → Morocco → Brazil → Germany → Netherlands →
       Iran → Spain → Argentina → France → Portugal → England →
       Ecuador → Uruguay → Sweden → Egypt

R16:   Mexico → Brazil → Netherlands → Spain → France → England → Uruguay → Egypt

QF:    Brazil → Spain  |  France → Uruguay (on pens)

SF:    Spain → France

Final: FRANCE 🏆
```

---

## Feature importance analysis (model v1.6)

Computed by averaging RF and XGBoost `feature_importances_` from the production model (trained on 965 matches, April 2026 FIFA rankings).

| Rank | Feature | Avg Imp | RF | XGB | Group |
|------|---------|---------|-----|-----|-------|
| 1 | `fifa_rank_diff` | **0.2248** | 0.3795 | 0.0701 | FIFA Rankings |
| 2 | `fifa_rank_B` | 0.0400 | 0.0610 | 0.0190 | FIFA Rankings |
| 3 | `wtd_goal_diff_opp_diff` | 0.0353 | 0.0190 | 0.0516 | **Neighbourhood Perf** |
| 4 | `weighted_opp_elo_diff` | 0.0295 | 0.0428 | 0.0162 | **Neighbourhood Perf** |
| 5 | `fifa_rank_A` | 0.0286 | 0.0398 | 0.0174 | FIFA Rankings |
| 6 | `avg_goal_diff_vs_opp_diff` | 0.0197 | 0.0111 | 0.0283 | **Neighbourhood Perf** |
| 7 | `opp_scored_A` | 0.0184 | 0.0203 | 0.0165 | **Neighbourhood Basic** |
| 8 | `wtd_goal_diff_opp_A` | 0.0171 | 0.0133 | 0.0210 | **Neighbourhood Perf** |
| 9 | `opp_conceded_B` | 0.0166 | 0.0169 | 0.0163 | **Neighbourhood Basic** |
| 10 | `wtd_goal_diff_opp_B` | 0.0161 | 0.0076 | 0.0246 | **Neighbourhood Perf** |
| ... | `h2h_wins` | 0.0150 | 0.0016 | 0.0284 | H2H |
| ... | `elo_diff` | 0.0119 | 0.0138 | 0.0099 | Elo |
| 64–67 | `is_knockout`, `round_number`, `games_in_tournament_A/B` | **0.0000** | 0.0000 | 0.0000 | Stage (v1.6) |

### Key takeaways

- **FIFA rankings dominate**: `fifa_rank_diff` alone accounts for 22% of total importance — by far the most predictive single feature.
- **Neighbourhood features are highly effective**: 5 of the top 10 features are neighbourhood-based (`wtd_goal_diff_opp_*`, `weighted_opp_elo_*`, `avg_goal_diff_vs_opp_*`). These capture quality-weighted schedule strength and outperform raw Elo and form.
- **RF vs XGB split**: RF loads heavily on `fifa_rank_diff` (0.38); XGB spreads weight more evenly across neighbourhood, H2H, and form. The ensemble benefits from this complementarity.

---

## Pi-ratings — research note (v1.7 candidate)

From the 2023 Soccer Prediction Challenge paper ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807)):

> **CatBoost + pi-ratings (RPS 0.2085) outperformed a 205-feature engineered set (RPS 0.2416)**. PageRank and Elo-based features were in the candidate pool but did not survive feature selection.

**What pi-ratings are**: A dynamic team strength rating computed via exponential smoothing on the difference between expected and actual goal margin per match. Separate home and away ratings are maintained per team, updated after every game. Unlike vanilla Elo, pi-ratings weight score margin continuously rather than rounding to win/draw/loss.

**Why this is relevant to our model**:

| Aspect | Our model | Pi-ratings |
|--------|-----------|------------|
| Primary rating | Elo + FIFA rankings | Pi-ratings (goal-margin exponential smoothing) |
| Form encoding | Explicit last-5 / last-2 feature windows | Implicit — baked into the rating |
| Top feature | `fifa_rank_diff` (22% importance) | Pi-rating itself |
| Benchmarks | RPS 0.2081 (WC 2022 retrain) | RPS 0.2085 (league soccer validation) |

The datasets are different (WC vs 51-league open database), so the comparison is indicative, not direct. That said, our performance is essentially on par with the best published pi-rating results.

**Potential v1.7 experiment**: Add `pi_rating_A` and `pi_rating_B` as supplementary features on top of the existing 68 — not as a replacement for FIFA rankings (which our ablation shows are the dominant signal for international football), but as an additional dynamic strength signal. Expected risk: may be partially redundant with Elo + neighbourhood features already in the model.

---

## Related repos

| Repo | Purpose |
|------|---------|
| `AmitGross/Against-All-Odds` | Next.js frontend (Vercel) |
| `AmitGross/football-prediction` | Research repo — training, evaluation, simulation, ablation |
