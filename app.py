# app.py — production FastAPI for Render
# Endpoints: /retrain, /fetch-scores, /predict-all, /health

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Header

MODEL_VERSION = "v1.6"

# Module-level model cache — populated at startup and after each retrain
_model        = None
_feature_cols = None


# ── Startup: download model from Supabase Storage ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feature_cols
    try:
        from model_store import load_model_from_storage
        _model, _feature_cols = load_model_from_storage()
        print(f"[startup] Model loaded ({len(_feature_cols)} features, v{MODEL_VERSION})")
    except Exception as exc:
        # Don't crash the server — /retrain or a build-step upload will fix it.
        print(f"[startup] WARNING: Could not load model from Supabase Storage: {exc}")
    yield


app = FastAPI(title="Football Prediction API", version=MODEL_VERSION, lifespan=lifespan)

# Secret token for all write endpoints (set RETRAIN_SECRET env var on Render)
_RETRAIN_SECRET = os.environ.get("RETRAIN_SECRET", "")


def _require_secret(x_secret: str) -> None:
    if _RETRAIN_SECRET and x_secret != _RETRAIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── /health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_version": MODEL_VERSION,
        "features":     len(_feature_cols) if _feature_cols else 0,
    }


# ── /retrain ───────────────────────────────────────────────────────────────────
# Triggered by Supabase webhook when a match status → 'finished'.

@app.post("/retrain")
async def retrain(x_secret: str = Header(default="")):
    """
    Retrains the model on historical data + all finished WC 2026 results,
    uploads new model.pkl to Supabase Storage, then re-predicts all remaining matches.
    """
    _require_secret(x_secret)
    global _model, _feature_cols
    try:
        from train import retrain_and_upload
        _model, _feature_cols = retrain_and_upload()
        count = await _predict_all_remaining()
        return {
            "status":               "ok",
            "message":              "Model retrained and predictions updated.",
            "predictions_written":  count,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /fetch-scores ──────────────────────────────────────────────────────────────
# Called every 5 min by GitHub Actions cron during the tournament.

@app.post("/fetch-scores")
async def fetch_scores_endpoint(x_secret: str = Header(default="")):
    """
    Pulls live / final scores from football-data.org and writes them to
    Supabase matches table. If any match just finished, triggers a full retrain.
    """
    _require_secret(x_secret)
    try:
        from fetch_scores import fetch_and_update
        newly_finished = fetch_and_update()

        retrain_triggered = False
        if newly_finished:
            global _model, _feature_cols
            from train import retrain_and_upload
            _model, _feature_cols = retrain_and_upload()
            await _predict_all_remaining()
            retrain_triggered = True

        return {
            "status":            "ok",
            "newly_finished":    newly_finished,
            "retrain_triggered": retrain_triggered,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /predict-all ───────────────────────────────────────────────────────────────

@app.post("/predict-all")
async def predict_all_endpoint(x_secret: str = Header(default="")):
    """Predicts all remaining scheduled WC 2026 group matches and writes to ml_predictions."""
    _require_secret(x_secret)
    try:
        count = await _predict_all_remaining()
        return {"status": "ok", "predictions_written": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Internal: predict all remaining group-stage matches ────────────────────────

async def _predict_all_remaining() -> int:
    if _model is None or _feature_cols is None:
        raise RuntimeError("Model not loaded — cannot predict.")

    from supabase_client import get_client, normalize_team_name
    from train import build_training_data
    from predict import predict_match_with_model

    client = get_client()

    # Resolve WC 2026 tournament id
    t = client.table("tournaments").select("id").eq("year", 2026).single().execute()
    tournament_id = t.data["id"]

    # Fetch all scheduled group-stage matches
    scheduled = (
        client.table("matches")
        .select(
            "id, starts_at, stage, "
            "home_team:home_team_id(id, name), away_team:away_team_id(id, name)"
        )
        .eq("tournament_id", tournament_id)
        .eq("status", "scheduled")
        .execute()
    )

    history = build_training_data()

    # Stage → (is_knockout, round_number) for v1.6 features
    _STAGE_FLAGS = {
        "GROUP": (0, 1),
        "R32":   (1, 2),
        "R16":   (1, 2),
        "QF":    (1, 3),
        "SF":    (1, 4),
        "FINAL": (1, 5),
    }

    count = 0
    for match in scheduled.data:
        team_A = normalize_team_name(match["home_team"]["name"])
        team_B = normalize_team_name(match["away_team"]["name"])
        match_date = pd.Timestamp(match["starts_at"])
        stage = match.get("stage", "GROUP")
        is_knockout, round_number = _STAGE_FLAGS.get(stage, (0, 1))

        # Only use history available before this match's kickoff
        past = history[history["date"] < match_date]

        result = predict_match_with_model(
            _model, _feature_cols,
            team_A, team_B, past,
            is_knockout=is_knockout,
            round_number=round_number,
        )

        client.table("ml_predictions").upsert(
            {
                "match_id":             match["id"],
                "home_team_id":         match["home_team"]["id"],
                "away_team_id":         match["away_team"]["id"],
                "predicted_home_goals": round(result["lam_A"], 2),
                "predicted_away_goals": round(result["lam_B"], 2),
                "prob_home_win":        result["prob_home_win"],
                "prob_draw":            result["prob_draw_raw"],
                "prob_away_win":        result["prob_away_win"],
                "model_version":        MODEL_VERSION,
            },
            on_conflict="match_id",
        ).execute()
        count += 1

    print(f"[predict_all] Wrote {count} predictions to Supabase ml_predictions")
    return count
