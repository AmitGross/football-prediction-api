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

        # Always re-populate knockout bracket (advances winners + fills R32 if group stage done)
        from populate_knockouts import populate_knockouts
        ko_result = populate_knockouts()
        # If new knockout matches were created, run predictions for them
        if ko_result.get("matches_upserted", 0) > 0:
            await _predict_all_remaining()

        return {
            "status":            "ok",
            "newly_finished":    newly_finished,
            "retrain_triggered": retrain_triggered,
            "knockouts":         ko_result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /populate-knockouts ────────────────────────────────────────────────────────

@app.post("/populate-knockouts")
async def populate_knockouts_endpoint(x_secret: str = Header(default="")):
    """
    Reads group stage standings → fills R32 knockout_slots with real team IDs.
    Also advances winner_team_id through R16, QF, SF, Final, and Bronze slots.
    Safe to call repeatedly (idempotent).
    """
    _require_secret(x_secret)
    try:
        from populate_knockouts import populate_knockouts
        result = populate_knockouts()
        return result
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
        match_date = pd.Timestamp(match["starts_at"]).tz_localize(None)
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

        # Map internal outcome label to home/away perspective
        _outcome_map = {"win": "home_win", "draw": "draw", "loss": "away_win"}
        outcome_label = _outcome_map.get(result.get("outcome", "draw"), "draw")

        # Delete existing prediction for this match then insert fresh
        client.table("ml_predictions").delete().eq("match_id", match["id"]).execute()
        client.table("ml_predictions").insert(
            {
                "match_id":             match["id"],
                "home_team_id":         match["home_team"]["id"],
                "away_team_id":         match["away_team"]["id"],
                "home_team_name":       match["home_team"]["name"],
                "away_team_name":       match["away_team"]["name"],
                "predicted_home_goals": round(result["lam_A"], 2),
                "predicted_away_goals": round(result["lam_B"], 2),
                "predicted_home_score": int(result["goals_A"]),
                "predicted_away_score": int(result["goals_B"]),
                "predicted_outcome":    outcome_label,
                "prob_home_win":        result["prob_home_win"],
                "prob_draw":            result["prob_draw_raw"],
                "prob_away_win":        result["prob_away_win"],
                "model_version":        MODEL_VERSION,
            }
        ).execute()
        count += 1

    print(f"[predict_all] Wrote {count} group-stage predictions to Supabase ml_predictions")

    # Also predict knockout slot matchups (teams in slots, not in matches table)
    ko_count = await _predict_knockout_slots()
    return count + ko_count


# ── Internal: predict all undecided knockout slot matchups ─────────────────────

async def _predict_knockout_slots() -> int:
    """
    Generates ML predictions for knockout slot pairs where both teams are known
    but the match hasn't been decided yet. Stores rows with slot_id (not match_id)
    so the frontend can find them by home_team_id + away_team_id.
    Mirrors the pairing logic in the Next.js knockouts page.
    """
    if _model is None or _feature_cols is None:
        raise RuntimeError("Model not loaded — cannot predict.")

    from supabase_client import get_client, normalize_team_name
    from train import build_training_data
    from predict import predict_match_with_model

    client = get_client()

    t = client.table("tournaments").select("id").eq("year", 2026).single().execute()
    tournament_id = t.data["id"]

    # Fetch all knockout slots with their home team info
    slots_result = client.table("knockout_slots").select(
        "id, round, side, position, home_team_id, winner_team_id, "
        "home_team:teams!knockout_slots_home_team_id_fkey(id, name)"
    ).eq("tournament_id", tournament_id).execute()

    slots = slots_result.data
    by_pos = {(s["round"], s["side"], s["position"]): s for s in slots}

    history = build_training_data()

    _ROUND_FLAGS = {
        "r32":    (1, 2),
        "r16":    (1, 2),
        "qf":     (1, 3),
        "sf":     (1, 4),
        "final":  (1, 5),
        "bronze": (1, 5),
    }
    _outcome_map = {"win": "home_win", "draw": "draw", "loss": "away_win"}

    def _store(anchor_slot, team_a_id, team_a_name, team_b_id, team_b_name, round_name):
        team_A = normalize_team_name(team_a_name)
        team_B = normalize_team_name(team_b_name)
        is_knockout, round_number = _ROUND_FLAGS.get(round_name, (1, 2))
        result = predict_match_with_model(
            _model, _feature_cols,
            team_A, team_B, history,
            is_knockout=is_knockout,
            round_number=round_number,
        )
        outcome_label = _outcome_map.get(result.get("outcome", "draw"), "draw")
        client.table("ml_predictions").delete().eq("slot_id", anchor_slot["id"]).execute()
        client.table("ml_predictions").insert({
            "slot_id":              anchor_slot["id"],
            "home_team_id":         team_a_id,
            "away_team_id":         team_b_id,
            "home_team_name":       team_a_name,
            "away_team_name":       team_b_name,
            "predicted_home_goals": round(result["lam_A"], 2),
            "predicted_away_goals": round(result["lam_B"], 2),
            "predicted_home_score": int(result["goals_A"]),
            "predicted_away_score": int(result["goals_B"]),
            "predicted_outcome":    outcome_label,
            "prob_home_win":        result["prob_home_win"],
            "prob_draw":            result["prob_draw_raw"],
            "prob_away_win":        result["prob_away_win"],
            "model_version":        MODEL_VERSION,
        }).execute()

    count = 0

    # R32 / R16 / QF / SF — pairs within the same (round, side)
    for round_name in ["r32", "r16", "qf", "sf"]:
        for side in ["left", "right"]:
            round_slots = sorted(
                [s for s in slots if s["round"] == round_name and s["side"] == side],
                key=lambda s: s["position"],
            )
            for i in range(0, len(round_slots) - 1, 2):
                slot_a = round_slots[i]
                slot_b = round_slots[i + 1]
                if not slot_a.get("home_team") or not slot_b.get("home_team"):
                    continue  # teams not known yet
                if slot_a.get("winner_team_id"):
                    continue  # already decided — no prediction needed
                _store(
                    slot_a,
                    slot_a["home_team"]["id"], slot_a["home_team"]["name"],
                    slot_b["home_team"]["id"], slot_b["home_team"]["name"],
                    round_name,
                )
                count += 1

    # Final — left side pos 0 vs right side pos 0
    left_final  = by_pos.get(("final", "left",  0))
    right_final = by_pos.get(("final", "right", 0))
    if (left_final and right_final
            and left_final.get("home_team") and right_final.get("home_team")
            and not left_final.get("winner_team_id")):
        _store(
            left_final,
            left_final["home_team"]["id"],  left_final["home_team"]["name"],
            right_final["home_team"]["id"], right_final["home_team"]["name"],
            "final",
        )
        count += 1

    # Bronze — position 0 vs position 1 (side may vary)
    bronze_slots = sorted(
        [s for s in slots if s["round"] == "bronze"],
        key=lambda s: s["position"],
    )
    if (len(bronze_slots) >= 2
            and bronze_slots[0].get("home_team") and bronze_slots[1].get("home_team")
            and not bronze_slots[0].get("winner_team_id")):
        _store(
            bronze_slots[0],
            bronze_slots[0]["home_team"]["id"], bronze_slots[0]["home_team"]["name"],
            bronze_slots[1]["home_team"]["id"], bronze_slots[1]["home_team"]["name"],
            "bronze",
        )
        count += 1

    print(f"[predict_knockout_slots] Wrote {count} knockout slot predictions")
    return count
