# train.py — production version for Render
# Always uses FIFA 2026 rankings (bundled data/fifa_rankings_2026.csv).
# Training data = historical matches.csv  +  finished WC 2026 matches from Supabase.

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import features as feat_module

# Production always targets WC 2026 — set before any feature building.
feat_module.set_fifa_rankings_year(2026)

from ensemble import AveragingEnsemble
from features import build_features

MODEL_VERSION   = "v1.6"
PARAMS_PATH     = os.path.join(os.path.dirname(__file__), "best_params.json")
HISTORICAL_DATA = os.path.join(os.path.dirname(__file__), "data", "matches.csv")


# ── Hyperparameters ────────────────────────────────────────────────────────────

def _load_best_params() -> dict:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return {}


def _apply_feature_params(p: dict) -> None:
    if "elo_k" in p:
        feat_module.ELO_K = p["elo_k"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_historical_data() -> pd.DataFrame:
    df = pd.read_csv(HISTORICAL_DATA, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[train] Loaded {len(df)} historical matches from {HISTORICAL_DATA}")
    return df


def fetch_wc2026_results() -> pd.DataFrame:
    """
    Pull all finished WC 2026 group-stage matches from Supabase.
    Returns DataFrame with columns: date, team_A, team_B, goals_A, goals_B, round
    (round='group' so build_features picks up stage context features).
    """
    from supabase_client import get_client, normalize_team_name

    client = get_client()

    # Resolve WC 2026 tournament id
    t = client.table("tournaments").select("id").eq("year", 2026).single().execute()
    tournament_id = t.data["id"]

    # Finished matches with resolved team names
    result = (
        client.table("matches")
        .select(
            "starts_at, home_score_90, away_score_90, stage, "
            "home_team:home_team_id(name), away_team:away_team_id(name)"
        )
        .eq("tournament_id", tournament_id)
        .eq("status", "finished")
        .execute()
    )

    # Stage → round string expected by build_features
    _STAGE_TO_ROUND = {
        "GROUP":  "group",
        "R32":    "r32",
        "R16":    "r16",
        "QF":     "qf",
        "SF":     "sf",
        "FINAL":  "final",
        "BRONZE": "3rd",
    }

    rows = []
    for m in result.data:
        rows.append(
            {
                "date":    pd.Timestamp(m["starts_at"]).tz_localize(None).normalize(),
                "team_A":  normalize_team_name(m["home_team"]["name"]),
                "team_B":  normalize_team_name(m["away_team"]["name"]),
                "goals_A": int(m["home_score_90"]),
                "goals_B": int(m["away_score_90"]),
                "round":   _STAGE_TO_ROUND.get(m["stage"], "group"),
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    print(f"[train] Fetched {len(df)} finished WC 2026 matches from Supabase")
    return df


def build_training_data() -> pd.DataFrame:
    """Combine historical matches.csv with WC 2026 results from Supabase."""
    hist = load_historical_data()
    wc26 = fetch_wc2026_results()

    if len(wc26) > 0:
        df = pd.concat([hist, wc26], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
        print(f"[train] Combined: {len(df)} matches ({len(wc26)} WC 2026 results added)")
    else:
        df = hist
        print("[train] No WC 2026 results yet — training on historical data only")

    return df


# ── Model training ─────────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    p = _load_best_params()
    _apply_feature_params(p)

    print("[train] Building features...")
    X, y_goals_A, y_goals_B = build_features(df)
    Y = np.column_stack([y_goals_A, y_goals_B])
    print(f"[train] Feature matrix: {X.shape}")

    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=p.get("rf_n_estimators", 200),
            max_depth=p.get("rf_max_depth", 6),
            random_state=42,
            n_jobs=-1,
        )
    )
    xgb = MultiOutputRegressor(
        XGBRegressor(
            objective="count:poisson",
            n_estimators=p.get("xgb_n_estimators", 300),
            max_depth=p.get("xgb_max_depth", 5),
            learning_rate=p.get("xgb_learning_rate", 0.05),
            subsample=p.get("xgb_subsample", 0.8),
            colsample_bytree=p.get("xgb_colsample", 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    )

    print("[train] Training Random Forest...")
    rf.fit(X, Y)
    print("[train] Training XGBoost...")
    xgb.fit(X, Y)

    model = AveragingEnsemble([rf, xgb])
    print("[train] Ensemble trained (RF + XGBoost).")
    return model, X.columns.tolist()


def retrain_and_upload():
    """
    Full retrain pipeline:
      1. Build training data (historical CSV + WC 2026 results from Supabase)
      2. Train ensemble
      3. Upload model.pkl to Supabase Storage
    Returns (model, feature_cols).
    """
    from model_store import upload_model

    df = build_training_data()
    model, feature_cols = train(df)
    upload_model(model, feature_cols, MODEL_VERSION)
    return model, feature_cols


# ── Build-time entrypoint (Render build command) ───────────────────────────────
if __name__ == "__main__":
    # Runs during Render build: train on historical data only (no WC results yet)
    # and upload model so the API can download it at startup.
    from model_store import upload_model

    df = load_historical_data()
    model, feature_cols = train(df)
    upload_model(model, feature_cols, MODEL_VERSION)
