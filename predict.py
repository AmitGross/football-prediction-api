# predict.py — production version for Render
# predict_match_with_model() accepts a pre-loaded model to avoid repeated file I/O.

import pandas as pd
import numpy as np

from poisson import predict_from_lambdas, score_grid, result_probabilities
from ensemble import AveragingEnsemble  # required for pickle deserialisation
import features as _features_module
from features import (
    EloRating,
    calculate_form_features,
    calculate_h2h,
    calculate_days_rest,
    calculate_neighbourhood_features,
    calculate_goal_diff_std,
    FORM_N,
)


def _build_feature_row(team_A, team_B, current_matches,
                       is_knockout=0, round_number=0,
                       games_in_tournament_A=0, games_in_tournament_B=0) -> dict:
    """
    Build the full feature dict for (team_A, team_B) from current_matches history.
    Shared by predict_match() and predict_match_with_model().
    """
    current_matches = current_matches.sort_values("date").reset_index(drop=True)

    # Replay all past matches to build current Elo state
    elo_system = EloRating()
    for _, row in current_matches.iterrows():
        elo_system.update(row["team_A"], row["team_B"], row["goals_A"], row["goals_B"])

    elo_A, elo_B, elo_diff = elo_system.get_ratings(team_A, team_B)

    wins_A,  draws_A,  losses_A,  scored_A,  conc_A,  wscored_A,  wconc_A  = calculate_form_features(current_matches, team_A, FORM_N, elo_system.ratings)
    wins_B,  draws_B,  losses_B,  scored_B,  conc_B,  wscored_B,  wconc_B  = calculate_form_features(current_matches, team_B, FORM_N, elo_system.ratings)
    wins_A2, draws_A2, losses_A2, scored_A2, conc_A2, wscored_A2, wconc_A2 = calculate_form_features(current_matches, team_A, 2, elo_system.ratings)
    wins_B2, draws_B2, losses_B2, scored_B2, conc_B2, wscored_B2, wconc_B2 = calculate_form_features(current_matches, team_B, 2, elo_system.ratings)

    h2h_wins, h2h_draws, h2h_losses = calculate_h2h(current_matches, team_A, team_B)

    current_date = current_matches["date"].max() if len(current_matches) > 0 else pd.Timestamp.now()
    rest_A = calculate_days_rest(current_matches, team_A, current_date)
    rest_B = calculate_days_rest(current_matches, team_B, current_date)

    nbr_A = calculate_neighbourhood_features(current_matches, team_A, elo_system.ratings)
    nbr_B = calculate_neighbourhood_features(current_matches, team_B, elo_system.ratings)

    goal_diff_std_A = calculate_goal_diff_std(current_matches, team_A)
    goal_diff_std_B = calculate_goal_diff_std(current_matches, team_B)

    return {
        "elo_A":        elo_A,
        "elo_B":        elo_B,
        "elo_diff":     elo_diff,
        "wins_A":       wins_A,   "draws_A":  draws_A,  "losses_A": losses_A,
        "scored_A":     scored_A, "conc_A":   conc_A,
        "wscored_A":    wscored_A,"wconc_A":  wconc_A,
        "wins_B":       wins_B,   "draws_B":  draws_B,  "losses_B": losses_B,
        "scored_B":     scored_B, "conc_B":   conc_B,
        "wscored_B":    wscored_B,"wconc_B":  wconc_B,
        "wins_A2":      wins_A2,  "draws_A2": draws_A2, "losses_A2": losses_A2,
        "scored_A2":    scored_A2,"conc_A2":  conc_A2,
        "wscored_A2":   wscored_A2,"wconc_A2": wconc_A2,
        "wins_B2":      wins_B2,  "draws_B2": draws_B2, "losses_B2": losses_B2,
        "scored_B2":    scored_B2,"conc_B2":  conc_B2,
        "wscored_B2":   wscored_B2,"wconc_B2": wconc_B2,
        "h2h_wins":     h2h_wins, "h2h_draws": h2h_draws, "h2h_losses": h2h_losses,
        "rest_A":         rest_A,        "rest_B":         rest_B,
        "match_count_A":  int(((current_matches["team_A"] == team_A) | (current_matches["team_B"] == team_A)).sum()),
        "match_count_B":  int(((current_matches["team_A"] == team_B) | (current_matches["team_B"] == team_B)).sum()),
        "fifa_rank_A":    _features_module._FIFA_RANKINGS.get(team_A, _features_module._FIFA_DEFAULT),
        "fifa_rank_B":    _features_module._FIFA_RANKINGS.get(team_B, _features_module._FIFA_DEFAULT),
        "fifa_rank_diff": _features_module._FIFA_RANKINGS.get(team_A, _features_module._FIFA_DEFAULT)
                        - _features_module._FIFA_RANKINGS.get(team_B, _features_module._FIFA_DEFAULT),
        "opp_elo_A":      nbr_A["avg_opp_elo"],       "opp_elo_B":      nbr_B["avg_opp_elo"],
        "opp_elo_diff":   nbr_A["avg_opp_elo"]       - nbr_B["avg_opp_elo"],
        "opp_scored_A":   nbr_A["avg_opp_scored"],   "opp_scored_B":   nbr_B["avg_opp_scored"],
        "opp_conceded_A": nbr_A["avg_opp_conceded"], "opp_conceded_B": nbr_B["avg_opp_conceded"],
        "n_opps_A":       nbr_A["n_opponents"],        "n_opps_B":       nbr_B["n_opponents"],
        "weighted_opp_elo_A":        nbr_A["weighted_opp_elo"],
        "weighted_opp_elo_B":        nbr_B["weighted_opp_elo"],
        "weighted_opp_elo_diff":     nbr_A["weighted_opp_elo"]          - nbr_B["weighted_opp_elo"],
        "win_rate_vs_top_A":         nbr_A["win_rate_vs_top_teams"],
        "win_rate_vs_top_B":         nbr_B["win_rate_vs_top_teams"],
        "win_rate_vs_top_diff":      nbr_A["win_rate_vs_top_teams"]     - nbr_B["win_rate_vs_top_teams"],
        "avg_goal_diff_vs_opp_A":    nbr_A["avg_goal_diff_vs_opp"],
        "avg_goal_diff_vs_opp_B":    nbr_B["avg_goal_diff_vs_opp"],
        "avg_goal_diff_vs_opp_diff": nbr_A["avg_goal_diff_vs_opp"]     - nbr_B["avg_goal_diff_vs_opp"],
        "wtd_goal_diff_opp_A":       nbr_A["weighted_goal_diff_by_opp"],
        "wtd_goal_diff_opp_B":       nbr_B["weighted_goal_diff_by_opp"],
        "wtd_goal_diff_opp_diff":    nbr_A["weighted_goal_diff_by_opp"] - nbr_B["weighted_goal_diff_by_opp"],
        "is_knockout":               is_knockout,
        "round_number":              round_number,
        "games_in_tournament_A":     games_in_tournament_A,
        "games_in_tournament_B":     games_in_tournament_B,
        "goal_diff_std_A":           goal_diff_std_A,
        "goal_diff_std_B":           goal_diff_std_B,
    }


def _run_prediction(model, feature_cols, row_dict) -> dict:
    """Run model inference on a pre-built feature row dict."""
    X = pd.DataFrame([row_dict])[feature_cols]
    pred  = model.predict(X)[0]
    lam_A = max(float(pred[0]), 0.0)
    lam_B = max(float(pred[1]), 0.0)

    display = predict_from_lambdas(lam_A, lam_B)
    grid    = score_grid(lam_A, lam_B)
    p_win_f, p_draw_f, p_loss_f = result_probabilities(grid)

    if p_win_f >= p_draw_f and p_win_f >= p_loss_f:
        outcome = "win"
    elif p_draw_f >= p_win_f and p_draw_f >= p_loss_f:
        outcome = "draw"
    else:
        outcome = "loss"

    return {
        "goals_A":    display["goals_A"],
        "goals_B":    display["goals_B"],
        "outcome":    outcome,
        "p_win_A":    round(p_win_f  * 100, 1),
        "p_draw":     round(p_draw_f * 100, 1),
        "p_win_B":    round(p_loss_f * 100, 1),
        "prob_score": round(display["prob_score"] * 100, 1),
        "lam_A":      round(lam_A, 2),
        "lam_B":      round(lam_B, 2),
        # Raw probs for Supabase storage
        "prob_home_win": round(p_win_f,  4),
        "prob_draw_raw": round(p_draw_f, 4),
        "prob_away_win": round(p_loss_f, 4),
    }


def predict_match_with_model(model, feature_cols, team_A, team_B, current_matches,
                              is_knockout=0, round_number=0,
                              games_in_tournament_A=0, games_in_tournament_B=0) -> dict:
    """
    Predict a match using an already-loaded model (no file I/O).
    Used by app.py _predict_all_remaining() to avoid repeated pickle loads.
    """
    row = _build_feature_row(
        team_A, team_B, current_matches,
        is_knockout, round_number,
        games_in_tournament_A, games_in_tournament_B,
    )
    result = _run_prediction(model, feature_cols, row)
    return {"team_A": team_A, "team_B": team_B, **result}
