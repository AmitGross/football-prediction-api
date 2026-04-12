# poisson.py — Poisson score distribution with Dixon-Coles low-score correction

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10  # grid size: 0..MAX_GOALS for each team

# Dixon-Coles correction parameters (standard literature values)
# rho < 0 increases 0-0, 1-0, 0-1 and decreases 1-1 relative to independence
DC_RHO = -0.13

# Lambda amplification: RF/XGBoost regress λ toward the mean.
# Power transform amplifies the gap between teams without changing average goals.
# α=1.0 = no amplification, α=1.5 = moderate, α=2.0 = aggressive
LAM_ALPHA = 1.5


def _amplify_lambdas(lam_A, lam_B, alpha=LAM_ALPHA):
    """
    Power-scale λ around their mean to amplify mismatches.
    Preserves mean total goals; makes blowouts look like blowouts.
    """
    mu = (lam_A + lam_B) / 2.0
    if mu < 1e-6:
        return lam_A, lam_B
    lam_A_new = mu * (lam_A / mu) ** alpha
    lam_B_new = mu * (lam_B / mu) ** alpha
    return max(lam_A_new, 1e-6), max(lam_B_new, 1e-6)


def _dc_correction(i, j, lam_A, lam_B, rho=DC_RHO):
    """
    Dixon-Coles correction factor tau(i, j).
    Only applied to low scores: (0,0), (1,0), (0,1), (1,1).
    """
    if i == 0 and j == 0:
        return 1 - lam_A * lam_B * rho
    elif i == 1 and j == 0:
        return 1 + lam_B * rho
    elif i == 0 and j == 1:
        return 1 + lam_A * rho
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1.0


def score_grid(lam_A, lam_B, max_goals=MAX_GOALS, rho=DC_RHO):
    """
    Returns a (max_goals+1) x (max_goals+1) matrix where
    grid[i][j] = P(team_A scores i, team_B scores j).

    Uses independent Poisson with Dixon-Coles correction on low scores.
    """
    lam_A = max(lam_A, 1e-6)
    lam_B = max(lam_B, 1e-6)

    goals = np.arange(max_goals + 1)
    p_A   = poisson.pmf(goals, lam_A)   # shape (max_goals+1,)
    p_B   = poisson.pmf(goals, lam_B)

    grid = np.outer(p_A, p_B)           # grid[i,j] = P(A=i) * P(B=j)

    # Apply Dixon-Coles correction to low-score cells
    for i in range(2):
        for j in range(2):
            grid[i, j] *= _dc_correction(i, j, lam_A, lam_B, rho)

    # Renormalise so probabilities sum to 1
    grid /= grid.sum()
    return grid


def result_probabilities(grid):
    """
    Returns (p_win_A, p_draw, p_win_B) from a score grid.
    """
    n = grid.shape[0]
    p_win_A = float(np.sum(np.tril(grid, k=-1)))   # i > j
    p_win_B = float(np.sum(np.triu(grid, k=1)))    # j > i
    p_draw  = float(np.trace(grid))                # i == j
    return p_win_A, p_draw, p_win_B


def most_likely_score(grid):
    """
    Returns (goals_A, goals_B) of the highest-probability scoreline.
    """
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return int(idx[0]), int(idx[1])


def predict_from_lambdas(lam_A, lam_B):
    """
    Full prediction from expected goals.
    Returns dict with:
      - goals_A, goals_B  : most likely scoreline
      - prob_score        : probability of that exact score
      - p_win_A, p_draw, p_win_B : outcome probabilities
      - lam_A, lam_B     : expected goals (passed through)
      - grid             : full (MAX_GOALS+1 x MAX_GOALS+1) probability matrix
    """
    lam_A_amp, lam_B_amp = _amplify_lambdas(lam_A, lam_B)
    grid               = score_grid(lam_A, lam_B)   # real λ for probabilities
    # Amplified λ for displayed score only — more realistic scorelines
    goals_A = int(round(lam_A_amp))
    goals_B = int(round(lam_B_amp))
    p_win_A, p_draw, p_win_B = result_probabilities(grid)
    prob_score         = float(grid[goals_A, goals_B])

    return {
        'goals_A':    goals_A,
        'goals_B':    goals_B,
        'prob_score': prob_score,
        'p_win_A':    p_win_A,
        'p_draw':     p_draw,
        'p_win_B':    p_win_B,
        'lam_A':      lam_A,
        'lam_B':      lam_B,
        'grid':       grid,
    }
