# fetch_scores.py — fetch live / final WC 2026 scores from football-data.org
# Called by /fetch-scores endpoint every 5 min during the tournament.

import os
import requests

FOOTBALL_DATA_API_KEY = os.environ["FOOTBALL_DATA_API_KEY"]
WC_COMPETITION_ID     = "2000"          # FIFA World Cup on football-data.org v4
BASE_URL              = "https://api.football-data.org/v4"
HEADERS               = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

# football-data.org team name → Supabase team name (before normalize_team_name runs)
_FDORG_TO_SUPABASE: dict[str, str] = {
    "Korea Republic":      "South Korea",
    "Bosnia-Herzegovina":  "Bosnia and Herzegovina",
    "USA":                 "United States",
    "IR Iran":             "Iran",
    "Côte d'Ivoire":       "Ivory Coast",
    "Curacao":             "Curaçao",       # football-data may use either spelling
}


def _normalise(name: str) -> str:
    """Map football-data.org names → Supabase names."""
    return _FDORG_TO_SUPABASE.get(name, name)


def fetch_and_update() -> list[str]:
    """
    Fetch all WC 2026 matches from football-data.org.
    Update Supabase matches table (status + scores) for live and finished matches.
    Returns a list of match IDs that transitioned to 'finished' in this call.
    """
    from supabase_client import get_client

    client = get_client()

    # Resolve WC 2026 tournament id
    t = client.table("tournaments").select("id").eq("year", 2026).single().execute()
    tournament_id = t.data["id"]

    # Build team name → Supabase UUID map
    teams_result = client.table("teams").select("id, name").execute()
    name_to_id: dict[str, str] = {row["name"]: row["id"] for row in teams_result.data}

    # Fetch matches from football-data.org
    resp = requests.get(
        f"{BASE_URL}/competitions/{WC_COMPETITION_ID}/matches",
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    fd_matches = resp.json().get("matches", [])

    newly_finished: list[str] = []

    for m in fd_matches:
        fd_status  = m["status"]
        home_name  = _normalise(m["homeTeam"]["name"])
        away_name  = _normalise(m["awayTeam"]["name"])

        home_id = name_to_id.get(home_name)
        away_id = name_to_id.get(away_name)

        if not home_id or not away_id:
            # Team not in our WC 2026 tournament (e.g. qualifier)
            continue

        if fd_status == "FINISHED":
            status     = "finished"
            home_score = m["score"]["fullTime"]["home"]
            away_score = m["score"]["fullTime"]["away"]
        elif fd_status in ("IN_PLAY", "PAUSED", "HALFTIME", "EXTRA_TIME", "PENALTY"):
            status     = "live"
            home_score = m["score"]["fullTime"].get("home") or m["score"]["halfTime"].get("home", 0)
            away_score = m["score"]["fullTime"].get("away") or m["score"]["halfTime"].get("away", 0)
        else:
            continue  # SCHEDULED / TIMED — nothing to update yet

        # Find the corresponding Supabase match row
        existing = (
            client.table("matches")
            .select("id, status")
            .eq("tournament_id", tournament_id)
            .eq("home_team_id", home_id)
            .eq("away_team_id", away_id)
            .maybe_single()
            .execute()
        )

        if not existing.data:
            continue

        prev_status = existing.data["status"]
        match_id    = existing.data["id"]

        # Update score and status in Supabase
        client.table("matches").update(
            {
                "status":        status,
                "home_score_90": home_score,
                "away_score_90": away_score,
            }
        ).eq("id", match_id).execute()

        if status == "finished" and prev_status != "finished":
            newly_finished.append(match_id)
            print(f"[fetch_scores] Finished: {home_name} {home_score}-{away_score} {away_name}")

    print(f"[fetch_scores] Done. {len(newly_finished)} newly finished match(es).")
    return newly_finished
