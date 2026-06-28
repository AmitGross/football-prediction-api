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

    # ── Knockout stage: update knockout_slots ─────────────────────────────────
    ko_newly_finished = _update_knockout_slots(client, tournament_id, fd_matches, name_to_id)
    newly_finished.extend(ko_newly_finished)

    print(f"[fetch_scores] Done. {len(newly_finished)} newly finished match(es) "
          f"({len(ko_newly_finished)} knockout).")
    return newly_finished


# fd.org stage → our knockout_slots round key
_FD_STAGE_TO_ROUND: dict[str, str] = {
    "ROUND_OF_32":   "r32",
    "LAST_16":       "r16",
    "QUARTER_FINAL": "qf",
    "SEMI_FINAL":    "sf",
    "FINAL":         "final",
    "THIRD_PLACE":   "bronze",
}


def _update_knockout_slots(client, tournament_id: str, fd_matches: list[dict], name_to_id: dict) -> list[str]:
    """
    For each live/finished knockout match on fd.org, find the matching
    knockout_slot pair and write scores + winner_team_id.
    Returns list of slot IDs (slotA) that transitioned to finished this call.
    """
    # Load all slots that have a team assigned
    slots_raw = (
        client.table("knockout_slots")
        .select("id, round, side, position, home_team_id, winner_team_id")
        .eq("tournament_id", tournament_id)
        .not_.is_("home_team_id", "null")
        .execute()
    )
    all_slots: list[dict] = slots_raw.data or []

    # Build: (round_key, team_id) → slot  (only even/slotA positions drive scores)
    even_slots: dict[tuple, dict] = {}
    slot_by_pos: dict[tuple, dict] = {}
    for s in all_slots:
        key = (s["round"], s["side"], s["position"])
        slot_by_pos[key] = s
        if s["position"] % 2 == 0:
            even_slots[(s["round"], s["home_team_id"])] = s

    newly_finished: list[str] = []

    for m in fd_matches:
        round_key = _FD_STAGE_TO_ROUND.get(m.get("stage", ""))
        if not round_key:
            continue

        fd_status = m["status"]
        if fd_status not in ("FINISHED", "IN_PLAY", "PAUSED", "HALFTIME", "EXTRA_TIME", "PENALTY"):
            continue

        home_name = _normalise(m["homeTeam"].get("name", ""))
        away_name = _normalise(m["awayTeam"].get("name", ""))
        home_id   = name_to_id.get(home_name)
        away_id   = name_to_id.get(away_name)
        if not home_id or not away_id:
            continue

        # Scores: use fullTime (90 min). For KO draws, also check extraTime.
        score      = m.get("score", {})
        ft         = score.get("fullTime", {})
        et         = score.get("extraTime", {}) or {}
        pens       = score.get("penalties", {}) or {}

        home_score_90 = ft.get("home")
        away_score_90 = ft.get("away")

        # Extra time goals (additive on top of 90 min)
        home_score_et = et.get("home")
        away_score_et = et.get("away")
        if home_score_et is not None and away_score_et is not None:
            home_full = (home_score_90 or 0) + home_score_et
            away_full = (away_score_90 or 0) + away_score_et
        else:
            home_full = home_score_90
            away_full = away_score_90

        # Find the slotA (even position) for this match
        slot_a = even_slots.get((round_key, home_id))
        swapped = False
        if slot_a is None:
            # Try with teams swapped (fd.org may list home/away differently to our slots)
            slot_a = even_slots.get((round_key, away_id))
            if slot_a:
                swapped = True
                home_id, away_id   = away_id, home_id
                home_score_90, away_score_90 = away_score_90, home_score_90
                if home_full is not None and away_full is not None:
                    home_full, away_full = away_full, home_full
                pens = {k: v for k, v in pens.items()}  # copy; swap below
                pens["home"], pens["away"] = pens.get("away"), pens.get("home")

        if slot_a is None:
            continue  # match teams not yet in our slots — populate_knockouts will handle

        # Determine winner
        winner_id: str | None = None
        if fd_status == "FINISHED":
            if home_full is not None and away_full is not None:
                if home_full > away_full:
                    winner_id = home_id
                elif away_full > home_full:
                    winner_id = away_id
                else:
                    # Penalties
                    ph = pens.get("home") or 0
                    pa = pens.get("away") or 0
                    winner_id = home_id if ph > pa else away_id

        prev_winner = slot_a.get("winner_team_id")

        update: dict = {}
        if home_score_90 is not None:
            update["home_score"] = home_score_90
        if away_score_90 is not None:
            update["away_score"] = away_score_90
        if winner_id:
            update["winner_team_id"] = winner_id

        if update:
            client.table("knockout_slots").update(update).eq("id", slot_a["id"]).execute()
            print(f"[fetch_scores_ko] {round_key}: {home_name} {home_score_90}-{away_score_90} "
                  f"{away_name}{' (winner: ' + (home_name if winner_id == home_id else away_name) + ')' if winner_id else ''}")

        if fd_status == "FINISHED" and not prev_winner and winner_id:
            newly_finished.append(slot_a["id"])

    return newly_finished
