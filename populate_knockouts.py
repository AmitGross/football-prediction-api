# populate_knockouts.py
# Reads group-stage results from Supabase and auto-fills knockout_slots:
#   - R32: maps slot_label ("1A", "2B", "3 ABCDF") → real team UUID
#   - R16 → QF → SF → Final / Bronze: propagates winner_team_id upward

from supabase_client import get_client


# ── Group standings ───────────────────────────────────────────────────────────

def _compute_group_standings(matches: list[dict]) -> dict[str, list[dict]]:
    """
    Build group standings from finished group-stage matches.
    Returns {group_name: [team sorted by pts↓ gd↓ gf↓ name↑]}.
    Each team dict: id, name, pts, gd, gf.
    """
    groups: dict[str, dict[str, dict]] = {}

    for m in matches:
        g = m.get("group_name")
        if not g:
            continue

        home = m["home_team"]
        away = m["away_team"]
        hs   = m.get("home_score_90")
        as_  = m.get("away_score_90")

        for team in (home, away):
            groups.setdefault(g, {})
            if team["id"] not in groups[g]:
                groups[g][team["id"]] = {
                    "id":   team["id"],
                    "name": team["name"],
                    "pts":  0, "gd": 0, "gf": 0,
                }

        if hs is None or as_ is None:
            continue  # match not yet played

        groups[g][home["id"]]["gf"] += hs
        groups[g][home["id"]]["gd"] += hs - as_
        groups[g][away["id"]]["gf"] += as_
        groups[g][away["id"]]["gd"] += as_ - hs

        if hs > as_:
            groups[g][home["id"]]["pts"] += 3
        elif hs < as_:
            groups[g][away["id"]]["pts"] += 3
        else:
            groups[g][home["id"]]["pts"] += 1
            groups[g][away["id"]]["pts"] += 1

    return {
        g: sorted(teams.values(), key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["name"]))
        for g, teams in groups.items()
    }


# ── Label parsing ─────────────────────────────────────────────────────────────

def _parse_label(label: str | None) -> tuple[int, list[str]]:
    """
    "1A"      → (1, ["A"])
    "2B"      → (2, ["B"])
    "3 ABCDF" → (3, ["A","B","C","D","F"])
    ""  / None → (0, [])
    """
    if not label:
        return 0, []
    label = label.strip()
    if len(label) == 2 and label[0].isdigit() and label[1].isalpha():
        return int(label[0]), [label[1].upper()]
    if label.startswith("3 "):
        return 3, list(label[2:].upper())
    return 0, []


# ── Main ──────────────────────────────────────────────────────────────────────

def populate_knockouts() -> dict:
    """
    Populate knockout_slots with real teams and advance bracket winners.
    Safe to call repeatedly — skips slots that already have the correct team.

    Returns a summary dict.
    """
    client = get_client()

    # Active tournament
    t = client.table("tournaments").select("id").eq("is_active", True).single().execute()
    tournament_id = t.data["id"]

    # ── 1. Group stage data ───────────────────────────────────────────────────
    raw = (
        client.table("matches")
        .select(
            "group_name, home_score_90, away_score_90, "
            "home_team:home_team_id(id, name), away_team:away_team_id(id, name)"
        )
        .eq("tournament_id", tournament_id)
        .eq("stage", "group")
        .execute()
    )
    group_matches = raw.data or []

    standings = _compute_group_standings(group_matches)

    if not standings:
        return {"error": "No group stage data found", "r32_updated": 0, "advanced": 0}

    # ── 2. Best 8 third-place teams ───────────────────────────────────────────
    thirds = []
    for group_name, teams in standings.items():
        if len(teams) >= 3:
            t3 = dict(teams[2])
            t3["group"] = group_name
            thirds.append(t3)

    qualifying_thirds = sorted(
        thirds, key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["name"])
    )[:8]

    # group → team_id for the 8 qualifiers
    thirds_by_group = {t["group"]: t["id"] for t in qualifying_thirds}

    print(f"[populate_knockouts] Qualifying 3rd-place teams:")
    for t in qualifying_thirds:
        print(f"  Group {t['group']}: {t['name']} ({t['pts']}pts, GD{t['gd']:+d})")

    # ── 3. Load all knockout slots ────────────────────────────────────────────
    slots_raw = (
        client.table("knockout_slots")
        .select("id, round, side, position, slot_label, home_team_id, winner_team_id")
        .eq("tournament_id", tournament_id)
        .execute()
    )
    all_slots: list[dict] = slots_raw.data or []

    # Fast lookup: (round, side, position) → slot
    slot_index: dict[tuple, dict] = {
        (s["round"], s["side"], s["position"]): s
        for s in all_slots
    }

    r32_updated = 0
    advanced    = 0

    # ── 4. Populate R32 from slot_label ──────────────────────────────────────
    for s in all_slots:
        if s["round"] != "r32":
            continue

        rank, groups = _parse_label(s.get("slot_label"))
        if rank == 0:
            continue

        team_id: str | None = None

        if rank in (1, 2):
            # e.g. "1A" → 1st place in group A
            grp_teams = standings.get(groups[0], [])
            if len(grp_teams) >= rank:
                team_id = grp_teams[rank - 1]["id"]

        # rank == 3 handled separately via bipartite matching below

        if team_id and team_id != s.get("home_team_id"):
            client.table("knockout_slots").update({"home_team_id": team_id}).eq("id", s["id"]).execute()
            r32_updated += 1
            print(f"[populate_knockouts] R32 {s['side']} pos {s['position']} "
                  f"({s['slot_label']}) → {team_id}")
    # ── 4b. Bipartite matching for "3 XXXXX" slots ───────────────────────────
    # Each "3 XXXXX" slot must receive exactly one qualifying 3rd-place team whose
    # group appears in the label. Use augmenting-path matching to avoid duplicates.
    third_slots = [
        s for s in all_slots
        if s["round"] == "r32" and (s.get("slot_label") or "").strip().startswith("3 ")
    ]
    # Build candidate map: slot_id → set of team_ids that could fill it
    slot_candidates: dict[str, list[str]] = {}
    for s in third_slots:
        _, groups = _parse_label(s.get("slot_label"))
        candidates = [thirds_by_group[g] for g in groups if g in thirds_by_group]
        slot_candidates[s["id"]] = candidates

    # Augmenting-path bipartite matching
    # match_slot[slot_id] = team_id currently assigned
    # match_team[team_id] = slot_id currently assigned
    match_slot: dict[str, str] = {}
    match_team: dict[str, str] = {}

    def _try_augment(slot_id: str, visited: set) -> bool:
        for team_id in slot_candidates.get(slot_id, []):
            if team_id in visited:
                continue
            visited.add(team_id)
            prev_slot = match_team.get(team_id)
            if prev_slot is None or _try_augment(prev_slot, visited):
                match_slot[slot_id] = team_id
                match_team[team_id] = slot_id
                return True
        return False

    for s in third_slots:
        _try_augment(s["id"], set())

    for s in third_slots:
        team_id = match_slot.get(s["id"])
        if team_id and team_id != s.get("home_team_id"):
            client.table("knockout_slots").update({"home_team_id": team_id}).eq("id", s["id"]).execute()
            r32_updated += 1
            print(f"[populate_knockouts] R32 {s['side']} pos {s['position']} "
                  f"({s['slot_label']}) → {team_id}")
    # ── 5. Advance winners through bracket ───────────────────────────────────
    # Refresh slots after R32 update so winner_team_id info is current
    slots_raw2 = (
        client.table("knockout_slots")
        .select("id, round, side, position, home_team_id, winner_team_id")
        .eq("tournament_id", tournament_id)
        .execute()
    )
    all_slots = slots_raw2.data or []
    slot_index = {(s["round"], s["side"], s["position"]): s for s in all_slots}

    for s in all_slots:
        round_ = s["round"]
        side   = s["side"]
        pos    = s["position"]
        winner = s.get("winner_team_id")

        if not winner:
            continue
        if pos % 2 != 0:
            continue  # slotB (odd) — winner always stored on slotA (even)

        next_pos = pos // 2

        # Standard rounds: r32 → r16 → qf → sf
        if round_ in ("r32", "r16", "qf"):
            next_round = {"r32": "r16", "r16": "qf", "qf": "sf"}[round_]
            key = (next_round, side, next_pos)
            target = slot_index.get(key)
            if target and target.get("home_team_id") != winner:
                client.table("knockout_slots").update({"home_team_id": winner}).eq("id", target["id"]).execute()
                advanced += 1
                print(f"[populate_knockouts] {round_} {side} pos {pos} winner → "
                      f"{next_round} {side} pos {next_pos}")

        elif round_ == "sf":
            # Winner → Final (same side, pos 0)
            final_key = ("final", side, 0)
            target = slot_index.get(final_key)
            if target and target.get("home_team_id") != winner:
                client.table("knockout_slots").update({"home_team_id": winner}).eq("id", target["id"]).execute()
                advanced += 1
                print(f"[populate_knockouts] SF {side} winner → Final {side}")

            # Loser → Bronze
            # The other team in the SF pair is at (sf, side, pos+1)
            slotB = slot_index.get((round_, side, pos + 1))
            if slotB:
                loser = slotB.get("home_team_id")
                if loser and loser != winner:
                    # Bronze: side "left" pos 0 (left SF loser), pos 1 (right SF loser)
                    bronze_pos = 0 if side == "left" else 1
                    bronze_key = ("bronze", "left", bronze_pos)
                    target = slot_index.get(bronze_key)
                    if target and target.get("home_team_id") != loser:
                        client.table("knockout_slots").update({"home_team_id": loser}).eq("id", target["id"]).execute()
                        advanced += 1
                        print(f"[populate_knockouts] SF {side} loser → Bronze pos {bronze_pos}")

    # ── 6. Upsert matches table entries for each slot pair ───────────────────
    # So that _predict_all_remaining() in app.py predicts knockout matches
    # and ml_predictions gets populated (same as group stage flow).
    matches_upserted = _upsert_slot_matches(client, tournament_id, slot_index)

    summary = {
        "status":           "ok",
        "r32_updated":      r32_updated,
        "advanced":         advanced,
        "matches_upserted": matches_upserted,
        "qualifying_thirds": [
            {"group": t["group"], "name": t["name"], "pts": t["pts"], "gd": t["gd"], "gf": t["gf"]}
            for t in qualifying_thirds
        ],
    }
    print(f"[populate_knockouts] Done — R32: {r32_updated}, advanced: {advanced}, matches upserted: {matches_upserted}")
    return summary


def _upsert_slot_matches(client, tournament_id: str, slot_index: dict) -> int:
    """
    For every knockout slot pair that has both teams assigned, ensure a row
    exists in the `matches` table (stage = round key, status = 'scheduled').
    This lets _predict_all_remaining() generate ml_predictions for knockout matches.
    Returns count of newly inserted matches.
    """
    from datetime import datetime, timezone

    PAIRABLE = ["r32", "r16", "qf", "sf"]
    inserted = 0

    # Build existing matches lookup: (home_team_id, away_team_id) → match_id
    existing_raw = (
        client.table("matches")
        .select("id, home_team_id, away_team_id")
        .eq("tournament_id", tournament_id)
        .in_("stage", PAIRABLE + ["final", "bronze"])
        .execute()
    )
    existing: dict[tuple, str] = {
        (r["home_team_id"], r["away_team_id"]): r["id"]
        for r in (existing_raw.data or [])
    }

    for round_ in PAIRABLE + ["final", "bronze"]:
        sides = ["left", "right"] if round_ not in ("final", "bronze") else ["left"]

        for side in sides:
            # Collect even-position slots for this round+side
            even_positions = sorted(
                pos for (r, s, pos) in slot_index if r == round_ and s == side and pos % 2 == 0
            )

            for even_pos in even_positions:
                slot_a = slot_index.get((round_, side, even_pos))
                slot_b = slot_index.get((round_, side, even_pos + 1))

                if not slot_a or not slot_b:
                    continue

                home_id = slot_a.get("home_team_id")
                away_id = slot_b.get("home_team_id")

                if not home_id or not away_id:
                    continue  # teams not yet determined

                if (home_id, away_id) in existing:
                    continue  # already in matches table

                # Determine kickoff time
                match_date = slot_a.get("match_date")
                starts_at = match_date if match_date else datetime.now(timezone.utc).isoformat()

                client.table("matches").insert({
                    "tournament_id": tournament_id,
                    "stage":         round_,
                    "home_team_id":  home_id,
                    "away_team_id":  away_id,
                    "starts_at":     starts_at,
                    "status":        "scheduled",
                }).execute()

                existing[(home_id, away_id)] = "inserted"
                inserted += 1
                print(f"[populate_knockouts] Created match: {round_} {side} pos {even_pos}/{even_pos+1}")

    return inserted
