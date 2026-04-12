# supabase_client.py — shared Supabase client for the production API

import os
from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Normalise team names from Supabase → ML model name (matches.csv / fifa_rankings_2026.csv)
# Add an entry here whenever a Supabase team name differs from what the ML model expects.
SUPABASE_TO_ML_NAME: dict[str, str] = {
    "Curaçao":              "Curacao",
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Korea Republic":       "South Korea",
    "USA":                  "United States",
    "IR Iran":              "Iran",
    "Côte d'Ivoire":        "Ivory Coast",
    "DR Congo":             "DR Congo",          # already matches — belt + suspenders
}


def normalize_team_name(name: str) -> str:
    """Map a Supabase team name to the ML model's canonical name."""
    return SUPABASE_TO_ML_NAME.get(name, name)


_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client
