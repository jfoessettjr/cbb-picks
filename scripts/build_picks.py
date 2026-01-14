import json
import os
import time
from datetime import date, datetime, timedelta
from dateutil.parser import isoparse
import requests

from elo import expected_score, update_elo

# Public demo API (free). Note the project states 5 req/s per IP. :contentReference[oaicite:3]{index=3}
API_BASE = os.getenv("NCAA_API_BASE", "https://ncaa-api.henrygd.me")

SPORT = os.getenv("NCAA_SPORT", "basketball-men")
DIVISION = os.getenv("NCAA_DIVISION", "d1")

# Tunables
START_SEASON_DATE = os.getenv("START_SEASON_DATE", "2025-11-01")  # adjust each season
HCA = float(os.getenv("HOME_COURT_ADV_ELO", "65"))
K = float(os.getenv("ELO_K", "20"))

# Safety to respect public API limits (5 req/s). :contentReference[oaicite:4]{index=4}
SLEEP_SECONDS = float(os.getenv("API_SLEEP_SECONDS", "0.25"))

def scoreboard_path(d: date) -> str:
    # NCAA-style path mirrored by the API :contentReference[oaicite:5]{index=5}
    # Example in README (football): /scoreboard/football/fbs/2023/13/all-conf :contentReference[oaicite:6]{index=6}
    return f"/scoreboard/{SPORT}/{DIVISION}/{d.year}/{d.month:02d}/{d.day:02d}/all-conf"

def api_get(path: str) -> dict:
    url = f"{API_BASE}{path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_games(scoreboard_json: dict) -> list[dict]:
    """
    Returns normalized games:
      {
        "game_id": "...",
        "start_epoch": int,
        "state": "final" | "live" | "pre",
        "home": {"name": str, "id": str, "score": int|None},
        "away": {"name": str, "id": str, "score": int|None}
      }
    """
    games_out = []
    for item in scoreboard_json.get("games", []):
        g = item.get("game", {})
        home = g.get("home", {}) or {}
        away = g.get("away", {}) or {}

        # Team identifiers: NCAA JSON often includes "seo" for team identity; we use that as id
        # (works fine for Elo keys).
        home_names = (home.get("names") or {})
        away_names = (away.get("names") or {})

        home_id = home_names.get("seo") or home_names.get("short") or home_names.get("full") or "home"
        away_id = away_names.get("seo") or away_names.get("short") or away_names.get("full") or "away"

        def to_int(x):
            try:
                return int(x)
            except Exception:
                return None

        games_out.append({
            "game_id": g.get("gameID") or g.get("id") or g.get("url") or f"{away_id}_at_{home_id}",
            "start_epoch": int(g.get("startTimeEpoch") or 0),
            "state": (g.get("gameState") or "").lower(),   # often "final" / "pre" / etc.
            "home": {
                "id": str(home_id),
                "name": home_names.get("short") or home_names.get("full") or "Home",
                "score": to_int(home.get("score"))
            },
            "away": {
                "id": str(away_id),
                "name": away_names.get("short") or away_names.get("full") or "Away",
                "score": to_int(away.get("score"))
            }
        })
    return games_out

def is_final(game: dict) -> bool:
    return game.get("state") == "final" or (game["home"]["score"] is not None and game["away"]["score"] is not None)

def build_elo_from_range(start_d: date, end_d: date) -> dict[str, float]:
    ratings: dict[str, float] = {}
    d = start_d
    while d <= end_d:
        path = scoreboard_path(d)
        try:
            data = api_get(path)
        except requests.HTTPError as e:
            # If a date has no scoreboard or upstream changed, skip gracefully
            print(f"[warn] {d.isoformat()} scoreboard failed: {e}")
            d += timedelta(days=1)
            time.sleep(SLEEP_SECONDS)
            continue

        games = parse_games(data)
        # sort by start time for determinism
        games.sort(key=lambda x: x.get("start_epoch", 0))

        for g in games:
            if not is_final(g):
                continue

            hid = g["home"]["id"]
            aid = g["away"]["id"]

            r_home = ratings.get(hid, 1500.0)
            r_away = ratings.get(aid, 1500.0)

            hs = g["home"]["score"]
            a_s = g["away"]["score"]
            if hs is None or a_s is None:
                continue

            if hs > a_s:
                s_home = 1.0
            elif hs < a_s:
                s_home = 0.0
            else:
                s_home = 0.5

            ratings[hid], ratings[aid] = update_elo(r_home, r_away, s_home, k=K)

        d += timedelta(days=1)
        time.sleep(SLEEP_SECONDS)

    return ratings

def main():
    picks_date = os.getenv("PICKS_DATE")
    target = date.fromisoformat(picks_date) if picks_date else date.today()

    season_start = date.fromisoformat(START_SEASON_DATE)
    yesterday = target - timedelta(days=1)

    ratings = {}
    if yesterday >= season_start:
        print(f"Building Elo from {season_start.isoformat()} through {yesterday.isoformat()} ...")
        ratings = build_elo_from_range(season_start, yesterday)
        print(f"Elo teams tracked: {len(ratings)}")

    # Fetch today's games
    try:
        today_sb = api_get(scoreboard_path(target))
    except requests.HTTPError as e:
        print(f"[error] Could not fetch today's scoreboard: {e}")
        today_sb = {"games": []}

    todays_games = parse_games(today_sb)

    picks = []
    for g in todays_games:
        hid = g["home"]["id"]
        aid = g["away"]["id"]

        home_elo_raw = ratings.get(hid, 1500.0)
        away_elo = ratings.get(aid, 1500.0)
        home_elo_with_hca = home_elo_raw + HCA

        p_home = expected_score(home_elo_with_hca, away_elo)

        if p_home >= 0.5:
            pick_team = g["home"]["name"]
            win_prob = p_home
        else:
            pick_team = g["away"]["name"]
            win_prob = 1.0 - p_home

        picks.append({
            "game_id": g["game_id"],
            "date_time_epoch": g["start_epoch"],
            "home_team": g["home"]["name"],
            "away_team": g["away"]["name"],
            "pick_team": pick_team,
            "win_prob": win_prob,
            "home_elo_raw": home_elo_raw,
            "home_elo_with_hca": home_elo_with_hca,
            "away_elo": away_elo,
            "hca": HCA,
        })

    picks.sort(key=lambda x: x["win_prob"], reverse=True)

    os.makedirs("public/picks", exist_ok=True)
    out = {
        "date": target.isoformat(),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "picks": picks,
    }
    with open("public/picks/latest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote public/picks/latest.json ({len(picks)} games)")

if __name__ == "__main__":
    main()
