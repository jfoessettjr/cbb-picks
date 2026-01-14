import json
import os
import time
from datetime import date, datetime, timedelta
import requests

from elo import expected_score, update_elo

API_BASE = os.getenv("NCAA_API_BASE", "https://ncaa-api.henrygd.me")
SPORT = os.getenv("NCAA_SPORT", "basketball-men")
DIVISION = os.getenv("NCAA_DIVISION", "d1")

# We only need to generate today + a few future days each run.
# Past dates remain available because we commit them to the repo.
FUTURE_DAYS = int(os.getenv("FUTURE_DAYS", "3"))

HCA = float(os.getenv("HOME_COURT_ADV_ELO", "65"))
K = float(os.getenv("ELO_K", "20"))
SLEEP_SECONDS = float(os.getenv("API_SLEEP_SECONDS", "0.25"))

SNAPSHOT_PATH = os.getenv("ELO_SNAPSHOT_PATH", "data/elo_snapshot.json")

def season_start_for(d: date) -> date:
    # Heuristic: season starts Nov 1.
    # If date is Jul-Dec => season starts Nov 1 of same year
    # If date is Jan-Jun => season starts Nov 1 of previous year
    season_year = d.year if d.month >= 7 else (d.year - 1)
    return date(season_year, 11, 1)

def season_label_for(d: date) -> str:
    ss = season_start_for(d)
    return f"{ss.year}-{ss.year + 1}"

def scoreboard_path(d: date) -> str:
    return f"/scoreboard/{SPORT}/{DIVISION}/{d.year}/{d.month:02d}/{d.day:02d}/all-conf"

def api_get(path: str) -> dict:
    url = f"{API_BASE}{path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_games(scoreboard_json: dict) -> list[dict]:
    games_out = []
    for item in scoreboard_json.get("games", []):
        g = item.get("game", {})
        home = g.get("home", {}) or {}
        away = g.get("away", {}) or {}

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
            "state": (g.get("gameState") or "").lower(),
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
    return game.get("state") == "final" or (
        game["home"]["score"] is not None and game["away"]["score"] is not None
    )

def load_snapshot(expected_season_start: date) -> tuple[dict[str, float], date | None]:
    """
    Returns (ratings, last_finalized_date) where last_finalized_date is the most recent date
    for which we have applied final-game Elo updates into the snapshot.
    """
    if not os.path.exists(SNAPSHOT_PATH):
        return {}, None

    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snap = json.load(f)

    if snap.get("season_start") != expected_season_start.isoformat():
        # new season (or snapshot mismatch) => reset
        return {}, None

    ratings = {str(k): float(v) for k, v in (snap.get("ratings") or {}).items()}
    last = snap.get("last_finalized_date")
    last_d = date.fromisoformat(last) if last else None
    return ratings, last_d

def save_snapshot(season_start: date, last_finalized: date, ratings: dict[str, float]):
    os.makedirs(os.path.dirname(SNAPSHOT_PATH) or ".", exist_ok=True)
    payload = {
        "season_start": season_start.isoformat(),
        "season": f"{season_start.year}-{season_start.year + 1}",
        "last_finalized_date": last_finalized.isoformat(),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ratings": ratings,
    }
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def apply_finals_for_date(d: date, ratings: dict[str, float]):
    try:
        sb = api_get(scoreboard_path(d))
    except requests.HTTPError as e:
        print(f"[warn] {d.isoformat()} scoreboard fetch failed while finalizing: {e}")
        return

    games = parse_games(sb)
    games.sort(key=lambda x: x.get("start_epoch", 0))

    for g in games:
        if not is_final(g):
            continue

        hs = g["home"]["score"]
        a_s = g["away"]["score"]
        if hs is None or a_s is None:
            continue

        hid = g["home"]["id"]
        aid = g["away"]["id"]

        r_home = ratings.get(hid, 1500.0)
        r_away = ratings.get(aid, 1500.0)

        if hs > a_s:
            s_home = 1.0
        elif hs < a_s:
            s_home = 0.0
        else:
            s_home = 0.5

        ratings[hid], ratings[aid] = update_elo(r_home, r_away, s_home, k=K)

def write_picks_for_date(d: date, ratings: dict[str, float]):
    """
    Writes:
      public/picks/YYYY-MM-DD.json
    using Elo ratings as-of end of previous day.
    """
    try:
        sb = api_get(scoreboard_path(d))
    except requests.HTTPError as e:
        print(f"[warn] {d.isoformat()} scoreboard fetch failed for picks: {e}")
        sb = {"games": []}

    games = parse_games(sb)
    picks = []

    for g in games:
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

    out = {
        "date": d.isoformat(),
        "season": season_label_for(d),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "picks": picks,
    }

    out_path = f"public/picks/{d.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out

def write_manifest():
    # List all available dated JSONs for the date picker
    files = []
    for name in os.listdir("public/picks"):
        if name.endswith(".json") and name not in ("latest.json", "manifest.json"):
            files.append(name.replace(".json", ""))

    # Keep only ISO dates (YYYY-MM-DD), ignore anything unexpected
    dates = [x for x in files if len(x) == 10 and x[4] == "-" and x[7] == "-"]
    dates.sort()

    manifest = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "available_dates": dates,
        "min_date": dates[0] if dates else None,
        "max_date": dates[-1] if dates else None,
    }

    with open("public/picks/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def main():
    os.makedirs("public/picks", exist_ok=True)

    picks_date = os.getenv("PICKS_DATE")
    today = date.fromisoformat(picks_date) if picks_date else date.today()

    season_start = season_start_for(today)

    ratings, last_finalized = load_snapshot(season_start)

    # If no snapshot, initialize by finalizing up through yesterday
    # (this can take a while on first run, then it’s fast forever)
    yesterday = today - timedelta(days=1)
    if last_finalized is None:
        print(f"No snapshot found for season {season_start.isoformat()}. Building initial Elo snapshot...")
        d = season_start
        while d <= yesterday:
            apply_finals_for_date(d, ratings)
            d += timedelta(days=1)
            time.sleep(SLEEP_SECONDS)
        save_snapshot(season_start, yesterday, ratings)
        last_finalized = yesterday
        print("Initial snapshot built.")
    else:
        # Advance snapshot from last_finalized+1 to yesterday (catch up)
        d = last_finalized + timedelta(days=1)
        while d <= yesterday:
            apply_finals_for_date(d, ratings)
            last_finalized = d
            d += timedelta(days=1)
            time.sleep(SLEEP_SECONDS)
        save_snapshot(season_start, last_finalized, ratings)
        print(f"Snapshot caught up through {last_finalized.isoformat()}.")

    # Now write picks for today and next N days using ratings as-of yesterday
    latest_out = None
    for offset in range(0, FUTURE_DAYS + 1):
        target = today + timedelta(days=offset)
        out = write_picks_for_date(target, ratings)
        if offset == 0:
            latest_out = out
        time.sleep(SLEEP_SECONDS)

    # Write latest.json for default view
    if latest_out is not None:
        with open("public/picks/latest.json", "w", encoding="utf-8") as f:
            json.dump(latest_out, f, indent=2)

    write_manifest()
    print("Updated picks + manifest.")

if __name__ == "__main__":
    main()
