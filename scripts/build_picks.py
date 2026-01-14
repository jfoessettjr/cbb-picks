import json
import os
import time
from datetime import date, datetime, timedelta
import requests

from elo import expected_score, update_elo_mov_recency

API_BASE = os.getenv("NCAA_API_BASE", "https://ncaa-api.henrygd.me")
SPORT = os.getenv("NCAA_SPORT", "basketball-men")
DIVISION = os.getenv("NCAA_DIVISION", "d1")

# Generate today + a few future days each run (past days remain because we commit files)
FUTURE_DAYS = int(os.getenv("FUTURE_DAYS", "3"))

# Elo tunables
HCA = float(os.getenv("HOME_COURT_ADV_ELO", "65"))
K = float(os.getenv("ELO_K", "20"))

# MOV + Recency tunables
HALF_LIFE_DAYS = float(os.getenv("HALF_LIFE_DAYS", "30"))
MOV_CAP = float(os.getenv("MOV_CAP", "2.0"))

# Season-phase K multipliers (higher early season, lower late season)
# Smooth K ramp (days since season start)
K_EARLY_MULT = float(os.getenv("K_EARLY_MULT", "1.40"))  # very early season
K_MID_MULT   = float(os.getenv("K_MID_MULT", "1.00"))    # mid-season baseline
K_LATE_MULT  = float(os.getenv("K_LATE_MULT", "0.85"))   # late-season stability

# Ramp breakpoints (days since season start)
K_RAMP_EARLY_DAYS = int(os.getenv("K_RAMP_EARLY_DAYS", "45"))   # end of early ramp
K_RAMP_MID_DAYS   = int(os.getenv("K_RAMP_MID_DAYS", "105"))    # end of mid ramp



SLEEP_SECONDS = float(os.getenv("API_SLEEP_SECONDS", "0.25"))
SNAPSHOT_PATH = os.getenv("ELO_SNAPSHOT_PATH", "data/elo_snapshot.json")


def season_start_for(d: date) -> date:
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

def _looks_neutral(game_obj: dict) -> bool:
    """
    Neutral-site detection:
    - If upstream provides a neutral-site boolean, use it.
    - Otherwise try some best-effort heuristics on strings/fields.
    This is safe: default False if unknown.
    """
    # Common-ish keys some feeds use:
    for key in ("neutralSite", "isNeutral", "neutral", "neutral_site"):
        val = game_obj.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str) and val.lower() in ("true", "false"):
            return val.lower() == "true"

    # Sometimes venue/site text includes "neutral"
    venue = game_obj.get("venue") or game_obj.get("site") or ""
    if isinstance(venue, dict):
        venue = venue.get("name") or venue.get("city") or ""
    if isinstance(venue, str) and "neutral" in venue.lower():
        return True

    notes = game_obj.get("gameNotes") or game_obj.get("notes") or ""
    if isinstance(notes, str) and "neutral" in notes.lower():
        return True

    return False

def parse_games(scoreboard_json: dict) -> list[dict]:
    games_out = []
    for item in scoreboard_json.get("games", []):
        g = item.get("game", {}) or {}
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
            "neutral_site": _looks_neutral(g),
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
    if not os.path.exists(SNAPSHOT_PATH):
        return {}, None

    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snap = json.load(f)

    if snap.get("season_start") != expected_season_start.isoformat():
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
        
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def k_multiplier_ramp(game_day: date, season_start: date) -> float:
    """
    Smooth piecewise-linear ramp by days since season start:
      0 .. K_RAMP_EARLY_DAYS : K_EARLY_MULT -> K_MID_MULT
      K_RAMP_EARLY_DAYS .. K_RAMP_MID_DAYS : K_MID_MULT -> K_LATE_MULT
      beyond K_RAMP_MID_DAYS : K_LATE_MULT
    """
    days = max(0, (game_day - season_start).days)

    e = max(1, K_RAMP_EARLY_DAYS)
    m = max(e + 1, K_RAMP_MID_DAYS)

    if days <= e:
        t = days / e
        return lerp(K_EARLY_MULT, K_MID_MULT, t)

    if days <= m:
        t = (days - e) / (m - e)
        return lerp(K_MID_MULT, K_LATE_MULT, t)

    return K_LATE_MULT


def apply_finals_for_date(d: date, as_of: date, ratings: dict[str, float]):
    """
    Apply Elo updates from final games on date d into ratings.
    as_of: used to compute recency decay (days_ago = as_of - d)
    """
    try:
        sb = api_get(scoreboard_path(d))
    except requests.HTTPError as e:
        print(f"[warn] {d.isoformat()} scoreboard fetch failed while finalizing: {e}")
        return

    games = parse_games(sb)
    games.sort(key=lambda x: x.get("start_epoch", 0))

    days_ago = max(0, (as_of - d).days)

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

        season_start = season_start_for(as_of)
        k_mult = k_multiplier_ramp(d, season_start)

        new_home, new_away = update_elo_mov_recency(
            r_home, r_away,
            home_score=hs, away_score=a_s,
            base_k=K * k_mult,
            days_ago=days_ago,
            half_life_days=HALF_LIFE_DAYS,
            mov_cap=MOV_CAP
        )


        ratings[hid], ratings[aid] = new_home, new_away

def write_picks_for_date(d: date, ratings: dict[str, float]):
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

        # Neutral site handling: set HCA=0 if neutral
        hca = 0.0 if g.get("neutral_site") else HCA

        home_elo_raw = ratings.get(hid, 1500.0)
        away_elo = ratings.get(aid, 1500.0)
        home_elo_with_hca = home_elo_raw + hca

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
            "neutral_site": bool(g.get("neutral_site")),
            "pick_team": pick_team,
            "win_prob": win_prob,
            "home_elo_raw": home_elo_raw,
            "home_elo_with_hca": home_elo_with_hca,
            "away_elo": away_elo,
            "hca": hca,
        })

    picks.sort(key=lambda x: x["win_prob"], reverse=True)

    out = {
        "date": d.isoformat(),
        "season": season_label_for(d),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": {
            "type": "Elo",
            "mov": True,
            "recency_half_life_days": HALF_LIFE_DAYS,
            "mov_cap": MOV_CAP,
            "home_court_adv_elo": HCA,
            "neutral_site_hca": 0,
            "k_ramp": {
                "k_early_mult": K_EARLY_MULT,
                "k_mid_mult": K_MID_MULT,
                "k_late_mult": K_LATE_MULT,
                "k_ramp_early_days": K_RAMP_EARLY_DAYS,
                "k_ramp_mid_days": K_RAMP_MID_DAYS
                }
        },
        "picks": picks,
    }

    out_path = f"public/picks/{d.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out

def write_manifest():
    files = []
    for name in os.listdir("public/picks"):
        if name.endswith(".json") and name not in ("latest.json", "manifest.json"):
            files.append(name.replace(".json", ""))

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

    yesterday = today - timedelta(days=1)

    if last_finalized is None:
        print(f"No snapshot found for season {season_start.isoformat()}. Building initial snapshot...")
        d = season_start
        while d <= yesterday:
            apply_finals_for_date(d, as_of=today, ratings=ratings)
            d += timedelta(days=1)
            time.sleep(SLEEP_SECONDS)
        save_snapshot(season_start, yesterday, ratings)
        last_finalized = yesterday
        print("Initial snapshot built.")
    else:
        d = last_finalized + timedelta(days=1)
        while d <= yesterday:
            apply_finals_for_date(d, as_of=today, ratings=ratings)
            last_finalized = d
            d += timedelta(days=1)
            time.sleep(SLEEP_SECONDS)
        save_snapshot(season_start, last_finalized, ratings)
        print(f"Snapshot caught up through {last_finalized.isoformat()}.")

    latest_out = None
    for offset in range(0, FUTURE_DAYS + 1):
        target = today + timedelta(days=offset)
        out = write_picks_for_date(target, ratings)
        if offset == 0:
            latest_out = out
        time.sleep(SLEEP_SECONDS)

    if latest_out is not None:
        with open("public/picks/latest.json", "w", encoding="utf-8") as f:
            json.dump(latest_out, f, indent=2)

    write_manifest()
    print("Updated picks + manifest.")

if __name__ == "__main__":
    main()
