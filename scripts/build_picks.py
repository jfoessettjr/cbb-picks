import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode

import requests

from elo import expected_score, update_elo_mov_recency

# =========================
# Core config
# =========================
API_BASE = os.getenv("NCAA_API_BASE", "https://ncaa-api.henrygd.me")
SPORT = os.getenv("NCAA_SPORT", "basketball-men")
DIVISION = os.getenv("NCAA_DIVISION", "d1")

FUTURE_DAYS = int(os.getenv("FUTURE_DAYS", "3"))

HCA = float(os.getenv("HOME_COURT_ADV_ELO", "65"))
K = float(os.getenv("ELO_K", "20"))

HALF_LIFE_DAYS = float(os.getenv("HALF_LIFE_DAYS", "30"))
MOV_CAP = float(os.getenv("MOV_CAP", "2.0"))

K_EARLY_MULT = float(os.getenv("K_EARLY_MULT", "1.40"))
K_MID_MULT   = float(os.getenv("K_MID_MULT", "1.00"))
K_LATE_MULT  = float(os.getenv("K_LATE_MULT", "0.85"))

K_RAMP_EARLY_DAYS = int(os.getenv("K_RAMP_EARLY_DAYS", "45"))
K_RAMP_MID_DAYS   = int(os.getenv("K_RAMP_MID_DAYS", "105"))

SLEEP_SECONDS = float(os.getenv("API_SLEEP_SECONDS", "0.25"))
SNAPSHOT_PATH = os.getenv("ELO_SNAPSHOT_PATH", "data/elo_snapshot.json")

# =========================
# Odds / EV tuning
# =========================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_API_BASE = os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
ODDS_SPORT_KEY = os.getenv("ODDS_SPORT_KEY", "basketball_ncaab")

REQUIRE_ODDS = os.getenv("REQUIRE_ODDS", "1").lower() in ("1", "true", "yes")

DOG_MIN = int(os.getenv("DOG_MIN", "110"))
DOG_MAX = int(os.getenv("DOG_MAX", "350"))
FAV_MIN = int(os.getenv("FAV_MIN", "-320"))
FAV_MAX = int(os.getenv("FAV_MAX", "-115"))

MARKET_BLEND_ALPHA = float(os.getenv("MARKET_BLEND_ALPHA", "0.55"))

DOG_MIN_EDGE = float(os.getenv("DOG_MIN_EDGE", "0.02"))
DOG_MIN_EV = float(os.getenv("DOG_MIN_EV", "0.01"))
DOG_MIN_FINAL_WINPROB = float(os.getenv("DOG_MIN_FINAL_WINPROB", "0.22"))

FAV_MIN_EDGE = float(os.getenv("FAV_MIN_EDGE", "0.01"))
FAV_MIN_EV = float(os.getenv("FAV_MIN_EV", "0.004"))

# 🔑 NEW: volume controls
MAX_PICKS_PER_DAY = int(os.getenv("MAX_PICKS_PER_DAY", "5"))
ONLY_POSITIVE_EV = os.getenv("ONLY_POSITIVE_EV", "1").lower() in ("1", "true", "yes")
DISABLE_GATES = os.getenv("DISABLE_GATES", "1").lower() in ("1", "true", "yes")

ODDS_TIME_MATCH_WINDOW_SEC = int(os.getenv("ODDS_TIME_MATCH_WINDOW_SEC", str(6 * 3600)))

# =========================
# Helpers
# =========================
def season_start_for(d: date) -> date:
    y = d.year if d.month >= 7 else d.year - 1
    return date(y, 11, 1)

def season_label_for(d: date) -> str:
    s = season_start_for(d)
    return f"{s.year}-{s.year + 1}"

def scoreboard_path(d: date) -> str:
    return f"/scoreboard/{SPORT}/{DIVISION}/{d.year}/{d.month:02d}/{d.day:02d}/all-conf"

def api_get(path: str) -> dict:
    r = requests.get(f"{API_BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

def expected_ev(p: float, dec: float) -> float:
    return p * (dec - 1) - (1 - p)

# =========================
# Team name normalization
# =========================
_STOP = {"university","college","of","the","and","at","state","st","saint","mt","mount","tech","a","m","am","univ"}

def norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    return [t for t in norm(s).split() if t not in _STOP]

def team_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if norm(a) == norm(b):
        return 1.0
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    return min(0.85, j)

# =========================
# Odds helpers
# =========================
def american_to_decimal(am):
    if am is None:
        return None
    am = float(am)
    return 1 + (am / 100 if am > 0 else 100 / abs(am))

def implied_prob(dec):
    return None if not dec or dec <= 1 else 1 / dec

def devig(p1, p2):
    s = (p1 or 0) + (p2 or 0)
    if s <= 0:
        return None, None
    return p1 / s, p2 / s

def fetch_odds_window(d_from, d_to):
    if not ODDS_API_KEY:
        return []
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": datetime.combine(d_from, datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00","Z"),
        "commenceTimeTo": datetime.combine(d_to, datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00","Z"),
    }
    r = requests.get(f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds/?{urlencode(params)}", timeout=30)
    r.raise_for_status()
    return r.json()

def best_price(og, team):
    best = None
    for bk in og.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []):
                if team_similarity(o.get("name",""), team) < 0.7:
                    continue
                dec = american_to_decimal(o.get("price"))
                if dec and (best is None or dec > best["dec"]):
                    best = {
                        "am": o.get("price"),
                        "dec": dec,
                        "book": bk.get("key")
                    }
    return best

# =========================
# Picks
# =========================
def write_picks_for_date(d, ratings, odds_window):
    sb = api_get(scoreboard_path(d))
    games = sb.get("games", [])
    picks = []

    for item in games:
        g = item.get("game", {})
        home = g.get("home", {})
        away = g.get("away", {})

        home_name = home.get("names", {}).get("short")
        away_name = away.get("names", {}).get("short")

        hid = home_name
        aid = away_name

        r_home = ratings.get(hid, 1500)
        r_away = ratings.get(aid, 1500)

        p_home_elo = expected_score(r_home + HCA, r_away)
        p_away_elo = 1 - p_home_elo

        if not ODDS_API_KEY:
            picks.append({
                "home_team": home_name,
                "away_team": away_name,
                "pick_team": home_name if p_home_elo >= 0.5 else away_name,
                "win_prob": max(p_home_elo, p_away_elo),
            })
            continue

        og = None
        for x in odds_window:
            if team_similarity(x.get("home_team",""), home_name) + team_similarity(x.get("away_team",""), away_name) >= 1.2:
                og = x
                break

        if not og:
            if REQUIRE_ODDS:
                continue
            else:
                continue

        oh, oa = og.get("home_team",""), og.get("away_team","")
        s_same = team_similarity(oh, home_name) + team_similarity(oa, away_name)
        s_swap = team_similarity(oh, away_name) + team_similarity(oa, home_name)
        swapped = s_swap > (s_same + 0.15)

        home_price = best_price(og, oa if swapped else oh)
        away_price = best_price(og, oh if swapped else oa)

        if not home_price or not away_price:
            continue

        p_home_m, p_away_m = devig(
            implied_prob(home_price["dec"]),
            implied_prob(away_price["dec"])
        )

        candidates = []

        for side, team, p_elo, p_mkt, price in [
            ("home", home_name, p_home_elo, p_home_m, home_price),
            ("away", away_name, p_away_elo, p_away_m, away_price),
        ]:
            if not p_mkt:
                continue

            p_final = MARKET_BLEND_ALPHA * p_mkt + (1 - MARKET_BLEND_ALPHA) * p_elo
            ev = expected_ev(p_final, price["dec"])
            edge = p_final - p_mkt

            ok = True
            if not DISABLE_GATES:
                ok = ev > 0

            if ok:
                candidates.append({
                    "home_team": home_name,
                    "away_team": away_name,
                    "pick_team": team,
                    "pick_side": side,
                    "win_prob": p_final,
                    "elo_win_prob": p_elo,
                    "pick_odds_american": price["am"],
                    "pick_odds_decimal": price["dec"],
                    "pick_book": price["book"],
                    "market_p_pick": p_mkt,
                    "edge": edge,
                    "ev": ev,
                })

        if not candidates:
            continue

        candidates.sort(key=lambda x: x["ev"], reverse=True)
        picks.append(candidates[0])

    if ONLY_POSITIVE_EV:
        picks = [p for p in picks if p["ev"] > 0]

    picks.sort(key=lambda x: x["ev"], reverse=True)

    if MAX_PICKS_PER_DAY > 0:
        picks = picks[:MAX_PICKS_PER_DAY]

    out = {
        "date": d.isoformat(),
        "season": season_label_for(d),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "picks": picks,
    }

    with open(f"public/picks/{d.isoformat()}.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

# =========================
# Main
# =========================
def main():
    os.makedirs("public/picks", exist_ok=True)
    today = date.today()

    ratings = {}
    odds_window = fetch_odds_window(today, today + timedelta(days=FUTURE_DAYS + 1))

    for i in range(FUTURE_DAYS + 1):
        write_picks_for_date(today + timedelta(days=i), ratings, odds_window)
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()