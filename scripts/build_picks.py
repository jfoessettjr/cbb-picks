import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode

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
K_EARLY_MULT = float(os.getenv("K_EARLY_MULT", "1.40"))  # very early season
K_MID_MULT   = float(os.getenv("K_MID_MULT", "1.00"))    # mid-season baseline
K_LATE_MULT  = float(os.getenv("K_LATE_MULT", "0.85"))    # late-season stability

# Ramp breakpoints (days since season start)
K_RAMP_EARLY_DAYS = int(os.getenv("K_RAMP_EARLY_DAYS", "45"))   # end of early ramp
K_RAMP_MID_DAYS   = int(os.getenv("K_RAMP_MID_DAYS", "105"))    # end of mid ramp

SLEEP_SECONDS = float(os.getenv("API_SLEEP_SECONDS", "0.25"))
SNAPSHOT_PATH = os.getenv("ELO_SNAPSHOT_PATH", "data/elo_snapshot.json")


# -------------------------
# Odds API (The Odds API)
# -------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_API_BASE = os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
ODDS_SPORT_KEY = os.getenv("ODDS_SPORT_KEY", "basketball_ncaab")

# "Sweet spot" ranges (ROI defaults)
# Underdogs primarily (+125..+325), light favorites only (-120..-220)
DOG_MIN = int(os.getenv("DOG_MIN", "125"))
DOG_MAX = int(os.getenv("DOG_MAX", "325"))
FAV_MIN = int(os.getenv("FAV_MIN", "-220"))   # most negative allowed (exclude heavier chalk)
FAV_MAX = int(os.getenv("FAV_MAX", "-120"))   # least negative allowed

# Market anchoring (higher = closer to market)
MARKET_BLEND_ALPHA = float(os.getenv("MARKET_BLEND_ALPHA", "0.65"))

# Gates for ROI-driven selection
DOG_MIN_EDGE = float(os.getenv("DOG_MIN_EDGE", "0.03"))
DOG_MIN_EV = float(os.getenv("DOG_MIN_EV", "0.02"))
DOG_MIN_FINAL_WINPROB = float(os.getenv("DOG_MIN_FINAL_WINPROB", "0.23"))

FAV_MIN_EDGE = float(os.getenv("FAV_MIN_EDGE", "0.015"))
FAV_MIN_EV = float(os.getenv("FAV_MIN_EV", "0.01"))

# How close (seconds) odds game time must be to NCAA time to match reliably
ODDS_TIME_MATCH_WINDOW_SEC = int(os.getenv("ODDS_TIME_MATCH_WINDOW_SEC", str(6 * 3600)))


# -------------------------
# Season helpers
# -------------------------
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


# -------------------------
# NCAA parsing
# -------------------------
def _looks_neutral(game_obj: dict) -> bool:
    """
    Neutral-site detection:
    - If upstream provides a neutral-site boolean, use it.
    - Otherwise try some best-effort heuristics on strings/fields.
    """
    for key in ("neutralSite", "isNeutral", "neutral", "neutral_site"):
        val = game_obj.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str) and val.lower() in ("true", "false"):
            return val.lower() == "true"

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


# -------------------------
# Snapshot load/save
# -------------------------
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


# -------------------------
# Elo ramp helpers
# -------------------------
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


# -------------------------
# Odds helpers
# -------------------------
def american_to_decimal(am: int | float | None) -> float | None:
    if am is None:
        return None
    try:
        amf = float(am)
    except Exception:
        return None
    if amf > 0:
        return 1.0 + amf / 100.0
    return 1.0 + 100.0 / abs(amf)

def implied_prob_from_decimal(dec: float | None) -> float | None:
    if dec is None or dec <= 1.0:
        return None
    return 1.0 / dec

def devig_two_way(p_a: float | None, p_b: float | None) -> tuple[float | None, float | None]:
    if p_a is None or p_b is None:
        return None, None
    s = p_a + p_b
    if s <= 0:
        return None, None
    return p_a / s, p_b / s

def expected_value_per_dollar(p_win: float, dec_odds: float) -> float:
    # EV per $1: p*(dec-1) - (1-p)*1
    return p_win * (dec_odds - 1.0) - (1.0 - p_win)

def _norm_team_name(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", "and")
    # normalize some common abbreviations/formatting
    s = s.replace("u.", "u").replace("st.", "st")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_STOP = {
    "university", "college", "of", "the", "and", "at",
    "state", "st", "saint", "mt", "mount",
    "tech", "a", "m", "am",
    "univ"
}

def _tokens(name: str) -> list[str]:
    t = [x for x in _norm_team_name(name).split(" ") if x and x not in _STOP]
    return t

def _initials(tokens: list[str]) -> str:
    # initials for multiword names (e.g., "ul monroe" -> "ulm")
    return "".join([w[0] for w in tokens if w])[:8]

def _abbr_match(short: str, long_name: str) -> bool:
    """
    True if short (like 'ULM') matches initials of long_name tokens
    or is contained as a token-ish piece.
    """
    short_n = _norm_team_name(short).replace(" ", "")
    if not short_n or len(short_n) > 6:
        return False

    toks = _tokens(long_name)
    ini = _initials(toks)

    if short_n == ini:
        return True

    # also allow 'ulmonroe' style join match
    joined = "".join(toks)
    if short_n in joined or joined in short_n:
        return True
    
    return False

def team_similarity(a: str, b: str) -> float:
    """
    Returns 0..1-ish similarity score.
    Exact match -> 1.0
    Strong acronym match -> 0.9
    Token Jaccard overlap -> 0..0.85
    """
    an = _norm_team_name(a)
    bn = _norm_team_name(b)
    if not an or not bn:
        return 0.0
    if an == bn:
        return 1.0

    # abbreviation handling (ULM, UIC, etc.)
    if len(an.replace(" ", "")) <= 5 and _abbr_match(an, b):
        return 0.90
    if len(bn.replace(" ", "")) <= 5 and _abbr_match(b, a):
        return 0.90

    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0

    inter = len(ta & tb)
    union = len(ta | tb)
    j = inter / union if union else 0.0

    # Boost if one token set is contained in the other
    if ta.issubset(tb) or tb.issubset(ta):
        j = max(j, 0.80)

    return min(0.85, j)

def best_matching_odds_game(odds_games: list[dict], home_name: str, away_name: str, start_epoch: int) -> dict | None:
    """
    Match by:
      1) commence_time within window
      2) best team name similarity (handles abbreviations like ULM)
    """
    if not odds_games:
        return None

    start_epoch = int(start_epoch or 0)
    best = None
    best_score = -1.0

    for og in odds_games:
        ce = parse_iso_to_epoch(og.get("commence_time", ""))
        if ce is None:
            continue
        if abs(ce - start_epoch) > ODDS_TIME_MATCH_WINDOW_SEC:
            continue

        oh = og.get("home_team", "") or ""
        oa = og.get("away_team", "") or ""

        # orientation 1: og home->ncaa home, og away->ncaa away
        s1 = team_similarity(oh, home_name) + team_similarity(oa, away_name)

        # orientation 2: swapped
        s2 = team_similarity(oh, away_name) + team_similarity(oa, home_name)

        score = max(s1, s2)

        if score > best_score:
            best_score = score
            best = og

    # require at least “pretty good” total similarity
    # exact-exact would be 2.0; acronym+good token match often lands ~1.6+
    if best is None or best_score < 1.35:
        return None

    return best

def calibrate_prob_for_long_odds(p: float) -> float:
    # Conservative shrink to reduce longshot overconfidence
    if p < 0.10:
        return p * 0.85
    if p < 0.20:
        return p * 0.90
    if p < 0.35:
        return p * 0.95
    if p < 0.55:
        return p * 0.98
    return p

def parse_iso_to_epoch(iso_str: str) -> int | None:
    # The Odds API commence_time is ISO with Z
    try:
        s = (iso_str or "").strip()
        if not s:
            return None
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None

def fetch_moneyline_odds_for_window(d_from: date, d_to_exclusive: date) -> list[dict]:
    """
    Fetch odds once for a whole window (UTC day bounds).
    """
    if not ODDS_API_KEY:
        return []

    start_dt = datetime(d_from.year, d_from.month, d_from.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(d_to_exclusive.year, d_to_exclusive.month, d_to_exclusive.day, 0, 0, 0, tzinfo=timezone.utc)

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": start_dt.isoformat().replace("+00:00", "Z"),
        "commenceTimeTo": end_dt.isoformat().replace("+00:00", "Z"),
    }

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds/?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def best_price_for_team(odds_game: dict, team_name: str) -> dict | None:
    """
    Find best (highest payout) American odds for team_name across bookmakers.
    Returns {am, dec, book} or None.
    """
    target = _norm_team_name(team_name)
    best = None

    for bk in odds_game.get("bookmakers", []) or []:
        book = bk.get("key") or bk.get("title") or "book"
        for m in bk.get("markets", []) or []:
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []) or []:
                if _norm_team_name(o.get("name", "")) != target:
                    continue
                am = o.get("price")
                dec = american_to_decimal(am)
                if dec is None:
                    continue
                if best is None or dec > best["dec"]:
                    best = {"am": int(am), "dec": float(dec), "book": str(book)}
    return best

def teams_match_odds_game(og: dict, home_name: str, away_name: str) -> bool:
    oh = _norm_team_name(og.get("home_team", ""))
    oa = _norm_team_name(og.get("away_team", ""))
    nh = _norm_team_name(home_name)
    na = _norm_team_name(away_name)
    return (oh == nh and oa == na) or (oh == na and oa == nh)

def best_matching_odds_game(odds_games: list[dict], home_name: str, away_name: str, start_epoch: int) -> dict | None:
    """
    Match by team names (either orientation) and closest commence_time to NCAA start.
    """
    candidates = []
    for og in odds_games:
        if not teams_match_odds_game(og, home_name, away_name):
            continue
        ce = parse_iso_to_epoch(og.get("commence_time", ""))
        if ce is None:
            continue
        diff = abs(ce - int(start_epoch or 0))
        if diff <= ODDS_TIME_MATCH_WINDOW_SEC:
            candidates.append((diff, og))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# -------------------------
# Elo finalization
# -------------------------
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


# -------------------------
# Picks generation (Elo + Market)
# -------------------------
def _in_range_by_odds(am: int) -> tuple[bool, str]:
    """
    Returns (in_range, kind) where kind is 'fav' or 'dog'.
    """
    if am < 0:
        # favorite range: between FAV_MIN (more negative) and FAV_MAX (less negative)
        return (FAV_MIN <= am <= FAV_MAX), "fav"
    else:
        return (DOG_MIN <= am <= DOG_MAX), "dog"

def write_picks_for_date(d: date, ratings: dict[str, float], odds_games_window: list[dict] | None):
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

        p_home_elo = expected_score(home_elo_with_hca, away_elo)
        p_away_elo = 1.0 - p_home_elo

        home_name = g["home"]["name"]
        away_name = g["away"]["name"]

        # Default (no odds): pick highest Elo win prob
        pick_team = home_name if p_home_elo >= 0.5 else away_name
        pick_side = "home" if p_home_elo >= 0.5 else "away"
        win_prob_final = max(p_home_elo, p_away_elo)
        elo_win_prob = win_prob_final

        pick_odds = None
        market_p_pick = None
        edge = None
        ev = None

        # If odds enabled, evaluate BOTH sides and choose best EV play
        if ODDS_API_KEY and odds_games_window is not None:
            og = best_matching_odds_game(odds_games_window, home_name, away_name, g.get("start_epoch", 0))
            if og is None:
                # user wants market filtering; if we cannot find odds, skip the game
                continue

            # Determine mapping orientation
            og_home_norm = _norm_team_name(og.get("home_team", ""))
            ncaa_home_norm = _norm_team_name(home_name)
            swapped = (og_home_norm != ncaa_home_norm)

            if not swapped:
                og_home_name = og.get("home_team", home_name)
                og_away_name = og.get("away_team", away_name)
                home_best = best_price_for_team(og, og_home_name)
                away_best = best_price_for_team(og, og_away_name)
            else:
                # swapped: odds API home corresponds to NCAA away
                og_home_name = og.get("home_team", away_name)
                og_away_name = og.get("away_team", home_name)
                # Map to NCAA correctly:
                # NCAA home corresponds to odds away
                away_best = best_price_for_team(og, og_home_name)   # NCAA away
                home_best = best_price_for_team(og, og_away_name)   # NCAA home

            if not home_best or not away_best:
                continue

            # Market fair probs (de-vig)
            p_home_raw = implied_prob_from_decimal(home_best["dec"])
            p_away_raw = implied_prob_from_decimal(away_best["dec"])
            p_home_mkt, p_away_mkt = devig_two_way(p_home_raw, p_away_raw)

            candidates = []

            # HOME candidate
            am_home = int(home_best["am"])
            in_range, kind = _in_range_by_odds(am_home)
            if in_range and p_home_mkt is not None:
                p_blend = (MARKET_BLEND_ALPHA * p_home_mkt) + ((1.0 - MARKET_BLEND_ALPHA) * p_home_elo)
                p_blend = calibrate_prob_for_long_odds(p_blend)
                ed = p_blend - p_home_mkt
                evv = expected_value_per_dollar(p_blend, float(home_best["dec"]))

                ok = True
                if kind == "fav":
                    ok = (ed >= FAV_MIN_EDGE) and (evv >= FAV_MIN_EV)
                else:
                    ok = (ed >= DOG_MIN_EDGE) and (evv >= DOG_MIN_EV) and (p_blend >= DOG_MIN_FINAL_WINPROB)

                if ok:
                    candidates.append({
                        "side": "home",
                        "team": home_name,
                        "p_elo": p_home_elo,
                        "p_mkt": p_home_mkt,
                        "p_final": p_blend,
                        "edge": ed,
                        "ev": evv,
                        "odds": home_best
                    })

            # AWAY candidate
            am_away = int(away_best["am"])
            in_range, kind = _in_range_by_odds(am_away)
            if in_range and p_away_mkt is not None:
                p_blend = (MARKET_BLEND_ALPHA * p_away_mkt) + ((1.0 - MARKET_BLEND_ALPHA) * p_away_elo)
                p_blend = calibrate_prob_for_long_odds(p_blend)
                ed = p_blend - p_away_mkt
                evv = expected_value_per_dollar(p_blend, float(away_best["dec"]))

                ok = True
                if kind == "fav":
                    ok = (ed >= FAV_MIN_EDGE) and (evv >= FAV_MIN_EV)
                else:
                    ok = (ed >= DOG_MIN_EDGE) and (evv >= DOG_MIN_EV) and (p_blend >= DOG_MIN_FINAL_WINPROB)

                if ok:
                    candidates.append({
                        "side": "away",
                        "team": away_name,
                        "p_elo": p_away_elo,
                        "p_mkt": p_away_mkt,
                        "p_final": p_blend,
                        "edge": ed,
                        "ev": evv,
                        "odds": away_best
                    })

            if not candidates:
                continue

            # Choose the best play per game: EV first, then edge, then win prob
            candidates.sort(key=lambda c: (c["ev"], c["edge"], c["p_final"]), reverse=True)
            best = candidates[0]

            pick_side = best["side"]
            pick_team = best["team"]
            win_prob_final = best["p_final"]
            elo_win_prob = best["p_elo"]
            market_p_pick = best["p_mkt"]
            edge = best["edge"]
            ev = best["ev"]
            pick_odds = best["odds"]

        picks.append({
            "game_id": g["game_id"],
            "date_time_epoch": g["start_epoch"],
            "home_team": home_name,
            "away_team": away_name,
            "neutral_site": bool(g.get("neutral_site")),

            "pick_team": pick_team,
            "pick_side": pick_side,

            # Final prob (blended if odds enabled & matched)
            "win_prob": win_prob_final,
            "elo_win_prob": elo_win_prob,

            # Elo info
            "home_elo_raw": home_elo_raw,
            "home_elo_with_hca": home_elo_with_hca,
            "away_elo": away_elo,
            "hca": hca,

            # Market fields (None if odds disabled)
            "pick_odds_american": (pick_odds["am"] if pick_odds else None),
            "pick_odds_decimal": (pick_odds["dec"] if pick_odds else None),
            "pick_book": (pick_odds["book"] if pick_odds else None),
            "market_p_pick": market_p_pick,
            "edge": edge,
            "ev": ev,
        })

    # Sort: for ROI mode, EV-first when odds enabled; otherwise win_prob
    if ODDS_API_KEY:
        picks.sort(key=lambda x: (x["ev"] if x["ev"] is not None else -999), reverse=True)
    else:
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
            },
            "market": {
                "enabled": bool(ODDS_API_KEY),
                "provider": ("the-odds-api" if ODDS_API_KEY else None),
                "sport_key": (ODDS_SPORT_KEY if ODDS_API_KEY else None),
                "ranges": {
                    "dog_min": (DOG_MIN if ODDS_API_KEY else None),
                    "dog_max": (DOG_MAX if ODDS_API_KEY else None),
                    "fav_min": (FAV_MIN if ODDS_API_KEY else None),
                    "fav_max": (FAV_MAX if ODDS_API_KEY else None),
                },
                "blend_alpha": (MARKET_BLEND_ALPHA if ODDS_API_KEY else None),
                "gates": {
                    "dog_min_edge": (DOG_MIN_EDGE if ODDS_API_KEY else None),
                    "dog_min_ev": (DOG_MIN_EV if ODDS_API_KEY else None),
                    "dog_min_final_winprob": (DOG_MIN_FINAL_WINPROB if ODDS_API_KEY else None),
                    "fav_min_edge": (FAV_MIN_EDGE if ODDS_API_KEY else None),
                    "fav_min_ev": (FAV_MIN_EV if ODDS_API_KEY else None),
                }
            }
        },
        "picks": picks,
    }

    out_path = f"public/picks/{d.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


# -------------------------
# Manifest writer
# -------------------------
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


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("public/picks", exist_ok=True)

    picks_date = os.getenv("PICKS_DATE")
    today = date.fromisoformat(picks_date) if picks_date else date.today()

    season_start = season_start_for(today)
    ratings, last_finalized = load_snapshot(season_start)

    yesterday = today - timedelta(days=1)

    # Build/catch up Elo snapshot
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

    # Fetch odds once for the full window (today .. today+FUTURE_DAYS)
    odds_window = None
    if ODDS_API_KEY:
        try:
            d_from = today
            d_to_excl = today + timedelta(days=FUTURE_DAYS + 1)
            odds_window = fetch_moneyline_odds_for_window(d_from, d_to_excl)
            print(f"Fetched odds window: {len(odds_window)} games from The Odds API.")
        except Exception as e:
            print(f"[warn] Failed to fetch odds window: {e}")
            odds_window = []

    latest_out = None
    for offset in range(0, FUTURE_DAYS + 1):
        target = today + timedelta(days=offset)
        out = write_picks_for_date(target, ratings, odds_window)
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  