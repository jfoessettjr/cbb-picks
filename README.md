# NCAAB Picks Engine – Model & Odds Integration (Updated)

This project generates daily **college basketball picks** using a blended **Elo + betting market** model, with safeguards to avoid bad data, extreme variance, and empty slates.

Recent updates focused on **correct odds matching**, **controlled pick volume**, and an optional **Win% (accuracy-first) mode**.

---

## 🔢 Model Overview

### Core signals
- **Elo ratings**
  - Margin-of-victory adjusted
  - Recency-weighted (half-life decay)
  - Season-phase K-factor ramp (early → late season)
  - Home court advantage (disabled for neutral sites)

- **Market odds (The Odds API)**
  - Best available moneyline across books
  - De-vigged implied probabilities
  - Blended with Elo to reduce overconfidence

Final win probability is computed as:

P_final = α · P_market + (1 − α) · P_elo


Where `α` (`MARKET_BLEND_ALPHA`) controls market anchoring.

---

## 🎯 Pick Selection Philosophy

The engine supports two selection styles:

### 1) EV/ROI Mode (default)
- Rank candidates by **expected value (EV)** to target long-term profitability
- Enforce **odds ranges** to avoid extreme chalk and lottery longshots
- Require **minimum edge vs market** and **minimum confidence**
- Cap daily output instead of over-filtering

### 2) Win% Mode (accuracy-first)
- Prefer **higher probability winners**
- Can **disable underdogs entirely**
- Rank primarily by **final win probability** (then edge, then EV)
- Does **not** force positive EV (to prevent over-thinning)

---

## 🧱 Odds Safety & Matching (Critical)

Odds matching is robust to:
- NCAA games missing or misformatted start times
- Small time discrepancies between NCAA and sportsbooks
- Team name variations (e.g. `ULM` vs `UL Monroe`, `St.` vs `Saint`)

Matching rules:
1. Match by team similarity (both orientations)
2. Use start time if available (± `ODDS_TIME_MATCH_WINDOW_SEC`)
3. Fall back to team-only matching if NCAA time is missing
4. Enforce hard odds-range checks before publishing

If odds cannot be matched and `REQUIRE_ODDS=true`, the game is excluded.

---

## 📊 Odds Ranges (Current Defaults)

- **Underdogs:** `+110` to `+260` (`DOG_MIN` → `DOG_MAX`)
- **Favorites:** `-115` to `-320` (`FAV_MAX` → `FAV_MIN`)

These ranges intentionally exclude:
- Massive chalk (`-600`, `-1100`)
- Ultra-low-probability lottery bets

---

## 🚦 Gates & Filters (How a Play Gets Approved)

Every candidate must first:
- Match to odds (if enabled)
- Pass the odds range check

Then it must pass gates depending on mode:

### EV/ROI Mode Gates
Favorites must pass:
- `edge >= FAV_MIN_EDGE`
- `ev >= FAV_MIN_EV`
- `P_final >= FAV_MIN_FINAL_WINPROB`

Underdogs must pass:
- `edge >= DOG_MIN_EDGE`
- `ev >= DOG_MIN_EV`
- `P_final >= DOG_MIN_FINAL_WINPROB`

### Win% Mode Gates (Accuracy-first)
Favorites must pass:
- `P_final >= WIN_MIN_FINAL_PROB_FAV`
- `edge >= WIN_MIN_EDGE_FAV`
- `ev >= WIN_MIN_EV_FAV` (can be slightly negative)

Underdogs (only if allowed) must pass:
- `P_final >= WIN_MIN_FINAL_PROB_DOG`
- `edge >= WIN_MIN_EDGE_DOG`
- `ev >= WIN_MIN_EV_DOG`

Optional: disable dogs entirely in Win% mode:
- `WIN_DISABLE_DOGS=1`

---

## 📈 Volume Control (Cap Instead of Over-filter)

Instead of killing plays early, the engine:
1. Evaluates all viable candidates
2. Selects the **best candidate per game**
3. Sorts the day’s picks
4. Publishes only the top N (`MAX_PICKS_PER_DAY`)

Sorting depends on mode:
- **EV/ROI mode:** EV → edge → win probability
- **Win% mode:** win probability → edge → EV

---

## 🧠 Elo Fallback (Optional)

An Elo fallback can admit high-confidence plays when gates fail (useful to prevent empty slates).

Fallback requires:
- Elo win prob >= `ELO_FALLBACK_WINPROB`
- EV >= `ELO_FALLBACK_MIN_EV`
- Blocks extreme chalk beyond `ELO_FALLBACK_MAX_FAV`

Toggles:
- `ELO_FALLBACK_ENABLED=1|0`

---

## 🧪 Debugging & Transparency

Each daily output includes odds-matching stats:

```json
"debug": {
  "odds": {
    "total": 56,
    "matched": 48,
    "no_match": 8,
    "no_prices": 0,
    "no_candidates": 5
  }
}
If matched = 0, odds matching failed — not the model.

🔧 Tunables (Environment Variables)
Below are the key environment variables and what they do.

NCAA / Generation
NCAA_API_BASE (default: https://ncaa-api.henrygd.me)
Base URL for NCAA scoreboard API.

NCAA_SPORT (default: basketball-men)

NCAA_DIVISION (default: d1)

FUTURE_DAYS (default: 3)
Generate picks for today through today + FUTURE_DAYS.

PICKS_DATE (default: unset)
If set to YYYY-MM-DD, treats that as “today” (testing/backfills).

Elo Core
HOME_COURT_ADV_ELO (default: 65)
Home advantage in Elo points (0 on neutral sites).

ELO_K (default: 20)
Base K-factor.

MOV + Recency
HALF_LIFE_DAYS (default: 30)
Lower = recent games matter more.

MOV_CAP (default: 2.0)
Caps blowout impact.

Season-phase K Ramp
K_EARLY_MULT (default: 1.40)

K_MID_MULT (default: 1.00)

K_LATE_MULT (default: 0.85)

K_RAMP_EARLY_DAYS (default: 45)

K_RAMP_MID_DAYS (default: 105)

Odds API
ODDS_API_KEY (default: empty)
Enables odds integration when set.

ODDS_API_BASE (default: https://api.the-odds-api.com/v4)

ODDS_SPORT_KEY (default: basketball_ncaab)

REQUIRE_ODDS (default: 1)
If true, only publish games with matched odds/prices.

ODDS_TIME_MATCH_WINDOW_SEC (default: 43200 = 12 hours)
Time tolerance when matching NCAA games to Odds API.

Odds Ranges
DOG_MIN (default: 110)

DOG_MAX (default: 260)

FAV_MIN (default: -320)

FAV_MAX (default: -115)

Market Blend
MARKET_BLEND_ALPHA (default: 0.65)
Higher = more market-following.

EV/ROI Gates (default mode)
DOG_MIN_EDGE (default: 0.02)

DOG_MIN_EV (default: 0.02)

DOG_MIN_FINAL_WINPROB (default: 0.32)

FAV_MIN_EDGE (default: 0.01)

FAV_MIN_EV (default: 0.004)

FAV_MIN_FINAL_WINPROB (default: 0.60)

DISABLE_GATES (default: 0)
Bypass gates entirely (still respects odds ranges).

ONLY_POSITIVE_EV (default: 1)
EV/ROI mode only: filters picks to EV > 0.

MAX_PICKS_PER_DAY (default: 5)
Publish top N picks.

Win% Mode (accuracy-first)
WIN_RATE_MODE (default: 0)
Enables accuracy-first selection + sorting.

WIN_DISABLE_DOGS (default: 1)
If true, dogs are rejected in Win% mode.

Win% thresholds:

WIN_MIN_FINAL_PROB_FAV (default: 0.62)

WIN_MIN_EDGE_FAV (default: 0.005)

WIN_MIN_EV_FAV (default: -0.01)

WIN_MIN_FINAL_PROB_DOG (default: 0.54)

WIN_MIN_EDGE_DOG (default: 0.010)

WIN_MIN_EV_DOG (default: -0.01)

Elo Fallback
ELO_FALLBACK_ENABLED (default: 1)

ELO_FALLBACK_WINPROB (default: 0.66)

ELO_FALLBACK_MAX_FAV (default: -450)

ELO_FALLBACK_MIN_EV (default: -0.01)

Runtime / Storage
API_SLEEP_SECONDS (default: 0.25)
Delay between API calls.

ELO_SNAPSHOT_PATH (default: data/elo_snapshot.json)
Where Elo ratings snapshot is stored.

✅ Suggested Presets
Accuracy-first (Higher raw win %)
WIN_RATE_MODE=1
WIN_DISABLE_DOGS=1
WIN_MIN_FINAL_PROB_FAV=0.62
MAX_PICKS_PER_DAY=3

# optional: keep favorites “decent chalk”
FAV_MIN=-260
FAV_MAX=-115
EV/ROI-first (Value hunting)
WIN_RATE_MODE=0
ONLY_POSITIVE_EV=1
MAX_PICKS_PER_DAY=5
DOG_MIN=110
DOG_MAX=260
FAV_MIN=-320
FAV_MAX=-115
📁 Outputs
Daily picks: public/picks/YYYY-MM-DD.json

Latest: public/picks/latest.json

Manifest: public/picks/manifest.json

Elo snapshot: data/elo_snapshot.json

🧭 Roadmap
Confidence tiers (A / B plays)

ROI tracking by odds band

Closing-line value (CLV)

Auto-tuning from historical results