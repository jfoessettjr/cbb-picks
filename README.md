# NCAAB Picks Engine – Model & Odds Integration (Updated)

This project generates daily **college basketball picks** using a blended **Elo + betting market** model, with strict safeguards to avoid bad data, extreme variance, and empty slates.

Recent updates focused on **correct odds matching**, **controlled pick volume**, and **better win-rate vs. ROI balance**.

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

```
P_final = α · P_market + (1 − α) · P_elo
```

Where `α` (MARKET_BLEND_ALPHA) controls market anchoring.

---

## 🎯 Pick Selection Philosophy

The engine is **EV-first**, but not EV-only.

Key principles:
- Rank by **expected value (EV)** to target long-term profitability
- Enforce **odds ranges** to avoid extreme chalk and lottery longshots
- Require **minimum confidence** (win probability floors)
- Cap daily output instead of over-filtering

This avoids:
- “-1100 favorites that add no value”
- “+600 longshots with awful hit rate”
- Empty slates caused by overly strict filters

---

## 🧱 Odds Safety & Matching (Critical Update)

Odds matching is now robust to:
- NCAA games missing or misformatted start times
- Small time discrepancies between NCAA and sportsbooks
- Team name variations (e.g. `ULM` vs `UL Monroe`, `St.` vs `Saint`)

Matching rules:
1. Match by team similarity (both orientations)
2. Use start time if available (±12 hours)
3. Fall back to team-only matching if NCAA time is missing
4. Enforce hard odds-range checks before publishing

If odds cannot be matched and `REQUIRE_ODDS=true`, the game is excluded.

---

## 📊 Odds Ranges (Current Defaults)

- **Underdogs:** `+110` to `+260`
- **Favorites:** `-115` to `-320`

These ranges intentionally exclude:
- Massive chalk (`-600`, `-1100`)
- Ultra-low-probability lottery bets

---

## 🚦 Gates & Filters

A play must pass:
- Odds range check
- Minimum EV
- Minimum edge vs market
- Minimum final win probability

---

## 📈 Volume Control

Instead of eliminating plays early, the model:
1. Evaluates all viable candidates
2. Sorts by EV (then edge, then win probability)
3. Publishes only the top N picks

---

## 🧠 Elo Fallback (Optional)

An optional Elo fallback can admit high-confidence plays even when EV is marginal.
Disabled by default.

---

## 🧪 Debugging & Transparency

Each daily output includes:

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
```

If `matched = 0`, odds matching failed — not the model.

---

## 🔧 Key Environment Variables

- `REQUIRE_ODDS`
- `MARKET_BLEND_ALPHA`
- `DOG_MIN / DOG_MAX`
- `FAV_MIN / FAV_MAX`
- `MAX_PICKS_PER_DAY`
- `ONLY_POSITIVE_EV`

---

## 🧭 Roadmap

- Confidence tiers (A / B plays)
- ROI tracking by odds band
- Closing-line value (CLV)
- Auto-tuning from historical results
