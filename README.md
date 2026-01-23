## Market (Odds API) EV Picks

This project can optionally blend the Elo model with real betting markets from **The Odds API** and then surface picks based on **expected value (EV)** instead of “most likely winner”.

### What changes when Market is enabled
- Pulls **moneyline (h2h)** odds for the date window.
- Matches each NCAA game to an Odds API event (team-name similarity + time proximity).
- Computes a **de-vigged market probability** (removes bookmaker margin).
- Blends Elo + market into a conservative final win probability:
  - `p_final = alpha * p_market + (1 - alpha) * p_elo`
- Evaluates both sides (home/away) and selects the side with the best EV *if it qualifies*.

### Key environment variables (tuning knobs)

#### Required
- `ODDS_API_KEY` — your The Odds API key (recommended to store as `GitHub Secrets`).
- `REQUIRE_ODDS` — `"1"` to **only include games that exist in the market** (recommended).
  - When `REQUIRE_ODDS=1`, any game without odds (or without a qualifying bet) is dropped.

#### Sweet-spot odds ranges (default ROI-style)
These ranges are designed to avoid heavy chalk and focus on lines where your model edge is most likely to matter:
- Underdogs: `DOG_MIN=125` to `DOG_MAX=325`
- Favorites: `FAV_MIN=-220` to `FAV_MAX=-120`

Examples:
- `-1100` is excluded (too heavy favorite)
- `+450` is excluded (too long a longshot)
- `-150` is included (light favorite)
- `+180` is included (underdog)

#### Blend strength (how much you trust market)
- `MARKET_BLEND_ALPHA=0.65`
  - Higher = closer to market (more conservative)
  - Lower = more Elo-driven (more aggressive)

#### Qualification gates (filters to avoid thin edges)
Favorites must meet:
- `FAV_MIN_EDGE=0.015` (final prob minus market prob)
- `FAV_MIN_EV=0.01`

Underdogs must meet:
- `DOG_MIN_EDGE=0.03`
- `DOG_MIN_EV=0.02`
- `DOG_MIN_FINAL_WINPROB=0.23`

### Debugging / sanity checks
When Market is enabled, the build logs print a daily summary:

`[odds] YYYY-MM-DD total=.. matched=.. no_match=.. no_prices=.. no_candidates=.. require_odds=True`

- `matched`: games where we found odds AND a qualifying bet
- `no_match`: NCAA games that didn’t match an Odds API event
- `no_prices`: matched event but couldn’t extract both sides’ prices
- `no_candidates`: odds existed but both sides failed range/gates

If your JSON shows `pick_odds_american: null` for every game, odds matching is failing (check the `[odds]` summary + matching window).

### UI behavior
- When Market is enabled, the UI labels Top 5 as **EV Picks** and shows:
  - Odds (American), EV, edge, market probability, and book.
- When Market is off, it shows **Most Likely Winners** based on Elo win probability.
