import math

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability A beats B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

def mov_multiplier(mov: int, cap: float = 2.0) -> float:
    """
    Margin-of-victory multiplier.
    Safe, smooth, and capped so blowouts don't explode Elo.
    """
    # ln(mov + 1) grows slowly; +0.8 keeps close games meaningful
    mult = 0.8 + 0.2 * math.log(mov + 1.0)
    return min(cap, max(0.5, mult))

def recency_weight(days_ago: int, half_life_days: float = 30.0) -> float:
    """
    Exponential decay: after `half_life_days`, weight is 0.5.
    days_ago=0 -> 1.0
    """
    if half_life_days <= 0:
        return 1.0
    if days_ago <= 0:
        return 1.0
    return math.pow(0.5, days_ago / half_life_days)

def update_elo(r_a: float, r_b: float, score_a: float, k: float = 20.0):
    """
    Basic Elo update.
    score_a: 1.0 win, 0.0 loss, 0.5 tie
    """
    exp_a = expected_score(r_a, r_b)
    exp_b = 1.0 - exp_a
    new_a = r_a + k * (score_a - exp_a)
    new_b = r_b + k * ((1.0 - score_a) - exp_b)
    return new_a, new_b

def update_elo_mov_recency(
    r_home: float,
    r_away: float,
    home_score: int,
    away_score: int,
    base_k: float = 20.0,
    days_ago: int = 0,
    half_life_days: float = 30.0,
    mov_cap: float = 2.0,
):
    """
    Elo update with:
      - MOV scaling
      - Recency weighting
    """
    if home_score > away_score:
        s_home = 1.0
    elif home_score < away_score:
        s_home = 0.0
    else:
        s_home = 0.5

    mov = abs(home_score - away_score)
    k_eff = base_k * mov_multiplier(mov, cap=mov_cap) * recency_weight(days_ago, half_life_days=half_life_days)

    return update_elo(r_home, r_away, s_home, k=k_eff)
