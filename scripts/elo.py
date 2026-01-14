import math

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability A beats B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

def update_elo(r_a: float, r_b: float, score_a: float, k: float = 20.0):
    """
    score_a: 1.0 win, 0.0 loss, 0.5 tie
    """
    exp_a = expected_score(r_a, r_b)
    exp_b = 1.0 - exp_a
    new_a = r_a + k * (score_a - exp_a)
    new_b = r_b + k * ((1.0 - score_a) - exp_b)
    return new_a, new_b
