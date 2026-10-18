"""Threshold-crossing and low-p slope estimates from sinter stats."""

import math

import numpy as np
import sinter


def logical_error_rate(stat: sinter.TaskStats, per_round: bool) -> float:
    """Per-shot failure fraction, or the equivalent per-round rate.

    Per-round conversion uses sinter's model: r independent rounds, each flipping
    the observable with the same probability, XOR-combined over the shot.
    """
    shot_rate = stat.errors / stat.shots
    if not per_round:
        return shot_rate
    return sinter.shot_error_rate_to_piece_error_rate(shot_rate, pieces=stat.json_metadata["r"])


def rates_by_p(
    stats: list[sinter.TaskStats], d: int, per_round: bool
) -> dict[float, float]:
    """Map p -> logical error rate for distance d, skipping points with zero errors."""
    return {
        s.json_metadata["p"]: logical_error_rate(s, per_round)
        for s in stats
        if s.json_metadata["d"] == d and s.errors > 0
    }


def find_crossings(
    stats: list[sinter.TaskStats], d_small: int, d_large: int, per_round: bool = True
) -> list[float]:
    """p values where the d_large curve crosses from below to above the d_small curve.

    Interpolates log(rate_large / rate_small) linearly in log p between adjacent
    shared p values. An empty list means no crossing inside the sampled range.
    """
    small = rates_by_p(stats, d_small, per_round)
    large = rates_by_p(stats, d_large, per_round)
    ps = sorted(set(small) & set(large))
    gaps = [math.log(large[p] / small[p]) for p in ps]
    crossings = []
    for (p0, g0), (p1, g1) in zip(zip(ps, gaps), zip(ps[1:], gaps[1:])):
        if g0 < 0 <= g1:
            t = -g0 / (g1 - g0)
            crossings.append(math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))))
    return crossings


def fit_low_p_slope(
    stats: list[sinter.TaskStats], d: int, p_max: float = 0.004, min_errors: int = 100
) -> float | None:
    """Slope of log p_L vs log p (per shot) over well-sampled low-p points.

    Returns None ("insufficient data") if fewer than two points qualify.
    """
    points = [
        (s.json_metadata["p"], s.errors / s.shots)
        for s in stats
        if s.json_metadata["d"] == d
        and s.json_metadata["p"] <= p_max
        and s.errors >= min_errors
    ]
    if len(points) < 2:
        return None
    log_p, log_rate = np.log(np.array(points)).T
    slope, _intercept = np.polyfit(log_p, log_rate, 1)
    return float(slope)
