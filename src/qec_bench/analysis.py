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
    return crossings_from_rates(small, large)


def crossings_from_rates(small: dict[float, float], large: dict[float, float]) -> list[float]:
    """Crossing estimator shared by find_crossings and the bootstrap."""
    ps = sorted(p for p in set(small) & set(large) if small[p] > 0 and large[p] > 0)
    gaps = [math.log(large[p] / small[p]) for p in ps]
    crossings = []
    for (p0, g0), (p1, g1) in zip(zip(ps, gaps), zip(ps[1:], gaps[1:])):
        if g0 < 0 <= g1:
            t = -g0 / (g1 - g0)
            crossings.append(math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))))
    return crossings


def _to_rate(errors: float, shots: int, rounds: int, per_round: bool) -> float:
    shot_rate = errors / shots
    if not per_round:
        return shot_rate
    return sinter.shot_error_rate_to_piece_error_rate(shot_rate, pieces=rounds)


def bootstrap_crossings(
    stats: list[sinter.TaskStats],
    d_small: int,
    d_large: int,
    n_boot: int = 2000,
    seed: int = 0,
    per_round: bool = True,
) -> np.ndarray:
    """Parametric bootstrap of the first crossing.

    Each replicate redraws every point's error count from Binomial(shots, errors/shots)
    and reruns the same interpolation estimator. NaN marks replicates with no crossing.
    """
    rng = np.random.default_rng(seed)
    points = {
        d: [(s.json_metadata["p"], s.shots, s.errors, s.json_metadata["r"])
            for s in stats if s.json_metadata["d"] == d]
        for d in (d_small, d_large)
    }
    results = np.full(n_boot, np.nan)
    for i in range(n_boot):
        rates = {
            d: {p: _to_rate(rng.binomial(n, k / n), n, r, per_round) for p, n, k, r in pts}
            for d, pts in points.items()
        }
        found = crossings_from_rates(rates[d_small], rates[d_large])
        if found:
            results[i] = found[0]
    return results


def rate_sigma(stat: sinter.TaskStats, per_round: bool) -> float:
    """1-sigma binomial uncertainty of the rate, propagated through the per-round map."""
    n, k, r = stat.shots, stat.errors, stat.json_metadata["r"]
    shot_sigma = math.sqrt(k * (n - k) / n) / n
    h = shot_sigma * 1e-3
    slope = (_to_rate(k + h * n, n, r, per_round) - _to_rate(k - h * n, n, r, per_round)) / (2 * h)
    return abs(slope) * shot_sigma


def ps_within_n_sigma(
    stats: list[sinter.TaskStats], d_small: int, d_large: int, n_sigma: float = 2.0,
    per_round: bool = True,
) -> list[float]:
    """Shared p values where the two curves differ by less than n_sigma combined sigma."""
    by_d = {
        d: {s.json_metadata["p"]: s for s in stats if s.json_metadata["d"] == d}
        for d in (d_small, d_large)
    }
    close = []
    for p in sorted(set(by_d[d_small]) & set(by_d[d_large])):
        a, b = by_d[d_small][p], by_d[d_large][p]
        diff = logical_error_rate(b, per_round) - logical_error_rate(a, per_round)
        sigma = math.hypot(rate_sigma(a, per_round), rate_sigma(b, per_round))
        if abs(diff) < n_sigma * sigma:
            close.append(p)
    return close


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
