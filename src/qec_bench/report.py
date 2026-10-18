"""Every number the README quotes, recomputed from the committed CSVs (no re-simulation).

Each `*_numbers` function returns plain JSON-serialisable data; each `print_*` function
formats that data exactly as the original analysis scripts printed it.
"""

import math
from pathlib import Path

import numpy as np
import sinter

from qec_bench.analysis import (
    any_of_copies_fail,
    bootstrap_crossings,
    bootstrap_crossings_from_counts,
    counts_by_p,
    crossings_from_counts,
    find_crossings,
    fit_low_p_slope,
    logical_error_rate,
    matched_distance,
    ps_within_n_sigma,
    rate_sigma,
)
from qec_bench.cc_sweep import SURFACE_DISTANCES, read_rows
from qec_bench.plotting import load_stats

# --- Phase 1: circuit-level surface code sweeps -----------------------------------------

# Pre-registered band for "roughly 1%" (Stim doc/getting_started.ipynb, section 9).
THRESHOLD_BAND = (0.007, 0.011)
PAIRS = [(3, 5), (5, 7)]


def _pair_key(d_small: int, d_large: int) -> str:
    return f"{d_small},{d_large}"


def surface_sweep_numbers(stats: list[sinter.TaskStats]) -> dict:
    """Crossings (per round and per shot), bootstrap intervals, 2-sigma overlaps, low-p slopes."""
    numbers: dict = {"crossings": {}, "bootstrap": {}, "within_2sigma": {}, "slopes": {}}
    for unit, per_round in (("per_round", True), ("per_shot", False)):
        numbers["crossings"][unit] = {
            _pair_key(a, b): find_crossings(stats, a, b, per_round=per_round) for a, b in PAIRS
        }
    for a, b in PAIRS:
        boot = bootstrap_crossings(stats, a, b, n_boot=2000, seed=0)
        found = boot[~np.isnan(boot)]
        lo, med, hi = np.percentile(found, [2.5, 50, 97.5]) if found.size else (np.nan,) * 3
        numbers["bootstrap"][_pair_key(a, b)] = {
            "median": float(med),
            "lo": float(lo),
            "hi": float(hi),
            "no_crossing_fraction": float(np.isnan(boot).mean()),
        }
        numbers["within_2sigma"][_pair_key(a, b)] = ps_within_n_sigma(stats, a, b, n_sigma=2.0)
    for d in sorted({s.json_metadata["d"] for s in stats}):
        numbers["slopes"][str(d)] = fit_low_p_slope(stats, d)
    return numbers


def print_surface_sweep_report(numbers: dict) -> None:
    print("Threshold crossings:")
    for unit, label in (("per_round", "per round"), ("per_shot", "per shot")):
        for a, b in PAIRS:
            crossings = numbers["crossings"][unit][_pair_key(a, b)]
            if not crossings:
                print(f"  ({a},{b}) {label}: no crossing in sampled range")
                continue
            text = ", ".join(f"{c:.4%}" for c in crossings)
            in_band = all(THRESHOLD_BAND[0] <= c <= THRESHOLD_BAND[1] for c in crossings)
            print(f"  ({a},{b}) {label}: {text}  in band {THRESHOLD_BAND}: {in_band}")
    print("Crossing uncertainty (binomial parametric bootstrap, 2000 replicates):")
    for a, b in PAIRS:
        boot = numbers["bootstrap"][_pair_key(a, b)]
        print(
            f"  ({a},{b}) per round: bootstrap median {boot['median']:.3%}, "
            f"95% interval [{boot['lo']:.3%}, {boot['hi']:.3%}], no crossing in "
            f"{boot['no_crossing_fraction']:.1%} of replicates"
        )
        close = numbers["within_2sigma"][_pair_key(a, b)]
        print(f"    p where curves agree within 2 sigma: {[f'{p:.4%}' for p in close]}")
    print("Low-p slopes (p <= 0.004, >= 100 errors; report only):")
    for d, slope in numbers["slopes"].items():
        measured = "insufficient data" if slope is None else f"{slope:.2f}"
        print(f"  d={d}: measured {measured}, ceil(d/2) = {math.ceil(int(d) / 2)}")


ROUNDS_COMPARISON_PS = (0.008, 0.009, 0.010, 0.011, 0.012)


def rounds_comparison(
    stats_d: list[sinter.TaskStats], stats_3d: list[sinter.TaskStats]
) -> list[dict]:
    """Per-round p_L at rounds = d vs rounds = 3d, at the (d, p) both sweeps share.

    ratio = rate(rounds=d) / rate(rounds=3d); z = (rate(3d) - rate(d)) / combined sigma,
    so z < 0 means the rounds = d rate is higher.
    """
    by_d = {(s.json_metadata["d"], s.json_metadata["p"]): s for s in stats_d}
    by_3d = {(s.json_metadata["d"], s.json_metadata["p"]): s for s in stats_3d}
    rows = []
    for d, p in sorted(set(by_d) & set(by_3d)):
        if p not in ROUNDS_COMPARISON_PS:
            continue
        a, b = by_d[(d, p)], by_3d[(d, p)]
        rate_a, rate_b = logical_error_rate(a, True), logical_error_rate(b, True)
        sigma = math.hypot(rate_sigma(a, True), rate_sigma(b, True))
        rows.append({"d": d, "p": p, "ratio": rate_a / rate_b, "z": (rate_b - rate_a) / sigma})
    return rows


def print_rounds_comparison(rows: list[dict]) -> None:
    print("Per-round p_L, rounds = d (main) / rounds = 3d (replication), p = 0.008-0.012:")
    for d in sorted({r["d"] for r in rows}):
        ratios = [r["ratio"] for r in rows if r["d"] == d]
        zs = [r["z"] for r in rows if r["d"] == d]
        print(
            f"  d={d}: ratio {min(ratios):.2f}-{max(ratios):.2f}, z {min(zs):.1f} to {max(zs):.1f}"
        )


PHASE1_SWEEPS = {
    "main": "surface_sweep.csv",
    "replication": "notebook_replication.csv",
    "followup": "rounds_d_followup.csv",
}


def phase1_numbers(results_dir: Path) -> dict:
    stats = {name: load_stats(results_dir / csv) for name, csv in PHASE1_SWEEPS.items()}
    numbers = {name: surface_sweep_numbers(s) for name, s in stats.items()}
    numbers["rounds_comparison"] = rounds_comparison(stats["main"], stats["replication"])
    return numbers


def print_phase1(numbers: dict) -> None:
    for name, csv in PHASE1_SWEEPS.items():
        print(f"=== {name} ({csv})")
        print_surface_sweep_report(numbers[name])
        print()
    print_rounds_comparison(numbers["rounds_comparison"])


# --- Phase 2: code capacity ---------------------------------------------------------------

# Pre-registered (Phase 2 plan): MWPM (9, 11) crossing point estimate in [9%, 11%].
# Reference: p_c0 = 0.1031 +/- 0.0001, the T=0 critical point of the 2D random-bond Ising
# model, identified with the minimum-weight decoding threshold (Wang, Harrington, Preskill,
# Ann. Phys. 303, 31 (2003), arXiv:quant-ph/0207088, abstract).
BAND = (0.09, 0.11)
DECODERS = ("mwpm", "bposd")


def threshold_numbers(rows: list[dict]) -> dict:
    """Adjacent-pair crossings with bootstrap intervals, and the pre-registered verdict."""
    numbers: dict = {
        "points": len(rows),
        "hit_shot_cap": sum(r["hit_shot_cap"] == "True" for r in rows),
        "crossings": {},
    }
    for decoder in DECODERS:
        numbers["crossings"][decoder] = {}
        for a, b in zip(SURFACE_DISTANCES, SURFACE_DISTANCES[1:]):
            small = counts_by_p(rows, "surface", a, decoder)
            large = counts_by_p(rows, "surface", b, decoder)
            boot = bootstrap_crossings_from_counts(small, large)
            ok = boot[~np.isnan(boot)]
            numbers["crossings"][decoder][_pair_key(a, b)] = {
                "points": crossings_from_counts(small, large),
                "lo": float(np.percentile(ok, 2.5)) if len(ok) else None,
                "hi": float(np.percentile(ok, 97.5)) if len(ok) else None,
                "no_crossing_fraction": float(np.isnan(boot).mean()),
            }
    point = numbers["crossings"]["mwpm"]["9,11"]["points"]
    numbers["preregistered"] = {
        "crossing": point[0] if point else None,
        "band": list(BAND),
        "pass": bool(point) and BAND[0] <= point[0] <= BAND[1],
    }
    return numbers


def print_threshold_report(numbers: dict) -> None:
    print(
        f"{numbers['hit_shot_cap']} of {numbers['points']} (task, decoder) points hit the shot cap"
    )
    for decoder in DECODERS:
        print(f"\n{decoder}: adjacent-pair crossings (per shot)")
        for pair, c in numbers["crossings"][decoder].items():
            interval = f"[{c['lo']:.4f}, {c['hi']:.4f}]" if c["lo"] is not None else "n/a"
            print(
                f"  ({pair}): {[round(x, 4) for x in c['points']]}  "
                f"bootstrap 95% {interval}, no-crossing {c['no_crossing_fraction']:.1%}"
            )
    pre = numbers["preregistered"]
    point = numbers["crossings"]["mwpm"]["9,11"]["points"]
    verdict = "PASS" if pre["pass"] else "FAIL"
    print(f"\nPre-registered: MWPM (9,11) crossing {point[:1]} in {BAND}: {verdict}")


COPIES = 12  # the gross code encodes 12 logical qubits


def surface_qubits(d: float) -> tuple[float, float]:
    """(data, data + check) qubits for 12 rotated surface-code patches of distance d."""
    return COPIES * d**2, COPIES * (2 * d**2 - 1)


def _matched_d(gross: tuple[int, int], surface: dict[int, tuple[int, int]]) -> float | None:
    target = gross[1] / gross[0]
    p12 = {d: any_of_copies_fail(k / n, COPIES) for d, (n, k) in surface.items()}
    return matched_distance(target, p12)


def gross_numbers(rows: list[dict], n_boot: int = 2000) -> dict:
    """Matched surface distance d* (with bootstrap interval) for the gross code at each p.

    One rng is shared across all (decoder, p) in a fixed order, so the intervals are
    reproducible for a given numpy version.
    """
    gross = counts_by_p(rows, "gross", 12, "bposd")
    rng = np.random.default_rng(0)
    numbers: dict = {}
    for decoder in DECODERS:
        numbers[decoder] = {}
        for p in sorted(gross):
            surface = {d: counts_by_p(rows, "surface", d, decoder)[p] for d in SURFACE_DISTANCES}
            n, k = gross[p]
            d_star = _matched_d(gross[p], surface)
            boots = []
            for _ in range(n_boot):
                g = (n, rng.binomial(n, k / n))
                s = {d: (sn, rng.binomial(sn, sk / sn)) for d, (sn, sk) in surface.items()}
                boots.append(_matched_d(g, s))
            ok = np.array([b for b in boots if b is not None])
            entry = {
                "gross_rate": k / n,
                "gross_sigma": math.sqrt(k) / n,
                "p12": {
                    str(d): any_of_copies_fail(sk / sn, COPIES) for d, (sn, sk) in surface.items()
                },
                "d_star": d_star,
                "out_of_range_fraction": 1 - len(ok) / n_boot,
            }
            if d_star is not None:
                lo, hi = np.percentile(ok, [2.5, 97.5])
                data, total = surface_qubits(d_star)
                entry |= {
                    "lo": float(lo),
                    "hi": float(hi),
                    "data_qubits": data,
                    "total_qubits": total,
                }
            numbers[decoder][str(p)] = entry
    return numbers


def print_gross_report(numbers: dict) -> None:
    print("p      P_gross (BP-OSD)   decoder for surface: d*  [95%]  -> data qubits / total qubits")
    for decoder in DECODERS:
        print(f"--- surface decoded with {decoder}; gross: 144 data / 288 total")
        for p, e in numbers[decoder].items():
            rate = e["gross_rate"]
            if e["d_star"] is None:
                side = "> d=11 (gross better)" if rate < e["p12"]["11"] else "< d=3 (gross worse)"
                print(
                    f"{float(p):.2f}   {rate:.3e}   d* {side}; P12(d=3..11) = "
                    + ", ".join(f"{v:.2e}" for v in e["p12"].values())
                )
                continue
            print(
                f"{float(p):.2f}   {rate:.3e} +/- {e['gross_sigma']:.1e}   "
                f"d* = {e['d_star']:.2f} [{e['lo']:.2f}, {e['hi']:.2f}]  -> "
                f"{e['data_qubits']:.0f} / {e['total_qubits']:.0f}"
                f"  (bootstrap out-of-range {e['out_of_range_fraction']:.1%})"
            )


SLOPE_PS = (0.02, 0.03, 0.04)


def _log_log_slope(counts: dict[float, tuple[int, int]]) -> float:
    ps = [p for p in sorted(counts) if p in SLOPE_PS]
    rates = [counts[p][1] / counts[p][0] for p in ps]
    slope, _intercept = np.polyfit(np.log(ps), np.log(rates), 1)
    return float(slope)


def gross_slope_numbers(rows: list[dict]) -> dict:
    """Per-shot log-log slopes over p = 2-4% (report only): gross BP-OSD, one surface patch MWPM."""
    return {
        "gross": _log_log_slope(counts_by_p(rows, "gross", 12, "bposd")),
        "surface_mwpm": {
            str(d): _log_log_slope(counts_by_p(rows, "surface", d, "mwpm"))
            for d in SURFACE_DISTANCES
        },
    }


def print_gross_slopes(numbers: dict) -> None:
    surface = ", ".join(f"d={d}: {s:.2f}" for d, s in numbers["surface_mwpm"].items())
    print(f"Log-log slopes over p = 2-4% (report only): gross {numbers['gross']:.2f}; {surface}")


def timing_numbers(rows: list[dict]) -> list[dict]:
    return [
        {
            "code": r["code"],
            "d": int(r["d"]),
            "n": int(r["n"]),
            "p": float(r["p"]),
            "method": r["method"],
            "us_per_shot": float(r["median_us_per_shot"]),
        }
        for r in rows
    ]


def print_timing(rows: list[dict]) -> None:
    print("Median decode time per shot (us):")
    for r in rows:
        print(
            f"  {r['code']} d={r['d']} (n={r['n']}) p={r['p']}: {r['method']}: {r['us_per_shot']}"
        )


def phase2_numbers(results_dir: Path) -> dict:
    gross_rows = read_rows(results_dir / "cc_gross_vs_surface.csv")
    return {
        "threshold": threshold_numbers(read_rows(results_dir / "cc_surface_threshold.csv")),
        "gross": gross_numbers(gross_rows),
        "gross_shot_cap": [
            {
                "d": int(r["d"]),
                "p": float(r["p"]),
                "decoder": r["decoder"],
                "shots": int(r["shots"]),
                "errors": int(r["errors"]),
            }
            for r in gross_rows
            if r["hit_shot_cap"] == "True"
        ],
        "slopes": gross_slope_numbers(gross_rows),
        "timing": timing_numbers(read_rows(results_dir / "decoder_timing.csv")),
    }


def print_phase2(numbers: dict) -> None:
    print("=== Experiment 1: surface-code code-capacity threshold")
    print_threshold_report(numbers["threshold"])
    print("\n=== Experiment 2: gross code vs 12 surface-code copies")
    print_gross_report(numbers["gross"])
    for r in numbers["gross_shot_cap"]:
        print(
            f"shot cap hit: {r['decoder']} surface d={r['d']} p={r['p']}: "
            f"{r['errors']} errors in {r['shots']} shots"
        )
    print_gross_slopes(numbers["slopes"])
    print("\n=== Experiment 3: decoder timing")
    print_timing(numbers["timing"])
