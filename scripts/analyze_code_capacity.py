"""Print the Phase 2 numbers quoted in the README, recomputed from the committed CSVs.

    python scripts/analyze_code_capacity.py threshold
    python scripts/analyze_code_capacity.py gross
"""

import sys
from pathlib import Path

import numpy as np

from qec_bench.analysis import (
    any_of_copies_fail,
    bootstrap_crossings_from_counts,
    counts_by_p,
    crossings_from_counts,
    matched_distance,
)
from qec_bench.cc_sweep import SURFACE_DISTANCES, read_rows

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Pre-registered (Phase 2 plan): MWPM (9, 11) crossing point estimate in [9%, 11%].
# Reference: p_c0 = 0.1031 +/- 0.0001, the T=0 critical point of the 2D random-bond Ising
# model, identified with the minimum-weight decoding threshold (Wang, Harrington, Preskill,
# Ann. Phys. 303, 31 (2003), arXiv:quant-ph/0207088, abstract).
BAND = (0.09, 0.11)


def report_threshold() -> None:
    rows = read_rows(RESULTS / "cc_surface_threshold.csv")
    capped = [r for r in rows if r["hit_shot_cap"] == "True"]
    print(f"{len(capped)} of {len(rows)} (task, decoder) points hit the shot cap")
    for decoder in ("mwpm", "bposd"):
        print(f"\n{decoder}: adjacent-pair crossings (per shot)")
        for d_small, d_large in zip(SURFACE_DISTANCES, SURFACE_DISTANCES[1:]):
            small = counts_by_p(rows, "surface", d_small, decoder)
            large = counts_by_p(rows, "surface", d_large, decoder)
            point = crossings_from_counts(small, large)
            boot = bootstrap_crossings_from_counts(small, large)
            ok = boot[~np.isnan(boot)]
            interval = (f"[{np.percentile(ok, 2.5):.4f}, {np.percentile(ok, 97.5):.4f}]"
                        if len(ok) else "n/a")
            print(f"  ({d_small},{d_large}): {[round(c, 4) for c in point]}  "
                  f"bootstrap 95% {interval}, no-crossing {np.isnan(boot).mean():.1%}")
    small = counts_by_p(rows, "surface", 9, "mwpm")
    large = counts_by_p(rows, "surface", 11, "mwpm")
    point = crossings_from_counts(small, large)
    verdict = "PASS" if point and BAND[0] <= point[0] <= BAND[1] else "FAIL"
    print(f"\nPre-registered: MWPM (9,11) crossing {point[:1]} in {BAND}: {verdict}")


COPIES = 12  # the gross code encodes 12 logical qubits


def surface_qubits(d: float) -> tuple[float, float]:
    """(data, data + check) qubits for 12 rotated surface-code patches of distance d."""
    return COPIES * d**2, COPIES * (2 * d**2 - 1)


def _matched_d(gross: tuple[int, int], surface: dict[int, tuple[int, int]]) -> float | None:
    target = gross[1] / gross[0]
    p12 = {d: any_of_copies_fail(k / n, COPIES) for d, (n, k) in surface.items()}
    return matched_distance(target, p12)


def report_gross(n_boot: int = 2000) -> None:
    rows = read_rows(RESULTS / "cc_gross_vs_surface.csv")
    gross = counts_by_p(rows, "gross", 12, "bposd")
    rng = np.random.default_rng(0)
    print("p      P_gross (BP-OSD)   decoder for surface: d*  [95%]  -> data qubits / total qubits")
    for decoder in ("mwpm", "bposd"):
        print(f"--- surface decoded with {decoder}; gross: 144 data / 288 total")
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
            out_of_range = 1 - len(ok) / n_boot
            p12 = {d: any_of_copies_fail(sk / sn, COPIES) for d, (sn, sk) in surface.items()}
            if d_star is None:
                side = "> d=11 (gross better)" if k / n < p12[11] else "< d=3 (gross worse)"
                print(f"{p:.2f}   {k / n:.3e}   d* {side}; P12(d=3..11) = "
                      + ", ".join(f"{v:.2e}" for v in p12.values()))
                continue
            lo, hi = np.percentile(ok, [2.5, 97.5])
            data, total = surface_qubits(d_star)
            print(f"{p:.2f}   {k / n:.3e} +/- {np.sqrt(k) / n:.1e}   d* = {d_star:.2f} "
                  f"[{lo:.2f}, {hi:.2f}]  -> {data:.0f} / {total:.0f}"
                  f"  (bootstrap out-of-range {out_of_range:.1%})")


if __name__ == "__main__":
    {"threshold": report_threshold, "gross": report_gross}[sys.argv[1]]()
