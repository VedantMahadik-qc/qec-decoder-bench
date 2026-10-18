"""Print the Phase 2 numbers quoted in the README, recomputed from the committed CSVs.

    python scripts/analyze_code_capacity.py threshold
"""

import sys
from pathlib import Path

import numpy as np

from qec_bench.analysis import bootstrap_crossings_from_counts, counts_by_p, crossings_from_counts
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


if __name__ == "__main__":
    {"threshold": report_threshold}[sys.argv[1]]()
