"""Run the Phase 2 code-capacity experiments.

python scripts/run_code_capacity.py --exp threshold --smoke   # timed smoke run + estimate
python scripts/run_code_capacity.py --exp threshold
python scripts/run_code_capacity.py --exp gross [--smoke]
python scripts/run_code_capacity.py --exp timing
"""

import argparse
import csv
import os
import time
from dataclasses import replace
from pathlib import Path

from qec_bench.cc_sweep import (
    SURFACE_DISTANCES,
    TIMING_FIELDS,
    Budget,
    estimate_seconds,
    gross_comparison_tasks,
    run_tasks,
    threshold_tasks,
    time_decoders,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SMOKE_BUDGET = Budget(max_shots=2000, max_errors=10**9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=["threshold", "gross", "timing"], required=True)
    parser.add_argument("--smoke", action="store_true", help="short timed run to results/scratch")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def full_tasks(exp: str) -> tuple[list, Path]:
    if exp == "threshold":
        # Only the MWPM crossing is pre-registered, so BP-OSD gets a smaller budget.
        tasks = threshold_tasks(mwpm=Budget(10**6, 2000), bposd=Budget(2 * 10**5, 500))
        return tasks, RESULTS / "cc_surface_threshold.csv"
    tasks = gross_comparison_tasks(surface=Budget(10**6, 200), gross=Budget(10**6, 200))
    return tasks, RESULTS / "cc_gross_vs_surface.csv"


def run_smoke(exp: str, workers: int) -> None:
    tasks, out = full_tasks(exp)
    smoke = [replace(t, budgets=tuple((n, SMOKE_BUDGET) for n, _ in t.budgets)) for t in tasks]
    scratch = RESULTS / "scratch" / f"smoke_{exp}.csv"
    scratch.unlink(missing_ok=True)
    start = time.perf_counter()
    rows = run_tasks(smoke, scratch, workers)
    print(f"smoke run: {time.perf_counter() - start:.0f} s wall")
    est = estimate_seconds(rows, tasks, workers)
    print(
        f"estimated full run for {out.name}: {est / 60:.0f} min on {workers} workers "
        "(decode time only, assumes even load)"
    )


def run_timing() -> None:
    out = RESULTS / "decoder_timing.csv"
    rows = []
    for p in (0.05, 0.10):
        for d in SURFACE_DISTANCES:
            rows += time_decoders("surface", d, p, shots=10_000, repeats=5)
        rows += time_decoders("gross", 12, p, shots=10_000, repeats=5)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


def main() -> None:
    args = parse_args()
    if args.exp == "timing":
        run_timing()
    elif args.smoke:
        run_smoke(args.exp, args.workers)
    else:
        tasks, out = full_tasks(args.exp)
        run_tasks(tasks, out, args.workers)


# Required on Windows: worker processes are spawned and re-import this file.
if __name__ == "__main__":
    main()
