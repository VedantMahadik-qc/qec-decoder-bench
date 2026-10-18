"""Plot a surface-code sweep CSV and print the validation numbers.

    python scripts/plot_surface_sweep.py results/surface_sweep.csv
"""

import argparse
import math
from pathlib import Path

from qec_bench.analysis import find_crossings, fit_low_p_slope
from qec_bench.plotting import load_stats, plot_logical_error_rate

# Pre-registered band for "roughly 1%" (Stim doc/getting_started.ipynb, section 9).
THRESHOLD_BAND = (0.007, 0.011)


def report_crossings(stats: list, per_round: bool) -> None:
    unit = "per round" if per_round else "per shot"
    for d_small, d_large in [(3, 5), (5, 7)]:
        crossings = find_crossings(stats, d_small, d_large, per_round=per_round)
        if not crossings:
            print(f"  ({d_small},{d_large}) {unit}: no crossing in sampled range")
            continue
        text = ", ".join(f"{c:.4%}" for c in crossings)
        in_band = all(THRESHOLD_BAND[0] <= c <= THRESHOLD_BAND[1] for c in crossings)
        print(f"  ({d_small},{d_large}) {unit}: {text}  in band {THRESHOLD_BAND}: {in_band}")


def report_slopes(stats: list) -> None:
    ds = sorted({s.json_metadata["d"] for s in stats})
    for d in ds:
        slope = fit_low_p_slope(stats, d)
        expected = math.ceil(d / 2)
        measured = "insufficient data" if slope is None else f"{slope:.2f}"
        print(f"  d={d}: measured {measured}, ceil(d/2) = {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()

    png = args.png or args.csv.with_suffix(".png")
    plot_logical_error_rate(args.csv, png)
    stats = load_stats(args.csv)
    print(f"Saved {png}")
    print("Threshold crossings:")
    report_crossings(stats, per_round=True)
    report_crossings(stats, per_round=False)
    print("Low-p slopes (p <= 0.004, >= 100 errors; report only):")
    report_slopes(stats)


if __name__ == "__main__":
    main()
