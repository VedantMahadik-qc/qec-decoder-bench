"""Plot a surface-code sweep CSV and print the validation numbers.

python scripts/plot_surface_sweep.py results/surface_sweep.csv
"""

import argparse
from pathlib import Path

from qec_bench.plotting import load_stats, plot_logical_error_rate
from qec_bench.report import print_surface_sweep_report, surface_sweep_numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()

    png = args.png or args.csv.with_suffix(".png")
    plot_logical_error_rate(args.csv, png)
    print(f"Saved {png}")
    print_surface_sweep_report(surface_sweep_numbers(load_stats(args.csv)))


if __name__ == "__main__":
    main()
