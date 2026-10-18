"""Print the Phase 2 numbers quoted in the README, recomputed from the committed CSVs.

python scripts/analyze_code_capacity.py threshold
python scripts/analyze_code_capacity.py gross
"""

import sys
from pathlib import Path

from qec_bench.cc_sweep import read_rows
from qec_bench.report import (
    gross_numbers,
    print_gross_report,
    print_threshold_report,
    threshold_numbers,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def report_threshold() -> None:
    print_threshold_report(threshold_numbers(read_rows(RESULTS / "cc_surface_threshold.csv")))


def report_gross() -> None:
    print_gross_report(gross_numbers(read_rows(RESULTS / "cc_gross_vs_surface.csv")))


if __name__ == "__main__":
    {"threshold": report_threshold, "gross": report_gross}[sys.argv[1]]()
