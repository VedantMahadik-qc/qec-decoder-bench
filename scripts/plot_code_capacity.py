"""Plots for the Phase 2 code-capacity experiments, from the committed CSVs.

python scripts/plot_code_capacity.py threshold
python scripts/plot_code_capacity.py gross
python scripts/plot_code_capacity.py timing
"""

import sys
from pathlib import Path

from qec_bench.plotting import plot_cc_threshold, plot_decoder_timing, plot_gross_vs_surface

RESULTS = Path(__file__).resolve().parent.parent / "results"

PLOTS = {
    "threshold": (plot_cc_threshold, "cc_surface_threshold"),
    "gross": (plot_gross_vs_surface, "cc_gross_vs_surface"),
    "timing": (plot_decoder_timing, "decoder_timing"),
}

if __name__ == "__main__":
    plot, stem = PLOTS[sys.argv[1]]
    plot(RESULTS / f"{stem}.csv", RESULTS / f"{stem}.png")
