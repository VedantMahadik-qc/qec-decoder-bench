"""Plots for the Phase 2 code-capacity experiments, from the committed CSVs.

    python scripts/plot_code_capacity.py threshold
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qec_bench.analysis import counts_by_p
from qec_bench.cc_sweep import SURFACE_DISTANCES, read_rows

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _errorbar(ax: plt.Axes, counts: dict, label: str, **kw) -> None:
    ps = sorted(counts)
    rates = [counts[p][1] / counts[p][0] for p in ps]
    sigmas = [math.sqrt(r * (1 - r) / counts[p][0]) for p, r in zip(ps, rates)]
    ax.errorbar(ps, rates, yerr=sigmas, label=label, capsize=2, marker="o", ms=3, **kw)


def plot_threshold() -> None:
    rows = read_rows(RESULTS / "cc_surface_threshold.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, decoder, title in zip(axes, ("mwpm", "bposd"), ("PyMatching (MWPM)", "BP-OSD-CS")):
        for d in SURFACE_DISTANCES:
            _errorbar(ax, counts_by_p(rows, "surface", d, decoder), f"d={d}")
        ax.axvline(0.1031, color="grey", ls="--", lw=1, label="0.1031 (WHP 2003)")
        ax.set_yscale("log")
        ax.set_xlabel("physical bit-flip rate p")
        ax.set_title(f"Rotated surface code, code capacity, {title}")
        ax.grid(which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("logical failure rate per shot")
    fig.tight_layout()
    fig.savefig(RESULTS / "cc_surface_threshold.png", dpi=150)


if __name__ == "__main__":
    {"threshold": plot_threshold}[sys.argv[1]]()
