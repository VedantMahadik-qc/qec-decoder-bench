"""Plots for the Phase 2 code-capacity experiments, from the committed CSVs.

    python scripts/plot_code_capacity.py threshold
    python scripts/plot_code_capacity.py gross
    python scripts/plot_code_capacity.py timing
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qec_bench.analysis import any_of_copies_fail, counts_by_p
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


def plot_gross() -> None:
    """P(any of 12 logical qubits fails): one gross code vs 12 independent surface patches."""
    rows = read_rows(RESULTS / "cc_gross_vs_surface.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    _errorbar(ax, counts_by_p(rows, "gross", 12, "bposd"),
              "gross [[144,12,12]], BP-OSD (144 data qubits)", color="black", lw=2)
    for d in SURFACE_DISTANCES:
        counts = counts_by_p(rows, "surface", d, "mwpm")
        ps = sorted(counts)
        p12 = [any_of_copies_fail(counts[p][1] / counts[p][0], 12) for p in ps]
        ax.plot(ps, p12, marker="o", ms=3, ls="--", label=f"12 x surface d={d}, MWPM ({12 * d * d} data)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical bit-flip rate p")
    ax.set_ylabel("P(any of 12 logical qubits fails) per shot")
    ax.set_title("Code capacity: gross code vs 12 surface-code patches")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "cc_gross_vs_surface.png", dpi=150)


def plot_timing() -> None:
    """Median decode time per shot vs number of data qubits n, one panel per p."""
    rows = read_rows(RESULTS / "decoder_timing.csv")
    ps = sorted({float(r["p"]) for r in rows})
    fig, axes = plt.subplots(1, len(ps), figsize=(6 * len(ps), 5), sharey=True)
    for ax, p in zip(axes, ps):
        for method in dict.fromkeys(r["method"] for r in rows):
            for code, marker in (("surface", "o"), ("gross", "*")):
                pts = sorted((int(r["n"]), float(r["median_us_per_shot"])) for r in rows
                             if float(r["p"]) == p and r["method"] == method and r["code"] == code)
                if pts:
                    ns, us = zip(*pts)
                    ax.plot(ns, us, marker=marker, ms=10 if code == "gross" else 4,
                            ls="-" if code == "surface" else "none",
                            label=f"{method}, {code}")
        ax.set_yscale("log")
        ax.set_xlabel("data qubits n (surface d=3..11; gross n=144)")
        ax.set_title(f"Decode time per shot, single process, p = {p}")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("median time per shot (us)")
    fig.tight_layout()
    fig.savefig(RESULTS / "decoder_timing.png", dpi=150)


if __name__ == "__main__":
    {"threshold": plot_threshold, "gross": plot_gross, "timing": plot_timing}[sys.argv[1]]()
