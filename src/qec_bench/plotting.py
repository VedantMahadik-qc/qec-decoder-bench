"""Log-log plots of logical error rate vs physical error rate."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter


def load_stats(csv_path: Path) -> list[sinter.TaskStats]:
    """Read a sinter CSV, merging rows that belong to the same task."""
    return sinter.read_stats_from_csv_files(str(csv_path))


def _plot_panel(ax: plt.Axes, stats: list[sinter.TaskStats], per_round: bool) -> None:
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda s: s.json_metadata["p"],
        group_func=lambda s: f"d={s.json_metadata['d']}",
        failure_units_per_shot_func=(lambda s: s.json_metadata["r"]) if per_round else (lambda s: 1),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate per " + ("round" if per_round else "shot"))
    ax.grid(which="both", alpha=0.3)
    ax.legend()


def plot_logical_error_rate(
    csv_path: Path, png_path: Path, panels: tuple[bool, ...] = (False, True)
) -> None:
    """Save a figure with one panel per entry in `panels` (False=per shot, True=per round)."""
    stats = load_stats(csv_path)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, per_round in zip(axes[0], panels):
        _plot_panel(ax, stats, per_round)
    fig.suptitle(f"Rotated surface code memory Z, PyMatching ({csv_path.name})")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
