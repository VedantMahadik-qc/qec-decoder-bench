"""sinter task construction and collection for surface-code sweeps."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import sinter

from qec_bench.circuits import surface_code_memory_z

DISTANCES = (3, 5, 7)

# Main sweep: coarse log grid over the requested range, plus a fine linear grid
# around the expected ~1% threshold so the crossing can be located.
MAIN_PS = sorted(
    {float(f"{p:.6g}") for p in np.geomspace(1e-3, 1.5e-2, 10)}
    | {float(f"{p:.6g}") for p in np.linspace(0.006, 0.012, 7)}
)

# Stim doc/getting_started.ipynb section 9: rounds = 3d, these five p values.
NOTEBOOK_PS = (0.008, 0.009, 0.010, 0.011, 0.012)


def make_tasks(
    distances: Iterable[int], ps: Iterable[float], rounds_factor: int = 1
) -> list[sinter.Task]:
    """One sinter task per (d, p), with rounds = rounds_factor * d."""
    tasks = []
    for d in distances:
        rounds = rounds_factor * d
        for p in ps:
            tasks.append(
                sinter.Task(
                    circuit=surface_code_memory_z(d, p, rounds=rounds),
                    json_metadata={"d": d, "p": p, "r": rounds},
                )
            )
    return tasks


def run_sweep(
    tasks: list[sinter.Task],
    csv_path: Path,
    num_workers: int,
    max_shots: int,
    max_errors: int,
) -> list[sinter.TaskStats]:
    """Decode all tasks with PyMatching; sinter writes (and resumes from) csv_path.

    Must be called from under `if __name__ == "__main__":` on Windows (spawn).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=max_shots,
        max_errors=max_errors,
        save_resume_filepath=str(csv_path),
        print_progress=True,
    )
