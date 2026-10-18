"""Run the surface-code memory sweep with sinter.

python scripts/run_surface_sweep.py --sweep main
python scripts/run_surface_sweep.py --sweep replication
python scripts/run_surface_sweep.py --sweep followup
"""

import argparse
import os
from pathlib import Path

from qec_bench.sweep import (
    DISTANCES,
    FOLLOWUP_DISTANCES,
    FOLLOWUP_PS,
    MAIN_PS,
    NOTEBOOK_PS,
    make_tasks,
    run_sweep,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["main", "replication", "followup"], default="main")
    parser.add_argument("--out", type=Path, default=None, help="override CSV path")
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--max-errors", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep == "main":
        tasks = make_tasks(DISTANCES, MAIN_PS, rounds_factor=1)
        default_out, default_errors = RESULTS / "surface_sweep.csv", 1_000
    elif args.sweep == "followup":
        tasks = make_tasks(FOLLOWUP_DISTANCES, FOLLOWUP_PS, rounds_factor=1)
        default_out, default_errors = RESULTS / "rounds_d_followup.csv", 5_000
    else:
        tasks = make_tasks(DISTANCES, NOTEBOOK_PS, rounds_factor=3)
        default_out, default_errors = RESULTS / "notebook_replication.csv", 5_000
    run_sweep(
        tasks,
        csv_path=args.out or default_out,
        num_workers=args.workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors or default_errors,
    )


# Required on Windows: sinter's worker processes are spawned and re-import this file.
if __name__ == "__main__":
    main()
