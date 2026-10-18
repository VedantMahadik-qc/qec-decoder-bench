"""`qec-bench` command line: single-point simulations and reproduction of the README results.

    qec-bench surface --d 5 --p 0.001 --rounds 15 --shots 100000
    qec-bench codecap --code gross --p 0.05 --decoder bposd --shots 10000
    qec-bench reproduce phase2

Single points run in one process (no worker pool), so nothing here needs the Windows
`if __name__ == "__main__":` guard beyond the console-script entry point itself.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import sinter

from qec_bench.cc_sweep import Budget, CCTask, run_task, task_seed
from qec_bench.circuits import surface_code_memory_z
from qec_bench.decoding import count_logical_errors
from qec_bench.plotting import (
    plot_cc_threshold,
    plot_decoder_timing,
    plot_gross_vs_surface,
    plot_logical_error_rate,
)
from qec_bench.report import phase1_numbers, phase2_numbers, print_phase1, print_phase2

GROSS_DISTANCE = 12


def binomial_sigma(errors: int, shots: int) -> float:
    """1-sigma binomial uncertainty of the failure fraction errors / shots."""
    rate = errors / shots
    return math.sqrt(rate * (1 - rate) / shots)


def format_rate(errors: int, shots: int) -> str:
    return f"{errors / shots:.4e} +/- {binomial_sigma(errors, shots):.1e}  ({errors}/{shots})"


def run_surface(args: argparse.Namespace) -> int:
    """One circuit-level point: Stim rotated memory-Z circuit, PyMatching decoding."""
    rounds = args.rounds if args.rounds is not None else args.d
    circuit = surface_code_memory_z(args.d, args.p, rounds=rounds)
    errors = count_logical_errors(circuit, shots=args.shots, seed=args.seed)
    per_round = sinter.shot_error_rate_to_piece_error_rate(errors / args.shots, pieces=rounds)
    print(
        f"surface code, circuit-level noise: d={args.d} p={args.p} rounds={rounds} "
        f"shots={args.shots} seed={args.seed}"
    )
    print(f"logical error rate per shot:  {format_rate(errors, args.shots)}")
    print(f"logical error rate per round: {per_round:.4e}  (sinter piece-rate conversion)")
    return 0


def run_codecap(args: argparse.Namespace) -> int:
    """One code-capacity point, via the same run_task that produced the committed CSVs."""
    if args.code == "gross":
        if args.decoder == "mwpm":
            print(
                "error: the gross code cannot be decoded with MWPM: each qubit sits in 3 Z "
                "checks, so errors are not graphlike. Use --decoder bposd.",
                file=sys.stderr,
            )
            return 2
        if args.d not in (None, GROSS_DISTANCE):
            print(
                f"error: the gross code has fixed d = {GROSS_DISTANCE}; omit --d.", file=sys.stderr
            )
            return 2
    d = GROSS_DISTANCE if args.code == "gross" else (args.d if args.d is not None else 5)
    seed = args.seed if args.seed is not None else task_seed(args.code, d, args.p)
    # max_errors above max_shots, so only the shot budget stops the run.
    budget = Budget(max_shots=args.shots, max_errors=args.shots + 1)
    task = CCTask(
        args.code, d, args.p, ((args.decoder, budget),), seed, batch=min(1000, args.shots)
    )
    (row,) = run_task(task)
    print(
        f"{args.code} code (n={row['n']}, k={row['k']}, d={d}), code-capacity X noise: "
        f"p={args.p} decoder={args.decoder} seed={seed}"
    )
    print(f"P(any logical fails) per shot: {format_rate(row['errors'], row['shots'])}")
    return 0


PHASE1_PLOTS = ("surface_sweep", "notebook_replication")
PHASE2_PLOTS = {
    "cc_surface_threshold": plot_cc_threshold,
    "cc_gross_vs_surface": plot_gross_vs_surface,
    "decoder_timing": plot_decoder_timing,
}


def run_reproduce(args: argparse.Namespace) -> int:
    """Recompute the README plots and numbers from the committed CSVs, without simulating."""
    results, out = args.results, args.out / args.phase
    if not results.is_dir():
        print(f"error: results directory {results} not found", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "phase1":
        numbers = phase1_numbers(results)
        print_phase1(numbers)
        for stem in PHASE1_PLOTS:
            plot_logical_error_rate(results / f"{stem}.csv", out / f"{stem}.png")
    else:
        numbers = phase2_numbers(results)
        print_phase2(numbers)
        for stem, plot in PHASE2_PLOTS.items():
            plot(results / f"{stem}.csv", out / f"{stem}.png")
    (out / "numbers.json").write_text(json.dumps(numbers, indent=2, allow_nan=True))
    print(f"\nWrote plots and numbers.json to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qec-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    surface = sub.add_parser("surface", help="one circuit-level surface-code point")
    surface.add_argument("--d", type=int, required=True, help="code distance")
    surface.add_argument("--p", type=float, required=True, help="circuit-level noise strength")
    surface.add_argument("--rounds", type=int, default=None, help="default: d")
    surface.add_argument("--shots", type=int, required=True)
    surface.add_argument("--seed", type=int, default=0)
    surface.set_defaults(func=run_surface)

    codecap = sub.add_parser("codecap", help="one code-capacity point")
    codecap.add_argument("--code", choices=["surface", "gross"], required=True)
    codecap.add_argument("--d", type=int, default=None, help="surface distance (default 5)")
    codecap.add_argument("--p", type=float, required=True, help="iid X error rate per qubit")
    codecap.add_argument("--decoder", choices=["mwpm", "bposd"], required=True)
    codecap.add_argument("--shots", type=int, required=True)
    codecap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="default: the per-task seed used by the committed sweeps",
    )
    codecap.set_defaults(func=run_codecap)

    reproduce = sub.add_parser("reproduce", help="rebuild README plots/numbers from CSVs")
    reproduce.add_argument("phase", choices=["phase1", "phase2"])
    reproduce.add_argument(
        "--results", type=Path, default=Path("results"), help="directory holding the committed CSVs"
    )
    reproduce.add_argument("--out", type=Path, default=Path("results") / "reproduced")
    reproduce.set_defaults(func=run_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
