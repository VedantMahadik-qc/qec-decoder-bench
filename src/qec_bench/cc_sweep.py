"""Code-capacity Monte Carlo sweeps: tasks, a picklable worker, and CSV output.

Each task is one (code, p). The worker draws shared batches of iid X errors and
decodes every batch with every decoder that is still below its budget, so for
shots both decoders process they see identical syndromes.
"""

import csv
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from qec_bench.codes import CSSCode, gross_code, logical_operators, rotated_surface_code
from qec_bench.decoding import (
    Decoder,
    decode_all,
    logical_failures,
    make_bposd,
    make_mwpm,
    sample_code_capacity,
)

SURFACE_DISTANCES = (3, 5, 7, 9, 11)
THRESHOLD_PS = tuple(round(0.05 + 0.005 * i, 3) for i in range(17))  # 0.050 ... 0.130
GROSS_PS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)

CSV_FIELDS = [
    "code",
    "d",
    "n",
    "k",
    "p",
    "decoder",
    "shots",
    "errors",
    "decode_seconds",
    "seed",
    "hit_shot_cap",
]


@dataclass(frozen=True)
class Budget:
    max_shots: int
    max_errors: int


@dataclass(frozen=True)
class CCTask:
    code_name: str  # "surface" or "gross"
    d: int
    p: float
    budgets: tuple[tuple[str, Budget], ...]  # (decoder name, budget)
    seed: int
    batch: int = 1000


def build_code(code_name: str, d: int) -> CSSCode:
    return gross_code() if code_name == "gross" else rotated_surface_code(d)


def task_seed(code_name: str, d: int, p: float, base: int = 2024) -> int:
    """Deterministic per-task seed, so reruns reproduce the same samples."""
    tag = 1 if code_name == "gross" else 0
    return int(np.random.SeedSequence([base, tag, d, round(p * 1e6)]).generate_state(1)[0])


def _make_decoder(name: str, code: CSSCode, p: float) -> Decoder:
    return make_mwpm(code) if name == "mwpm" else make_bposd(code, p)


def run_task(task: CCTask) -> list[dict]:
    """Run one (code, p) task to its budgets; return one CSV row per decoder."""
    code = build_code(task.code_name, task.d)
    _lx, lz = logical_operators(code)
    decoders = {name: _make_decoder(name, code, task.p) for name, _ in task.budgets}
    budgets = dict(task.budgets)
    tally = {name: {"shots": 0, "errors": 0, "seconds": 0.0} for name in decoders}
    rng = np.random.default_rng(task.seed)

    def active(name: str) -> bool:
        t, b = tally[name], budgets[name]
        return t["shots"] < b.max_shots and t["errors"] < b.max_errors

    while any(active(name) for name in decoders):
        errors, syndromes = sample_code_capacity(code, task.p, task.batch, rng)
        for name, decoder in decoders.items():
            if not active(name):
                continue
            start = time.perf_counter()
            corrections = decode_all(decoder, syndromes)
            tally[name]["seconds"] += time.perf_counter() - start
            tally[name]["shots"] += task.batch
            tally[name]["errors"] += int(logical_failures(code, lz, errors, corrections).sum())

    return [
        {
            "code": task.code_name,
            "d": task.d,
            "n": code.n,
            "k": code.k,
            "p": task.p,
            "decoder": name,
            "shots": t["shots"],
            "errors": t["errors"],
            "decode_seconds": round(t["seconds"], 4),
            "seed": task.seed,
            "hit_shot_cap": t["errors"] < budgets[name].max_errors,
        }
        for name, t in tally.items()
    ]


def threshold_tasks(mwpm: Budget, bposd: Budget, batch: int = 1000) -> list[CCTask]:
    budgets = (("mwpm", mwpm), ("bposd", bposd))
    return [
        CCTask("surface", d, p, budgets, task_seed("surface", d, p), batch)
        for d in SURFACE_DISTANCES
        for p in THRESHOLD_PS
    ]


def gross_comparison_tasks(surface: Budget, gross: Budget, batch: int = 1000) -> list[CCTask]:
    """Gross code with BP-OSD; surface d = 3..11 with MWPM (and BP-OSD for reference)."""
    tasks = [
        CCTask("gross", 12, p, (("bposd", gross),), task_seed("gross", 12, p), batch)
        for p in GROSS_PS
    ]
    tasks += [
        CCTask(
            "surface",
            d,
            p,
            (("mwpm", surface), ("bposd", surface)),
            task_seed("surface", d, p),
            batch,
        )
        for d in SURFACE_DISTANCES
        for p in GROSS_PS
    ]
    return tasks


def run_tasks(tasks: list[CCTask], csv_path: Path, workers: int) -> list[dict]:
    """Run tasks in a process pool, appending rows to csv_path as each task finishes.

    Tasks whose (code, d, p) already appear in the CSV are skipped (resume).
    Must be called from under `if __name__ == "__main__":` on Windows (spawn).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["code"], int(r["d"]), float(r["p"])) for r in read_rows(csv_path)}
    todo = [t for t in tasks if (t.code_name, t.d, t.p) not in done]
    new_file = not csv_path.exists()
    rows = []
    with open(csv_path, "a", newline="") as f, Pool(workers) as pool:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for i, task_rows in enumerate(pool.imap_unordered(run_task, todo), start=1):
            writer.writerows(task_rows)
            f.flush()
            rows += task_rows
            summary = ", ".join(f"{r['decoder']} {r['errors']}/{r['shots']}" for r in task_rows)
            print(
                f"[{i}/{len(todo)}] {task_rows[0]['code']} d={task_rows[0]['d']} "
                f"p={task_rows[0]['p']}: {summary}",
                flush=True,
            )
    return rows


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def estimate_seconds(smoke_rows: list[dict], tasks: list[CCTask], workers: int) -> float:
    """Wall-clock estimate for the full run, from a smoke run's per-shot times and rates.

    Shots needed per (task, decoder) = min(max_shots, max_errors / rate), with rate
    from the smoke run (max_shots if the smoke run saw no errors). Sampling overhead
    is ignored, and the total work is assumed to spread evenly over the workers.
    """
    smoke = {(r["code"], int(r["d"]), float(r["p"]), r["decoder"]): r for r in smoke_rows}
    total = 0.0
    for task in tasks:
        for name, budget in task.budgets:
            r = smoke[(task.code_name, task.d, task.p, name)]
            shots, errors = int(r["shots"]), int(r["errors"])
            per_shot = float(r["decode_seconds"]) / shots
            needed = (
                budget.max_shots
                if errors == 0
                else min(budget.max_shots, budget.max_errors * shots / errors)
            )
            total += per_shot * needed
    return total / workers


TIMING_FIELDS = ["code", "d", "n", "p", "method", "shots", "repeats", "median_us_per_shot"]


def time_decoders(code_name: str, d: int, p: float, shots: int, repeats: int) -> list[dict]:
    """Median decode time per shot (microseconds), single process, decode only.

    Methods: "PyMatching (per-shot call)" and "BP-OSD (per-shot call)" call decode()
    once per shot, so both pay the same Python call overhead; "PyMatching
    (decode_batch, per shot)" is PyMatching's realistic usage. BP-OSD has no batch API.
    """
    code = build_code(code_name, d)
    _errors, syndromes = sample_code_capacity(code, p, shots, np.random.default_rng(0))
    bposd = make_bposd(code, p)
    methods = {"BP-OSD (per-shot call)": lambda: [bposd.decode(s) for s in syndromes]}
    if code_name == "surface":
        mwpm = make_mwpm(code)
        methods["PyMatching (per-shot call)"] = lambda: [mwpm.decode(s) for s in syndromes]
        methods["PyMatching (decode_batch, per shot)"] = lambda: mwpm.decode_batch(syndromes)
    rows = []
    for method, run in methods.items():
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        rows.append(
            {
                "code": code_name,
                "d": d,
                "n": code.n,
                "p": p,
                "method": method,
                "shots": shots,
                "repeats": repeats,
                "median_us_per_shot": round(float(np.median(times)) / shots * 1e6, 3),
            }
        )
    return rows
