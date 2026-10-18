# CLAUDE.md

## Project

QEC decoder benchmark built on **Stim** (circuit simulation / detector error models),
**PyMatching** (MWPM decoding), **BP-OSD** from the `ldpc` package, and **sinter**
(parallel Monte Carlo sampling).

Roadmap order:
1. Rotated surface codes (memory experiments, circuit-level noise) — PyMatching, then BP-OSD.
2. The [[144,12,12]] bivariate bicycle (gross) code — BP-OSD.

## Code conventions

- Python 3.12.
- src layout: package lives in `src/qec_bench/`; tests in `tests/`.
- Packaging and tool config in `pyproject.toml` (no `setup.py`).
- Tests with `pytest`.
- Type hints on all function signatures.
- Small, single-purpose functions; prefer clarity over cleverness.

## Platform: Windows

- This is a Windows machine. multiprocessing uses `spawn`, so **any script that calls
  `sinter.collect` (or otherwise starts worker processes) must put its entry point under
  `if __name__ == "__main__":`**. Without it, workers re-import the script and recurse.
- Custom sinter decoders must be importable (module-level, picklable), not defined inside
  the `__main__` block.

## Physics validation (non-negotiable)

- Every physics result must be validated against a known value or analytic limit before
  it is trusted. Examples:
  - Zero noise ⇒ zero logical errors; detectors deterministic in the noiseless circuit.
  - Code parameters [[n, k, d]] match the definition (e.g. BB code: n=144, k=12, d=12).
  - Surface-code threshold under circuit-level depolarizing noise lands near the
    literature value (~0.7–1% for MWPM), and logical error rate is suppressed with
    distance below threshold.
  - Low-p scaling: p_L ∝ p^⌈d/2⌉ for distance d.
  - Cross-check decoders against each other on the same detector error model.
- **Never tune code to hit an expected number.** If results disagree with literature,
  stop and explain the discrepancy (noise model differences, conventions such as per-round
  vs. per-shot error rates, rounds count, decoder settings) instead of adjusting until
  it matches.
- State the source of each reference value (paper + figure/table) next to the check.

## Working style

- After each change, briefly explain what the code does and why. The user must be able
  to explain every line in an interview, so explain the physics and design choices, not
  just the mechanics.
- Commit after each working step (tests passing) with a clear, descriptive message.
