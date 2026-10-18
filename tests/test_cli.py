"""CLI smoke tests, and a check that `qec-bench reproduce` regenerates the README numbers."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from qec_bench import cli

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")


# --- Smoke tests: tiny shot counts, in-process -------------------------------------------


def test_surface_point_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["surface", "--d", "3", "--p", "0.005", "--rounds", "3", "--shots", "200"]) == 0
    assert "logical error rate per shot" in capsys.readouterr().out


def test_surface_zero_noise_has_no_errors(capsys: pytest.CaptureFixture[str]) -> None:
    # Analytic limit: no noise means every shot decodes correctly.
    assert cli.main(["surface", "--d", "3", "--p", "0", "--shots", "200"]) == 0
    assert "(0/200)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--code", "surface", "--d", "3", "--decoder", "mwpm"],
        ["--code", "surface", "--d", "3", "--decoder", "bposd"],
        ["--code", "gross", "--decoder", "bposd"],
    ],
    ids=["surface-mwpm", "surface-bposd", "gross-bposd"],
)
def test_codecap_point_runs(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["codecap", *args, "--p", "0.05", "--shots", "100"]) == 0
    assert "P(any logical fails) per shot" in capsys.readouterr().out


def test_codecap_rejects_gross_with_mwpm(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli.main(
            ["codecap", "--code", "gross", "--p", "0.05", "--decoder", "mwpm", "--shots", "10"]
        )
        == 2
    )
    assert "cannot be decoded with MWPM" in capsys.readouterr().err


# --- reproduce: README numbers, recomputed from the committed CSVs -------------------------
#
# Tolerances: CI may run a different numpy/platform, where bootstrap RNG streams and float
# results are not guaranteed identical. Deterministic numbers must match to half a unit of
# the last digit the README shows. Bootstrap-derived numbers get +/-0.05 percentage points
# (+/-0.05 for d*), and bootstrap "no crossing" fractions +/-0.5 percentage points.
#
# README numbers NOT checked here, because they don't come from the committed CSVs:
#   - Phase 1 validation table (p_L = 0.498; 1.05e-3 vs 2.97e-3): from tests/test_surface_code.py.
#   - Phase 2 cross-check (MWPM 0.0231 vs BP-OSD 0.0233): from tests/test_code_capacity.py.
#   - BP convergence fractions (86%, 20%, 2%): a separate decoding measurement.

BOOT_PCT = 0.0005  # 0.05 percentage points
BOOT_D = 0.05
BOOT_FRACTION = 0.005  # 0.5 percentage points


def parse_shown(text: str) -> tuple[float, float]:
    """A README number -> (value, half a unit of its last displayed digit)."""
    s = text.replace("−", "-").replace(",", "")
    scale = 0.01 if s.endswith("%") else 1.0
    s = s.rstrip("%")
    mantissa, _, exponent = s.partition("e")
    unit = 10.0 ** -len(mantissa.partition(".")[2]) * (10.0 ** int(exponent) if exponent else 1)
    return float(s) * scale, 0.5 * unit * scale


Numbers = dict
Check = tuple[str, str, Callable[[Numbers], float], float | None]  # snippet, shown, value, tol


def p1(sweep: str, *keys: str) -> Callable[[Numbers], float]:
    def get(n: Numbers) -> float:
        value = n["phase1"][sweep]
        for k in keys:
            value = value[int(k)] if isinstance(value, list) else value[k]
        return value

    return get


def rounds(d: int, field: str, agg: Callable) -> Callable[[Numbers], float]:
    return lambda n: agg(r[field] for r in n["phase1"]["rounds_comparison"] if r["d"] == d)


def cc(decoder: str, pair: str, i: int) -> Callable[[Numbers], float]:
    return lambda n: n["phase2"]["threshold"]["crossings"][decoder][pair]["points"][i]


def gross(p: str, field: str, decoder: str = "mwpm") -> Callable[[Numbers], float]:
    return lambda n: n["phase2"]["gross"][decoder][p][field]


def timing(d: int, p: float, method: str) -> Callable[[Numbers], float]:
    def get(n: Numbers) -> float:
        (row,) = [
            r
            for r in n["phase2"]["timing"]
            if r["d"] == d and r["p"] == p and r["method"].startswith(method)
        ]
        return row["us_per_shot"]

    return get


def timing_ratio(num: str, den: str) -> Callable[[Numbers], float]:
    return lambda n: timing(11, 0.05, num)(n) / timing(11, 0.05, den)(n)


MAIN_ROW = "| main | d | ~1,000 | **1.30%** | [0.98%, 1.42%] (median 1.08%; no crossing in 5% of"
REPL_ROW = "| replication | 3d | ~5,000+ | **1.055%** | [1.025%, 1.103%] | 1.1, 1.2% |"
FOLLOWUP_ROW = "| follow-up | d | **1.168%** | **[1.082%, 1.237%]** | 0% |"
MWPM_ROW = (
    "| MWPM | 9.13% | 9.84% (also 10.51%) | 10.08% | **9.41%** (also 10.61%) | [8.81%, 10.17%] |"
)
BPOSD_ROW = (
    "| BP-OSD | 8.64% (also 10.08%) | 7.93% (also 9.36%) | 8.97% (also 9.79%) "
    "| 9.71% (also 10.63%, 11.53%) | [6.98%, 10.64%] |"
)
GROSS_2PC = "| 2% | 9.4e-4 ± 0.7e-4 | 10.19 [9.85, 10.57] | **1,246 / 2,480** | 144 / 288 |"
SLOPES = (
    "3.44, against 5.52 for surface d = 11. (Slopes are report-only. Surface d = 3…11: 2.02, 2.59,"
)
SLOPES_2 = "3.65, 4.63, 5.52, against ⌈d/2⌉ = 2…6."


def surface_slope(d: str) -> Callable[[Numbers], float]:
    return lambda n: n["phase2"]["slopes"]["surface_mwpm"][d]


CHECKS: list[Check] = [
    # Phase 1: threshold crossings and bootstrap intervals (per round, (5,7)).
    (MAIN_ROW, "1.30%", p1("main", "crossings", "per_round", "5,7", "0"), None),
    (MAIN_ROW, "0.98%", p1("main", "bootstrap", "5,7", "lo"), BOOT_PCT),
    (MAIN_ROW, "1.42%", p1("main", "bootstrap", "5,7", "hi"), BOOT_PCT),
    (MAIN_ROW, "1.08%", p1("main", "bootstrap", "5,7", "median"), BOOT_PCT),
    (MAIN_ROW, "5%", p1("main", "bootstrap", "5,7", "no_crossing_fraction"), BOOT_FRACTION),
    ("| 1.0, 1.1, 1.11, 1.2, 1.5% |", "1.0%", p1("main", "within_2sigma", "5,7", "0"), None),
    ("| 1.0, 1.1, 1.11, 1.2, 1.5% |", "1.1%", p1("main", "within_2sigma", "5,7", "1"), None),
    ("| 1.0, 1.1, 1.11, 1.2, 1.5% |", "1.11%", p1("main", "within_2sigma", "5,7", "2"), None),
    ("| 1.0, 1.1, 1.11, 1.2, 1.5% |", "1.2%", p1("main", "within_2sigma", "5,7", "3"), None),
    ("| 1.0, 1.1, 1.11, 1.2, 1.5% |", "1.5%", p1("main", "within_2sigma", "5,7", "4"), None),
    (REPL_ROW, "1.055%", p1("replication", "crossings", "per_round", "5,7", "0"), None),
    (REPL_ROW, "1.025%", p1("replication", "bootstrap", "5,7", "lo"), BOOT_PCT),
    (REPL_ROW, "1.103%", p1("replication", "bootstrap", "5,7", "hi"), BOOT_PCT),
    (REPL_ROW, "1.1%", p1("replication", "within_2sigma", "5,7", "0"), None),
    (REPL_ROW, "1.2%", p1("replication", "within_2sigma", "5,7", "1"), None),
    (
        "| replication | 3d | 1.055% | [1.025%, 1.103%] | 0.2% |",
        "0.2%",
        p1("replication", "bootstrap", "5,7", "no_crossing_fraction"),
        BOOT_FRACTION,
    ),
    (FOLLOWUP_ROW, "1.168%", p1("followup", "crossings", "per_round", "5,7", "0"), None),
    (FOLLOWUP_ROW, "1.082%", p1("followup", "bootstrap", "5,7", "lo"), BOOT_PCT),
    (FOLLOWUP_ROW, "1.237%", p1("followup", "bootstrap", "5,7", "hi"), BOOT_PCT),
    (FOLLOWUP_ROW, "0%", p1("followup", "bootstrap", "5,7", "no_crossing_fraction"), BOOT_FRACTION),
    (
        "**The shift is about +0.11 percentage points",
        "0.11",
        lambda n: (
            100
            * (
                p1("followup", "crossings", "per_round", "5,7", "0")(n)
                - p1("replication", "crossings", "per_round", "5,7", "0")(n)
            )
        ),
        None,
    ),
    # Phase 1: low-p slopes (main sweep, report only).
    ("1.93, 2.94 and 4.07 for d = 3, 5, 7", "1.93", p1("main", "slopes", "3"), None),
    ("1.93, 2.94 and 4.07 for d = 3, 5, 7", "2.94", p1("main", "slopes", "5"), None),
    ("1.93, 2.94 and 4.07 for d = 3, 5, 7", "4.07", p1("main", "slopes", "7"), None),
    # Phase 1: per-round rate ratio rounds = d / rounds = 3d, and z-scores.
    (
        "| 3 | 1.01–1.13 (higher with fewer rounds) | −0.2 to −3.4 |",
        "1.01",
        rounds(3, "ratio", min),
        None,
    ),
    (
        "| 3 | 1.01–1.13 (higher with fewer rounds) | −0.2 to −3.4 |",
        "1.13",
        rounds(3, "ratio", max),
        None,
    ),
    (
        "| 3 | 1.01–1.13 (higher with fewer rounds) | −0.2 to −3.4 |",
        "−0.2",
        rounds(3, "z", max),
        None,
    ),
    (
        "| 3 | 1.01–1.13 (higher with fewer rounds) | −0.2 to −3.4 |",
        "−3.4",
        rounds(3, "z", min),
        None,
    ),
    ("| 5 | 0.92–1.00 | 0.0 to 2.3 |", "0.92", rounds(5, "ratio", min), None),
    ("| 5 | 0.92–1.00 | 0.0 to 2.3 |", "1.00", rounds(5, "ratio", max), None),
    ("| 5 | 0.92–1.00 | 0.0 to 2.3 |", "0.0", rounds(5, "z", min), None),
    ("| 5 | 0.92–1.00 | 0.0 to 2.3 |", "2.3", rounds(5, "z", max), None),
    (
        "| 7 | 0.87–0.94 (lower with fewer rounds, all 5 points) | 1.6 to 4.2 |",
        "0.87",
        rounds(7, "ratio", min),
        None,
    ),
    (
        "| 7 | 0.87–0.94 (lower with fewer rounds, all 5 points) | 1.6 to 4.2 |",
        "0.94",
        rounds(7, "ratio", max),
        None,
    ),
    (
        "| 7 | 0.87–0.94 (lower with fewer rounds, all 5 points) | 1.6 to 4.2 |",
        "1.6",
        rounds(7, "z", min),
        None,
    ),
    (
        "| 7 | 0.87–0.94 (lower with fewer rounds, all 5 points) | 1.6 to 4.2 |",
        "4.2",
        rounds(7, "z", max),
        None,
    ),
    # Phase 2, experiment 1: code-capacity crossings (per shot).
    (MWPM_ROW, "9.13%", cc("mwpm", "3,5", 0), None),
    (MWPM_ROW, "9.84%", cc("mwpm", "5,7", 0), None),
    (MWPM_ROW, "10.51%", cc("mwpm", "5,7", 1), None),
    (MWPM_ROW, "10.08%", cc("mwpm", "7,9", 0), None),
    (MWPM_ROW, "9.41%", cc("mwpm", "9,11", 0), None),
    (MWPM_ROW, "10.61%", cc("mwpm", "9,11", 1), None),
    (
        MWPM_ROW,
        "8.81%",
        lambda n: n["phase2"]["threshold"]["crossings"]["mwpm"]["9,11"]["lo"],
        BOOT_PCT,
    ),
    (
        MWPM_ROW,
        "10.17%",
        lambda n: n["phase2"]["threshold"]["crossings"]["mwpm"]["9,11"]["hi"],
        BOOT_PCT,
    ),
    (BPOSD_ROW, "8.64%", cc("bposd", "3,5", 0), None),
    (BPOSD_ROW, "10.08%", cc("bposd", "3,5", 1), None),
    (BPOSD_ROW, "7.93%", cc("bposd", "5,7", 0), None),
    (BPOSD_ROW, "9.36%", cc("bposd", "5,7", 1), None),
    (BPOSD_ROW, "8.97%", cc("bposd", "7,9", 0), None),
    (BPOSD_ROW, "9.79%", cc("bposd", "7,9", 1), None),
    (BPOSD_ROW, "9.71%", cc("bposd", "9,11", 0), None),
    (BPOSD_ROW, "10.63%", cc("bposd", "9,11", 1), None),
    (BPOSD_ROW, "11.53%", cc("bposd", "9,11", 2), None),
    (
        BPOSD_ROW,
        "6.98%",
        lambda n: n["phase2"]["threshold"]["crossings"]["bposd"]["9,11"]["lo"],
        BOOT_PCT,
    ),
    (
        BPOSD_ROW,
        "10.64%",
        lambda n: n["phase2"]["threshold"]["crossings"]["bposd"]["9,11"]["hi"],
        BOOT_PCT,
    ),
    # Phase 2, experiment 2: gross code vs 12 surface patches.
    (GROSS_2PC, "9.4e-4", gross("0.02", "gross_rate"), None),
    (GROSS_2PC, "0.7e-4", gross("0.02", "gross_sigma"), None),
    (GROSS_2PC, "10.19", gross("0.02", "d_star"), None),
    (GROSS_2PC, "9.85", gross("0.02", "lo"), BOOT_D),
    (GROSS_2PC, "10.57", gross("0.02", "hi"), BOOT_D),
    (GROSS_2PC, "1,246", gross("0.02", "data_qubits"), None),
    (GROSS_2PC, "2,480", gross("0.02", "total_qubits"), None),
    ("| 3% | 3.1e-3 | > 11 |", "3.1e-3", gross("0.03", "gross_rate"), None),
    ("| 4% | 1.05e-2 | > 11 |", "1.05e-2", gross("0.04", "gross_rate"), None),
    ("| 5% | 4.1e-2 | > 11 |", "4.1e-2", gross("0.05", "gross_rate"), None),
    ("| 6%–8% | 0.096–0.30 | > 11 |", "0.096", gross("0.06", "gross_rate"), None),
    ("| 6%–8% | 0.096–0.30 | > 11 |", "0.30", gross("0.08", "gross_rate"), None),
    ("about **8.7× fewer", "8.7", lambda n: gross("0.02", "data_qubits")(n) / 144, None),
    ("**8.6× fewer total qubits**", "8.6", lambda n: gross("0.02", "total_qubits")(n) / 288, None),
    (SLOPES, "3.44", lambda n: n["phase2"]["slopes"]["gross"], None),
    (SLOPES, "2.02", surface_slope("3"), None),
    (SLOPES, "2.59", surface_slope("5"), None),
    (SLOPES_2, "3.65", surface_slope("7"), None),
    (SLOPES_2, "4.63", surface_slope("9"), None),
    (SLOPES_2, "5.52", surface_slope("11"), None),
    # Phase 2, experiment 3: decoder timing (us per shot), and the ratios quoted in the text.
    (
        "| surface d=3 (9) | 0.05 | 11 µs | 4.1 µs | 0.08 µs |",
        "11",
        timing(3, 0.05, "BP-OSD"),
        None,
    ),
    (
        "| surface d=3 (9) | 0.05 | 11 µs | 4.1 µs | 0.08 µs |",
        "4.1",
        timing(3, 0.05, "PyMatching (per"),
        None,
    ),
    (
        "| surface d=3 (9) | 0.05 | 11 µs | 4.1 µs | 0.08 µs |",
        "0.08",
        timing(3, 0.05, "PyMatching (decode"),
        None,
    ),
    (
        "| surface d=7 (49) | 0.05 | 217 µs | 5.0 µs | 0.73 µs |",
        "217",
        timing(7, 0.05, "BP-OSD"),
        None,
    ),
    (
        "| surface d=7 (49) | 0.05 | 217 µs | 5.0 µs | 0.73 µs |",
        "5.0",
        timing(7, 0.05, "PyMatching (per"),
        None,
    ),
    (
        "| surface d=7 (49) | 0.05 | 217 µs | 5.0 µs | 0.73 µs |",
        "0.73",
        timing(7, 0.05, "PyMatching (decode"),
        None,
    ),
    (
        "| surface d=11 (121) | 0.05 | 807 µs | 6.8 µs | 2.6 µs |",
        "807",
        timing(11, 0.05, "BP-OSD"),
        None,
    ),
    (
        "| surface d=11 (121) | 0.05 | 807 µs | 6.8 µs | 2.6 µs |",
        "6.8",
        timing(11, 0.05, "PyMatching (per"),
        None,
    ),
    (
        "| surface d=11 (121) | 0.05 | 807 µs | 6.8 µs | 2.6 µs |",
        "2.6",
        timing(11, 0.05, "PyMatching (decode"),
        None,
    ),
    ("| gross (144) | 0.05 | 293 µs | — | — |", "293", timing(12, 0.05, "BP-OSD"), None),
    (
        "| surface d=11 (121) | 0.10 | 1,053 µs | 11.8 µs | 6.7 µs |",
        "1,053",
        timing(11, 0.1, "BP-OSD"),
        None,
    ),
    (
        "| surface d=11 (121) | 0.10 | 1,053 µs | 11.8 µs | 6.7 µs |",
        "11.8",
        timing(11, 0.1, "PyMatching (per"),
        None,
    ),
    (
        "| surface d=11 (121) | 0.10 | 1,053 µs | 11.8 µs | 6.7 µs |",
        "6.7",
        timing(11, 0.1, "PyMatching (decode"),
        None,
    ),
    ("| gross (144) | 0.10 | 1,676 µs | — | — |", "1,676", timing(12, 0.1, "BP-OSD"), None),
    # "about 3–120×" and "about 300×": rounded to the leading digit(s), so +/- half of that.
    (
        "BP-OSD is about 3–120× slower than PyMatching's per-shot call",
        "3",
        lambda n: timing(3, 0.05, "BP-OSD")(n) / timing(3, 0.05, "PyMatching (per")(n),
        None,
    ),
    (
        "BP-OSD is about 3–120× slower than PyMatching's per-shot call",
        "120",
        timing_ratio("BP-OSD", "PyMatching (per"),
        5.0,
    ),
    (
        "about 300× slower than its batch mode at d = 11",
        "300",
        timing_ratio("BP-OSD", "PyMatching (decode"),
        50.0,
    ),
]


@pytest.fixture(scope="module")
def reproduced(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Numbers]:
    out = tmp_path_factory.mktemp("reproduced")
    for phase in ("phase1", "phase2"):
        argv = ["reproduce", phase, "--results", str(REPO / "results"), "--out", str(out)]
        assert cli.main(argv) == 0
    numbers = {p: json.loads((out / p / "numbers.json").read_text()) for p in ("phase1", "phase2")}
    return out, numbers


@pytest.mark.parametrize("snippet, shown, get, tol", CHECKS, ids=[c[1] for c in CHECKS])
def test_reproduced_number_matches_readme(
    reproduced: tuple[Path, Numbers],
    snippet: str,
    shown: str,
    get: Callable[[Numbers], float],
    tol: float | None,
) -> None:
    expected, half_unit = parse_shown(shown)
    tol = half_unit if tol is None else tol
    value = get(reproduced[1])
    assert abs(value - expected) <= tol * (1 + 1e-9), f"{shown!r}: reproduced {value!r}"


@pytest.mark.parametrize("snippet, shown", [c[:2] for c in CHECKS], ids=[c[1] for c in CHECKS])
def test_readme_still_quotes_number(snippet: str, shown: str) -> None:
    assert shown in snippet or shown.rstrip("%") in snippet
    assert snippet in README, f"README no longer contains: {snippet!r}"


def test_reproduced_claims(reproduced: tuple[Path, Numbers]) -> None:
    """README statements that are facts about the data rather than numbers with digits."""
    n1, n2 = reproduced[1]["phase1"], reproduced[1]["phase2"]
    # "(lower with fewer rounds, all 5 points)" and "(higher with fewer rounds)".
    assert [r["ratio"] < 1 for r in n1["rounds_comparison"] if r["d"] == 7] == [True] * 5
    assert all(r["ratio"] > 1 for r in n1["rounds_comparison"] if r["d"] == 3)
    # Experiment 1: no point hit its shot cap; pre-registered check passes; the (9,11)
    # bootstrap interval excludes the 10.31% reference. Number of sign changes per pair.
    threshold = n2["threshold"]
    assert threshold["hit_shot_cap"] == 0
    assert threshold["preregistered"]["pass"]
    assert threshold["crossings"]["mwpm"]["9,11"]["hi"] < 0.1031
    for decoder, counts in (("mwpm", [1, 2, 1, 2]), ("bposd", [2, 2, 2, 3])):
        assert [len(c["points"]) for c in threshold["crossings"][decoder].values()] == counts
    # Experiment 2: d* > 11 (gross better) for p >= 3%; BP-OSD surface decoding moves d* < 0.01.
    for decoder in ("mwpm", "bposd"):
        for p, e in n2["gross"][decoder].items():
            if float(p) >= 0.03:
                assert e["d_star"] is None and e["gross_rate"] < e["p12"]["11"]
    assert (
        abs(n2["gross"]["bposd"]["0.02"]["d_star"] - n2["gross"]["mwpm"]["0.02"]["d_star"]) < 0.01
    )
    # "Surface d = 9 and 11 at p = 2% hit the shot cap with 168 and 47 errors."
    capped = {(r["d"], r["p"]): r["errors"] for r in n2["gross_shot_cap"] if r["decoder"] == "mwpm"}
    assert capped == {(9, 0.02): 168, (11, 0.02): 47}


def test_reproduce_writes_every_readme_plot(reproduced: tuple[Path, Numbers]) -> None:
    out = reproduced[0]
    for stem in cli.PHASE1_PLOTS:
        assert (out / "phase1" / f"{stem}.png").stat().st_size > 0
    for stem in cli.PHASE2_PLOTS:
        assert (out / "phase2" / f"{stem}.png").stat().st_size > 0
    # Every plot the README embeds is one that reproduce rebuilds.
    embedded = {
        line.split("(results/")[1].split(".png")[0]
        for line in README.splitlines()
        if line.startswith("![")
    }
    assert embedded == set(cli.PHASE1_PLOTS) | set(cli.PHASE2_PLOTS)
