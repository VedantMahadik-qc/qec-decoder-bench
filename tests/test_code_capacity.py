"""Validation tests for code-capacity decoding with MWPM and BP-OSD."""

import itertools
import math

import numpy as np
import pytest

from qec_bench.codes import CSSCode, gross_code, logical_operators, rotated_surface_code
from qec_bench.decoding import (
    Decoder,
    decode_all,
    logical_failures,
    make_bposd,
    make_mwpm,
    sample_code_capacity,
)


def surface_and_gross() -> list[CSSCode]:
    return [rotated_surface_code(d) for d in (3, 5, 7)] + [gross_code()]


def failure_count(code: CSSCode, decoder: Decoder, errors: np.ndarray) -> int:
    _lx, lz = logical_operators(code)
    syndromes = ((errors.astype(np.int64) @ code.hz.T) % 2).astype(np.uint8)
    return int(logical_failures(code, lz, errors, decode_all(decoder, syndromes)).sum())


@pytest.mark.parametrize("code", surface_and_gross(), ids=lambda c: c.name)
def test_zero_noise_gives_no_syndromes_and_no_failures(code: CSSCode) -> None:
    # Analytic limit: p = 0 means no errors, so no syndrome and nothing to fail.
    errors, syndromes = sample_code_capacity(code, 0.0, 200, np.random.default_rng(0))
    assert not syndromes.any()
    # BP needs a nonzero prior (p = 0 gives infinite log-likelihoods), so use 1e-3.
    decoders = [make_bposd(code, 1e-3)]
    if code.name.startswith("surface"):
        decoders.append(make_mwpm(code))
    for decoder in decoders:
        assert failure_count(code, decoder, errors) == 0


@pytest.mark.parametrize("code", surface_and_gross(), ids=lambda c: c.name)
def test_corrections_reproduce_syndrome(code: CSSCode) -> None:
    # Consistency: a valid decoder output must explain the measured syndrome.
    errors, syndromes = sample_code_capacity(code, 0.05, 300, np.random.default_rng(1))
    decoders = [make_bposd(code, 0.05)]
    if code.name.startswith("surface"):
        decoders.append(make_mwpm(code))
    for decoder in decoders:
        corrections = decode_all(decoder, syndromes)
        assert np.array_equal((corrections.astype(np.int64) @ code.hz.T) % 2, syndromes)


@pytest.mark.parametrize("d", [3, 5])
def test_mwpm_corrects_every_error_up_to_half_distance(d: int) -> None:
    # Known guarantee: a distance-d code decoded by minimum weight corrects every
    # error of weight <= floor((d-1)/2). Exhaustive over all such X errors.
    code = rotated_surface_code(d)
    t = (d - 1) // 2
    errors = np.array(
        [
            np.isin(np.arange(code.n), support).astype(np.uint8)
            for w in range(1, t + 1)
            for support in itertools.combinations(range(code.n), w)
        ]
    )
    assert failure_count(code, make_mwpm(code), errors) == 0


@pytest.mark.parametrize(
    "code", [rotated_surface_code(3), rotated_surface_code(5), gross_code()], ids=lambda c: c.name
)
def test_bposd_corrects_every_single_qubit_error(code: CSSCode) -> None:
    errors = np.eye(code.n, dtype=np.uint8)
    assert failure_count(code, make_bposd(code, 0.01), errors) == 0


@pytest.mark.parametrize(
    "code",
    [rotated_surface_code(5), pytest.param(gross_code(), marks=pytest.mark.slow)],
    ids=lambda c: c.name,
)
def test_maximal_noise_randomises_every_logical(code: CSSCode) -> None:
    # Analytic limit: at p = 1/2 the error is uniform, so the residual's logical class is
    # uniform over 2^k classes and P(any logical flipped) = 1 - 2^-k.
    shots = 4000
    errors, _ = sample_code_capacity(code, 0.5, shots, np.random.default_rng(2))
    expected = 1 - 2.0**-code.k
    sigma = math.sqrt(expected * (1 - expected) / shots)
    decoder = make_mwpm(code) if code.name.startswith("surface") else make_bposd(code, 0.49)
    rate = failure_count(code, decoder, errors) / shots
    # The 1/shots slack covers the gross code, where expected ~ 1 makes sigma tiny.
    assert abs(rate - expected) < 4 * sigma + 1 / shots


def test_decoders_cross_check_on_identical_syndromes() -> None:
    # Cross-check at p = 0.05 (below the ~10% threshold), surface d = 5, same syndromes:
    # both decoders fail sometimes but far less than the p = 1/2 limit. The gap is printed.
    code = rotated_surface_code(5)
    errors, _ = sample_code_capacity(code, 0.05, 20_000, np.random.default_rng(3))
    rates = {
        name: failure_count(code, decoder, errors) / len(errors)
        for name, decoder in (("mwpm", make_mwpm(code)), ("bposd", make_bposd(code, 0.05)))
    }
    print(f"p=0.05, d=5 failure rates on identical syndromes: {rates}")
    for rate in rates.values():
        assert 0 < rate < 0.1
