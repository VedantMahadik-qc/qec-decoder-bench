"""Validation tests for code construction (Phase 2, code capacity)."""

import itertools

import numpy as np
import pytest

from qec_bench.codes import (
    CSSCode,
    find_low_weight_x_logical,
    gf2_rank,
    gross_code,
    in_rowspace,
    logical_operators,
    rotated_surface_code,
)

SURFACE_DISTANCES = [3, 5, 7, 9, 11]


def all_codes() -> list[CSSCode]:
    return [rotated_surface_code(d) for d in SURFACE_DISTANCES] + [gross_code()]


@pytest.mark.parametrize("code", all_codes(), ids=lambda c: c.name)
def test_css_condition(code: CSSCode) -> None:
    # Definition: X and Z checks commute iff H_X H_Z^T = 0 (mod 2).
    assert not ((code.hx.astype(np.int64) @ code.hz.T) % 2).any()


def test_gross_code_parameters_from_rank() -> None:
    # Known value: [[144,12,12]], Bravyi et al. arXiv:2308.07915 Table 3. k comes from GF(2) ranks.
    code = gross_code()
    assert code.n == 144
    assert code.k == 12


@pytest.mark.parametrize("d", SURFACE_DISTANCES)
def test_surface_code_parameters(d: int) -> None:
    # Known value: the rotated surface code is [[d^2, 1, d]].
    code = rotated_surface_code(d)
    assert code.n == d * d
    assert code.k == 1


def test_gross_code_check_weights() -> None:
    # Definition: A and B each have 3 monomials, so every check row has weight 3 + 3 = 6,
    # and every qubit sits in 3 checks of each type.
    code = gross_code()
    for h in (code.hx, code.hz):
        assert set(h.sum(axis=1).tolist()) == {6}
        assert set(h.sum(axis=0).tolist()) == {3}


@pytest.mark.parametrize("code", all_codes(), ids=lambda c: c.name)
def test_logical_operators_are_valid(code: CSSCode) -> None:
    lx, lz = logical_operators(code)
    assert lx.shape == lz.shape == (code.k, code.n)
    assert not ((code.hx.astype(np.int64) @ lz.T) % 2).any()  # Z logicals commute with X checks
    assert not ((code.hz.astype(np.int64) @ lx.T) % 2).any()  # X logicals commute with Z checks
    pairing = (lx.astype(np.int64) @ lz.T) % 2
    assert np.array_equal(pairing, np.eye(code.k, dtype=np.int64))
    assert gf2_rank(np.vstack([code.hx, lx])) == gf2_rank(code.hx) + code.k
    assert gf2_rank(np.vstack([code.hz, lz])) == gf2_rank(code.hz) + code.k


def _is_nontrivial_x_logical(code: CSSCode, v: np.ndarray) -> bool:
    # Nontrivial X logical: in ker(H_Z) but not a product of X stabilizers.
    commutes_with_z_checks = not ((code.hz.astype(np.int64) @ v) % 2).any()
    return commutes_with_z_checks and not in_rowspace(v, code.hx)


def test_gross_code_weight_12_logical_witness() -> None:
    # Upper bound only: a weight-12 X logical shows d_X <= 12. It does not prove d = 12.
    # Weight < 12 would contradict d = 12 (Bravyi et al. Table 3) and signal a construction bug.
    code = gross_code()
    v = find_low_weight_x_logical(code, target=12, seed=0)
    assert v is not None, "no X logical of weight <= 12 found"
    assert _is_nontrivial_x_logical(code, v)
    assert v.sum() == 12, f"found weight {v.sum()} < 12, contradicting d = 12"


def test_surface_d3_distance_brute_force() -> None:
    # Known value: exhaustive search over all 2^9 X errors gives min nontrivial logical weight 3.
    code = rotated_surface_code(3)
    weights = [
        sum(bits)
        for bits in itertools.product((0, 1), repeat=code.n)
        if any(bits) and _is_nontrivial_x_logical(code, np.array(bits, dtype=np.uint8))
    ]
    assert min(weights) == 3


def test_surface_d5_weight_5_logical_witness() -> None:
    # Upper bound d_X <= 5. The matching lower bound (MWPM corrects every weight <= 2 error)
    # is in test_code_capacity.py; together they give d_X = 5.
    code = rotated_surface_code(5)
    v = find_low_weight_x_logical(code, target=5, seed=0)
    assert v is not None and _is_nontrivial_x_logical(code, v)
    assert v.sum() == 5
