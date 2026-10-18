"""Validation tests for the rotated surface code memory experiment."""

import math

import pytest

from qec_bench.circuits import detector_error_model, surface_code_memory_z
from qec_bench.decoding import count_logical_errors


@pytest.mark.parametrize("d", [3, 5])
def test_zero_noise_gives_zero_logical_errors(d: int) -> None:
    # Analytic limit: with no noise, every detector and observable is deterministic.
    circuit = surface_code_memory_z(d, p=0.0)
    detectors, observables = circuit.compile_detector_sampler(seed=1).sample(
        1000, separate_observables=True
    )
    assert not detectors.any()
    assert not observables.any()
    assert count_logical_errors(circuit, shots=1000, seed=1) == 0


@pytest.mark.parametrize("d", [3, 5, 7])
def test_circuit_distance_equals_code_distance(d: int) -> None:
    # Known value: the rotated surface code has distance d, and Stim's circuit is
    # built so that circuit-level noise does not reduce it.
    dem = detector_error_model(surface_code_memory_z(d, p=1e-3))
    assert len(dem.shortest_graphlike_error()) == d


@pytest.mark.slow
def test_larger_distance_suppresses_errors_below_threshold() -> None:
    # Known behavior: below threshold (~1%, Stim getting_started.ipynb section 9),
    # increasing d lowers p_L. Require ~3-sigma separation so the test can't pass by luck.
    shots = 200_000
    p = 2e-3
    rate3 = count_logical_errors(surface_code_memory_z(3, p), shots=shots, seed=11) / shots
    rate5 = count_logical_errors(surface_code_memory_z(5, p), shots=shots, seed=12) / shots
    sigma3 = math.sqrt(rate3 * (1 - rate3) / shots)
    sigma5 = math.sqrt(rate5 * (1 - rate5) / shots)
    assert rate5 + 3 * sigma5 < rate3 - 3 * sigma3


def test_high_noise_saturates_at_one_half() -> None:
    # Analytic limit: when noise fully scrambles the logical observable, the decoder
    # is right only by chance, so p_L -> 1/2.
    shots = 20_000
    failures = count_logical_errors(surface_code_memory_z(3, p=0.3), shots=shots, seed=7)
    sigma = math.sqrt(0.25 / shots)
    assert abs(failures / shots - 0.5) < 4 * sigma
