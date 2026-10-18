"""Stim circuit generation for surface-code memory experiments."""

import stim


def surface_code_memory_z(distance: int, p: float, rounds: int | None = None) -> stim.Circuit:
    """Rotated surface code Z-basis memory experiment with uniform circuit-level noise.

    All four noise knobs are set to p, matching the model in Stim's
    doc/getting_started.ipynb (section 9). Rounds default to the distance.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance if rounds is None else rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """DEM with hyperedges decomposed into graphlike pieces, as matching requires."""
    return circuit.detector_error_model(decompose_errors=True)
