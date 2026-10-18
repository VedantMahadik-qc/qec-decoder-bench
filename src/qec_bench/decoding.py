"""Direct (non-sinter) decoding: seeded Stim decoding for tests, and code-capacity decoding."""

import numpy as np
import pymatching
import stim
from ldpc import BpOsdDecoder

from qec_bench.circuits import detector_error_model
from qec_bench.codes import CSSCode


def count_logical_errors(circuit: stim.Circuit, shots: int, seed: int) -> int:
    """Sample `shots` from `circuit`, decode with PyMatching, return the failure count.

    A shot fails if the predicted observable flips differ from the actual ones
    in any logical observable.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(shots, separate_observables=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model(circuit))
    predictions = matcher.decode_batch(detection_events)
    return int(np.any(predictions != observable_flips, axis=1).sum())


# --- Code-capacity decoding (Phase 2) -------------------------------------------------
# Noise: iid X errors on data qubits, perfect syndrome measurement. X errors are seen
# by the Z checks, so every decoder here works on H_Z and the syndrome s = H_Z e.

Decoder = pymatching.Matching | BpOsdDecoder

# BP-OSD settings, fixed before any sweep ran (not tuned to results).
BPOSD_SETTINGS = {
    "bp_method": "minimum_sum",
    "ms_scaling_factor": 0.625,
    "max_iter": 1000,
    "schedule": "parallel",
    "osd_method": "osd_cs",
    "osd_order": 7,
}


def sample_code_capacity(
    code: CSSCode, p: float, shots: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """iid X errors with probability p per data qubit, and their Z-check syndromes."""
    errors = (rng.random((shots, code.n)) < p).astype(np.uint8)
    syndromes = ((errors.astype(np.int64) @ code.hz.T) % 2).astype(np.uint8)
    return errors, syndromes


def make_mwpm(code: CSSCode) -> pymatching.Matching:
    """MWPM on H_Z. Uniform weights are exact minimum-weight decoding for iid noise."""
    return pymatching.Matching.from_check_matrix(code.hz)


def make_bposd(code: CSSCode, p: float) -> BpOsdDecoder:
    """BP-OSD on H_Z, with the true channel error rate p as the BP prior."""
    return BpOsdDecoder(code.hz, error_rate=p, **BPOSD_SETTINGS)


def decode_all(decoder: Decoder, syndromes: np.ndarray) -> np.ndarray:
    """Corrections for a batch of syndromes, one row per shot, for either decoder type."""
    if isinstance(decoder, pymatching.Matching):
        return decoder.decode_batch(syndromes).astype(np.uint8)
    return np.array([decoder.decode(s) for s in syndromes], dtype=np.uint8)


def logical_failures(
    code: CSSCode, lz: np.ndarray, errors: np.ndarray, corrections: np.ndarray
) -> np.ndarray:
    """Per-shot failure flags: the residual X error flips at least one Z logical.

    The residual r = e + correction must have zero syndrome (the decoder output
    must explain the syndrome); then it is a stabilizer or a logical, and it is a
    logical iff it anticommutes with some Z logical (lz . r = 1).
    """
    residual = (errors ^ corrections).astype(np.int64)
    if ((residual @ code.hz.T) % 2).any():
        raise ValueError("decoder correction does not reproduce the syndrome")
    return ((residual @ lz.T) % 2).any(axis=1)
