"""Direct (non-sinter) decoding, used by tests for deterministic, seeded results."""

import numpy as np
import pymatching
import stim

from qec_bench.circuits import detector_error_model


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
