import numpy as np

from monitoring.drift_simulation import population_stability_index


def test_psi_is_zero_for_identical_distributions():
    values = np.arange(100, dtype=float)
    assert population_stability_index(values, values) == 0.0


def test_psi_increases_for_shifted_distribution():
    reference = np.arange(100, dtype=float)
    shifted = reference + 50
    assert population_stability_index(reference, shifted) > 0.20
