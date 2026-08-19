import numpy as np

from experiments.dual_memory.analyze_oracle import statistics


def test_hierarchical_paired_statistics_preserve_known_contrasts():
    values = np.zeros((3, 4, 20), dtype=np.float64)
    values[:, 0, :4] = 1  # N = 0.20
    values[:, 1, :8] = 1  # S = 0.40
    values[:, 2, :6] = 1  # P = 0.30
    values[:, 3, :12] = 1  # SP = 0.60
    result = statistics(values, np.random.default_rng(7), draws=200)
    assert result["success_rates"] == {"N": 0.2, "S": 0.4, "P": 0.3, "SP": 0.6}
    assert np.isclose(result["contrasts"]["SP-S"]["estimate"], 0.2)
    assert np.isclose(result["contrasts"]["SP-P"]["estimate"], 0.3)
    assert np.isclose(result["contrasts"]["SP-max(S,P)"]["estimate"], 0.2)
    assert np.isclose(result["contrasts"]["interaction"]["estimate"], 0.1)
