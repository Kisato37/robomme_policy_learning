"""CPU-only tests of strict single-environment terminal flag adaptation."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.uniform_keyframe_expansion.evaluator import (
    ExpansionEvaluationError, _benchmark_stop_flag,
)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("kind", ["python", "numpy_scalar", "numpy_0d", "numpy_1d",
                                  "torch_0d", "torch_1d"])
def test_single_boolean_representation_preserves_value(value, kind):
    if kind.startswith("torch"):
        import torch
        source = torch.tensor(value if kind == "torch_0d" else [value], dtype=torch.bool)
    else:
        source = {"python": value, "numpy_scalar": np.bool_(value),
                  "numpy_0d": np.array(value), "numpy_1d": np.array([value])}[kind]
    assert _benchmark_stop_flag(source) is value


@pytest.mark.parametrize("value", [
    None, 0, 1, 0.0, 1.0, float("nan"), "False", "True", [], [False], [True],
    np.int64(0), np.array(1), np.array([1]), np.array([], dtype=bool),
    np.array([False, True]), np.array([[False]]), np.array([True], dtype=object),
])
def test_non_boolean_or_non_single_environment_is_not_coerced(value):
    with pytest.raises(ExpansionEvaluationError, match="Benchmark stop flag"):
        _benchmark_stop_flag(value)


@pytest.mark.parametrize("kind", ["int", "float", "empty", "multiple", "matrix", "meta"])
def test_invalid_torch_values_fail_closed(kind):
    import torch
    value = {
        "int": lambda: torch.tensor(1),
        "float": lambda: torch.tensor(1.0),
        "empty": lambda: torch.empty(0, dtype=torch.bool),
        "multiple": lambda: torch.tensor([False, True]),
        "matrix": lambda: torch.tensor([[True]]),
        "meta": lambda: torch.empty((), dtype=torch.bool, device="meta"),
    }[kind]()
    with pytest.raises(ExpansionEvaluationError, match="Benchmark stop flag"):
        _benchmark_stop_flag(value)


def test_python_and_numpy_flags_do_not_import_torch_or_initialize_backends():
    root = Path(__file__).resolve().parents[2]
    code = """
import sys
import numpy as np
from experiments.uniform_keyframe_expansion.evaluator import _benchmark_stop_flag
assert _benchmark_stop_flag(False) is False
assert _benchmark_stop_flag(np.array([True])) is True
assert not {'torch', 'jax', 'jaxlib', 'sapien', 'mani_skill'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=root,
                   env=dict(os.environ, PYTHONPATH=f"{root / 'src'}:{root}"),
                   check=True, capture_output=True, text=True, timeout=30)
