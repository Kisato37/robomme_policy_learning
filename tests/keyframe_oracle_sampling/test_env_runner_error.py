from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "robomme"
BENCHMARK_SRC = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "robomme_benchmark"
    / "src"
)
for source_dir in (EXAMPLES_DIR, BENCHMARK_SRC):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

# ``EnvRunner.step`` itself only needs NumPy, but importing the released
# wrapper normally registers every SAPIEN environment.  Keep this CPU unit
# test independent of the GPU/server-only benchmark dependencies by exposing
# the tiny import surface that ``env_runner.py`` requires.
robomme_module = types.ModuleType("robomme")
robomme_env_module = types.ModuleType("robomme.robomme_env")
record_wrapper_module = types.ModuleType("robomme.env_record_wrapper")
demonstration_module = types.ModuleType(
    "robomme.env_record_wrapper.DemonstrationWrapper"
)


class _UnusedBuilder:
    pass


class _UnusedDemonstrationWrapper:
    pass


record_wrapper_module.BenchmarkEnvBuilder = _UnusedBuilder
demonstration_module.DemonstrationWrapper = _UnusedDemonstrationWrapper
stubbed_modules = {
    "robomme": robomme_module,
    "robomme.robomme_env": robomme_env_module,
    "robomme.env_record_wrapper": record_wrapper_module,
    "robomme.env_record_wrapper.DemonstrationWrapper": demonstration_module,
}
tracked_module_names = (*stubbed_modules, "utils", "causal_stage_instrumentation")
previous_modules = {name: sys.modules.get(name) for name in tracked_module_names}
sys.modules.update(stubbed_modules)
module_spec = importlib.util.spec_from_file_location(
    "_keyframe_env_runner_error_test_module",
    EXAMPLES_DIR / "env_runner.py",
)
assert module_spec is not None and module_spec.loader is not None
env_runner_module = importlib.util.module_from_spec(module_spec)
try:
    module_spec.loader.exec_module(env_runner_module)
finally:
    for module_name, previous_module in previous_modules.items():
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

EnvRunner = env_runner_module.EnvRunner


class _FailAwareErrorEnv:
    def step(self, action):
        del action
        return (
            None,
            0.0,
            True,
            False,
            {
                "status": "error",
                "error_message": "FailAwareWrapper caught IK failure",
                "exception_type": "RuntimeError",
            },
        )


def test_failaware_benchmark_error_is_preserved_before_observation_unpacking():
    runner = object.__new__(EnvRunner)
    runner.env = _FailAwareErrorEnv()
    runner.require_current_task_index = True

    observation, stopped, outcome = runner.step(np.zeros(8, dtype=np.float32))

    assert observation == (None, None, None)
    assert stopped is True
    assert outcome == "error"
    assert runner.info["exception_type"] == "RuntimeError"
