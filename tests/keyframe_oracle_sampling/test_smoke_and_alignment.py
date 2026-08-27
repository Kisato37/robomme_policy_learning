from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS, FORMAL_TASKS
from experiments.keyframe_oracle_sampling.architecture_smoke import (
    _action_contract_checks,
)
from experiments.keyframe_oracle_sampling.prepare_smoke import (
    BENCHMARK_UV_LOCK,
    POLICY_PYTHON,
    POLICY_UV_LOCK,
    POLICY_VENV,
    environment_lock_identity,
    python_environment_identity,
    require_clean_repository,
)
from experiments.keyframe_oracle_sampling.smoke_matrix import expand_rows, load_frozen_matrix
from examples.robomme.causal_stage_instrumentation import (
    install_current_task_index_instrumentation,
    validate_aligned_stages,
)


REPO = Path(__file__).resolve().parents[2]


def test_frozen_smoke_matrix_is_exact_and_terminal_rotation_is_preregistered():
    payload = load_frozen_matrix()
    rows = expand_rows(payload)
    assert len(rows) == 80
    short = [row for row in rows if row["trajectory_kind"] == "short"]
    terminal = [row for row in rows if row["trajectory_kind"] == "terminal"]
    assert len(short) == 64 and len(terminal) == 16
    for task_index, task in enumerate(FORMAL_TASKS):
        assert {row["arm"] for row in short if row["task"] == task} == set(ALL_ARMS)
        selected_terminal = [row for row in terminal if row["task"] == task]
        assert selected_terminal[0]["arm"] == ALL_ARMS[task_index % 4]
        assert selected_terminal[0]["episode_id"] == 0
        assert selected_terminal[0]["dataset"] == "val"
    assert any(row["task"] == "InsertPeg" for row in rows)
    assert any(row["task"] == "PickXtimes" for row in rows)


def test_smoke_prepare_rejects_a_dirty_candidate():
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        require_clean_repository({"dirty": True})
    require_clean_repository({"dirty": False})


def test_architecture_smoke_checks_absolute_action_contract():
    valid = _action_contract_checks(np.zeros((20, 8), dtype=np.float32))
    assert all(valid.values())
    assert _action_contract_checks(np.zeros((19, 8), dtype=np.float32))[
        "action_shape_is_frozen_20x8"
    ] is False
    assert _action_contract_checks(np.zeros((20, 8), dtype=np.int32))[
        "action_dtype_is_floating"
    ] is False
    nonfinite = np.zeros((20, 8), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    assert _action_contract_checks(nonfinite)["action_values_are_finite"] is False


def test_policy_python_provenance_records_actual_virtual_environment():
    identity = python_environment_identity(POLICY_PYTHON, POLICY_VENV)
    assert identity["venv_prefix"] == str(POLICY_VENV.resolve())
    assert identity["launcher_python_path"] == str(POLICY_PYTHON.absolute())
    assert len(identity["interpreter_sha256"]) == 64
    assert len(identity["pyvenv_cfg_sha256"]) == 64
    assert identity["implementation"] == "CPython"


def test_smoke_provenance_hashes_policy_and_benchmark_lockfiles():
    identity = environment_lock_identity()
    assert POLICY_UV_LOCK.is_file() and BENCHMARK_UV_LOCK.is_file()
    assert set(identity) == {
        "policy_uv_lock_sha256",
        "benchmark_uv_lock_sha256",
    }
    assert all(len(digest) == 64 for digest in identity.values())


class _FakeUnwrapped:
    current_task_index = 0


class _FakeDemonstrationWrapper:
    def __init__(self):
        self.unwrapped = _FakeUnwrapped()

    def _augment_obs_and_info(self, obs, info, action):
        return {"front_rgb_list": obs.copy()}, dict(info)


def test_reset_demonstration_batch_preserves_one_live_stage_per_front_frame():
    install_current_task_index_instrumentation(_FakeDemonstrationWrapper)
    wrapper = _FakeDemonstrationWrapper()
    frames = []
    stages = []
    for frame_value, stage in enumerate((2, 2, 3, 4)):
        wrapper.unwrapped.current_task_index = stage
        frame = np.full((2, 2, 3), frame_value, dtype=np.uint8)
        augmented_obs, _ = wrapper._augment_obs_and_info(frame, {}, None)
        frames.append(augmented_obs["front_rgb_list"])
        stages.append(augmented_obs["current_task_index"])
    assert validate_aligned_stages(frames, stages) == [2, 2, 3, 4]


def test_alignment_hard_stops_instead_of_inventing_reset_labels():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(RuntimeError, match="Global blocker"):
        validate_aligned_stages(frames, [2])
    with pytest.raises(RuntimeError, match="Global blocker"):
        validate_aligned_stages(frames, None)


def test_instrumentation_is_captured_before_reset_flattening_and_not_model_packed():
    instrumentation_source = (
        REPO / "examples/robomme/causal_stage_instrumentation.py"
    ).read_text()
    runner_source = (REPO / "examples/robomme/env_runner.py").read_text()
    utils_source = (REPO / "examples/robomme/utils.py").read_text()
    assert 'getattr(self.unwrapped, "current_task_index"' in instrumentation_source
    assert "Install before make_env/reset" in runner_source
    assert "reference" not in instrumentation_source.lower()
    assert 'payload["current_task_index"]' in utils_source
    assert '"images": image_output' in utils_source


def test_every_smoke_launcher_passes_shell_syntax_and_cpu_dry_run():
    launchers = (
        REPO / "experiments/keyframe_oracle_sampling/run_smoke.sbatch",
        REPO / "experiments/keyframe_oracle_sampling/run_architecture_smoke.sbatch",
    )
    for launcher in launchers:
        subprocess.run(["bash", "-n", str(launcher)], cwd=REPO, check=True)
        result = subprocess.run(
            ["bash", str(launcher), "--dry-run"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        assert "Dry run only" in result.stdout


@pytest.mark.parametrize(
    "module",
    (
        "experiments.keyframe_oracle_sampling.prepare_smoke",
        "experiments.keyframe_oracle_sampling.architecture_smoke",
        "experiments.keyframe_oracle_sampling.audit_smoke",
    ),
)
def test_cpu_entrypoints_work_as_modules_from_repo_root(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--dry-run"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert "submits_jobs" in result.stdout
