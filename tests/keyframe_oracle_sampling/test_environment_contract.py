from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.keyframe_oracle_sampling import environment_contract
from experiments.keyframe_oracle_sampling.environment_contract import (
    EnvironmentContractError,
    validate_environment_contract,
)


def _identity() -> dict:
    return {
        "environment_locks": {
            "policy_uv_lock_sha256": "a" * 64,
            "benchmark_uv_lock_sha256": "b" * 64,
        },
        "python_environments": {
            "policy": {"interpreter_sha256": "c" * 64},
            "simulator": {"interpreter_sha256": "d" * 64},
        },
    }


def test_environment_contract_requires_and_matches_both_paths(monkeypatch):
    identity = _identity()
    monkeypatch.setattr(
        environment_contract,
        "live_environment_identity",
        lambda: copy.deepcopy(identity),
    )
    assert validate_environment_contract(identity) == identity


def test_environment_contract_accepts_aliases_only_for_the_same_filesystem_objects(
    tmp_path, monkeypatch
):
    real_root = tmp_path / "real"
    real_venv = real_root / "venv"
    real_base = real_root / "base"
    real_venv.mkdir(parents=True)
    real_base.mkdir()
    executable = real_venv / "python"
    base_executable = real_base / "python"
    executable.write_text("venv interpreter")
    base_executable.write_text("base interpreter")
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    expected = _identity()
    observed = copy.deepcopy(expected)
    path_fields = {
        "launcher_python_path": (real_venv / "python", alias_root / "venv" / "python"),
        "reported_executable": (real_venv / "python", alias_root / "venv" / "python"),
        "executable_realpath": (real_base / "python", alias_root / "base" / "python"),
        "venv_prefix": (real_venv, alias_root / "venv"),
        "base_prefix": (real_base, alias_root / "base"),
    }
    for role in ("policy", "simulator"):
        for field, (expected_path, observed_path) in path_fields.items():
            expected["python_environments"][role][field] = str(expected_path)
            observed["python_environments"][role][field] = str(observed_path)
    monkeypatch.setattr(environment_contract, "live_environment_identity", lambda: observed)
    assert validate_environment_contract(expected) == observed

    different = tmp_path / "different-python"
    different.write_text("different interpreter")
    observed["python_environments"]["policy"]["reported_executable"] = str(different)
    with pytest.raises(EnvironmentContractError, match="differ"):
        validate_environment_contract(expected)


@pytest.mark.parametrize("field", ["environment_locks", "python_environments"])
def test_environment_contract_rejects_missing_provenance(monkeypatch, field):
    identity = _identity()
    monkeypatch.setattr(
        environment_contract,
        "live_environment_identity",
        lambda: copy.deepcopy(identity),
    )
    manifest = copy.deepcopy(identity)
    del manifest[field]
    with pytest.raises(EnvironmentContractError, match="lacks"):
        validate_environment_contract(manifest)


@pytest.mark.parametrize("field", ["environment_locks", "python_environments"])
def test_environment_contract_rejects_live_drift(monkeypatch, field):
    identity = _identity()
    observed = copy.deepcopy(identity)
    if field == "environment_locks":
        observed[field]["policy_uv_lock_sha256"] = "e" * 64
    else:
        observed[field]["policy"]["interpreter_sha256"] = "e" * 64
    monkeypatch.setattr(
        environment_contract,
        "live_environment_identity",
        lambda: observed,
    )
    with pytest.raises(EnvironmentContractError, match="differ"):
        validate_environment_contract(identity)


def test_launchers_revalidate_environment_and_do_not_inherit_pythonpath_or_all():
    repo = Path(__file__).resolve().parents[2]
    experiment = repo / "experiments" / "keyframe_oracle_sampling"
    architecture_script = (experiment / "run_architecture_smoke.sbatch").read_text()
    smoke_script = (experiment / "run_smoke.sbatch").read_text()
    architecture_submit = (experiment / "submit_architecture_smoke.py").read_text()
    smoke_submit = (experiment / "submit_smoke.py").read_text()
    architecture_gate = (experiment / "architecture_smoke.py").read_text()
    row_preflight = (experiment / "preflight_smoke_row.py").read_text()

    for script in (architecture_script, smoke_script):
        assert "${PYTHONPATH" not in script
    for submitter in (architecture_submit, smoke_submit):
        assert "--export=ALL" not in submitter
        assert "--export=KEYFRAME_REPO_ROOT=" in submitter
        assert "validate_environment_contract(manifest)" in submitter
    assert "validate_environment_contract(launch_manifest)" in architecture_gate
    assert "validate_environment_contract(manifest)" in row_preflight
