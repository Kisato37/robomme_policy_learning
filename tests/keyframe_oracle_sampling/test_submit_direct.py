"""CPU-only direct launcher contracts; fake UUIDs never initialize CUDA."""

# ruff: noqa: PLC0415

from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
import threading
import uuid

import pytest

from experiments.keyframe_neighborhood_sampling import submit_direct as direct
from experiments.keyframe_neighborhood_sampling.direct_runtime import GpuLease
from experiments.keyframe_neighborhood_sampling.direct_runtime import ResourceBusyError
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_oracle_sampling.artifacts import sha256_file

GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"
GPU_C = "GPU-33333333-3333-3333-3333-333333333333"
GPU_D = "GPU-44444444-4444-4444-4444-444444444444"


@pytest.mark.parametrize("backend", ["slurm", "direct"])
@pytest.mark.parametrize("stage", ["smoke", "formal"])
def test_preparation_instructions_use_selected_backend(tmp_path, monkeypatch, capsys, backend, stage):
    from experiments.keyframe_neighborhood_sampling import prepare_formal
    from experiments.keyframe_neighborhood_sampling import prepare_smoke

    module = prepare_smoke if stage == "smoke" else prepare_formal
    arguments = [
        "prepare", "--run-root", str(tmp_path / "new-run"),
        "--checkpoint-archive", str(tmp_path / "checkpoint.zip"),
        "--runner-backend", backend,
    ]
    if stage == "formal":
        arguments.extend([
            "--architecture-run-root", str(tmp_path / "architecture"),
            "--development-smoke-run-root", str(tmp_path / "smoke"),
            "--authorization-note", "test fixture only",
        ])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(module, "prepare", lambda *_args, **_kwargs: tmp_path / "seed_table.json")
    module.main()
    output = capsys.readouterr().out
    expected_stage = "architecture_smoke" if stage == "smoke" else "formal"
    expected_entry = f"submit_direct --stage {expected_stage}" if backend == "direct" else f"submit_{expected_stage}"
    assert expected_entry in output


def dispatch(tmp_path, stage="development_smoke"):
    return DirectDispatch(
        execution_id=str(uuid.uuid4()),
        run_root=str(tmp_path / "runs/keyframe_neighborhood_sampling/test"),
        repository_commit_sha="a" * 40,
        stage=stage,
        launch_manifest_sha256="b" * 64,
        submission_plan_sha256="c" * 64,
        matrix_sha256="d" * 64,
        attempt_id=0,
        row_id=None if stage == "architecture_smoke" else 0,
        shard_id=0 if stage == "formal" else None,
        gpu_uuids=(GPU_A,) if stage == "architecture_smoke" else (GPU_A, GPU_B),
        host_name=socket.gethostname(),
        host_boot_id=str(uuid.uuid4()),
        policy_port=None if stage == "architecture_smoke" else 23456,
    )


@pytest.mark.parametrize(
    ("stage", "pairs"),
    [
        ("architecture_smoke", [[GPU_A, GPU_B]]),
        ("architecture_smoke", [[GPU_A], [GPU_B]]),
        ("development_smoke", [[GPU_A]]),
        ("formal", [[GPU_A, GPU_A]]),
        ("formal", [[GPU_A, GPU_B], [GPU_C, GPU_A]]),
        ("formal", [["0", "1"]]),
    ],
)
def test_bad_allocations_rejected(stage, pairs):
    with pytest.raises(ValueError, match=r"Expected|requires|overlap|UUID"):
        direct.validate_allocations(stage, pairs)


def test_colocated_mode_has_two_roles_but_one_physical_device(tmp_path, monkeypatch):
    direct.validate_allocations("development_smoke", [[GPU_A, GPU_A]], "colocated")
    direct.validate_allocations("development_smoke", [[GPU_A, GPU_A], [GPU_B, GPU_B]], "colocated")
    for key in list(direct.os.environ):
        if key.startswith("SLURM_"):
            monkeypatch.delenv(key)
    d = replace(dispatch(tmp_path), gpu_layout="colocated", gpu_uuids=(GPU_A, GPU_A))
    for role in ("policy", "evaluator", "preflight", "reconcile"):
        env = direct.child_environment(d, Path(d.run_root) / "dispatch.json", "e" * 64, role)
        assert env["CUDA_VISIBLE_DEVICES"] == GPU_A
        if role == "policy":
            assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


@pytest.mark.parametrize("pairs", [[[GPU_A, GPU_B]], [[GPU_A, GPU_A], [GPU_A, GPU_A]], [[GPU_A]]])
def test_colocated_rejects_mixed_or_cross_slot_overlap(pairs):
    with pytest.raises(ValueError, match=r"overlap|physical GPU"):
        direct.validate_allocations("development_smoke", pairs, "colocated")


@pytest.mark.parametrize("module_name", ["prepare_smoke", "prepare_formal"])
def test_colocated_is_not_silently_added_to_slurm(tmp_path, module_name):
    from experiments.keyframe_neighborhood_sampling import prepare_formal
    from experiments.keyframe_neighborhood_sampling import prepare_smoke

    module = prepare_smoke if module_name == "prepare_smoke" else prepare_formal
    args = [tmp_path / "run", tmp_path / "checkpoint.zip"]
    kwargs = {"runner_backend": "slurm", "gpu_layout": "colocated"}
    if module_name == "prepare_formal":
        args.extend([tmp_path / "arch", tmp_path / "smoke"])
        kwargs["authorization_note"] = "test fixture only"
    with pytest.raises(ValueError, match="explicit direct"):
        module.prepare(*args, **kwargs)


def test_explicit_uuid_roles_and_no_invented_slurm(tmp_path, monkeypatch):
    for key in list(direct.os.environ):
        if key.startswith("SLURM_"):
            monkeypatch.delenv(key)
    d = dispatch(tmp_path)
    path = Path(d.run_root) / "direct/attempt_00/row_0000/dispatch.json"
    for role, expected in (("policy", GPU_A), ("evaluator", GPU_B), ("preflight", f"{GPU_A},{GPU_B}")):
        env = direct.child_environment(d, path, "e" * 64, role)
        assert env["CUDA_VISIBLE_DEVICES"] == expected
        assert env["KEYFRAME_SMOKE_ROW_ID"] == "0"
        assert env["KEYFRAME_RUNNER_BACKEND"] == "direct"
        assert not any(key.startswith("SLURM_") for key in env)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="Slurm"):
        direct.child_environment(d, path, "e" * 64, "policy")


@pytest.mark.parametrize("stage", ["development_smoke", "formal"])
def test_commands_preserve_frozen_experiment_and_socket_identity(tmp_path, stage):
    d = dispatch(tmp_path, stage)
    row = {
        "row_id": 0,
        "task": "BinFill",
        "episode_id": 0,
        "arm": "OC3",
        "trajectory_kind": "formal" if stage == "formal" else "short",
        "max_steps": 1300 if stage == "formal" else 64,
        "dataset": "test" if stage == "formal" else "val",
    }
    commands = direct.row_commands(d, row, Path(d.run_root) / "dispatch.json", "e" * 64, 17)
    assert "--seed=7" in commands["policy"]
    assert "--listen-fd=17" in commands["policy"]
    assert f"--execution-id={d.execution_id}" in commands["policy"]
    assert "--args.obs-horizon=16" in commands["evaluator"]
    assert "--args.keyframe-selector-arm=OC3" in commands["evaluator"]
    assert "--args.model-seed=7" in commands["evaluator"]
    assert "--args.model-ckpt-id=79999" in commands["evaluator"]
    assert "--args.episode-ids=0" in commands["evaluator"]
    assert f"--args.dataset={row['dataset']}" in commands["evaluator"]
    assert ("--formal-authorization" in commands["preflight"]) == (stage == "formal")
    assert all("sbatch" not in command for command in commands.values())


def test_busy_gpu_is_never_cancelled_or_substituted(monkeypatch):
    calls = []

    def query(argv, **_):
        calls.append(argv)
        if "--query-compute-apps=gpu_uuid,pid" in argv:
            return f"{GPU_B}, 456\n"
        return f"{GPU_A}, 0, 0\n{GPU_B}, 500, 0\n"

    monkeypatch.setattr(direct.subprocess, "check_output", query)
    with pytest.raises(ResourceBusyError):
        direct.check_idle_gpus((GPU_A, GPU_B))
    assert len(calls) == 2
    assert all(command[0] == "nvidia-smi" for command in calls)


def test_runtime_scripts_are_bound_and_not_global(tmp_path):
    setup = tmp_path / "environment.sh"
    graphics = tmp_path / "graphics.sh"
    setup.write_text("export TEST_ONLY=value\n")
    graphics.write_text('exec "$@"\n')
    profile = direct.runtime_profile(setup, graphics, tmp_path)
    command = direct.wrapped_command([sys.executable, "-V"], profile, graphics=True)
    assert command[-2:] == [sys.executable, "-V"]
    assert str(graphics) in command
    assert 'source "$1"; shift; exec "$@"' in command
    setup.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        direct.verify_runtime_profile(profile)


def test_failed_setup_never_executes_payload(tmp_path):
    setup, graphics = tmp_path / "environment.sh", tmp_path / "graphics.sh"
    setup.write_text("false\n")
    graphics.write_text('exec "$@"\n')
    marker = tmp_path / "payload-started"
    command = direct.wrapped_command(
        [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)],
        direct.runtime_profile(setup, graphics, tmp_path),
        graphics=False,
    )
    assert direct.subprocess.run(command, check=False, timeout=5).returncode != 0
    assert not marker.exists()


def test_resource_affinity_is_twelve_disjoint_cpus_per_pair(monkeypatch):
    monkeypatch.setattr(direct.os, "sched_getaffinity", lambda _: set(range(48)), raising=False)
    profile = direct.bind_host_resources({}, [[GPU_A, GPU_B], [GPU_C, GPU_D]])
    groups = profile["cpu_ids_by_policy_gpu"]
    assert len(groups[GPU_A]) == len(groups[GPU_C]) == 12
    assert not set(groups[GPU_A]) & set(groups[GPU_C])
    assert profile["memory_bytes_per_row"] == 96 * 1024**3


def test_numerical_overrides_are_not_silently_inherited():
    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
        direct.reject_numerical_overrides({"JAX_ENABLE_X64": "true"})
    direct.reject_numerical_overrides({"CUDA_VISIBLE_DEVICES": "0", "LANG": "en_US.UTF-8"})


def test_publish_full_formal_plan_before_shard_records(tmp_path):
    root = tmp_path
    (root / "protocol").mkdir()
    paths = [root / f"protocol/submission_record_shard_{i:02d}.json" for i in range(2)]
    records = [(paths[i], {"shard_id": i}) for i in range(2)]
    plan = {"attempt_id": 0, "shards": [{"row_ids": list(range(1000))}, {"row_ids": list(range(1000, 1600))}]}
    direct.publish_submission(root, records, plan)
    from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_plan_path

    digest = sha256_file(submission_plan_path(root, 0))
    assert all(json.loads(path.read_text())["submission_plan_sha256"] == digest for path in paths)
    with pytest.raises(FileExistsError):
        direct.publish_submission(root, records, plan)


@pytest.mark.skipif(sys.platform != "linux", reason="Real owned process proof requires Linux")
def test_real_cpu_execution_produces_auditable_lifecycle(tmp_path, monkeypatch):
    from dataclasses import replace

    from experiments.keyframe_neighborhood_sampling.direct_provenance import audit_direct_completion

    d = replace(
        dispatch(tmp_path, "architecture_smoke"),
        host_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    directory = Path(d.run_root) / "direct/architecture"
    directory.mkdir(parents=True)
    setup, graphics = tmp_path / "environment.sh", tmp_path / "graphics.sh"
    setup.write_text(":\n")
    graphics.write_text('exec "$@"\n')
    profile = direct.runtime_profile(setup, graphics, tmp_path)
    with GpuLease(tmp_path, (GPU_A,)) as lease:
        ex = direct.Execution(d, directory, profile, lease, threading.Event())
        ex.start("architecture", [sys.executable, "-c", "print('cpu-only-fixture')"])
        assert ex.wait("architecture") == 0
        ex.complete()
    envelope = {
        "backend": "direct",
        "dispatch_path": str(ex.path),
        "dispatch_sha256": ex.digest,
        "dispatch": d.as_record(),
    }
    result = audit_direct_completion(envelope, Path(d.run_root), required_roles={"architecture"})
    assert result["cleanup_confirmed"] is True
    assert (directory / "architecture.out").read_text().strip() == "cpu-only-fixture"


def test_existing_dispatch_never_runs_again(tmp_path, monkeypatch):
    d = dispatch(tmp_path)
    root = Path(d.run_root)
    directory = root / "direct/attempt_00/row_0000"
    directory.mkdir(parents=True)
    setup, graphics = tmp_path / "environment.sh", tmp_path / "graphics.sh"
    setup.write_text(":\n")
    graphics.write_text('exec "$@"\n')
    record_path = root / "record.json"
    write_once_record(
        record_path, {"attempt_id": 0, "runtime_profile": direct.runtime_profile(setup, graphics, tmp_path)}
    )
    monkeypatch.setattr(direct, "check_idle_gpus", lambda _: pytest.fail("Must fail before querying GPUs"))
    with pytest.raises(FileExistsError):
        direct.execute_one(root, "development_smoke", record_path, {"row_id": 0}, (GPU_A, GPU_B), threading.Event())
