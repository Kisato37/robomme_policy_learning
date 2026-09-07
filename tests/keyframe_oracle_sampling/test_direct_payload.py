"""CPU-only tests: effective runtime setup may never bypass the launch identity."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.keyframe_neighborhood_sampling import direct_payload

SCRIPT = str(Path(direct_payload.__file__).resolve())


def snapshot():
    return {
        "protected_environment": {
            "CUDA_VISIBLE_DEVICES": "GPU-11111111-1111-1111-1111-111111111111",
            "PYTHONPATH": "/not-used-by-isolated-validator",
            "KEYFRAME_RUNNER_BACKEND": "direct",
            "KEYFRAME_DIRECT_DISPATCH_PATH": "/canonical/dispatch.json",
            "KEYFRAME_DIRECT_DISPATCH_SHA256": "a" * 64,
            "KEYFRAME_SMOKE_ROW_ID": "0",
            "KEYFRAME_FORMAL_ROW_ID": None,
        },
        "numerical_environment": {},
    }


def record_file(tmp_path, record=None):
    record = snapshot() if record is None else record
    path = tmp_path.resolve() / "environment.json"
    data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def effective_environment(record):
    environ = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("KEYFRAME_", "SLURM_", "SLURMD_", *direct_payload.NUMERICAL_PREFIXES))
    }
    environ.update({key: value for key, value in record["protected_environment"].items() if value is not None})
    for key, value in record["protected_environment"].items():
        if value is None:
            environ.pop(key, None)
    environ.update(record["numerical_environment"])
    return environ


def invocation(path, digest, *, isolated=True):
    return [
        sys.executable,
        *(["-I"] if isolated else []),
        SCRIPT,
        "--environment-record",
        str(path),
        "--expected-sha256",
        digest,
        "--",
        sys.executable,
        "-I",
        "-c",
        "print('payload-started')",
    ]


def test_matching_canonical_environment_executes_payload(tmp_path):
    record = snapshot()
    path, digest = record_file(tmp_path, record)
    result = subprocess.run(
        invocation(path, digest),
        env=effective_environment(record),
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "payload-started"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("PYTHONPATH", "/unreviewed-imports"),
        ("KEYFRAME_RUNNER_BACKEND", "slurm"),
        ("KEYFRAME_DIRECT_DISPATCH_SHA256", "b" * 64),
        ("KEYFRAME_FORMAL_ROW_ID", "0"),
        ("KEYFRAME_UNRECORDED", "1"),
        ("JAX_ENABLE_X64", "1"),
        ("OMP_NUM_THREADS", "12"),
        ("SLURM_JOB_ID", ""),
        ("SLURMD_NODENAME", "test"),
    ],
)
def test_effective_overrides_fail_before_payload(tmp_path, key, value):
    record = snapshot()
    path, digest = record_file(tmp_path, record)
    environ = effective_environment(record)
    environ[key] = value
    result = subprocess.run(
        invocation(path, digest), env=environ, capture_output=True, text=True, check=False, timeout=5
    )
    assert result.returncode == 125
    assert result.stdout == ""


def test_null_means_absent_not_empty():
    record = snapshot()
    environ = effective_environment(record)
    environ["KEYFRAME_FORMAL_ROW_ID"] = ""
    with pytest.raises(ValueError, match="KEYFRAME_FORMAL_ROW_ID"):
        direct_payload.validate_environment(record, environ)


def test_reviewed_architecture_numeric_flags_are_exact():
    record = snapshot()
    record["numerical_environment"] = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_EXPLAIN_CACHE_MISSES": "true",
    }
    environ = effective_environment(record)
    direct_payload.validate_environment(record, environ)
    del environ["JAX_EXPLAIN_CACHE_MISSES"]
    with pytest.raises(ValueError, match="JAX_EXPLAIN_CACHE_MISSES"):
        direct_payload.validate_environment(record, environ)


def test_empty_unrecorded_numerical_variable_is_not_an_active_override():
    record = snapshot()
    environ = effective_environment(record)
    environ["JAX_ENABLE_X64"] = ""
    direct_payload.validate_environment(record, environ)


def test_record_hash_and_canonical_encoding_are_required(tmp_path):
    path, digest = record_file(tmp_path)
    assert direct_payload.load_environment_record(str(path), digest) == snapshot()
    with pytest.raises(ValueError, match="SHA256"):
        direct_payload.load_environment_record(str(path), "b" * 64)
    path.write_text(json.dumps(snapshot(), indent=2))
    with pytest.raises(ValueError, match="canonical JSON"):
        direct_payload.load_environment_record(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def test_symlink_and_relative_record_paths_are_rejected(tmp_path):
    path, digest = record_file(tmp_path)
    alias = path.with_name("alias.json")
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="canonical absolute"):
        direct_payload.load_environment_record(str(alias), digest)
    with pytest.raises(ValueError, match="canonical absolute"):
        direct_payload.load_environment_record("environment.json", digest)


def test_unknown_schema_fields_are_rejected(tmp_path):
    record = {**snapshot(), "unreviewed": True}
    path, digest = record_file(tmp_path, record)
    with pytest.raises(ValueError, match="schema"):
        direct_payload.load_environment_record(str(path), digest)


def test_fifo_record_is_rejected_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path.resolve() / "not-a-record"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        direct_payload.load_environment_record(str(fifo), "a" * 64)


def test_unisolated_invocation_refuses_payload(tmp_path):
    path, digest = record_file(tmp_path)
    result = subprocess.run(
        invocation(path, digest, isolated=False),
        env=effective_environment(snapshot()),
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 125
    assert result.stdout == ""
    assert "Python -I" in result.stderr


def test_isolated_validator_ignores_poisoned_pythonpath_before_rejection(tmp_path):
    # A startup module would execute before main() without -I. It must not get
    # an opportunity to run merely because the setup script changed PYTHONPATH.
    poison = tmp_path / "unreviewed-imports"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text("print('UNREVIEWED-STARTUP-CODE')\n")
    path, digest = record_file(tmp_path)
    environ = effective_environment(snapshot())
    environ["PYTHONPATH"] = str(poison)
    result = subprocess.run(
        invocation(path, digest), env=environ, capture_output=True, text=True, check=False, timeout=5
    )
    assert result.returncode == 125
    assert result.stdout == ""
    assert "PYTHONPATH" in result.stderr
