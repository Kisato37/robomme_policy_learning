"""Validate effective direct-run environment before exec; stdlib only.

Invoke with an absolute script path and Python -I AFTER sourcing the runtime
and graphics setup. Isolated mode prevents an unvalidated PYTHONPATH from
shadowing this validator's imports, while os.environ remains available for
comparison. No model, repository module, CUDA, or simulator is imported here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

NUMERICAL_PREFIXES = (
    "JAX_",
    "XLA_",
    "TF_",
    "CUBLAS_",
    "CUDNN_",
    "NVIDIA_TF32_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "PYTORCH_",
)
MAX_RECORD_BYTES = 1024 * 1024


def load_environment_record(path: str, expected_sha256: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Expected SHA256 must be 64 lowercase hexadecimal characters")
    target = Path(path)
    if not target.is_absolute() or str(target) != path or str(target.resolve(strict=True)) != path:
        raise ValueError("Environment record must use its canonical absolute nonsymlink path")
    # O_NONBLOCK also prevents a substituted FIFO from hanging before fstat.
    descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Environment record must be a regular file")
        data = stream.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("Environment record is too large")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Environment record SHA256 differs from the recorded launch identity")
    record = json.loads(data)
    canonical = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if data != canonical:
        raise ValueError("Environment record must contain canonical JSON bytes")
    if not isinstance(record, dict) or set(record) != {"protected_environment", "numerical_environment"}:
        raise ValueError("Environment record has missing or unknown schema fields")
    return record


def validate_environment(record: Mapping, environ: Mapping[str, str]) -> None:
    protected = record.get("protected_environment")
    numerical = record.get("numerical_environment")
    if not isinstance(protected, dict) or not isinstance(numerical, dict):
        raise ValueError("Environment snapshots must be objects")
    if not {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "KEYFRAME_RUNNER_BACKEND"}.issubset(protected):
        raise ValueError("Protected environment lacks required CUDA, Python-path, or backend identity")
    for key, value in protected.items():
        if (
            not isinstance(key, str)
            or (key not in {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"} and re.fullmatch(r"KEYFRAME_[A-Z0-9_]+", key) is None)
            or (value is not None and not isinstance(value, str))
        ):
            raise ValueError("Invalid protected environment key or value")
    if protected["KEYFRAME_RUNNER_BACKEND"] != "direct":
        raise ValueError("Protected environment must declare the direct backend")
    for key, value in numerical.items():
        if (
            not isinstance(key, str)
            or not key.startswith(NUMERICAL_PREFIXES)
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError("Invalid numerical environment snapshot")
    if any(key.startswith(("SLURM_", "SLURMD_")) for key in environ):
        raise ValueError("Effective direct environment contains Slurm variables")
    changed = sorted(key for key, value in protected.items() if environ.get(key) != value)
    unexpected = sorted(key for key in environ if key.startswith("KEYFRAME_") and key not in protected)
    if changed or unexpected:
        raise ValueError("Effective protected environment changed: " + ", ".join(sorted(set(changed + unexpected))))
    observed_numerical = {key: value for key, value in environ.items() if value and key.startswith(NUMERICAL_PREFIXES)}
    if observed_numerical != numerical:
        changed_numerical = sorted(
            key
            for key in observed_numerical.keys() | numerical.keys()
            if observed_numerical.get(key) != numerical.get(key)
        )
        raise ValueError("Effective numerical environment changed: " + ", ".join(changed_numerical))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-record", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if not sys.flags.isolated:
            raise ValueError("Invoke this validator with Python -I and its absolute script path")
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command or not command[0]:
            raise ValueError("An explicit payload command is required")
        record = load_environment_record(args.environment_record, args.expected_sha256)
        environ = dict(os.environ)
        validate_environment(record, environ)
        os.execvpe(command[0], command, environ)
    except Exception as error:
        print(f"direct-payload: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 125
    raise RuntimeError("execvpe unexpectedly returned")


if __name__ == "__main__":
    sys.exit(main())
