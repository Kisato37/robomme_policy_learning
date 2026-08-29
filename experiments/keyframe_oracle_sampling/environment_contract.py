"""Fail-closed validation of the two frozen Python environments.

The preparation step records exact lockfile and interpreter identities.  This
module recomputes those identities at each run-level gate so a prepared smoke
cannot silently execute after either environment has changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.keyframe_oracle_sampling.prepare_smoke import (
    POLICY_PYTHON,
    POLICY_VENV,
    SIMULATOR_PYTHON,
    SIMULATOR_VENV,
    environment_lock_identity,
    python_environment_identity,
)


class EnvironmentContractError(RuntimeError):
    """Raised when the live launch environment differs from preparation."""


_PYTHON_PATH_IDENTITY_FIELDS = frozenset(
    {
        "launcher_python_path",
        "reported_executable",
        "executable_realpath",
        "venv_prefix",
        "base_prefix",
    }
)


def _same_python_environment_identity(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    """Match content exactly while accepting two paths to the same filesystem object."""
    if set(expected) != set(observed):
        return False
    for field, expected_value in expected.items():
        observed_value = observed[field]
        if expected_value == observed_value:
            continue
        if field not in _PYTHON_PATH_IDENTITY_FIELDS:
            return False
        if not isinstance(expected_value, str) or not isinstance(observed_value, str):
            return False
        try:
            if not Path(expected_value).samefile(observed_value):
                return False
        except OSError:
            return False
    return True


def live_environment_identity() -> dict[str, Any]:
    """Return the exact environment provenance used by both launch paths."""
    return {
        "environment_locks": environment_lock_identity(),
        "python_environments": {
            "policy": python_environment_identity(POLICY_PYTHON, POLICY_VENV),
            "simulator": python_environment_identity(
                SIMULATOR_PYTHON, SIMULATOR_VENV
            ),
        },
    }


def validate_environment_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and exactly match environment provenance from ``manifest``."""
    expected_locks = manifest.get("environment_locks")
    expected_pythons = manifest.get("python_environments")
    if not isinstance(expected_locks, Mapping):
        raise EnvironmentContractError(
            "Prepared manifest lacks the frozen environment-lock identities"
        )
    if not isinstance(expected_pythons, Mapping):
        raise EnvironmentContractError(
            "Prepared manifest lacks the frozen Python-environment identities"
        )

    observed = live_environment_identity()
    if dict(expected_locks) != observed["environment_locks"]:
        raise EnvironmentContractError(
            "Live policy/benchmark lockfiles differ from the prepared manifest"
        )
    observed_pythons = observed["python_environments"]
    if (
        set(expected_pythons) != set(observed_pythons)
        or any(
            not isinstance(expected_pythons[role], Mapping)
            or not isinstance(observed_pythons[role], Mapping)
            or not _same_python_environment_identity(
                expected_pythons[role], observed_pythons[role]
            )
            for role in expected_pythons
        )
    ):
        raise EnvironmentContractError(
            "Live policy/simulator Python environments differ from the prepared manifest"
        )
    return observed
