"""Fail-closed validation of the two frozen Python environments.

The preparation step records exact lockfile and interpreter identities.  This
module recomputes those identities at each run-level gate so a prepared smoke
cannot silently execute after either environment has changed.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    if dict(expected_pythons) != observed["python_environments"]:
        raise EnvironmentContractError(
            "Live policy/simulator Python environments differ from the prepared manifest"
        )
    return observed
