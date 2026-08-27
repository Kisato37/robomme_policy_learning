from __future__ import annotations

import pytest

from experiments.keyframe_oracle_sampling.audit_smoke import (
    audit_short_trajectory_length,
)


@pytest.mark.parametrize("reason", ["success", "fail", "timeout"])
def test_short_smoke_accepts_an_official_terminal_before_cap(reason):
    audit = audit_short_trajectory_length({"steps": 23, "terminal_reason": reason})
    assert audit == {
        "valid": True,
        "steps": 23,
        "terminal_reason": reason,
        "completion_mode": "official_terminal_before_cap",
    }


@pytest.mark.parametrize("reason", ["timeout", "success", "fail"])
def test_short_smoke_accepts_exactly_the_64_step_cap(reason):
    audit = audit_short_trajectory_length({"steps": 64, "terminal_reason": reason})
    assert audit["valid"] is True
    assert audit["completion_mode"] == "reached_64_step_cap"


@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        (23, "error"),
        (23, "unknown"),
        (64, "ongoing"),
        (64, "unknown"),
        (0, "success"),
        (65, "success"),
        (True, "success"),
        (None, "success"),
    ],
)
def test_short_smoke_rejects_nonterminal_early_exit_or_bad_length(steps, reason):
    assert audit_short_trajectory_length(
        {"steps": steps, "terminal_reason": reason}
    )["valid"] is False
