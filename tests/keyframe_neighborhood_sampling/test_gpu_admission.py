"""GPU sharing is opt-in and bounded; these tests never query a real device."""

import pytest

from experiments.keyframe_neighborhood_sampling import submit_direct as direct
from experiments.keyframe_neighborhood_sampling.direct_runtime import ResourceBusyError
from experiments.keyframe_neighborhood_sampling.gpu_admission import admission_policy

GPU = "GPU-11111111-1111-1111-1111-111111111111"


def shared_profile():
    return {
        "gpu_layout": "colocated",
        "gpu_admission": {"mode": "shared", "min_free_memory_mib": 49152, "max_utilization_gpu_percent": 80},
    }


def test_legacy_remains_exclusive(monkeypatch):
    calls = []
    monkeypatch.setattr(direct, "check_idle_gpus", calls.append)
    assert admission_policy({}) == {"mode": "exclusive"}
    result = direct.check_gpu_admission((GPU,), {})
    assert calls == [(GPU,)]
    assert result["policy"]["mode"] == "exclusive"


@pytest.mark.parametrize(("field", "value"), [
    ("mode", "automatic"), ("min_free_memory_mib", 0), ("min_free_memory_mib", True),
    ("min_free_memory_mib", 1.5), ("max_utilization_gpu_percent", -1),
    ("max_utilization_gpu_percent", 101), ("max_utilization_gpu_percent", False),
    ("unrecorded", 1),
])
def test_bad_policy_fails_closed(field, value):
    profile = shared_profile()
    profile["gpu_admission"][field] = value
    with pytest.raises(ValueError, match="GPU"):
        admission_policy(profile)


def test_sharing_requires_nonpreallocating_colocated_mode():
    profile = shared_profile()
    profile["gpu_layout"] = "separate"
    with pytest.raises(ValueError, match="colocated"):
        admission_policy(profile)


def test_shared_gpu_accepts_recorded_headroom_not_exclusivity(monkeypatch):
    commands = []

    def query(argv, **kwargs):
        commands.append(argv)
        assert kwargs["timeout"] == 10
        return f"{GPU}, 65712, 35, Default\n"

    monkeypatch.setattr(direct.subprocess, "check_output", query)
    monkeypatch.setattr(direct, "check_idle_gpus", lambda _: pytest.fail("Sharing must not demand exclusivity"))
    result = direct.check_gpu_admission((GPU, GPU), shared_profile())
    assert len(result["devices"]) == 1
    assert result["devices"][0]["free_memory_mib"] == 65712
    assert result["exclusive_capacity_guaranteed"] is False
    assert len(commands) == 1
    assert commands[0][0] == "nvidia-smi"


@pytest.mark.parametrize(("free", "utilization"), [(49151, 10), (60000, 81)])
def test_shared_gpu_rejects_insufficient_headroom(monkeypatch, free, utilization):
    monkeypatch.setattr(direct.subprocess, "check_output", lambda *_a, **_k: f"{GPU}, {free}, {utilization}, Default\n")
    with pytest.raises(ResourceBusyError):
        direct.check_gpu_admission((GPU,), shared_profile())


@pytest.mark.parametrize("text", ["", f"{GPU}, N/A, 0, Default\n", f"{GPU}, -1, 0, Default\n", f"{GPU}, 60000, 101, Default\n", f"{GPU}, 60000, 0, Default\n{GPU}, 60000, 0, Default\n"])
def test_shared_gpu_bad_inventory_fails_closed(monkeypatch, text):
    monkeypatch.setattr(direct.subprocess, "check_output", lambda *_a, **_k: text)
    with pytest.raises(ValueError, match=r"GPU|invalid literal"):
        direct.check_gpu_admission((GPU,), shared_profile())


def test_shared_policy_preserves_allocator_and_resource_limits(monkeypatch):
    monkeypatch.setattr(direct.os, "sched_getaffinity", lambda _: set(range(12)), raising=False)
    profile = direct.bind_host_resources(shared_profile(), [[GPU, GPU]])
    assert profile["gpu_admission"] == shared_profile()["gpu_admission"]
    assert profile["policy_allocator_environment"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    assert profile["memory_bytes_per_row"] == 96 * 1024**3


@pytest.mark.parametrize("mode", ["Exclusive_Process", "Prohibited", "Unknown"])
def test_nonshared_compute_mode_is_rejected_before_cuda(monkeypatch, mode):
    monkeypatch.setattr(direct.subprocess, "check_output", lambda *_a, **_k: f"{GPU}, 60000, 30, {mode}\n")
    with pytest.raises(ResourceBusyError, match="compute_mode"):
        direct.check_gpu_admission((GPU,), shared_profile())


def test_high_utilization_is_permitted_when_explicitly_configured(monkeypatch):
    profile = shared_profile()
    profile["gpu_admission"]["max_utilization_gpu_percent"] = 100
    monkeypatch.setattr(direct.subprocess, "check_output", lambda *_a, **_k: f"{GPU}, 60000, 100, Default\n")
    assert direct.check_gpu_admission((GPU,), profile)["devices"][0]["compute_mode"] == "Default"
