"""GPU telemetry tests use only fake queries; they never access a GPU or driver."""

import json
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from experiments.keyframe_neighborhood_sampling import gpu_telemetry

GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"
GPU_C = "GPU-33333333-3333-3333-3333-333333333333"


@pytest.fixture(autouse=True)
def forbid_real_queries(monkeypatch):
    monkeypatch.setattr(
        gpu_telemetry.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Real GPU query is forbidden in tests")
    )


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Telemetry fixture did not reach its expected state")


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("allocation", [[], ["0"], [GPU_A, GPU_A], ["MIG-test"], GPU_A])
def test_invalid_or_aliased_allocation_is_rejected_without_writes(tmp_path, allocation):
    path = tmp_path / "gpu.jsonl"
    with pytest.raises(ValueError, match="physical GPU UUIDs"):
        gpu_telemetry.GpuTelemetry(path, allocation)
    assert not path.exists()


@pytest.mark.parametrize(
    "options", [{"interval_seconds": 0}, {"interval_seconds": float("inf")}, {"query_timeout_seconds": 11}]
)
def test_unbounded_or_invalid_timing_is_rejected(tmp_path, options):
    with pytest.raises(ValueError, match=r"interval|timeout"):
        gpu_telemetry.GpuTelemetry(tmp_path / "gpu.jsonl", [GPU_A], **options)


def test_new_file_is_exclusive_and_constructor_is_inert(tmp_path):
    path = tmp_path / "gpu.jsonl"
    telemetry = gpu_telemetry.GpuTelemetry(path, [GPU_A])
    assert not path.exists()
    path.write_text("existing-evidence\n")
    with pytest.raises(FileExistsError):
        telemetry.start()
    assert path.read_text() == "existing-evidence\n"
    target = tmp_path / "other-evidence"
    target.write_text("preserve")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(target)
    with pytest.raises(FileExistsError):
        gpu_telemetry.GpuTelemetry(alias, [GPU_A]).start()
    assert target.read_text() == "preserve"


def test_selected_gpu_samples_and_sampled_peak_summary(tmp_path, monkeypatch):
    calls = []

    def fake_query(argv, **kwargs):
        calls.append((argv, kwargs))
        used = 10 if len(calls) == 1 else 25
        return SimpleNamespace(stdout=f"{GPU_B}, 81920, 4, 0\n{GPU_A}, 81920, {used}, 12\n")

    monkeypatch.setattr(gpu_telemetry.subprocess, "run", fake_query)
    path = tmp_path / "gpu.jsonl"
    with gpu_telemetry.GpuTelemetry(path, [GPU_A, GPU_B], interval_seconds=0.01) as telemetry:
        wait_for(lambda: telemetry.summary["successful_sample_count"] >= 2)
    observed = records(path)
    assert observed[0]["kind"] == "start"
    assert observed[-1]["kind"] == "summary"
    assert observed[-1]["sampled_peaks"][GPU_A]["sampled_peak_memory_used_mib"] == 25
    assert observed[-1]["sampled_peaks"][GPU_B]["sampled_peak_memory_used_mib"] == 4
    assert observed[-1]["peak_semantics"] == "maximum_of_successful_samples_not_exact_peak"
    assert observed[-1]["thread_stopped"] is True
    assert all(record["timestamp_utc"].endswith("+00:00") for record in observed)
    for argv, kwargs in calls:
        assert argv == [
            "nvidia-smi",
            f"--id={GPU_A},{GPU_B}",
            "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 2.0}
    before = path.read_bytes()
    telemetry.close()
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="restarted"):
        telemetry.start()


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "unselected", "missing", "unsupported"])
def test_query_failures_are_logged_then_monitor_recovers(tmp_path, monkeypatch, failure):
    calls = 0

    def fake_query(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 0.02, output=f"private {GPU_C}")
            if failure == "nonzero":
                raise subprocess.CalledProcessError(1, argv, stderr=f"private {GPU_C}")
            if failure == "unselected":
                return SimpleNamespace(stdout=f"{GPU_C}, 81920, 1, 0\n")
            if failure == "missing":
                return SimpleNamespace(stdout="")
            return SimpleNamespace(stdout=f"{GPU_A}, N/A, N/A, N/A\n")
        return SimpleNamespace(stdout=f"{GPU_A}, 81920, 15, 1\n")

    monkeypatch.setattr(gpu_telemetry.subprocess, "run", fake_query)
    path = tmp_path / "gpu.jsonl"
    with gpu_telemetry.GpuTelemetry(path, [GPU_A], interval_seconds=0.01) as telemetry:
        wait_for(lambda: telemetry.summary["successful_sample_count"] >= 1)
    assert telemetry.summary["query_error_count"] == 1
    assert GPU_C not in path.read_text()
    assert "private" not in path.read_text()
    assert any(record["kind"] == "query_error" for record in records(path))


def test_stop_interrupts_sampling_wait_and_does_not_mask_body_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gpu_telemetry.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{GPU_A}, 81920, 0, 0\n")
    )
    telemetry = gpu_telemetry.GpuTelemetry(tmp_path / "gpu.jsonl", [GPU_A], interval_seconds=60)
    timing = {}

    def experiment():
        with telemetry:
            wait_for(lambda: telemetry.summary["successful_sample_count"] == 1)
            timing["started"] = time.monotonic()
            raise RuntimeError("experiment-error")

    with pytest.raises(RuntimeError, match="experiment-error"):
        experiment()
    assert time.monotonic() - timing["started"] < 1
    assert telemetry.summary["thread_stopped"] is True


def test_close_has_a_bound_even_if_query_ignores_its_timeout(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def stuck_query(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=3)
        return SimpleNamespace(stdout=f"{GPU_A}, 81920, 1, 0\n")

    monkeypatch.setattr(gpu_telemetry.subprocess, "run", stuck_query)
    path = tmp_path / "gpu.jsonl"
    telemetry = gpu_telemetry.GpuTelemetry(path, [GPU_A], query_timeout_seconds=0.01).start()
    try:
        assert entered.wait(timeout=1)
        started = time.monotonic()
        telemetry.close()
        assert time.monotonic() - started < 2
        assert telemetry.summary["thread_stopped"] is False
        assert telemetry.summary["logging_error"] == "TelemetryThreadJoinTimeout"
        before = path.read_bytes()
    finally:
        release.set()
    wait_for(lambda: not any(thread.name == "gpu-telemetry" for thread in threading.enumerate()))
    assert path.read_bytes() == before
