"""CPU end-to-end seam: real websocket + buffer/selector + write-once store.

The simulator, visual features, final encoder and action values are fixtures.
This does NOT claim actual checkpoint or GPU validation.
"""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.uniform_keyframe_expansion import contract
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore
from experiments.uniform_keyframe_expansion.evaluator import BenchmarkComponents, evaluate_attempt
from experiments.uniform_keyframe_expansion.serving import ExpansionClient
from tests.uniform_keyframe_expansion.test_evaluator import Environment, Recorder, utils
from tests.uniform_keyframe_expansion.test_runtime import fixture_policy, fixture_buffer
from tests.uniform_keyframe_expansion.test_serving import FakePolicy, IDENTITY, loopback, wait_released


def run_provenance():
    return {"code_commit": "a" * 40, "benchmark_commit": "b" * 40,
            "protocol_sha256": "c" * 64, "environment_manifest_sha256": "d" * 64,
            "checkpoint_archive_sha256": contract.CHECKPOINT_ARCHIVE_SHA256,
            "host": "cpu-fixture", "hardware": {"scope": "cpu-no-checkpoint"},
            "command": ["pytest", "test_episode_integration.py"]}


class BufferPolicy(FakePolicy):
    """Use the production memory-gather code with synthetic cached features."""

    def reset(self):
        super().reset()
        self.stages = []
        self.calls = 0

    def add_buffer(self, payload):
        super().add_buffer(payload)
        self.stages.extend(payload["current_task_index"].tolist())

    def infer(self, observation):
        boundaries = tuple(i for i, stage in enumerate(self.stages) if i == 0 or stage != self.stages[i - 1])
        # Configure a blank real policy adapter using the transported config,
        # then supply CPU fixture visual features for every actually received frame.
        policy = fixture_policy(length=1, real_dimensions=True)
        policy.mem_buffer = fixture_buffer()
        policy.step_idx = -1
        policy._keyframe_selector_config = None
        policy.configure_uniform_keyframe_expansion(self.config)
        policy.mem_buffer = fixture_buffer(len(self.stages), boundaries, real_dimensions=True)
        policy.step_idx = len(self.stages) - 1
        policy._selector_call_index = self.calls
        policy._prepare_history({})
        policy._record_final_memory_tensor(np.zeros((1, 768, 1024), np.float32))
        trace = deepcopy(policy._pending_selector_trace)
        trace.update(environment_step=observation["keyframe_environment_step"], model_latency_ms=1.0)
        self.calls += 1
        return {"selector_trace": trace, "actions": np.zeros((20, 8), np.float32), "infer_time_ms": 1.0}


@pytest.mark.parametrize("status", ["success", "fail", "timeout", "error"])
def test_full_adapter_real_artifacts_cross_arm_residency_and_early_terminal(tmp_path, status):
    store = ExpansionRunStore.create(tmp_path / "uniform_keyframe_expansion" / f"cpu-{status}",
                                    stage="smoke", run_manifest=run_provenance())
    components = BenchmarkComponents(utils.EpisodeState, utils.pack_buffer, Recorder, tuple(utils.TASK_WITH_VIDEO_DEMO))
    with loopback(BufferPolicy()) as (policy, server, port):
        initial = []
        for row in contract.build_smoke_matrix()["rows"][:2]:
            def environment(*args, **kwargs):
                return Environment(*args, **kwargs, prefix_length=43, stop_at=18, status=status)
            result = evaluate_attempt(
                store, row, 0, env_factory=environment,
                client_factory=lambda: ExpansionClient("127.0.0.1", port, expected_execution_identity=IDENTITY),
                components=components, episode_provenance={"policy_execution_identity": IDENTITY},
            )
            wait_released(server)
            audit = store.audit_attempt(row, 0)
            assert result["terminal_reason"] == status and audit["status"] == "complete"
            assert audit["smoke_readiness_pass"] is (status != "error")
            directory = store.attempt_dir(row, 0)
            initial.append(json.loads((directory / "initial_condition_hashes.json").read_text())["hashes"])
            assert (directory / "attachments/initial_observations.npz").is_file()
            assert (directory / "attachments/initial_task_state.json").is_file()
            assert (directory / "rollout.mp4").is_file()
            assert not (directory / ".video_staging/rollout.mp4").exists()
            with pytest.raises(ValueError):
                store.new_attempt(row, 1, {})  # Scientific fail/error is not retryable.
        assert initial[0] == initial[1]
        assert policy.resets == 2  # Same resident object; different fresh episodes.
    report = store.completeness()
    assert report["completed_count"] == 2 and report["complete"] is False
    assert report["expected_count"] == 48  # Two local fixtures never claim full smoke.


@pytest.mark.parametrize("status", ["success", "fail", "timeout", "error"])
def test_known_terminal_then_video_io_failure_blocks_real_store_retry(tmp_path, monkeypatch, status):
    store = ExpansionRunStore.create(tmp_path / "uniform_keyframe_expansion" / f"terminal-io-{status}",
                                    stage="smoke", run_manifest=run_provenance())
    components = BenchmarkComponents(utils.EpisodeState, utils.pack_buffer, Recorder, tuple(utils.TASK_WITH_VIDEO_DEMO))
    def broken(*args):
        raise OSError(5, "fixture video EIO after terminal")
    monkeypatch.setattr(Recorder, "save_video", broken)
    row, other = contract.build_smoke_matrix()["rows"][:2]
    with loopback(BufferPolicy()) as (_, server, port):
        def environment(*args, **kwargs):
            return Environment(*args, **kwargs, prefix_length=43, stop_at=1, status=status)
        with pytest.raises(OSError, match="video EIO"):
            evaluate_attempt(
                store, row, 0, env_factory=environment,
                client_factory=lambda: ExpansionClient("127.0.0.1", port, expected_execution_identity=IDENTITY),
                components=components, episode_provenance={"policy_execution_identity": IDENTITY},
            )
        wait_released(server)
    audit = store.audit_attempt(row, 0)
    assert audit["retry_allowed"] is False
    with pytest.raises(ValueError):
        store.new_attempt(row, 1, {})
    with pytest.raises(ValueError):
        store.new_attempt(other, 0, {})
