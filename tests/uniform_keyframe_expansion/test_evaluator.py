"""CPU episode-loop integration with original packing/state utilities.

The simulator and action model are deterministic fixtures, never GPU jobs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.uniform_keyframe_expansion.contract import build_smoke_matrix
from experiments.uniform_keyframe_expansion.evaluator import (
    BenchmarkComponents, ExpansionEvaluationError, encode_task_state, evaluate_attempt, validate_reset_prefix, _classify_failure,
)
from mme_vla_suite.shared.uniform_keyframe_config import payload_digest, expanded_history_mapping, RELEASED_HISTORY_CONFIG
from tests.uniform_keyframe_expansion.test_trace_validation import synthetic_trace

REPO = Path(__file__).resolve().parents[2]
EXECUTION_IDENTITY = {"execution_id": "11111111-1111-4111-8111-111111111111", "dispatch_sha256": "a" * 64}
spec = importlib.util.spec_from_file_location("_expansion_original_utils", REPO / "examples/robomme/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def digest(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()
    if isinstance(value, list):
        return hashlib.sha256("".join(digest(v) for v in value).encode()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Writer:
    def __init__(self, path):
        self.attempt_dir = path
        path.mkdir()
        self.traces = []
        self.attachments = {}
        self.initial = self.result = self.failure = None

    def write_attachment(self, name, data):
        assert name not in self.attachments
        self.attachments[name] = data
        return hashlib.sha256(data).hexdigest()

    def record_initial_conditions(self, hashes, **evidence):
        assert "initial_observations.npz" in self.attachments
        assert self.initial is None and not self.traces
        self.initial = {"hashes": hashes, **evidence}

    def publish_video(self, source):
        data = source.read_bytes()
        (self.attempt_dir / "rollout.mp4").write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def append_trace(self, trace):
        assert self.initial is not None
        self.traces.append(trace)

    def finalize_scientific(self, result):
        assert self.failure is None
        assert result["policy_call_count"] == len(self.traces)
        self.result = result

    def record_failure(self, kind, code, message, evidence):
        assert self.result is None and self.failure is None
        self.failure = dict(kind=kind, code=code, message=message, evidence=evidence)


class Store:
    def __init__(self, root):
        self.root = root
        self.writer = None

    def new_attempt(self, row, attempt_id, provenance):
        assert self.writer is None
        self.writer = Writer(self.root / "attempt")
        return self.writer


class Recorder:
    def __init__(self, path, goal, fps):
        assert fps == 30
        self.path = path
        path.mkdir(exist_ok=True)
        self.frames = []

    def record(self, **frame):
        self.frames.append(frame)

    def save_video(self, name):
        (self.path / name).write_bytes(b"cpu-fixture-not-real-video")


class Environment:
    _digest_value = staticmethod(digest)
    _canonicalize_task_state_for_hashing = staticmethod(lambda v: deepcopy(v))

    def __init__(self, task, directory, *, max_steps, dataset, require_current_task_index,
                 prefix_length=3, stop_at=18, status="success", fault=None, stage_period=10):
        self.env_id, self.dataset = task, dataset
        self.require_current_task_index = require_current_task_index
        assert require_current_task_index is True
        self.max_steps = max_steps
        self.prefix_length, self.stop_at, self.status, self.fault = prefix_length, stop_at, status, fault
        self.stage_period = stage_period
        self.difficulty = self.resolved_difficulty_hint = "hard"
        self.resolved_environment_seed = 610002
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(get_state_dict=lambda: {"x": [1, 2.0]}))
        self.task_goal = "unchanged full task instruction"
        self.reset_count = self.steps = 0
        self.closed = False
        self.info = {}
        self.actions = []

    def make_env(self, episode):
        self.episode_id = episode
        if self.fault == "setup":
            raise RuntimeError("renderer construction error")

    @staticmethod
    def observation(index):
        return np.full((2, 2, 3), index % 256, np.uint8), np.full((2, 2, 3), 5, np.uint8), np.zeros(8, np.float32)

    def get_init_obs(self):
        self.reset_count += 1
        obs = [self.observation(i) for i in range(self.prefix_length)]
        self.prefix = {
            "images": [o[0] for o in obs], "wrist_images": [o[1] for o in obs], "states": [o[2] for o in obs],
            "current_task_indices": [i // self.stage_period for i in range(self.prefix_length)], "task_goal": self.task_goal,
        }
        if self.fault == "labels":
            self.prefix["current_task_indices"].pop()
        self.initial_condition_hashes = {
            "front_observations_sha256": digest(self.prefix["images"]),
            "wrist_observations_sha256": digest(self.prefix["wrist_images"]),
            "robot_states_sha256": digest(self.prefix["states"]),
            "task_instruction_sha256": digest(self.task_goal),
            "task_state_sha256": digest({"x": [1, 2.0]}),
        }
        if self.fault == "hash":
            self.initial_condition_hashes["task_instruction_sha256"] = "f" * 64
        return self.prefix

    def step(self, action):
        self.steps += 1
        self.actions.append(action.copy())
        if self.fault == "python_error":
            raise RuntimeError("not an official benchmark error")
        if self.fault == "simulator_timeout_exception":
            raise TimeoutError("simulator timeout is not proven transport failure")
        stop = self.steps == self.stop_at
        status = self.status if stop else "unknown"
        self.info = {"status": status}
        if status == "error":
            self.info.update(error_message="actual benchmark wrapper error", exception_type="FixtureCollision")
            return (None, None, None), True, status
        self.current_task_index = (self.prefix_length + self.steps - 1) // self.stage_period
        return self.observation(self.prefix_length + self.steps - 1), stop, status

    def close_env(self):
        self.closed = True


class Client:
    def __init__(self, fault=None):
        self.fault = fault
        self.segments = []
        self.stages = []
        self.call = 0
        self.closed = False

    def get_server_metadata(self):
        _, evidence = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
        metadata = {"wire_schema": 1, "experiment_family": "uniform_keyframe_expansion-v1",
                    "effective_memory_budget": 768, "evaluation_policy_seed": 7,
                    "resident_policy": True, "strict_weight_tree_load": True, "model_process_pid": 123,
                    "source_history_config_sha256": evidence["source_history_config_sha256"],
                    "effective_history_config_sha256": evidence["effective_history_config_sha256"],
                    "direct_execution": deepcopy(EXECUTION_IDENTITY)}
        if self.fault == "wrong_server":
            metadata["direct_execution"]["dispatch_sha256"] = "b" * 64
        if self.fault == "pid_mismatch":
            metadata["model_process_pid"] = 999
        return metadata

    def reset(self, config):
        from experiments.uniform_keyframe_expansion.serving import RESET_STATE

        self.config = deepcopy(config)
        reply = {"reset_finished": True, "reset_time_ms": 1.0, "resident_reset": {
            "state": deepcopy(RESET_STATE), "model_process_pid": 123,
            "experiment_family": "uniform_keyframe_expansion-v1", "selector_config_sha256": payload_digest(config),
        }}
        if self.fault == "dirty_reset":
            reply["resident_reset"]["state"]["history_empty"] = False
        return reply

    def add_buffer(self, payload):
        self.segments.append(deepcopy(payload))
        self.stages.extend(payload["current_task_index"].tolist())
        return {"add_buffer_finished": True}

    def infer(self, request):
        assert set(request) == {"observation/image", "observation/wrist_image", "observation/state",
                                "prompt", "keyframe_environment_step"}
        assert request["prompt"] == "unchanged full task instruction"
        if self.fault == "transport":
            raise ConnectionResetError("documented test transport failure")
        if self.fault == "server_protocol":
            raise RuntimeError("remote deterministic selector invariant")
        boundaries = [i for i, stage in enumerate(self.stages) if i == 0 or stage != self.stages[i - 1]]
        trace = synthetic_trace(self.config["arm"], len(self.stages) - 1, boundaries, self.call,
                                self.config["task"], self.config["split"], self.config["episode_id"])
        actions = np.arange(160, dtype=np.float32).reshape(20, 8) + 1000 * self.call
        self.call += 1
        if self.fault == "wrong_action":
            actions = actions[:16]
        if self.fault == "wrong_call":
            trace["environment_step"] = 16
        return {"selector_trace": trace, "actions": actions, "infer_time_ms": 5.0}

    def close(self):
        self.closed = True


def run_case(tmp_path, *, row_id=0, prefix_length=3, stop_at=18, status="success", env_fault=None, client_fault=None, stage_period=10):
    store = Store(tmp_path)
    env_holder = []
    client = Client(client_fault)
    def factory(*args, **kwargs):
        env = Environment(*args, **kwargs, prefix_length=prefix_length, stop_at=stop_at, status=status, fault=env_fault, stage_period=stage_period)
        env_holder.append(env)
        return env
    components = BenchmarkComponents(utils.EpisodeState, utils.pack_buffer, Recorder, tuple(utils.TASK_WITH_VIDEO_DEMO))
    def run():
        return evaluate_attempt(store, build_smoke_matrix()["rows"][row_id], 0,
                                env_factory=factory, client_factory=lambda: client, components=components,
                                episode_provenance={"policy_execution_identity": deepcopy(EXECUTION_IDENTITY)})
    return run, store, env_holder, client


@pytest.mark.parametrize("row_id", [0, 1, 2, 41])
def test_real_packing_single_reset_generated_prefix_and_16_action_execution(tmp_path, row_id):
    run, store, envs, client = run_case(tmp_path, row_id=row_id, prefix_length=43)
    result = run()
    env = envs[0]
    assert result["success"] is True and result["environment_steps"] == 18
    assert result["policy_call_count"] == 2
    assert env.reset_count == 1 and env.closed and client.closed
    assert [len(s["images"]) for s in client.segments] == [43, 16]
    assert [s["exec_start_idx"] for s in client.segments] == [42, 0]
    assert result["history_lengths_at_policy_calls"] == [43, 59]
    np.testing.assert_array_equal(env.actions[15], np.arange(120, 128))
    np.testing.assert_array_equal(env.actions[16], np.arange(8) + 1000)
    assert store.writer.initial["environment_provenance"]["resolved_environment_seed"] == 610002
    with np.load(io.BytesIO(store.writer.attachments["initial_observations.npz"]), allow_pickle=False) as raw:
        assert raw["front"].shape == (43, 2, 2, 3)
        assert raw["current_task_index"].tolist() == env.prefix["current_task_indices"]


@pytest.mark.parametrize("status", ["success", "fail", "timeout", "error"])
def test_early_official_terminal_preserved_and_never_stepped_past(tmp_path, status):
    run, store, envs, client = run_case(tmp_path, stop_at=1, status=status)
    result = run()
    assert result["terminal_reason"] == status
    assert result["success"] is (status == "success")
    assert envs[0].steps == client.call == 1
    assert store.writer.failure is None


@pytest.mark.parametrize(("row_id", "expected", "steps", "calls"), [(0, "short_limit", 64, 4), (2, "timeout", 1300, 82)])
def test_exact_short_and_full_limit_without_extra_step(tmp_path, row_id, expected, steps, calls):
    # Sparse stage changes avoid intentionally unsupported dense boundary unions.
    run, store, envs, client = run_case(tmp_path, row_id=row_id, stop_at=10000, stage_period=100000)
    result = run()
    assert result["terminal_reason"] == expected
    assert result["environment_steps"] == steps and client.call == calls
    assert envs[0].steps == steps and store.writer.failure is None


@pytest.mark.parametrize(("env_fault", "client_fault", "kind"), [
    ("setup", None, "hard_stop"), ("labels", None, "hard_stop"), ("hash", None, "hard_stop"),
    ("python_error", None, "hard_stop"), ("simulator_timeout_exception", None, "hard_stop"),
    (None, "dirty_reset", "hard_stop"), (None, "wrong_action", "hard_stop"),
    (None, "wrong_call", "hard_stop"), (None, "server_protocol", "hard_stop"),
    (None, "wrong_server", "hard_stop"), (None, "pid_mismatch", "hard_stop"),
    (None, "transport", "infrastructure"),
])
def test_failures_preserved_without_retry_and_cleanup_runs(tmp_path, env_fault, client_fault, kind):
    run, store, envs, client = run_case(tmp_path, env_fault=env_fault, client_fault=client_fault)
    with pytest.raises(Exception):
        run()
    assert store.writer.result is None and store.writer.failure["kind"] == kind
    assert store.writer.failure["evidence"]["automatic_retry"] is False
    assert len(envs) == 1 and envs[0].closed
    if env_fault not in ("setup", "labels"):
        assert client.closed


def test_task_state_archive_preserves_array_dtype_shape_and_bytes():
    array = np.array([[1.5, -2.5]], np.float32)
    encoded = encode_task_state({"actor": (array, None)})
    node = encoded["items"][0][1]["items"][0]
    import base64
    decoded = np.frombuffer(base64.b64decode(node["bytes_base64"]), dtype=node["dtype"]).reshape(node["shape"])
    np.testing.assert_array_equal(decoded, array)
    assert decoded.dtype == array.dtype


def test_result_published_before_ledger_error_is_not_reclassified_for_retry(tmp_path, monkeypatch):
    run, store, envs, client = run_case(tmp_path)
    def partial(self, result):
        (self.attempt_dir / "episode_result.json").write_text(json.dumps(result))
        raise OSError(5, "fixture ledger fsync failed after result publication")
    monkeypatch.setattr(Writer, "finalize_scientific", partial)
    with pytest.raises(OSError, match="ledger fsync") as caught:
        run()
    assert store.writer.failure is None
    assert "no retry" in " ".join(caught.value.__notes__)
    assert client.closed and envs[0].closed


def test_failure_ledger_error_does_not_hide_original_exception(tmp_path, monkeypatch):
    run, store, envs, _ = run_case(tmp_path, env_fault="setup")
    def failed_record(*args):
        raise OSError(28, "fixture full filesystem")
    monkeypatch.setattr(Writer, "record_failure", failed_record)
    with pytest.raises(RuntimeError, match="renderer construction") as caught:
        run()
    assert "could not be finalized" in " ".join(caught.value.__notes__)
    assert envs[0].closed


@pytest.mark.parametrize(("status", "phase"), [
    (status, phase) for status in ("success", "fail", "timeout", "error")
    for phase in ("save_video", "publish_video", "finalize_scientific")
] + [(status, "record") for status in ("success", "fail", "timeout")])
def test_filesystem_failure_after_known_terminal_never_allows_scientific_retry(tmp_path, monkeypatch, status, phase):
    run, store, envs, client = run_case(tmp_path, stop_at=1, status=status)
    def broken(*args, **kwargs):
        raise OSError(5, "fixture EIO after scientific termination")
    if phase == "record":
        original = Recorder.record
        def record(self, **frame):
            if "action" in frame:
                broken()
            return original(self, **frame)
        monkeypatch.setattr(Recorder, phase, record)
    else:
        monkeypatch.setattr(Recorder if phase == "save_video" else Writer, phase, broken)
    with pytest.raises(OSError, match="scientific termination"):
        run()
    failure = store.writer.failure
    assert failure["kind"] == "hard_stop"
    assert failure["evidence"]["known_terminal_outcome"] == {
        "terminal_reason": status, "success": status == "success", "official_terminal": True,
        "official_stop_flag": True, "environment_steps": 1, "policy_call_count": 1,
    }
    assert store.writer.result is None
    assert envs[0].steps == 1 and envs[0].closed and client.closed


@pytest.mark.parametrize(("row_id", "terminal"), [(0, "short_limit"), (2, "timeout")])
def test_recording_failure_at_horizon_limit_is_not_retryable(tmp_path, monkeypatch, row_id, terminal):
    run, store, envs, _ = run_case(tmp_path, row_id=row_id, stop_at=10000, stage_period=100000)
    def broken(*args):
        raise OSError(28, "fixture ENOSPC after horizon")
    monkeypatch.setattr(Recorder, "save_video", broken)
    with pytest.raises(OSError, match="after horizon"):
        run()
    assert store.writer.failure["kind"] == "hard_stop"
    assert store.writer.failure["evidence"]["known_terminal_outcome"]["terminal_reason"] == terminal
    assert envs[0].steps == build_smoke_matrix()["rows"][row_id]["max_steps"]


@pytest.mark.parametrize(("code", "kind"), [(None, "infrastructure"), (1001, "infrastructure"),
                                         (1012, "infrastructure"), (1011, "hard_stop"), (1008, "hard_stop")])
def test_websocket_failure_classification_is_phase_and_reason_specific(code, kind):
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close
    error = ConnectionClosedError(None if code is None else Close(code, "fixture close"), None)
    assert _classify_failure(error, "infer") == kind
    assert _classify_failure(error, "environment_step") == "hard_stop"


@pytest.mark.parametrize("fault", ["empty", "float_stage", "bool_stage", "wrist_short", "nan_state"])
def test_reset_prefix_rejects_invalid_alignment_without_guessing(fault):
    env = Environment("BinFill", Path("/unused"), max_steps=64, dataset="val", require_current_task_index=True)
    prefix = env.get_init_obs()
    if fault == "empty":
        prefix["images"] = []
    elif fault == "float_stage":
        prefix["current_task_indices"][0] = 0.0
    elif fault == "bool_stage":
        prefix["current_task_indices"][0] = True
    elif fault == "wrist_short":
        prefix["wrist_images"].pop()
    else:
        prefix["states"][0][0] = np.nan
    with pytest.raises(ExpansionEvaluationError):
        validate_reset_prefix(prefix)
