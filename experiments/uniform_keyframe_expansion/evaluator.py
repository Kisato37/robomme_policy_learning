"""Isolated expansion episode adapter; no scheduler, retries or launch authority.

An authorized launcher supplies a validated store/row, the original RoboMME
EnvRunner and utility classes, and a client bound to its resident policy session.
Importing this module neither imports the simulator nor loads model weights.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import errno
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

from experiments.uniform_keyframe_expansion.contract import build_selector_config, validate_row
from experiments.uniform_keyframe_expansion.trace_validation import validate_selector_trace


@dataclass(frozen=True)
class BenchmarkComponents:
    """Pass the unchanged examples/robomme utility implementations, not substitutes."""

    episode_state: Callable
    pack_buffer: Callable
    recorder: Callable
    tasks_with_video_demo: tuple[str, ...]


class ExpansionEvaluationError(RuntimeError):
    """A deterministic adapter/metadata failure; not a benchmark outcome."""


def _benchmark_stop_flag(value: Any) -> bool:
    """Read one boolean without broad truthiness or multi-environment reduction.

    The original DemonstrationWrapper emits torch.bool batch entries; EnvRunner
    passes one of them through unchanged. Keep that interface and the original
    terminal decision, normalizing only its scalar representation in this
    experiment adapter. The simulator has already imported Torch; do not import
    a production backend here or change its configured initialization order.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.dtype == np.dtype(bool) and value.shape in ((), (1,)):
            return bool(value.item())
    else:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(value, torch.Tensor):
            if (value.dtype == torch.bool and value.layout == torch.strided
                    and tuple(value.shape) in ((), (1,))):
                try:
                    return bool(value.item())
                except (RuntimeError, ValueError, NotImplementedError) as exc:
                    raise ExpansionEvaluationError("Benchmark stop flag has no readable boolean value") from exc
    raise ExpansionEvaluationError(
        "Benchmark stop flag must be a scalar or single-environment boolean; "
        f"received {type(value).__module__}.{type(value).__name__} "
        f"with dtype={getattr(value, 'dtype', None)} and shape={getattr(value, 'shape', None)}")


def _finite_array(value, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape[-len(shape_tail):] != shape_tail or array.dtype.kind not in "buif"
            or not np.isfinite(array).all()):
        raise ExpansionEvaluationError(f"Invalid {name}: shape/dtype/nonfinite data")
    return array


def validate_reset_prefix(prefix: dict) -> tuple[list[int], int]:
    """Check all generated reset frames, not just the final current observation."""
    if not isinstance(prefix, dict) or not isinstance(prefix.get("task_goal"), str) or not prefix["task_goal"]:
        raise ExpansionEvaluationError("Reset prefix requires original task text")
    try:
        count = len(prefix["images"])
        if count < 1 or any(len(prefix[key]) != count for key in ("wrist_images", "states", "current_task_indices")):
            raise ValueError("length mismatch")
        stages = []
        for stage in prefix["current_task_indices"]:
            if isinstance(stage, (bool, np.bool_)) or not isinstance(stage, (int, np.integer)):
                raise ValueError("stage must be an integer")
            stages.append(int(stage))
        for front, wrist, state in zip(prefix["images"], prefix["wrist_images"], prefix["states"], strict=True):
            for image in (front, wrist):
                array = _finite_array(image, (3,), "RGB reset image")
                if array.ndim != 3 or array.dtype != np.uint8:
                    raise ValueError("RGB frame must be H x W x 3 uint8")
            if _finite_array(state, (8,), "reset robot state").ndim != 1:
                raise ValueError("state must have eight entries")
    except (KeyError, TypeError, ValueError) as exc:
        raise ExpansionEvaluationError("Reset demonstration frames/states/stages are not aligned") from exc
    return stages, count


def encode_task_state(value: Any) -> Any:
    """Lossless JSON tree for initial-state evidence, without unsafe pickle."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ExpansionEvaluationError("Object arrays cannot be archived as initial state")
        array = np.ascontiguousarray(value)
        return {"type": "ndarray", "dtype": array.dtype.str, "shape": list(array.shape),
                "bytes_base64": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return encode_task_state(value.item())
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode_task_state(key), encode_task_state(child)]
                                           for key, child in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [encode_task_state(v) for v in value]}
    if value is None or type(value) in (str, int, bool, float):
        if isinstance(value, float) and not np.isfinite(value):
            raise ExpansionEvaluationError("Nonfinite scalar in initial task state")
        return {"type": "scalar", "value": value}
    raise ExpansionEvaluationError(f"Unserializable initial task-state value: {type(value).__name__}")


def _record_initial(writer, env, prefix, reset_reply, server_metadata, config, expected_identity) -> None:
    from experiments.uniform_keyframe_expansion.serving import validate_reset_response, validate_server_metadata

    server_metadata = validate_server_metadata(server_metadata, expected_identity)
    reset = validate_reset_response(reset_reply, config)
    if reset["model_process_pid"] != server_metadata["model_process_pid"]:
        raise ExpansionEvaluationError("Handshake and reset refer to different resident model processes")
    stages, count = validate_reset_prefix(prefix)
    # Preserve the underlying evidence as well as hashes. A digest alone was
    # insufficient for re-auditing some of the historical baseline exports.
    hashes = env.initial_condition_hashes
    if not isinstance(hashes, dict):
        raise ExpansionEvaluationError("Missing actual initial-condition hashes")
    actual = {
        "front_observations_sha256": env._digest_value(prefix["images"]),
        "wrist_observations_sha256": env._digest_value(prefix["wrist_images"]),
        "robot_states_sha256": env._digest_value(prefix["states"]),
        "task_instruction_sha256": env._digest_value(prefix["task_goal"]),
    }
    task_state = env._canonicalize_task_state_for_hashing(env.env.unwrapped.get_state_dict())
    actual["task_state_sha256"] = env._digest_value(task_state)
    if hashes != actual:
        raise ExpansionEvaluationError("Initial inputs/state changed after the single environment reset")
    archive = io.BytesIO()
    np.savez_compressed(archive, front=np.stack(prefix["images"]), wrist=np.stack(prefix["wrist_images"]),
                        robot_state=np.stack(prefix["states"]), current_task_index=np.asarray(stages, dtype=np.int64))
    writer.write_attachment("initial_observations.npz", archive.getvalue())
    writer.write_attachment("initial_task_state.json", json.dumps(
        encode_task_state(task_state), ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8"))
    writer.write_attachment("initial_task_instruction.json", json.dumps(prefix["task_goal"], ensure_ascii=True).encode("utf-8"))
    writer.record_initial_conditions(
        hashes,
        environment_provenance={
            "resolved_environment_seed": env.resolved_environment_seed,
            "difficulty": env.difficulty, "dataset": env.dataset,
            "resolved_difficulty_hint": env.resolved_difficulty_hint,
            "renderer_device": getattr(env, "renderer_device", None),
        },
        reset_evidence={
            "policy_seed": 7, "memory_cleared": True, "policy_rng_reset": True,
            "reset_prefix_frame_count": count, "reset_prefix_stage_count": len(stages),
            "reset_prefix_frames_sha256": actual["front_observations_sha256"],
            "reset_prefix_stages_sha256": env._digest_value(stages),
            "resident_reset": reset, "server_metadata": server_metadata,
        },
    )


def _classify_failure(error: Exception, phase: str) -> str:
    # An opaque server error, simulator exception or selector bug is never
    # automatically reclassified as a transient transport failure.
    if _transport_failure(error, phase):
        return "infrastructure"
    if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.ESTALE}:
        return "infrastructure"
    return "hard_stop"


def _transport_failure(error: Exception, phase: str) -> bool:
    from websockets.exceptions import ConnectionClosed

    if phase not in {"client_connect", "policy_reset", "add_buffer", "infer"}:
        return False
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if isinstance(error, ConnectionClosed):
        # A lost connection or server restart is transport evidence. An explicit
        # protocol/policy/internal-error close (e.g. 1008/1011) is not proof of a
        # transient failure and must remain a hard stop if its detail was lost.
        code = error.rcvd.code if error.rcvd is not None else 1006
        return code in {1000, 1001, 1006, 1012, 1013}
    return False


def evaluate_attempt(store, row: dict, attempt_id: int, *, env_factory: Callable,
                     client_factory: Callable, components: BenchmarkComponents,
                     episode_provenance: dict | None = None) -> dict:
    """Execute exactly one pre-authorized row, from a fresh reset, with no retry.

    The store is not a launch gate: its existence cannot authorize this call.
    An external reviewed launcher must enforce approvals, source/environment
    attestations and GPU architecture gates before constructing real factories.
    CPU fixtures may supply deterministic fake environment/model factories.
    """
    row = validate_row(row)
    episode_provenance = episode_provenance or {}
    expected_identity = episode_provenance.get("policy_execution_identity")
    if not isinstance(expected_identity, dict):
        raise ExpansionEvaluationError("Episode adapter needs its prevalidated resident execution identity")
    writer = store.new_attempt(row, attempt_id, episode_provenance)
    env = client = None
    finalized = False
    failure = None
    terminal_evidence = None
    phase = "environment_setup"
    try:
        env = env_factory(row["task"], writer.attempt_dir, max_steps=row["max_steps"],
                          dataset=row["dataset"], require_current_task_index=True)
        env.make_env(row["episode_id"])
        if (env.env_id != row["task"] or env.episode_id != row["episode_id"]
                or env.dataset != row["dataset"] or env.require_current_task_index is not True):
            raise ExpansionEvaluationError("Live environment differs from submitted row")
        phase = "environment_reset"
        prefix = env.get_init_obs()  # Exactly once; includes generated demonstration history.
        stages, prefix_count = validate_reset_prefix(prefix)
        if prefix["task_goal"] != env.task_goal:
            raise ExpansionEvaluationError("Environment task instruction differs from reset prefix")
        config = build_selector_config(row)
        phase = "client_connect"
        client = client_factory()
        metadata = client.get_server_metadata()
        from experiments.uniform_keyframe_expansion.serving import validate_server_metadata
        validate_server_metadata(metadata, expected_identity)
        phase = "policy_reset"
        reset_reply = client.reset(config)
        phase = "initial_evidence"
        _record_initial(writer, env, prefix, reset_reply, metadata, config, expected_identity)

        state = components.episode_state()
        video_staging = Path(writer.attempt_dir) / ".video_staging"
        recorder = components.recorder(video_staging, prefix["task_goal"], fps=30)
        for i, (front, wrist, robot, stage) in enumerate(zip(
                prefix["images"], prefix["wrist_images"], prefix["states"], stages, strict=True)):
            state.add_observation(front, wrist, robot, current_task_index=stage)
            recorder.record(image=front.copy(), wrist_image=wrist.copy(), state=robot.copy(),
                            is_video_demo=row["task"] in components.tasks_with_video_demo and i < prefix_count - 1,
                            subgoal=None)
        state.exec_start_idx = prefix_count - 1
        front, wrist, robot = state.get_current_obs()
        call_count = 0
        terminal = None
        history_lengths = []
        latencies = []
        actual_stages = list(stages)
        while terminal is None:
            if not state.action_plan:
                phase = "prepare_history"
                segment_length = len(state.image_buffer)
                payload = components.pack_buffer(state.image_buffer, state.state_buffer, state.exec_start_idx,
                                                  current_task_indices=state.current_task_index_buffer)
                phase = "add_buffer"
                ack = client.add_buffer(payload)
                if ack.get("add_buffer_finished") is not True:
                    raise ExpansionEvaluationError("Policy did not acknowledge the history segment")
                state.total_history_frames_sent += segment_length
                request = {"observation/image": front, "observation/wrist_image": wrist,
                           "observation/state": robot, "prompt": prefix["task_goal"],
                           "keyframe_environment_step": state.count}
                phase = "infer"
                started = time.monotonic()
                response = client.infer(request)
                request_ms = (time.monotonic() - started) * 1000
                phase = "validate_inference"
                trace = dict(response.get("selector_trace") or {})
                trace["end_to_end_request_latency_ms"] = request_ms
                validate_selector_trace(trace)
                if (trace["policy_call_index"] != call_count or trace["environment_step"] != state.count
                        or trace["history_length"] != state.total_history_frames_sent):
                    raise ExpansionEvaluationError("Policy trace differs from actual episode history/call")
                actual_boundaries = [i for i, stage in enumerate(actual_stages)
                                     if i == 0 or stage != actual_stages[i - 1]]
                if (len(actual_stages) != state.total_history_frames_sent
                        or trace["visible_boundary_indices"] != actual_boundaries):
                    raise ExpansionEvaluationError("Policy boundaries differ from actually observed online stages")
                for field, expected in (("arm", row["arm"]), ("task", row["task"]),
                                        ("episode_id", row["episode_id"]), ("split", row["dataset"])):
                    if trace.get(field) != expected:
                        raise ExpansionEvaluationError(f"Policy trace {field} differs from row")
                actions = _finite_array(response.get("actions"), (20, 8), "action chunk")
                if actions.shape != (20, 8):
                    raise ExpansionEvaluationError("Action proposal must have exact shape (20,8)")
                writer.append_trace(trace)
                call_count += 1
                history_lengths.append(state.total_history_frames_sent)
                latencies.append(request_ms)
                state.action_plan.extend(actions[:16])
                state.clear_buffers()  # New observations form a non-overlapping segment.

            phase = "environment_step"
            action = state.action_plan.popleft()
            observation, stopped, status = env.step(action)
            state.count += 1
            stopped = _benchmark_stop_flag(stopped)
            if stopped and status not in {"success", "fail", "timeout", "error"}:
                raise ExpansionEvaluationError("Unknown official benchmark terminal")
            if stopped or state.count >= row["max_steps"]:
                terminal = status if stopped else ("short_limit" if row["trajectory_kind"] == "short" else "timeout")
                # The scientific outcome is already known, even if recording
                # this final observation/video or publishing artifacts fails.
                # Never grant a retry based on a later filesystem exception.
                terminal_evidence = {
                    "terminal_reason": terminal, "success": terminal == "success",
                    "official_terminal": terminal != "short_limit",
                    "official_stop_flag": bool(stopped),
                    "environment_steps": state.count, "policy_call_count": call_count,
                }
            # A real benchmark error intentionally carries no image. Never turn
            # it into infrastructure or attempt to unpack a missing observation.
            if stopped and status == "error":
                terminal = status
                break
            if state.count >= row["max_steps"]:
                terminal = status if stopped else ("short_limit" if row["trajectory_kind"] == "short" else "timeout")
                break  # Matches old controller horizon cutoff; never step past it.
            front, wrist, robot = observation
            stage = env.current_task_index
            if isinstance(stage, (bool, np.bool_)) or not isinstance(stage, (int, np.integer)):
                raise ExpansionEvaluationError("Missing/malformed online current_task_index")
            state.add_observation(front, wrist, robot, current_task_index=int(stage))
            actual_stages.append(int(stage))
            recorder.record(image=front.copy(), wrist_image=wrist.copy(), state=robot.copy(),
                            action=action.copy(), subgoal=None)
            if stopped:
                terminal = status

        phase = "record_video"
        # Use a safe fixed name in this attempt, not task text as a path.
        video_name = "rollout.mp4"
        recorder.save_video(video_name)
        video_path = video_staging / video_name
        # Recorder's write path is private to this newly reserved attempt. The
        # artifact layer must bind the completed video bytes before finalizing.
        if not video_path.is_file():
            raise ExpansionEvaluationError("Recorder did not produce the episode video")
        video_digest = writer.publish_video(video_path)
        info = getattr(env, "info", {}) or {}
        result = {
            "terminal_reason": terminal, "success": terminal == "success",
            "official_terminal": terminal != "short_limit",
            "terminal_metadata": {"reason": terminal, "status": str(info.get("status", terminal)),
                                  "official_stop_flag": bool(stopped),
                                  "error_message": str(info["error_message"]) if info.get("error_message") else None,
                                  "exception_type": str(info["exception_type"]) if info.get("exception_type") else None},
            "environment_steps": state.count, "policy_call_count": call_count,
            "reset_verified": True, "collision": bool(info.get("collision", False)),
            "timeout": terminal == "timeout", "terminal_state": str(info.get("status", terminal)),
            "benchmark_error_message": str(info["error_message"]) if info.get("error_message") else None,
            "benchmark_exception_type": str(info["exception_type"]) if info.get("exception_type") else None,
            "history_lengths_at_policy_calls": history_lengths,
            "policy_latency_ms": latencies, "video_sha256": video_digest,
            "video_filename": video_name,
        }
        phase = "finalize"
        writer.finalize_scientific(result)
        finalized = True
        return result
    except Exception as exc:
        failure = exc
        if not finalized:
            if (Path(writer.attempt_dir) / "episode_result.json").exists():
                # A result may have been published before its ledger fsync
                # failed. Preserve that partial commit; never append a failure
                # that could authorize rerunning a known scientific outcome.
                exc.add_note("Scientific result was already published; closure audit/review required, no retry.")
            else:
                kind = "hard_stop" if terminal_evidence is not None else _classify_failure(exc, phase)
                code = ("transport" if _transport_failure(exc, phase) else "filesystem") if kind == "infrastructure" else type(exc).__name__
                try:
                    writer.record_failure(kind, code, str(exc) or repr(exc),
                                          {"phase": phase, "exception_type": type(exc).__name__,
                                           "automatic_retry": False, "protocol_evidence": getattr(exc, "evidence", None),
                                           "known_terminal_outcome": terminal_evidence})
                except Exception as recording_error:
                    exc.add_note(f"Failure evidence could not be finalized; review required: {recording_error}")
        raise
    finally:
        cleanup_errors = []
        for target, method in ((client, "close"), (env, "close_env")):
            if target is not None:
                try:
                    getattr(target, method)()
                except Exception as exc:
                    cleanup_errors.append(f"{method}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            message = "Cleanup failed; do not dispatch another episode: " + "; ".join(cleanup_errors)
            if failure is not None:
                failure.add_note(message)
            else:
                # A previously finalized scientific result remains accepted;
                # this exception must never be used to rerun that outcome.
                raise ExpansionEvaluationError(message)
