"""Explicit new-family websocket adapter; no scheduler or runnable launch CLI.

This transport may be constructed only by a separately authorized launcher.
It does not widen the old server's reset protocol or start/load a model on import.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import logging
import math
import os
from pathlib import Path
import threading
import time

from openpi_client import msgpack_numpy
from openpi.serving.websocket_policy_server import (
    WebsocketPolicyServer as _TransportServer, validate_direct_options,
)
from websockets.exceptions import ConnectionClosed
import websockets.sync.client

from experiments.uniform_keyframe_expansion.contract import (
    EXPANDED_POLICY_VARIANT, U_POLICY_VARIANT,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET, FORMAL_SEED_SCOPE, SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE, derive_random_seed, derive_smoke_random_seed,
)
from mme_vla_suite.shared.uniform_keyframe_config import (
    EXPANSION_FAMILY, RELEASED_HISTORY_CONFIG, canonical_json,
    expanded_history_mapping, payload_digest,
)
from mme_vla_suite.shared.uniform_keyframe_expansion import (
    EXPANSION_ARMS, MAX_POLICY_CALLS, derive_expansion_seed,
)


LOGGER = logging.getLogger(__name__)
RESET_STATE = {
    "seed": 7, "history_empty": True, "boundary_metadata_empty": True,
    "step_idx": -1, "exec_start_idx": 0, "selector_call_index": 0,
    "selector_unconfigured": True, "selector_rng_empty": True,
    "pending_trace_empty": True, "rng_matches_seed": True,
}


class ExpansionRemoteError(RuntimeError):
    """Protocol/model/serialization hard stop, distinct from transport errors."""

    classification = "hard_stop"

    def __init__(self, message: str, *, error_type: str = "ExpansionWireContractError", evidence=None):
        self.error_type = error_type
        self.evidence = deepcopy(evidence)
        super().__init__(message)


def _same(actual, expected, description: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{description} differs from the expansion contract")


def _identity(identity: Mapping) -> dict:
    if not isinstance(identity, Mapping) or set(identity) != {"execution_id", "dispatch_sha256"}:
        raise ValueError("An explicit execution_id/dispatch_sha256 identity is required")
    result = validate_direct_options(None, identity["execution_id"], identity["dispatch_sha256"])
    if result is None:
        raise ValueError("Execution identity may not be empty")
    return result


def _selector_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("Reset needs the complete new-family selector configuration")
    if config.get("arm") == "U":
        required = {"arm", "split", "task", "episode_id"}
        if set(config) != required:
            raise ValueError("U reset needs only its exact non-random selector context")
        if (config["split"] == "val" and config["episode_id"] != 0
                or config["split"] == "test" and not 0 <= config["episode_id"] < 50):
            raise ValueError("Reset episode is outside the frozen evaluation population")
        # Validate task/split/strict integer ID through the shared derivation,
        # without storing or exposing a random selector for the deterministic U arm.
        derive_expansion_seed(config["split"], config["task"], config["episode_id"], 0)
        return deepcopy(dict(config))
    required = {"arm", "split", "task", "episode_id", "random_seeds", "seed_table_sha256"}
    if set(config) != required or config["arm"] not in EXPANSION_ARMS:
        raise ValueError("Reset needs the complete new-family selector configuration")
    # Derivation validates strict integer IDs, canonical task/split and call range.
    seeds = [derive_expansion_seed(config["split"], config["task"], config["episode_id"], index)
             for index in range(MAX_POLICY_CALLS)]
    if (config["split"] == "val" and config["episode_id"] != 0
            or config["split"] == "test" and not 0 <= config["episode_id"] < 50):
        raise ValueError("Reset episode is outside the frozen evaluation population")
    _same(config["random_seeds"], seeds, "Selector seed table")
    _same(config["seed_table_sha256"], payload_digest(seeds), "Selector seed-table digest")
    return deepcopy(dict(config))


def _u_compatibility_config(config: Mapping) -> dict:
    """Translate a deterministic study-U reset to the released audit schema.

    The compatibility seeds are validated but never consumed by arm U. They are
    deliberately absent from the study selector manifest to avoid implying that
    randomized selection occurs in U.
    """
    config = _selector_config(config)
    if config["arm"] != "U":
        raise ValueError("Only U has a released-selector compatibility mapping")
    if config["split"] == FORMAL_SEED_DATASET:
        scope, derive = FORMAL_SEED_SCOPE, derive_random_seed
    elif config["split"] == SMOKE_SEED_DATASET:
        scope, derive = SMOKE_SEED_SCOPE, derive_smoke_random_seed
    else:  # pragma: no cover - _selector_config already rejects this
        raise ValueError("Unknown U split")
    seeds = [derive(config["task"], config["episode_id"], index)
             for index in range(MAX_POLICY_CALLS)]
    return {
        "arm": "U", "task": config["task"], "episode_id": config["episode_id"],
        "random_seeds": seeds, "seed_table_sha256": payload_digest(seeds),
        "seed_table_scope": scope, "seed_table_dataset": config["split"],
    }


def _positive_pid(value) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("The resident model PID must be a positive integer")


def _duration(value) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Runtime duration must be finite and nonnegative")


def validate_reset_response(reply: Mapping, config: Mapping) -> dict:
    """Return checked evidence; reset state is captured BEFORE configuration."""
    config = _selector_config(config)
    if not isinstance(reply, Mapping) or reply.get("reset_finished") is not True:
        raise ValueError("The server did not acknowledge a completed reset")
    evidence = reply.get("resident_reset")
    if not isinstance(evidence, Mapping):
        raise ValueError("Reset reply has no resident reset evidence")
    _same(evidence.get("state"), RESET_STATE, "Complete reset state")
    _same(evidence.get("experiment_family"), EXPANSION_FAMILY, "Reset family")
    _same(evidence.get("selector_config_sha256"), payload_digest(config), "Reset selector binding")
    _positive_pid(evidence.get("model_process_pid"))
    _duration(reply.get("reset_time_ms"))
    return deepcopy(dict(evidence))


def validate_server_metadata(metadata: Mapping, expected_execution_identity: Mapping,
                             expected_policy_variant: str = EXPANDED_POLICY_VARIANT) -> dict:
    identity = _identity(expected_execution_identity)
    if not isinstance(metadata, Mapping):
        raise ValueError("Server handshake must be a mapping")
    if expected_policy_variant not in (U_POLICY_VARIANT, EXPANDED_POLICY_VARIANT):
        raise ValueError("Unknown expected policy variant")
    _, expanded = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
    source_digest = payload_digest(RELEASED_HISTORY_CONFIG)
    effective_budget = 512 if expected_policy_variant == U_POLICY_VARIANT else 768
    effective_digest = source_digest if expected_policy_variant == U_POLICY_VARIANT else expanded["effective_history_config_sha256"]
    for key, value in {
        "wire_schema": 1, "experiment_family": EXPANSION_FAMILY,
        "policy_variant": expected_policy_variant,
        "effective_memory_budget": effective_budget, "evaluation_policy_seed": 7,
        "resident_policy": True, "strict_weight_tree_load": True,
        "effective_history_config_sha256": effective_digest,
        "source_history_config_sha256": source_digest,
        "direct_execution": identity,
    }.items():
        _same(metadata.get(key), value, f"Server metadata {key}")
    _positive_pid(metadata.get("model_process_pid"))
    return deepcopy(dict(metadata))


def load_study_policy(checkpoint_dir: str | Path, policy_variant: str):
    """Actual opt-in strict loader, called only by an authorized future launcher.

    This doesn't attest archive hashes or authorize GPU initialization. The
    caller must validate its frozen launch/checkpoint evidence before calling.
    No generic policy selector, default checkpoint, or fallback is accepted.
    """
    from mme_vla_suite.policies.policy_config import create_trained_policy
    from mme_vla_suite.training.config import get_config

    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.name != "79999":
        raise ValueError("Expansion requires checkpoint 79999")
    if policy_variant == U_POLICY_VARIANT:
        return create_trained_policy(
            get_config("mme_vla_suite"), checkpoint_dir, seed=7,
            strict_weight_tree_load=True,
        )
    if policy_variant == EXPANDED_POLICY_VARIANT:
        return create_trained_policy(
            get_config("mme_vla_suite"), checkpoint_dir, seed=7,
            experimental_memory_expansion=EXPANSION_FAMILY,
            strict_weight_tree_load=True,
        )
    raise ValueError("Unknown study policy variant")


def load_expansion_policy(checkpoint_dir: str | Path):
    """Backward-compatible explicit loader for the 768-token architecture probe."""
    return load_study_policy(checkpoint_dir, EXPANDED_POLICY_VARIANT)


class ExpansionPolicyServer(_TransportServer):
    """One initialized episode per connection, one active connection at a time."""

    def __init__(self, policy, *, execution_identity: Mapping,
                 policy_variant: str = EXPANDED_POLICY_VARIANT,
                 host="127.0.0.1", port=None, listen_fd=None):
        identity = _identity(execution_identity)
        if policy_variant not in (U_POLICY_VARIANT, EXPANDED_POLICY_VARIANT):
            raise ValueError("Unknown resident policy variant")
        from omegaconf import OmegaConf
        _, expanded = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
        source_digest = payload_digest(RELEASED_HISTORY_CONFIG)
        # This metadata is an audited loader declaration, not independent proof
        # of the checkpoint contents. Real checkpoint smoke remains mandatory.
        if policy_variant == EXPANDED_POLICY_VARIANT:
            evidence = policy.metadata.get("uniform_keyframe_expansion")
            _same(evidence, expanded, "Policy expansion configuration evidence")
            effective_budget = 768
            effective_digest = expanded["effective_history_config_sha256"]
        else:
            _same(OmegaConf.to_container(policy.config, resolve=True), RELEASED_HISTORY_CONFIG,
                  "Policy released U history configuration")
            effective_budget = 512
            effective_digest = source_digest
        _same(policy._seed, 7, "Policy RNG seed")
        metadata = {
            "wire_schema": 1, "experiment_family": EXPANSION_FAMILY,
            "policy_variant": policy_variant,
            "effective_memory_budget": effective_budget, "evaluation_policy_seed": 7,
            "resident_policy": True, "strict_weight_tree_load": True,
            "source_history_config_sha256": source_digest,
            "effective_history_config_sha256": effective_digest,
            "model_process_pid": os.getpid(),
        }
        super().__init__(policy, host=host, port=port, metadata=metadata,
                         listen_fd=listen_fd, **identity)
        self._active_trajectory = None
        self._hard_failure = None
        self._policy_variant = policy_variant

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        initialized = False
        trajectory_config = None
        operation = None
        try:
            await websocket.send(packer.pack(self._metadata))
            async for raw in websocket:
                if isinstance(raw, str):
                    raise ValueError("Expansion transport requires binary msgpack messages")
                request = msgpack_numpy.unpackb(raw)
                if not isinstance(request, dict):
                    raise ValueError("Expansion request must be a mapping")
                operation = request.get("operation")
                if self._active_trajectory not in (None, websocket):
                    raise ValueError("Another trajectory owns the resident policy")
                if self._hard_failure is not None:
                    raise ValueError("Resident model is latched in an experiment hard stop; review is required")
                if not initialized and operation != "reset":
                    raise ValueError("First trajectory request must reset and configure the expansion policy")
                if initialized and operation == "reset":
                    raise ValueError("A trajectory cannot reset twice on one connection")

                if operation == "reset":
                    if set(request) != {"operation", "config"}:
                        raise ValueError("Reset request fields differ from the new wire schema")
                    config = _selector_config(request["config"])
                    self._active_trajectory = websocket
                    started = time.monotonic()
                    self._policy.reset()
                    state = self._policy.reset_evidence()
                    _same(state, RESET_STATE, "Policy reset state")
                    if self._policy_variant == U_POLICY_VARIANT:
                        self._policy.configure_keyframe_selector(_u_compatibility_config(config))
                    else:
                        self._policy.configure_uniform_keyframe_expansion(config)
                    trajectory_config = config
                    initialized = True
                    reply = {
                        "operation": operation, "reset_finished": True,
                        "reset_time_ms": (time.monotonic() - started) * 1000,
                        "resident_reset": {"state": deepcopy(state), "model_process_pid": os.getpid(),
                                           "experiment_family": EXPANSION_FAMILY,
                                           "selector_config_sha256": payload_digest(config)},
                    }
                elif operation == "add_buffer":
                    if set(request) != {"operation", "payload"} or not isinstance(request["payload"], dict):
                        raise ValueError("Invalid add_buffer request")
                    payload = request["payload"]
                    if set(payload) != {"add_buffer", "images", "state", "exec_start_idx", "current_task_index"}:
                        raise ValueError("History payload must contain only the causal aligned history fields")
                    if payload["add_buffer"] is not True:
                        raise ValueError("History append needs add_buffer=true")
                    started = time.monotonic()
                    self._policy.add_buffer(payload)
                    reply = {"operation": operation, "add_buffer_finished": True,
                             "add_buffer_time_ms": (time.monotonic() - started) * 1000}
                elif operation == "infer":
                    if set(request) != {"operation", "observation"} or not isinstance(request["observation"], dict):
                        raise ValueError("Invalid inference request")
                    observation = request["observation"]
                    expected_fields = {"observation/image", "observation/wrist_image", "observation/state",
                                       "prompt", "keyframe_environment_step"}
                    if set(observation) != expected_fields:
                        raise ValueError("Inference cannot contain boundary labels, symbolic subgoals or extra inputs")
                    step = observation["keyframe_environment_step"]
                    if type(step) is not int or not 0 <= step <= 1300:
                        raise ValueError("Invalid inference environment step")
                    reply = dict(self._policy.infer(observation))
                    if self._policy_variant == U_POLICY_VARIANT:
                        trace = dict(reply.get("selector_trace") or {})
                        trace.update({
                            "experiment_family": EXPANSION_FAMILY,
                            "arm": "U", "split": trajectory_config["split"],
                            "step_idx": trace.get("current_history_index"),
                            "effective_memory_budget": 512,
                            "base_uniform_token_budget": 512,
                        })
                        reply["selector_trace"] = trace
                    reply["operation"] = operation
                else:
                    raise ValueError("Unknown expansion request operation")
                await websocket.send(packer.pack(reply))
        except ConnectionClosed:
            # No automatic retry or reset. A new reviewed attempt must create a
            # new connection and explicitly reset every mutable episode state.
            pass
        except Exception as error:
            evidence = deepcopy(getattr(error, "evidence", None))
            error_payload = {"classification": "hard_stop", "error_type": type(error).__name__,
                             "message": str(error), "evidence": evidence}
            # Reject a stray second client without poisoning the active owner.
            # Once this client owns model state, a contract/model failure latches
            # the resident session and cannot be cleared by simply reconnecting.
            if self._active_trajectory is websocket:
                self._hard_failure = deepcopy(error_payload)
            LOGGER.exception("Expansion serving contract/model failure")
            try:
                await websocket.send(packer.pack({"operation": operation, "error": error_payload}))
                await websocket.close(code=1011, reason="Expansion experiment hard stop")
            except ConnectionClosed:
                pass
        finally:
            if self._active_trajectory is websocket:
                self._active_trajectory = None


class ExpansionClient:
    """Synchronous bounded client; never reconnects or retries implicitly."""

    def __init__(self, host: str, port: int, *, expected_execution_identity: Mapping,
                 expected_policy_variant: str = EXPANDED_POLICY_VARIANT,
                 connect_timeout: float = 10, response_timeout: float = 600):
        identity = _identity(expected_execution_identity)
        for value in (connect_timeout, response_timeout):
            _duration(value)
            if value == 0:
                raise ValueError("Transport timeouts must be positive")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Port must be a valid integer TCP port")
        self._ws = None
        self._initialized = False
        self._response_timeout = response_timeout
        self._lock = threading.Lock()
        self._packer = msgpack_numpy.Packer()
        try:
            self._ws = websockets.sync.client.connect(
                f"ws://{host}:{port}", compression=None, max_size=None,
                open_timeout=connect_timeout, close_timeout=min(connect_timeout, 10), ping_timeout=600,
            )
            metadata = self._unpack(self._ws.recv(timeout=connect_timeout))
            try:
                self._metadata = validate_server_metadata(metadata, identity, expected_policy_variant)
            except (TypeError, ValueError) as exc:
                raise ExpansionRemoteError(f"Server handshake rejected: {exc}") from exc
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _unpack(raw) -> dict:
        if isinstance(raw, str):
            raise ExpansionRemoteError(f"Unexpected textual server response: {raw}")
        try:
            reply = msgpack_numpy.unpackb(raw)
        except Exception as exc:
            raise ExpansionRemoteError(f"Invalid msgpack server response: {exc}") from exc
        if not isinstance(reply, dict):
            raise ExpansionRemoteError("Server response is not a mapping")
        if "error" in reply:
            error = reply["error"]
            if not isinstance(error, dict) or error.get("classification") != "hard_stop":
                raise ExpansionRemoteError("Malformed or unsupported server error classification")
            raise ExpansionRemoteError(str(error.get("message", "Unknown server failure")),
                                       error_type=str(error.get("error_type", "Unknown")),
                                       evidence=error.get("evidence"))
        return reply

    def _rpc(self, operation: str, field: str, payload) -> dict:
        with self._lock:
            if self._ws is None:
                raise ExpansionRemoteError("Expansion client is closed")
            try:
                self._ws.send(self._packer.pack({"operation": operation, field: payload}))
                reply = self._unpack(self._ws.recv(timeout=self._response_timeout))
                if reply.get("operation") != operation:
                    raise ExpansionRemoteError("Server replied to a different request operation")
                return reply
            except BaseException:
                self.close()
                raise

    def reset(self, config: Mapping) -> dict:
        if self._initialized:
            raise ExpansionRemoteError("A trajectory cannot reset twice on one connection")
        config = _selector_config(config)
        reply = self._rpc("reset", "config", config)
        try:
            evidence = validate_reset_response(reply, config)
            _same(evidence["model_process_pid"], self._metadata["model_process_pid"], "Reset/handshake PID")
        except (TypeError, ValueError) as exc:
            self.close()
            raise ExpansionRemoteError(f"Reset evidence rejected: {exc}") from exc
        self._initialized = True
        return reply

    def add_buffer(self, payload: Mapping) -> dict:
        if not self._initialized:
            raise ExpansionRemoteError("Reset/configure must complete before history append")
        reply = self._rpc("add_buffer", "payload", dict(payload))
        if reply.get("add_buffer_finished") is not True:
            self.close()
            raise ExpansionRemoteError("Server did not acknowledge history append")
        return reply

    def infer(self, observation: Mapping) -> dict:
        if not self._initialized:
            raise ExpansionRemoteError("Reset/configure must complete before inference")
        return self._rpc("infer", "observation", dict(observation))

    def get_server_metadata(self) -> dict:
        return deepcopy(self._metadata)

    def close(self) -> None:
        connection, self._ws = self._ws, None
        if connection is not None:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
