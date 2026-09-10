"""Numerical probes of an already strictly loaded, authorized GPU policy.

This module does not load weights, launch a job, execute a benchmark task, or
publish a PASS gate. CPU tests exercise helpers, never the public live entry.
The fixtures are exactly the parent's synthetic development-history generator;
``InsertPeg/val/0`` is selector seed context, not a dataset read or task outcome.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from experiments.uniform_keyframe_expansion.trace_validation import validate_selector_trace
from mme_vla_suite.shared.uniform_keyframe_config import (
    EXPANSION_FAMILY, RELEASED_HISTORY_CONFIG, expanded_history_mapping, payload_digest,
)
from mme_vla_suite.shared.uniform_keyframe_expansion import derive_expansion_seed

HISTORY_LENGTHS = (16, 64)
ARMS = ("UK48", "UN48")
CASES = tuple((arm, length) for arm in ARMS for length in HISTORY_LENGTHS)
MEMORY_FIELDS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")
MEMORY_KEYS = ("image", "position", "state", "mask")
RAW_SHAPES = ((768, 2048), (768, 768), (768, 8), (768,))


class ArchitectureProbeError(RuntimeError):
    """No numerical architecture gate can be published for this execution."""


def _require(condition, message):
    if not condition:
        raise ArchitectureProbeError(message)


def _digest(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode("ascii"))
    result.update(json.dumps(array.shape).encode("ascii"))
    result.update(array.tobytes())
    return result.hexdigest()


def _file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _write_npz(path, arrays):
    """Write-once atomic array evidence, never an object/pickle archive."""
    path = Path(path)
    encoded = _audit_arrays(arrays)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # fail rather than replace existing evidence
    finally:
        os.unlink(temporary)
    return _file_record(path)


def _audit_arrays(arrays):
    """BF16 -> F32 preserves every numerical value and avoids NPZ void dtype."""
    result = {}
    for key, value in arrays.items():
        value = np.asarray(value)
        if str(value.dtype) == "bfloat16":
            value = value.astype(np.float32)
        _require(value.dtype.kind in "biuf" and np.isfinite(value).all(),
                 f"Nonfinite or unsupported non-numerical artifact array: {key}")
        result[key] = value
    return result


def _encoding_metadata(arrays):
    encoded = _audit_arrays(arrays)
    return {key: {"original_dtype": str(np.asarray(value).dtype),
                  "original_sha256": _digest(value), "stored_dtype": str(encoded[key].dtype),
                  "stored_sha256": _digest(encoded[key])}
            for key, value in arrays.items()}


def _reference(record):
    return {key: record[key] for key in ("path", "sha256")}


def _selector_config(arm):
    seeds = [derive_expansion_seed("val", "InsertPeg", 0, index) for index in range(82)]
    return {"arm": arm, "split": "val", "task": "InsertPeg", "episode_id": 0,
            "random_seeds": seeds, "seed_table_sha256": payload_digest(seeds)}


def _synthetic_history(length):
    _require(length in HISTORY_LENGTHS, "Only the frozen 16/64 development fixtures are allowed")
    rng = np.random.Generator(np.random.PCG64(2026082501 + length))
    images = rng.integers(0, 256, (length, 1, 256, 256, 3), dtype=np.uint8)
    phase = np.linspace(-0.25, 0.25, length, dtype=np.float32)[:, None]
    offsets = np.linspace(-0.04, 0.04, 8, dtype=np.float32)[None, :]
    states = (phase + offsets).astype(np.float32)
    stages = np.arange(length, dtype=np.int64) // 7
    return images, states, stages


def _peak_memory(device):
    stats = device.memory_stats()
    _require(isinstance(stats, dict), "GPU allocator memory statistics are unavailable")
    peak = stats.get("peak_bytes_in_use")
    _require(type(peak) is int and peak > 0, "A positive measured GPU allocator peak is required")
    return {"peak_gpu_bytes": peak, "meter": "jax_device.memory_stats.peak_bytes_in_use",
            "scope": "cumulative_allocator_peak_in_this_live_process",
            "bytes_in_use": stats.get("bytes_in_use")}


def _cache_sizes(policy):
    result = {}
    for name in ("_vision_encode", "_perceptual_memory_encode", "_sample_actions"):
        function = getattr(policy, name)
        try:
            jitted = inspect.getclosurevars(function).nonlocals.get("jitted_fn")
        except TypeError:
            jitted = None
        _require(jitted is not None and hasattr(jitted, "_cache_size"),
                 f"Cannot audit the real module_jit cache for {name}")
        result[name] = int(jitted._cache_size())
    return result


@contextmanager
def _capture(policy, jax):
    """Observe real intermediates, synchronizing work; restore on every exit."""
    originals = {name: getattr(policy, name) for name in
                 ("_prepare_experiment_frame_sampling", "_prepare_history", "_sample_actions", "_perceptual_memory_encode")}
    previous_attributes = {name: policy.__dict__.get(name) for name in originals}
    owned = {name: name in policy.__dict__ for name in originals}
    capture = {}

    def prepared(*args, **kwargs):
        result = originals["_prepare_experiment_frame_sampling"](*args, **kwargs)
        capture["raw"] = tuple(np.asarray(value).copy() for value in result)
        return result

    def sample(rng, observation, **kwargs):
        capture["sample_rng"] = rng
        capture["observation"] = observation
        jax.block_until_ready((rng, observation))
        started = time.perf_counter()
        result = originals["_sample_actions"](rng, observation, **kwargs)
        jax.block_until_ready(result)
        capture["synchronized_sample_latency_ms"] = (time.perf_counter() - started) * 1000
        capture["raw_model_actions"] = np.asarray(result).copy()
        return result

    def history(inputs):
        result = originals["_prepare_history"](inputs)
        capture["pre_model"] = {key: np.asarray(result[field]).copy()
                                for key, field in zip(MEMORY_KEYS, MEMORY_FIELDS, strict=True)}
        return result

    def encoded(*args, **kwargs):
        result = originals["_perceptual_memory_encode"](*args, **kwargs)
        jax.block_until_ready(result)
        capture["final_memory"] = np.asarray(result).copy()
        return result

    try:
        policy._prepare_experiment_frame_sampling = prepared
        policy._prepare_history = history
        policy._sample_actions = sample
        policy._perceptual_memory_encode = encoded
        yield capture
    finally:
        for name in originals:
            if owned[name]:
                setattr(policy, name, previous_attributes[name])
            else:
                delattr(policy, name)


def _validate_arrays(arrays, trace, *, jax):
    """Validate real captured values, not just trace declarations."""
    valid = trace["valid_memory_token_count"]
    raw = [arrays[f"raw_{key}"] for key in MEMORY_KEYS]
    _require(tuple(value.shape for value in raw) == RAW_SHAPES, "Incorrect gathered memory shapes")
    _require(raw[-1].dtype == np.bool_, "Raw memory mask must be boolean")
    _require(raw[-1][:valid].all() and not raw[-1][valid:].any(), "Invalid raw valid-prefix mask")
    for value in raw[:-1]:
        _require(np.isfinite(value).all(), "Nonfinite raw gathered features")
        _require(not np.any(value[valid:] != 0), "Gather padding must be exactly zero before normalization")
    for key, shape in zip(MEMORY_KEYS, RAW_SHAPES, strict=True):
        array = arrays[key]
        _require(array.shape == (1, *shape), f"Incorrect sampled observation {key} shape")
        _require(np.isfinite(array).all(), f"Nonfinite sampled observation {key}")
        previous = arrays[f"pre_model_{key}"]
        _require(previous.shape == shape and np.isfinite(previous).all(), f"Invalid pre-JAX {key}")
        # Original jnp.asarray may legitimately downcast NumPy float64 padding
        # when JAX x64 is disabled. Verify the actual conversion, not equality of
        # hashes across these different stages; never change the x64 setting.
        converted = np.asarray(jax.numpy.asarray(previous))[None, ...]
        _require(_digest(array) == _digest(converted), f"Input transform changed static {key} unexpectedly")
    _require(np.array_equal(arrays["mask"][0], raw[-1]), "Transforms changed memory mask")
    _require(arrays["mask"].dtype == np.bool_, "Sampled mask must remain boolean")
    for key, field in (("image", "image_tensor_sha256"), ("position", "position_tensor_sha256"),
                       ("state", "state_tensor_sha256"), ("mask", "mask_sha256")):
        _require(_digest(arrays[f"pre_model_{key}"]) == trace[field],
                 f"Captured pre-JAX {key} differs from runtime trace")
    _require(arrays["final_memory"].shape == (1, 768, 1024), "Incorrect final memory tensor shape")
    _require(np.isfinite(arrays["final_memory"]).all(), "Nonfinite encoded memory")
    _require(_digest(arrays["final_memory"]) == trace["final_memory_tensor_sha256"],
             "Captured final memory differs from runtime trace")
    _require(arrays["actions"].shape == (20, 8) and np.isfinite(arrays["actions"]).all(),
             "Expected finite 20-by-8 executable actions")
    _require(np.isfinite(arrays["raw_model_actions"]).all(), "Nonfinite model actions")


def _run_once(policy, *, arm, length, jax, device):
    from experiments.uniform_keyframe_expansion.serving import RESET_STATE

    policy.reset()
    reset = policy.reset_evidence()
    _require(reset == RESET_STATE, "State/seed was not completely cleared before the case")
    policy.configure_uniform_keyframe_expansion(_selector_config(arm))
    images, states, stages = _synthetic_history(length)
    cache_before = _cache_sizes(policy)
    # Include history encoding in a separate preparation measurement, not hide it
    # in selector-only latency. A fresh reset retains existing compiled kernels.
    jax.block_until_ready(policy._rng)
    started = time.perf_counter()
    for start in range(0, length, 16):
        stop = min(length, start + 16)
        policy.add_buffer({"images": images[start:stop], "state": states[start:stop],
                           "current_task_index": stages[start:stop], "exec_start_idx": 0})
    jax.block_until_ready(policy.mem_buffer._history_feats)
    preparation_ms = (time.perf_counter() - started) * 1000
    front, state = images[-1, 0], states[-1]
    wrist = np.flip(front, axis=1).copy()
    with _capture(policy, jax) as captured:
        started = time.perf_counter()
        output = policy.infer({"observation/image": front, "observation/wrist_image": wrist,
                               "observation/state": state, "prompt": "insert the peg into the hole",
                               "keyframe_environment_step": 0})
        jax.block_until_ready(output["actions"])
        request_ms = (time.perf_counter() - started) * 1000
    trace = deepcopy(output["selector_trace"])
    validate_selector_trace(trace)
    _require(policy.step_idx == length - 1 and policy._selector_call_index == 1,
             "History length/call count changed during the architecture request")
    _require(policy.mem_buffer._frame_sampling_selector is None, "Temporary selector was not restored")
    observation = captured["observation"]
    arrays = {f"raw_{key}": value for key, value in zip(MEMORY_KEYS, captured["raw"], strict=True)}
    arrays.update({key: np.asarray(getattr(observation, field)).copy()
                   for key, field in zip(MEMORY_KEYS, MEMORY_FIELDS, strict=True)})
    arrays.update({f"pre_model_{key}": value for key, value in captured["pre_model"].items()})
    noise_shape = (1, policy._model.action_horizon, policy._model.action_dim)
    initial_noise = jax.random.normal(captured["sample_rng"], noise_shape)
    jax.block_until_ready(initial_noise)
    arrays.update({"final_memory": captured["final_memory"], "actions": np.asarray(output["actions"]),
                   "raw_model_actions": captured["raw_model_actions"],
                   "sample_rng_data": np.asarray(jax.random.key_data(captured["sample_rng"])),
                   "initial_noise": np.asarray(initial_noise), "history_images": images,
                   "history_states": states, "history_stages": stages,
                   "current_front": front, "current_wrist": wrist, "current_state": state})
    _validate_arrays(arrays, trace, jax=jax)
    return {"trace": trace, "arrays": arrays, "observation": observation,
            "sample_rng": captured["sample_rng"], "initial_noise": initial_noise,
            "reset": reset, "cache_before": cache_before, "cache_after": _cache_sizes(policy),
            "history_preparation_latency_ms": preparation_ms,
            "full_request_latency_ms": request_ms,
            "sample_latency_ms": captured["synchronized_sample_latency_ms"],
            "memory_meter": _peak_memory(device)}


def _compare_repeat(first, repeat):
    _require(first["trace"]["selected_frame_indices"] == repeat["trace"]["selected_frame_indices"],
             "Interleaved reset/repeat changed selected frame indices")
    _require(first["trace"]["selector_seed"] == repeat["trace"]["selector_seed"],
             "Interleaved reset/repeat changed selector RNG")
    _require(set(first["arrays"]) == set(repeat["arrays"]), "Repeated capture fields differ")
    for key in first["arrays"]:
        _require(_digest(first["arrays"][key]) == _digest(repeat["arrays"][key]),
                 f"Same-process repeat changed actual {key}")
    _require(repeat["cache_before"] == repeat["cache_after"], "Warm repeat unexpectedly recompiled")


def _parameter_snapshot(model, nnx, jax):
    leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    _require(bool(leaves), "Diagnostic model has no real parameter leaves")
    return [(str(path), value) for path, value in leaves]


def _same_parameter_arrays(reference, observed):
    _require(len(reference) == len(observed), "Diagnostic changed parameter leaf count")
    for (path, value), (other_path, other_value) in zip(reference, observed, strict=True):
        _require(path == other_path and value is other_value, "Diagnostic parameters are not the same JAX arrays")
        _require(value.shape == other_value.shape and value.dtype == other_value.dtype,
                 "Diagnostic changed parameter shapes/dtypes")


def _make_512_graph(model, nnx, jax):
    from omegaconf import OmegaConf

    original = _parameter_snapshot(model, nnx, jax)
    expected, _ = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
    _require(OmegaConf.to_container(model.history_config, resolve=True) == expected,
             "Primary model is not the frozen 768-slot configuration")
    clone = nnx.clone(model)  # split/merge only: no constructor, RNG or weight load
    config512 = OmegaConf.create(deepcopy(RELEASED_HISTORY_CONFIG))
    clone.history_config = config512
    clone.config = dataclasses.replace(model.config, history_config=config512)
    clone.mem_encoder.config = config512
    _require(clone is not model and clone.mem_encoder is not model.mem_encoder,
             "Diagnostic graph did not isolate mutable module objects")
    _require(clone.history_config is not model.history_config, "History config is aliased")
    _same_parameter_arrays(original, _parameter_snapshot(clone, nnx, jax))
    return clone, original


def _slice_u_observation(observation):
    mask = np.asarray(observation.static_mask)
    _require(mask.shape == (1, 768) and mask.dtype == np.bool_, "Diagnostic requires real padded768 mask")
    _require(mask[0, :256].all() and not mask[0, 256:].any(), "Diagnostic requires short16 U without extra frames")
    return observation.replace(**{field: getattr(observation, field)[:, :512] for field in MEMORY_FIELDS})


def _linf(left, right):
    left, right = np.asarray(left), np.asarray(right)
    _require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
             "Cannot compare different shapes or nonfinite diagnostic outputs")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _padding_diagnostic(policy, case, jax):
    from flax import nnx
    from omegaconf import OmegaConf
    from openpi.shared import nnx_utils
    from mme_vla_suite.models.integration.history_observation import preprocess_observation

    trace = case["trace"]
    _require(trace["history_length"] == 16 and trace["selected_frame_indices"] == list(range(16))
             and trace["extra_count"] == 0, "Padded-U diagnostic must not include keyframe additions")
    model = policy._model
    clone, original_parameters = _make_512_graph(model, nnx, jax)
    primary_config = deepcopy(OmegaConf.to_container(model.history_config, resolve=True))
    observation768 = case["observation"]
    observation512 = _slice_u_observation(observation768)
    noise, rng = case["initial_noise"], case["sample_rng"]
    timestep = jax.numpy.ones((1,), dtype=noise.dtype)
    arrays = {"initial_noise": np.asarray(noise), "time": np.asarray(timestep)}
    for slots, current_model, obs in ((512, clone, observation512), (768, model, observation768)):
        # New JIT wrappers for each graph; original live-policy kernels untouched.
        sampler = nnx_utils.module_jit(current_model.sample_actions)
        predictor = nnx_utils.module_jit(current_model.predict_velocity_from_preprocessed)
        raw_actions = sampler(rng, obs, noise=noise, **policy._sample_kwargs)
        preprocessed = preprocess_observation(None, obs, train=False)
        velocity, _ = predictor(preprocessed, noise, timestep)
        jax.block_until_ready((raw_actions, velocity))
        decoded = policy._output_transform({"state": np.asarray(obs.state[0]),
                                             "actions": np.asarray(raw_actions[0])})
        actions = np.asarray(decoded["actions"])
        _require(actions.shape == (20, 8), "Diagnostic executable actions changed shape")
        arrays[f"actions_{slots}"] = actions
        arrays[f"raw_model_actions_{slots}"] = np.asarray(raw_actions)
        arrays[f"velocity_{slots}"] = np.asarray(velocity)
        for key, field in zip(MEMORY_KEYS, MEMORY_FIELDS, strict=True):
            arrays[f"{key}_{slots}"] = np.asarray(getattr(obs, field))
    # The ordinary sampler produces noise inside its JIT, whereas this matched
    # diagnostic supplies it as an argument. Fusion can change last-bit rounding
    # even with the identical PRNG key. Do not conflate these two compiled APIs
    # with the bitwise actual-infer repeat gate above. Measure this extra contrast
    # while both 512/768 diagnostic calls receive the very same explicit array.
    arrays["actual_infer_raw_model_actions_768"] = case["arrays"]["raw_model_actions"]
    _same_parameter_arrays(original_parameters, _parameter_snapshot(model, nnx, jax))
    _same_parameter_arrays(original_parameters, _parameter_snapshot(clone, nnx, jax))
    _require(OmegaConf.to_container(model.history_config, resolve=True) == primary_config
             and int(model.mem_encoder.config.budget) == 768
             and int(model.config.history_config.budget) == 768, "Diagnostic mutated primary configuration")
    structure = [{"path": path, "shape": list(value.shape), "dtype": str(value.dtype)}
                 for path, value in original_parameters]
    return {"same_input": True, "same_noise": True, "u_memory_slots": 512, "padded_memory_slots": 768,
            "action_difference_linf": _linf(arrays["actions_512"], arrays["actions_768"]),
            "velocity_difference_linf": _linf(arrays["velocity_512"], arrays["velocity_768"]),
            "input_kind": "short16_same_U_content_different_false_padding_length",
            "shared_parameter_array_identity": True, "parameter_leaf_count": len(structure),
            "parameter_structure_sha256": payload_digest(structure), "zero_weight_initialization": True,
            "primary_budget_before": 768, "primary_budget_after": 768, "diagnostic_budget": 512,
            "rope_implementation_changed": False, "time_value": 1.0,
            "model_action_shape": list(arrays["raw_model_actions_768"].shape),
            "executed_action_shape": list(arrays["actions_768"].shape),
            "velocity_shape": list(arrays["velocity_768"].shape),
            "explicit_noise_uses_actual_infer_rng": True,
            "explicit_noise_vs_implicit_noise_768_linf": _linf(
                arrays["raw_model_actions_768"], arrays["actual_infer_raw_model_actions_768"]),
            "explicit_vs_implicit_comparison_scope": "different compiled noise paths; reported, not bitwise gate",
            "difference_is_measurement_not_failure": True}, arrays


def _require_live_policy(policy, plan):
    from experiments.uniform_keyframe_expansion.launch_contract import ValidatedExecutionPlan

    _require(type(plan) is ValidatedExecutionPlan, "A sealed validated architecture plan is required")
    plan.require_runtime_stage("architecture_smoke")
    plan.revalidate()
    _require(not plan.rows, "Architecture probe cannot run benchmark trajectory rows")
    # Imports that can initialize a backend occur only beyond plan validation.
    import jax
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
    from mme_vla_suite.policies.uniform_keyframe_expansion_policy import UniformKeyframeExpansionPolicy
    from omegaconf import OmegaConf

    _require(type(policy) is UniformKeyframeExpansionPolicy and type(policy._model) is HistoryPi0,
             "A real, strictly loaded expansion policy is required, not a CPU fixture")
    expected, evidence = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
    _require(policy.metadata.get("uniform_keyframe_expansion") == evidence,
             "Policy lacks exact strict-loading/new-family metadata")
    _require(OmegaConf.to_container(policy.config, resolve=True) == expected and policy._seed == 7,
             "Policy configuration/seed differs from the frozen contract")
    _require(set(policy._sample_kwargs) <= {"num_steps"} and policy._sample_kwargs.get("num_steps", 10) == 10,
             "Unexpected sampling override; the exact actual-infer noise path must be retained")
    devices = jax.devices()
    _require(len(devices) == 1 and devices[0].platform == "gpu", "Architecture evidence requires one real GPU")
    _peak_memory(devices[0])  # unavailable/zero measurement must fail, not fake a value
    return jax, devices[0]


def run_loaded_probe(policy, plan, output_dir=None):
    """Measure a loaded GPU model; return raw measurements, never a PASS gate.

    ``cold_latency_ms`` means first request of that case in the resident process,
    not a separate cold model load for each case. Cache sizes expose compilation
    reuse. ``warm_latency_ms`` is a synchronized interleaved repeat. Peaks are
    actual cumulative JAX-allocator peaks, not independent per-case GPU peaks.
    Controller ownership, live checkpoint-byte attestation and final publication
    are outside this numerical function and remain required separately.
    """
    jax, device = _require_live_policy(policy, plan)
    directory = None if output_dir is None else Path(output_dir).absolute()
    if directory is not None:
        _require(not any(path.is_symlink() for path in (directory, *directory.parents)),
                 "Probe artifacts cannot traverse symlinks")
        directory.mkdir(parents=False, exist_ok=False)
    model = policy._model
    first = {}
    records = []
    case_results = []
    for arm, length in CASES:
        first[(arm, length)] = _run_once(policy, arm=arm, length=length, jax=jax, device=device)
    # Reverse order deliberately crosses both arm/length boundaries in one PID.
    for arm, length in reversed(CASES):
        initial = first[(arm, length)]
        repeat = _run_once(policy, arm=arm, length=length, jax=jax, device=device)
        _compare_repeat(initial, repeat)
        result = {"arm": arm, "history_length": length, "task": "InsertPeg", "split": "val", "episode_id": 0,
                  "action_shape": [20, 8], "memory_shape": [1, 768, 1024],
                  "finite_outputs": True, "memory_digest_repeat_equal": True, "action_repeat_equal": True,
                  "reset_verified": True, "reset_first": initial["reset"], "reset_repeat": repeat["reset"],
                  "cold_latency_ms": initial["sample_latency_ms"], "warm_latency_ms": repeat["sample_latency_ms"],
                  "selector_latency_ms": repeat["trace"]["selector_latency_ms"],
                  "full_request_latency_ms": repeat["full_request_latency_ms"],
                  "first_full_request_latency_ms": initial["full_request_latency_ms"],
                  "first_history_preparation_latency_ms": initial["history_preparation_latency_ms"],
                  "repeat_history_preparation_latency_ms": repeat["history_preparation_latency_ms"],
                  "peak_gpu_bytes": repeat["memory_meter"]["peak_gpu_bytes"],
                  "memory_meter": repeat["memory_meter"],
                  "cache_first_before": initial["cache_before"], "cache_first_after": initial["cache_after"],
                  "cache_repeat_before": repeat["cache_before"], "cache_repeat_after": repeat["cache_after"],
                  "selected_indices": initial["trace"]["selected_frame_indices"],
                  "first_trace": initial["trace"], "repeat_trace": repeat["trace"],
                  "first_array_encoding": _encoding_metadata(initial["arrays"]),
                  "repeat_array_encoding": _encoding_metadata(repeat["arrays"])}
        if directory is not None:
            for label, run in (("first", initial), ("repeat", repeat)):
                record = _write_npz(directory / f"{arm}_{length}_{label}.npz", run["arrays"])
                records.append(record)
                result[f"{label}_artifact"] = _reference(record)
        case_results.append(result)
    diagnostic, diagnostic_arrays = _padding_diagnostic(policy, first[("UK48", 16)], jax)
    _require(policy._model is model, "Probe replaced the resident model")
    policy.reset()
    final_reset = policy.reset_evidence()
    if directory is not None:
        record = _write_npz(directory / "padded_u_diagnostic.npz", diagnostic_arrays)
        diagnostic["artifact"] = _reference(record)
        diagnostic["array_encoding"] = _encoding_metadata(diagnostic_arrays)
        records.append(record)
    result = {"schema_version": 1, "kind": "architecture_measurements", "binding": plan.binding,
              "execution_identity": plan.execution_identity, "experiment_family": EXPANSION_FAMILY,
              "scope": "development_only", "input_kind": "synthetic_causal_history", "synthetic_fixture": True,
              "history_scope": "generated causal reset-prefix; environment step=0, history index=length-1",
              "numeric_audit_encoding": "bfloat16 converted losslessly to float32 only for saved evidence; model unchanged",
              "no_task_execution": True, "formal_test_outcomes_opened": False, "training_performed": False,
              "model_process_pid": os.getpid(), "case_count": 4,
              "strict_load": {"missing": [], "extra": [], "random_initialized": []},
              "strict_load_evidence_scope": "validated_loader_metadata; controller binds live bootstrap/checkpoint proof",
              "cases": sorted(case_results, key=lambda item: CASES.index((item["arm"], item["history_length"]))),
              "first_case_order": [list(case) for case in CASES],
              "repeat_case_order": [list(case) for case in reversed(CASES)],
              "final_reset_evidence": final_reset, "padded_u_diagnostic": diagnostic,
              "latency_scope": "synchronized sampling; cold=first case request, not per-case cold model load",
              "peak_scope": "measured cumulative JAX allocator peak; not total process or per-case independent peak",
              "post_diagnostic_memory_meter": _peak_memory(device), "raw_artifacts": records}
    if directory is not None:
        from experiments.uniform_keyframe_expansion.artifacts import _write

        manifest = {"schema_version": 1, "kind": "architecture_raw_artifacts",
                    "execution_identity": plan.execution_identity, "artifacts": records}
        path = directory / "raw_artifact_manifest.json"
        _write(path, manifest)
        record = _file_record(path)
        result["raw_artifact_manifest"] = {key: record[key] for key in ("path", "sha256")}
    # Fail rather than leak any nonfinite measured duration or malformed JSON.
    json.dumps(result, allow_nan=False)
    return result
