from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.uniform_keyframe_expansion.trace_validation import (
    ExpansionTraceError, validate_selector_trace, validate_trace_sequence,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import official_uniform_indices
from mme_vla_suite.shared.uniform_keyframe_config import payload_digest
from mme_vla_suite.shared.uniform_keyframe_expansion import (
    SEED_FAMILY, derive_expansion_seed, select_expansion_indices,
)


def array_hash(array):
    result = hashlib.sha256()
    result.update(str(array.dtype).encode("ascii"))
    result.update(json.dumps(array.shape).encode("ascii"))
    result.update(array.tobytes(order="C"))
    return result.hexdigest()


@lru_cache
def zero_array_hash(shape):
    return array_hash(np.zeros(shape, np.float32))


def synthetic_trace(arm="UK48", step=63, boundaries=(0, 17, 35), call=0,
                    task="BinFill", split="test", episode=0, final=True):
    selected, info = select_expansion_indices(
        arm, step_idx=step, base_uniform_indices=official_uniform_indices(step),
        boundary_flags=[index in boundaries for index in range(step + 1)], split=split,
        task=task, episode_id=episode, policy_call_index=call,
    )
    mask = np.arange(768) < len(selected) * 16
    seeds = [derive_expansion_seed(split, task, episode, index) for index in range(82)]
    trace = {
        **info, "schema_version": 1, "experiment_family": SEED_FAMILY,
        "selector_name": arm, "seed_table_scope": SEED_FAMILY,
        "seed_table_dataset": split, "seed_table_sha256": payload_digest(seeds),
        "history_length": step + 1, "current_history_index": step,
        "selected_frame_indices": selected, "selected_indices_sha256": payload_digest(selected),
        "valid_memory_token_count": len(selected) * 16,
        "mask_shape": [768], "mask_dtype": "bool", "mask_valid_prefix_all_true": True,
        "mask_padding_all_false": True, "mask_sha256": array_hash(mask),
        "prepared_memory_component_shapes": [[768, 2048], [768, 768], [768, 8], [768]],
        "age_distribution": [step - index for index in selected],
        "maximum_temporal_gap": max((b - a for a, b in zip(selected, selected[1:])), default=0),
        "boundary_recall": info["selected_boundary_count"] / len(boundaries),
        "effective_memory_budget": 768, "base_uniform_token_budget": 512,
        "boundary_lookup_latency_ms": 1.0, "selector_decision_latency_ms": 2.0,
        "selector_bookkeeping_latency_ms": 3.0, "selector_latency_ms": 6.0,
    }
    for prefix, width in (("image", 2048), ("position", 768), ("state", 8)):
        trace[f"{prefix}_tensor_shape"] = [768, width]
        trace[f"{prefix}_tensor_dtype"] = "float32"
        trace[f"{prefix}_tensor_sha256"] = zero_array_hash((768, width))
    trace["prepared_memory_components_sha256"] = hashlib.sha256(b"".join(
        bytes.fromhex(trace[field]) for field in ("image_tensor_sha256", "position_tensor_sha256",
                                                 "state_tensor_sha256", "mask_sha256"))).hexdigest()
    if final:
        trace.update({
            "prepared_memory_input_shape": [768, 2824], "prepared_memory_input_dtype": "float32",
            "prepared_memory_input_sha256": zero_array_hash((768, 2824)),
            "final_memory_tensor_shape": [1, 768, 1024], "final_memory_tensor_dtype": "float32",
            "final_memory_tensor_is_floating": True, "final_memory_tensor_finite": True,
            "final_memory_tensor_sha256": zero_array_hash((1, 768, 1024)),
            "environment_step": call * 16, "model_latency_ms": 5.0,
        })
    return trace


class TraceValidationTests(unittest.TestCase):
    def test_valid_both_arm_traces_and_early_padding(self):
        for arm in ("UK48", "UN48"):
            for step, boundaries in ((0, (0,)), (31, (0, 17)), (63, (0, 17, 35)),
                                     (1300, (0, 17, 35, 128, 640))):
                result = validate_selector_trace(synthetic_trace(arm, step, boundaries))
                self.assertEqual(result["status"], "passed")
                self.assertFalse(result["feature_values_verified_from_hashes"])

    def test_prefinal_is_explicit_and_partial_final_fields_are_rejected(self):
        trace = synthetic_trace(final=False)
        self.assertEqual(validate_selector_trace(trace, require_final_memory=False)["status"], "passed")
        with self.assertRaises(ExpansionTraceError):
            validate_selector_trace(trace)
        trace["final_memory_tensor_shape"] = [1, 768, 1024]
        with self.assertRaises(ExpansionTraceError):
            validate_selector_trace(trace, require_final_memory=False)

    def test_missing_trace_and_every_missing_required_field_fail(self):
        for missing in (None, {}, [], ""):
            with self.assertRaises(ExpansionTraceError):
                validate_selector_trace(missing)
        trace = synthetic_trace()
        for field in trace:
            damaged = deepcopy(trace)
            del damaged[field]
            with self.subTest(field=field), self.assertRaises(ExpansionTraceError):
                validate_selector_trace(damaged)

    def test_mutations_reject_capacity_type_seed_family_selection_and_counts(self):
        trace = synthetic_trace("UN48")
        mutations = {
            "schema_version": True, "experiment_family": "keyframe_oracle_sampling",
            "arm": "R", "selector_name": "UK48", "split": "train", "episode_id": True,
            "policy_call_index": 82, "task": "FakeTask", "history_length": 63,
            "current_history_index": 62, "base_uniform_token_budget": 768,
            "effective_memory_budget": 512, "frame_capacity": 32, "memory_token_capacity": 512,
            "selected_frame_indices": trace["selected_frame_indices"][::-1],
            "base_uniform_indices": official_uniform_indices(63, 48),
            "new_key_indices": [], "extra_count": 0, "nonkey_candidate_count": 1234,
            "selected_extra_indices": [17, 35], "selected_boundary_count": 0,
            "selector_seed": trace["selector_seed"] + 1, "selector_rng": "MT19937",
            "seed_table_sha256": "a" * 64, "seed_table_scope": "formal-evaluation-v1",
            "selected_indices_sha256": "b" * 64,
            "valid_frame_count": 32, "padding_frame_count": 0,
            "valid_token_count": 512, "padding_token_count": 0,
            "mask_shape": [512], "mask_dtype": "int8", "mask_padding_all_false": False,
            "mask_valid_prefix_all_true": 1, "mask_sha256": "c" * 64,
            "image_tensor_shape": [512, 2048], "position_tensor_shape": [768, 512],
            "state_tensor_shape": [768, 7], "image_tensor_dtype": "int32",
            "image_tensor_sha256": "a" * 64, "position_tensor_sha256": "F" * 64,
            "prepared_memory_components_sha256": "d" * 64,
            "prepared_memory_input_shape": [512, 2824], "prepared_memory_input_dtype": "object",
            "prepared_memory_input_sha256": "missing", "final_memory_tensor_shape": [1, 512, 1024],
            "final_memory_tensor_dtype": "int64", "final_memory_tensor_finite": False,
            "final_memory_tensor_is_floating": 1, "final_memory_tensor_sha256": "G" * 64,
            "environment_step": 1301, "model_latency_ms": -1,
            "selector_latency_ms": 12.0, "boundary_lookup_latency_ms": True,
            "maximum_temporal_gap": 999, "age_distribution": [], "boundary_recall": -1.0,
        }
        for field, wrong in mutations.items():
            damaged = deepcopy(trace)
            damaged[field] = wrong
            with self.subTest(field=field), self.assertRaises(ExpansionTraceError):
                validate_selector_trace(damaged)

    def test_reject_future_or_duplicated_boundary_and_nonfinite_value(self):
        for boundaries in ([0, 17, 35, 64], [0, 17, 17, 35], [17, 35], [0, True, 35]):
            trace = synthetic_trace()
            trace["visible_boundary_indices"] = boundaries
            with self.assertRaises(ExpansionTraceError):
                validate_selector_trace(trace)
        trace = synthetic_trace()
        trace["model_latency_ms"] = float("nan")
        with self.assertRaises(ExpansionTraceError):
            validate_selector_trace(trace)

    def test_hash_only_validation_explicitly_does_not_claim_feature_evidence(self):
        trace = synthetic_trace()
        # A consistently rebound feature hash passes structural validation; no
        # checker can prove its original tensor values without that tensor.
        trace["image_tensor_sha256"] = "e" * 64
        trace["prepared_memory_components_sha256"] = hashlib.sha256(b"".join(
            bytes.fromhex(trace[key]) for key in ("image_tensor_sha256", "position_tensor_sha256",
                                                "state_tensor_sha256", "mask_sha256"))).hexdigest()
        result = validate_selector_trace(trace)
        self.assertFalse(result["feature_values_verified_from_hashes"])

    def test_valid_sequence_and_expected_count(self):
        traces = [synthetic_trace(step=31, boundaries=(0, 17), call=0),
                  synthetic_trace(step=63, boundaries=(0, 17, 35), call=1)]
        result = validate_trace_sequence(traces, expected_call_count=2)
        self.assertEqual(result["call_count"], 2)
        self.assertTrue(result["expected_call_count_verified"])
        self.assertFalse(result["terminal_outcome_verified"])
        with self.assertRaises(ExpansionTraceError):
            validate_trace_sequence(traces, expected_call_count=3)
        self.assertFalse(validate_trace_sequence(traces)["expected_call_count_verified"])

    def test_sequence_missing_calls_mixed_context_and_revised_past_are_rejected(self):
        first = synthetic_trace(step=31, boundaries=(0, 17), call=0)
        bad_seconds = [
            synthetic_trace(call=2), synthetic_trace(call=0),
            synthetic_trace(call=1, arm="UN48"), synthetic_trace(call=1, task="InsertPeg"),
            synthetic_trace(call=1, split="val"), synthetic_trace(call=1, episode=1),
            synthetic_trace(step=31, boundaries=(0, 17), call=1),
            synthetic_trace(call=1, boundaries=(0, 35)),
        ]
        for second in bad_seconds:
            with self.assertRaises(ExpansionTraceError):
                validate_trace_sequence([first, second])
        second = synthetic_trace(call=1)
        second["environment_step"] = 0
        with self.assertRaises(ExpansionTraceError):
            validate_trace_sequence([first, second])
        for absent in ([], None, "missing", [None], [first] * 83):
            with self.assertRaises(ExpansionTraceError):
                validate_trace_sequence(absent)

    def test_checker_accepts_actual_production_dimension_runtime_trace(self):
        # This integration fixture uses the real MemoryBuffer gather and actual
        # new policy audit code, but never loads a model or invokes GPU inference.
        location = Path(__file__).with_name("test_runtime.py")
        spec = importlib.util.spec_from_file_location("expansion_runtime_trace_fixture", location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for arm in ("UK48", "UN48"):
            policy = module.fixture_policy(arm, length=64, real_dimensions=True)
            policy.state_norm_stats.mean = 10  # masked raw-state zeros become nonzero after normalization
            inputs = policy._prepare_history({})
            trace = deepcopy(policy._pending_selector_trace)
            self.assertEqual(validate_selector_trace(trace, require_final_memory=False)["status"], "passed")
            valid = trace["valid_memory_token_count"]
            self.assertTrue(np.all(inputs["static_state_emb"][valid:] != 0))
            policy._record_final_memory_tensor(np.ones((1, 768, 1024), np.float32))
            complete = deepcopy(policy._pending_selector_trace)
            complete.update(environment_step=0, model_latency_ms=0.0)
            self.assertEqual(validate_selector_trace(complete)["status"], "passed")
            self.assertEqual(validate_trace_sequence([complete], expected_call_count=1)["status"], "passed")
            # This digest parity assertion independently uses the runtime method.
            self.assertEqual(trace["mask_sha256"], policy._array_digest(inputs["static_mask"]))


if __name__ == "__main__":
    unittest.main()
