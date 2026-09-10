"""CPU-only tests: no model, simulator, downloads, or experiment execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import unittest

import numpy as np

from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from mme_vla_suite.shared.keyframe_oracle_sampling import official_uniform_indices
from mme_vla_suite.shared.uniform_keyframe_expansion import (
    BASE_FRAME_CAPACITY, CANONICAL_TASKS, EXPANSION_ARMS, FRAME_CAPACITY,
    MEMORY_TOKEN_CAPACITY, TOKENS_PER_FRAME, ExpansionInvariantError,
    derive_expansion_seed, derive_selector_seed, select_expansion_indices,
)


def flags_at(step_idx, boundaries):
    return [index in set(boundaries) for index in range(step_idx + 1)]


class SelectorTests(unittest.TestCase):
    def select(self, arm="UK48", step_idx=130, boundaries=(0, 10, 20), **changes):
        arguments = dict(
            step_idx=step_idx, base_uniform_indices=official_uniform_indices(step_idx),
            boundary_flags=flags_at(step_idx, boundaries), split="test", task="BinFill",
            episode_id=0, policy_call_index=0,
        )
        arguments.update(changes)
        return select_expansion_indices(arm, **arguments)

    def test_frozen_capacities_and_canonical_tasks(self):
        self.assertEqual((BASE_FRAME_CAPACITY, FRAME_CAPACITY, TOKENS_PER_FRAME,
                          MEMORY_TOKEN_CAPACITY), (32, 48, 16, 768))
        self.assertEqual(CANONICAL_TASKS, FORMAL_TASKS)

    def test_all_early_histories_preserve_every_frame_and_only_pad(self):
        for step in range(32):
            for arm in EXPANSION_ARMS:
                with self.subTest(step=step, arm=arm):
                    selected, info = self.select(arm, step, boundaries=range(step + 1))
                    self.assertEqual(selected, list(range(step + 1)))
                    self.assertEqual(info["extra_count"], 0)
                    self.assertEqual(info["padding_frame_count"], 48 - step - 1)
                    self.assertEqual(info["valid_token_count"] + info["padding_token_count"], 768)

    def test_both_arms_preserve_original_uniform_and_match_increment(self):
        for step in (32, 33, 63, 130, 511, 1300):
            base = official_uniform_indices(step)
            gaps = sorted(set(range(step + 1)) - set(base))
            # Leave enough non-keyframe candidates for the matched random arm.
            boundaries = sorted({0, step, *gaps[:min(8, len(gaps) // 2)]})
            uk, uk_info = self.select("UK48", step, boundaries)
            un, un_info = self.select("UN48", step, boundaries)
            self.assertEqual(uk, sorted(set(base) | set(boundaries)))
            self.assertEqual(len(uk), len(un))
            self.assertEqual(uk_info["extra_count"], un_info["extra_count"])
            self.assertTrue(set(base).issubset(un))
            self.assertFalse((set(un) - set(base)) & set(boundaries))
            for selected, info in ((uk, uk_info), (un, un_info)):
                self.assertEqual(selected, sorted(set(selected)))
                self.assertTrue(all(0 <= value <= step for value in selected))
                self.assertEqual(info["selected_boundary_indices"], sorted(set(selected) & set(boundaries)))
                self.assertEqual(info["selected_boundary_count"], len(info["selected_boundary_indices"]))

    def test_duplicate_boundaries_in_uniform_are_not_added_again(self):
        base = official_uniform_indices(130)
        new = sorted(set(range(131)) - set(base))[:7]
        boundaries = sorted({*base[:3], *new})
        for arm in EXPANSION_ARMS:
            selected, info = self.select(arm, boundaries=boundaries)
            self.assertEqual(len(boundaries), 10)
            self.assertEqual(info["extra_count"], 7)
            self.assertEqual(len(selected), 39)
            self.assertEqual(info["padding_frame_count"], 9)
            self.assertEqual(info["new_key_indices"], new)

    def test_capacity_exactly_48_is_accepted(self):
        base = official_uniform_indices(130)
        extras = sorted(set(range(131)) - set(base))[:16]
        for arm in EXPANSION_ARMS:
            selected, info = self.select(arm, boundaries=[0, *extras])
            self.assertEqual(len(selected), 48)
            self.assertEqual(info["padding_token_count"], 0)

    def test_overflow_is_structured_hard_stop_for_both_arms(self):
        base = official_uniform_indices(130)
        extras = sorted(set(range(131)) - set(base))[:17]
        for arm in EXPANSION_ARMS:
            with self.assertRaises(ExpansionInvariantError) as caught:
                self.select(arm, boundaries=[0, *extras])
            error = caught.exception
            self.assertEqual(error.code, "capacity_overflow")
            self.assertEqual(error.evidence["required_frame_count"], 49)
            self.assertEqual(error.evidence["extra_count"], 17)
            self.assertEqual(error.evidence["base_uniform_indices"], base)
            json.dumps(error.evidence, allow_nan=False)

    def test_nonkey_shortage_stops_uk_too_instead_of_changing_control(self):
        step = 32
        missing = sorted(set(range(step + 1)) - set(official_uniform_indices(step)))
        self.assertEqual(len(missing), 1)
        for arm in EXPANSION_ARMS:
            with self.assertRaises(ExpansionInvariantError) as caught:
                self.select(arm, step, boundaries=[0, *missing])
            self.assertEqual(caught.exception.code, "nonkey_candidate_shortage")
            self.assertEqual(caught.exception.evidence["nonkey_candidate_count"], 0)
            self.assertEqual(caught.exception.evidence["extra_count"], 1)

    def test_history_flags_require_exact_visible_prefix(self):
        for count in (130, 132):
            with self.assertRaises(ExpansionInvariantError) as caught:
                self.select(boundary_flags=[True] + [False] * (count - 1))
            self.assertEqual(caught.exception.code, "boundary_history_alignment")

    def test_boundary_flags_must_be_boolean_and_start_with_boundary(self):
        for invalid in (1, 0, "False", None, 1.0, np.int64(1)):
            bad_flags = flags_at(130, [0])
            bad_flags[7] = invalid
            with self.assertRaises(ExpansionInvariantError) as caught:
                self.select(boundary_flags=bad_flags)
            self.assertEqual(caught.exception.code, "non_boolean_boundary_flag")
        with self.assertRaises(ExpansionInvariantError) as caught:
            self.select(boundary_flags=[False] * 131)
        self.assertEqual(caught.exception.code, "initial_frame_must_be_boundary")
        selected, _ = self.select(boundary_flags=np.asarray(flags_at(130, [0]), dtype=np.bool_))
        self.assertEqual(selected, official_uniform_indices(130))

    def test_invalid_uniform_is_never_silently_sorted_clipped_or_replaced(self):
        base = official_uniform_indices(130)
        invalid = [base[::-1], base + [130], base[:-1],
                   [*base[:-1], 131], [*base[:-1], True], official_uniform_indices(130, 48)]
        for bad_base in invalid:
            with self.assertRaises(ExpansionInvariantError):
                self.select(base_uniform_indices=bad_base)
        changed = list(base)
        changed[1] += 1
        with self.assertRaises(ExpansionInvariantError) as caught:
            self.select(base_uniform_indices=changed)
        self.assertEqual(caught.exception.code, "base_is_not_original_uniform32")

    def test_literal_original_buffer_sampler_is_accepted_for_every_history_length(self):
        # Execute only the literal tiny functions from source.  Importing the
        # whole MemoryBuffer here would unnecessarily initialize model libraries.
        repo = Path(__file__).resolve().parents[2]
        shared = repo / "src/mme_vla_suite/shared"
        utility_tree = ast.parse((shared / "data_utils.py").read_text())
        utility = next(node for node in utility_tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "even_sampling_indices")
        buffer_tree = ast.parse((shared / "mem_buffer.py").read_text())
        cls = next(node for node in buffer_tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "MemoryBuffer")
        method = next(node for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == "get_frame_sampling_indices")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[utility, method], type_ignores=[]), "literal-sampler", "exec"), namespace)
        buffer = type("OneViewFixture", (), {"num_views": 1})()
        for step in range(1301):
            base = namespace["get_frame_sampling_indices"](buffer, step, 512, 16)
            selected, _ = self.select(step_idx=step, boundaries=[0], base_uniform_indices=base)
            self.assertEqual(selected, base)

    def test_seed_namespace_has_frozen_golden_values(self):
        self.assertEqual(derive_expansion_seed("test", "BinFill", 0, 0), 16024834809501479908)
        self.assertEqual(derive_expansion_seed("val", "BinFill", 0, 0), 4766352607133001150)
        payload = b'[2026091001,"uniform_keyframe_expansion-v1","test","InsertPeg",49,81,"UN48"]'
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.assertEqual(derive_expansion_seed("test", "InsertPeg", 49, 81), expected)
        self.assertEqual(derive_selector_seed(split="test", task="InsertPeg", episode_id=49,
                                              policy_call_index=81), expected)
        with self.assertRaises(ExpansionInvariantError):
            derive_selector_seed(split="test", task="InsertPeg", episode_id=49,
                                 policy_call_index=81, arm="UK48")

    def test_random_choice_is_exact_pcg64_without_replacement(self):
        selected, info = self.select("UN48")
        candidates = sorted(set(range(131)) - (set(info["base_uniform_indices"]) |
                                               set(info["visible_boundary_indices"])))
        expected_extras = sorted(np.random.Generator(np.random.PCG64(info["selector_seed"])).choice(
            candidates, size=info["extra_count"], replace=False).tolist())
        self.assertEqual(info["selected_extra_indices"], expected_extras)
        self.assertEqual(selected, sorted(info["base_uniform_indices"] + expected_extras))
        self.assertEqual((selected, info), self.select("UN48"))
        self.assertIsNone(self.select("UK48")[1]["selector_seed"])

    def test_random_selector_does_not_mutate_global_rng_or_inputs(self):
        numpy_before = copy.deepcopy(np.random.get_state())
        python_before = random.getstate()
        base = official_uniform_indices(130)
        flags = flags_at(130, [0, 10, 20])
        copies = copy.deepcopy((base, flags))
        for arm in EXPANSION_ARMS:
            self.select(arm, base_uniform_indices=base, boundary_flags=flags)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())
        self.assertEqual((base, flags), copies)

    def test_invalid_context_is_rejected_even_for_nonrandom_arm(self):
        changes = [dict(split="train"), dict(task="WrongTask"), dict(episode_id=True),
                   dict(episode_id=-1), dict(policy_call_index=-1), dict(policy_call_index=82),
                   dict(policy_call_index=1.5)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ExpansionInvariantError):
                self.select(**change)
        with self.assertRaises(ExpansionInvariantError):
            select_expansion_indices("UK48", step_idx=True, base_uniform_indices=[0, 1],
                                     boundary_flags=[True, False], split="val", task="BinFill",
                                     episode_id=0, policy_call_index=0)
        for arm in ("U", "OC", "R", "Uniform48", "un48"):
            with self.assertRaises(ExpansionInvariantError):
                self.select(arm)

    def test_seed_table_unique_and_val_disjoint_from_formal(self):
        formal = {derive_expansion_seed("test", task, episode, call)
                  for task in FORMAL_TASKS for episode in range(50) for call in range(82)}
        val = {derive_expansion_seed("val", task, 0, call)
               for task in FORMAL_TASKS for call in range(82)}
        self.assertEqual(len(formal), 16 * 50 * 82)
        self.assertEqual(len(val), 16 * 82)
        self.assertFalse(formal & val)

    def test_diagnostics_are_json_serializable_and_self_consistent(self):
        for arm in EXPANSION_ARMS:
            selected, info = self.select(arm)
            restored = json.loads(json.dumps(info, allow_nan=False))
            self.assertEqual(info, restored)
            self.assertEqual(restored["selected_indices"], selected)
            self.assertEqual(restored["valid_frame_count"] + restored["padding_frame_count"], 48)
            self.assertEqual(restored["valid_token_count"], len(selected) * 16)


if __name__ == "__main__":
    unittest.main()
