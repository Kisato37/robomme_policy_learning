"""Opt-in policy adapter for U32 plus causal extra frames in 48 fixed slots.

The released policy and all completed selectors remain untouched. This adapter
reuses the same observation, feature gather, normalization, model and action RNG
paths; only selection, the explicit input budget and its audit contract differ.
"""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from omegaconf import OmegaConf

from mme_vla_suite.policies.policy import MME_VLA_Policy
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.uniform_keyframe_config import EXPANSION_FAMILY
from mme_vla_suite.shared.uniform_keyframe_config import payload_digest
from mme_vla_suite.shared.uniform_keyframe_config import validate_expanded_history_mapping
from mme_vla_suite.shared.uniform_keyframe_expansion import EXPANSION_ARMS
from mme_vla_suite.shared.uniform_keyframe_expansion import ExpansionInvariantError
from mme_vla_suite.shared.uniform_keyframe_expansion import derive_expansion_seed
from mme_vla_suite.shared.uniform_keyframe_expansion import select_expansion_indices


class UniformKeyframeExpansionPolicy(MME_VLA_Policy):
    def reset(self) -> None:
        super().reset()
        self.last_expansion_failure = None

    def configure_keyframe_selector(self, selector_config: dict) -> None:
        raise ValueError("48-slot policy requires configure_uniform_keyframe_expansion, not an old-family selector")

    def configure_uniform_keyframe_expansion(self, selector_config: dict) -> None:
        if (self.step_idx != -1 or self.mem_buffer is None or self.mem_buffer._history_feats
                or self.mem_buffer._history_metadata or self._keyframe_selector_config is not None):
            raise RuntimeError("Expansion configuration requires a fresh, unconfigured episode reset")
        validate_expanded_history_mapping(OmegaConf.to_container(self.config, resolve=True))
        if self._seed != 7 or self._model.action_horizon != 20:
            raise ValueError("Expansion requires policy seed 7 and action horizon 20")
        required = {"arm", "task", "episode_id", "split", "random_seeds", "seed_table_sha256"}
        if set(selector_config) != required:
            raise ValueError(f"Expansion configuration requires exactly {sorted(required)}")
        arm = selector_config["arm"]
        if arm not in EXPANSION_ARMS:
            raise ValueError("Expansion arm must be UK48 or UN48")
        split, task, episode = (selector_config[key] for key in ("split", "task", "episode_id"))
        expected_seeds = [derive_expansion_seed(split, task, episode, i) for i in range(MAX_POLICY_CALLS)]
        if (split == "val" and episode != 0) or (split == "test" and not 0 <= episode < 50):
            raise ValueError("Expansion trajectory is outside the frozen split/episode population")
        actual_seeds = selector_config["random_seeds"]
        if (not isinstance(actual_seeds, (list, tuple))
                or any(type(seed) is not int for seed in actual_seeds)
                or list(actual_seeds) != expected_seeds):
            raise ValueError("Expansion seed list differs from the frozen per-trajectory seed table")
        if selector_config["seed_table_sha256"] != payload_digest(expected_seeds):
            raise ValueError("Expansion per-trajectory seed-table digest mismatch")
        self._keyframe_selector_config = deepcopy(selector_config)
        self._selector_call_index = 0
        self._selector_rng = None
        self._pending_selector_trace = None
        self.last_expansion_failure = None

    def _prepare_history(self, inputs: dict) -> dict:
        if self._keyframe_selector_config is None:
            # Without this check an unconfigured 768-slot model would silently
            # use Uniform48. It must never masquerade as either new arm.
            raise RuntimeError("Configure an expansion trajectory before inference")
        if self.last_expansion_failure is not None:
            raise RuntimeError("Expansion invariant failed; a fresh reset and reviewed attempt are required")
        prepared = super()._prepare_history(inputs)
        # The parent updates the normalized state's hash. Keep its dtype in the
        # new-family trace aligned too when norm statistics promote precision.
        self._pending_selector_trace["state_tensor_dtype"] = str(np.asarray(prepared["static_state_emb"]).dtype)
        return prepared

    def _prepare_experiment_frame_sampling(self, history_feats_gather_fn, token_budget, token_per_image):
        config = self._keyframe_selector_config
        if config is None:
            raise RuntimeError("Expansion selector is unconfigured")
        if self.last_expansion_failure is not None:
            raise RuntimeError("Expansion invariant failed; reset is required")
        if (token_budget, token_per_image, self.mem_buffer.num_views) != (768, 16, 1):
            raise RuntimeError("Expansion runtime budget must remain 48 frames / 768 tokens / one view")
        call_index = self._selector_call_index
        if not 0 <= call_index < MAX_POLICY_CALLS:
            raise RuntimeError("Expansion policy call exceeds preregistered seed table 0..81")
        lookup_started = time.monotonic()
        visible = self.mem_buffer.get_boundary_indices(self.step_idx)
        flags = [self.mem_buffer._history_metadata[i]["boundary"] for i in range(self.step_idx + 1)]
        lookup_ms = (time.monotonic() - lookup_started) * 1000
        decision = {}

        def selector(step_idx, budget, tokens):
            if (step_idx, budget, tokens) != (self.step_idx, 768, 16):
                raise RuntimeError("Expansion selector received a different frame/budget context")
            # The original method is the ONLY runtime source of the U base.
            base = self.mem_buffer.get_frame_sampling_indices(step_idx, 512, 16)
            selected, evidence = select_expansion_indices(
                config["arm"], step_idx=step_idx, base_uniform_indices=base,
                boundary_flags=flags, split=config["split"], task=config["task"],
                episode_id=config["episode_id"], policy_call_index=call_index,
            )
            if config["arm"] == "UN48" and evidence["selector_seed"] != config["random_seeds"][call_index]:
                raise RuntimeError("Expansion runtime RNG differs from the frozen seed entry")
            decision.update(evidence)
            return selected

        try:
            with self.mem_buffer.temporary_frame_sampling_selector(selector):
                prepared = self.mem_buffer.prepare_frame_sampling(
                    self.step_idx, token_budget, token_per_image, history_feats_gather_fn,
                )
        except ExpansionInvariantError as exc:
            # The future runner must persist this alongside the exception and
            # stop the study; it is not a scientific failure or a skipped cell.
            self.last_expansion_failure = deepcopy(exc.evidence)
            self._pending_selector_trace = None
            raise

        bookkeeping_started = time.monotonic()
        image, position, state, mask = prepared
        selected = list(self.mem_buffer._last_frame_sampling_indices)
        mask = np.asarray(mask)
        valid_tokens = 16 * len(selected)
        if mask.shape != (768,) or mask.dtype != np.bool_:
            raise RuntimeError("Expansion requires a boolean 768-token mask")
        if not mask[:valid_tokens].all() or mask[valid_tokens:].any() or int(mask.sum()) != valid_tokens:
            raise RuntimeError("Expansion mask must contain one valid prefix followed by false padding")
        expected_shapes = ((768, self.mem_buffer.img_emb_dim), (768, self.mem_buffer.pos_emb_dim),
                           (768, self.mem_buffer.state_emb_dim))
        for array, shape in zip((image, position, state), expected_shapes, strict=True):
            array = np.asarray(array)
            if array.shape != shape or not np.isfinite(array).all():
                raise RuntimeError(f"Invalid expansion memory tensor: expected finite {shape}, got {array.shape}")
            if np.any(array[valid_tokens:] != 0):
                raise RuntimeError("Unused memory slots must be zero-padded before normalization")
        if selected != decision["selected_indices"]:
            raise RuntimeError("Expansion gather changed the selected frame indices")
        selected_boundaries = len(set(selected) & set(visible))
        gaps = [right - left for left, right in zip(selected, selected[1:])]
        selector_ms = self.mem_buffer._last_frame_sampling_selector_latency_ms
        self._pending_selector_trace = {
            **decision,
            "schema_version": 1,
            "experiment_family": EXPANSION_FAMILY,
            "selector_name": config["arm"],
            "seed_table_scope": EXPANSION_FAMILY,
            "seed_table_dataset": config["split"],
            "seed_table_sha256": config["seed_table_sha256"],
            "history_length": self.step_idx + 1,
            "current_history_index": self.step_idx,
            "selected_frame_indices": selected,
            "selected_indices_sha256": payload_digest(selected),
            "valid_memory_token_count": valid_tokens,
            "mask_shape": list(mask.shape), "mask_dtype": str(mask.dtype),
            "mask_valid_prefix_all_true": True, "mask_padding_all_false": True,
            "mask_sha256": self._array_digest(mask),
            "image_tensor_shape": list(np.shape(image)),
            "image_tensor_dtype": str(np.asarray(image).dtype),
            "image_tensor_sha256": self._array_digest(image),
            "position_tensor_shape": list(np.shape(position)),
            "position_tensor_dtype": str(np.asarray(position).dtype),
            "position_tensor_sha256": self._array_digest(position),
            "state_tensor_shape": list(np.shape(state)),
            "state_tensor_dtype": str(np.asarray(state).dtype),
            "state_tensor_sha256": self._array_digest(state),
            "prepared_memory_component_shapes": [list(np.shape(a)) for a in prepared],
            "prepared_memory_components_sha256": self._memory_digest(*prepared),
            "age_distribution": [self.step_idx - i for i in selected],
            "maximum_temporal_gap": max(gaps, default=0),
            "boundary_recall": selected_boundaries / len(visible),
            "effective_memory_budget": 768,
            "base_uniform_token_budget": 512,
        }
        bookkeeping_ms = (time.monotonic() - bookkeeping_started) * 1000
        self._pending_selector_trace.update({
            "boundary_lookup_latency_ms": lookup_ms,
            "selector_decision_latency_ms": selector_ms,
            "selector_bookkeeping_latency_ms": bookkeeping_ms,
            "selector_latency_ms": lookup_ms + selector_ms + bookkeeping_ms,
        })
        self._selector_call_index += 1
        return prepared
