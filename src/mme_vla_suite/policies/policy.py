from collections.abc import Sequence
import hashlib
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import SelectorArm
from mme_vla_suite.shared.keyframe_oracle_sampling import derive_random_seed
from mme_vla_suite.shared.keyframe_oracle_sampling import derive_smoke_random_seed
from mme_vla_suite.shared.keyframe_oracle_sampling import oracle_neighborhood_coverage_decision
from mme_vla_suite.shared.keyframe_oracle_sampling import parse_arm
from mme_vla_suite.shared.keyframe_oracle_sampling import select_indices
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.shared.mem_buffer import MemoryBufferRecurrent
from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils


class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)
        self._perceptual_memory_encode = (
            nnx_utils.module_jit(model.mem_encoder.encode_tokens)
            if model.history_config is not None
            and model.history_config.representation_type == "perceptual"
            else None
        )
        
        
        self.config = model.history_config
        self.mem_buffer = None
        self._keyframe_selector_config = None
        self._selector_call_index = 0
        self._selector_rng = None
        self._pending_selector_trace = None
        
        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles
        
        self.reset()
        
    
    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score = self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type != "symbolic":
            assert len(self.mem_buffer._history_feats) > 0, \
                "history feats is empty, add buffer first"
                                        
        # Experiment bookkeeping is removed before transforms/model inputs.
        obs = dict(obs)
        environment_step = obs.pop("keyframe_environment_step", None)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        if self._pending_selector_trace is not None:
            if self._perceptual_memory_encode is None:  # pragma: no cover - config gate
                raise RuntimeError("Perceptual-memory audit encoder is unavailable")
            final_memory_tensor = np.asarray(
                self._perceptual_memory_encode(
                    observation.static_image_emb,
                    observation.static_pos_emb,
                    observation.static_state_emb,
                )
            )
            self._record_final_memory_tensor(final_memory_tensor)
        self._rng, sample_rng = jax.random.split(self._rng)
    
        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)      
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000
        if self._pending_selector_trace is not None:
            trace = dict(self._pending_selector_trace)
            trace["environment_step"] = (
                None if environment_step is None else int(environment_step)
            )
            trace["model_latency_ms"] = model_time * 1000
            outputs["selector_trace"] = trace
            self._pending_selector_trace = None
        
        return outputs
    
    @override
    def reset(self) -> None:
        del self.mem_buffer
        self._prepare_mem_buffer()
        self.step_idx = -1  
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)
        self._keyframe_selector_config = None
        self._selector_call_index = 0
        self._selector_rng = None
        self._pending_selector_trace = None

    def configure_keyframe_selector(self, selector_config: dict) -> None:
        """Configure one freshly reset experiment trajectory."""
        if self.step_idx != -1 or self.mem_buffer._history_feats:
            raise RuntimeError("Selector configuration is allowed only after a fresh episode reset")
        expected = {
            "budget": 512,
            "num_views": 1,
            "token_per_image": 16,
            "representation_type": "perceptual",
            "perceptual_memory_type": "frame_sampling",
            "integration_type": "modulation",
            "use_state_emb": False,
            "streaming_obs_horizon": 16,
            "action_horizon": 20,
        }
        observed = {
            "budget": int(self.config.budget),
            "num_views": int(self.config.num_views),
            "token_per_image": int(self.config.token_per_image),
            "representation_type": str(self.config.representation_type),
            "perceptual_memory_type": str(self.config.perceptual_memory.type),
            "integration_type": str(self.config.integration_type),
            "use_state_emb": bool(self.config.use_state_emb),
            "streaming_obs_horizon": int(self.config.streaming_obs_horizon),
            "action_horizon": int(self._model.action_horizon),
        }
        if observed != expected:
            raise RuntimeError(
                f"Frozen keyframe history configuration mismatch: {observed} != {expected}"
            )
        arm = parse_arm(selector_config["arm"])
        task = str(selector_config["task"])
        episode_id = int(selector_config["episode_id"])
        if not task or episode_id < 0:
            raise ValueError("Selector task and episode_id must identify a valid trajectory")
        seed_table_scope = str(selector_config["seed_table_scope"])
        seed_table_dataset = str(selector_config["seed_table_dataset"])
        seed_derivation_contracts = {
            (FORMAL_SEED_SCOPE, FORMAL_SEED_DATASET): derive_random_seed,
            (SMOKE_SEED_SCOPE, SMOKE_SEED_DATASET): derive_smoke_random_seed,
        }
        try:
            derive_seed = seed_derivation_contracts[
                (seed_table_scope, seed_table_dataset)
            ]
        except KeyError as exc:
            raise ValueError(
                "Unknown or mismatched RandomSamp seed-table scope/dataset: "
                f"{seed_table_scope!r}/{seed_table_dataset!r}"
            ) from exc
        random_seeds = tuple(int(value) for value in selector_config["random_seeds"])
        if len(random_seeds) != MAX_POLICY_CALLS:
            raise ValueError(f"Selector configuration requires {MAX_POLICY_CALLS} preregistered seeds")
        for call_index, seed in enumerate(random_seeds):
            expected_seed = derive_seed(task, episode_id, call_index)
            if seed != expected_seed:
                raise ValueError(
                    f"Preregistered RandomSamp seed mismatch at policy call {call_index}"
                )
        self._keyframe_selector_config = {
            "arm": arm,
            "task": task,
            "episode_id": episode_id,
            "random_seeds": random_seeds,
            "seed_table_sha256": str(selector_config["seed_table_sha256"]),
            "seed_table_scope": seed_table_scope,
            "seed_table_dataset": seed_table_dataset,
        }
        self._selector_call_index = 0
        # RandomSamp uses an independently instantiated PCG64 stream for each
        # preregistered call.  This slot is explicit so reset/isolation tests can
        # prove that no selector generator survives an episode reset.
        self._selector_rng = None
            
    
    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        stage_indices = obs.get("current_task_index")
        if self._keyframe_selector_config is not None and stage_indices is None:
            raise RuntimeError(
                "Global blocker: live current_task_index is missing for a history segment"
            )
        if obs.get("exec_start_idx", 0) > 0: # has video
            self.exec_start_idx = obs["exec_start_idx"]
        
        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        if self.config.representation_type == "recurrent":
            if stage_indices is not None:
                raise RuntimeError(
                    "Causal keyframe stage metadata is unsupported for recurrent memory"
                )
            self.mem_buffer.add_buffer(images, states, step_idx_list)
        else:
            self.mem_buffer.add_buffer(
                images,
                states,
                step_idx_list,
                stage_indices=stage_indices,
            )
        self.step_idx += len(images)

    @staticmethod
    def _array_digest(value) -> str:
        array = np.asarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @classmethod
    def _memory_digest(cls, *values) -> str:
        digest = hashlib.sha256()
        for value in values:
            digest.update(bytes.fromhex(cls._array_digest(value)))
        return digest.hexdigest()

    def _record_final_memory_tensor(self, value) -> None:
        """Audit the true feature-encoder output consumed by the Modulator."""
        if self._pending_selector_trace is None:
            raise RuntimeError("Cannot record a final memory tensor without a selector trace")
        array = np.asarray(value)
        expected_shape = (
            1,
            int(self.config.budget),
            int(self.config.memory_token_dim),
        )
        if array.shape != expected_shape:
            raise RuntimeError(
                f"Final perceptual-memory tensor shape mismatch: {array.shape} != {expected_shape}"
            )
        is_floating = bool(jnp.issubdtype(array.dtype, jnp.floating))
        if not is_floating:
            raise RuntimeError(
                f"Final perceptual-memory tensor must be floating, got {array.dtype}"
            )
        is_finite = bool(np.isfinite(array).all())
        if not is_finite:
            raise RuntimeError("Final perceptual-memory tensor contains non-finite values")
        self._pending_selector_trace.update(
            {
                "final_memory_tensor_shape": list(array.shape),
                "final_memory_tensor_dtype": str(array.dtype),
                "final_memory_tensor_is_floating": is_floating,
                "final_memory_tensor_finite": is_finite,
                "final_memory_tensor_sha256": self._array_digest(array),
            }
        )

    def _prepare_experiment_frame_sampling(
        self,
        history_feats_gather_fn,
        token_budget,
        token_per_image,
    ):
        config = self._keyframe_selector_config
        arm = config["arm"]
        call_index = self._selector_call_index
        if call_index >= MAX_POLICY_CALLS:
            raise RuntimeError(
                f"Policy call {call_index} exceeds preregistered seed table 0..{MAX_POLICY_CALLS - 1}"
            )
        boundary_lookup_started = time.monotonic()
        visible_boundaries = self.mem_buffer.get_boundary_indices(self.step_idx)
        boundary_lookup_latency_ms = (
            time.monotonic() - boundary_lookup_started
        ) * 1000
        selector_seed = config["random_seeds"][call_index] if arm is SelectorArm.RANDOM else None
        selector_metadata: dict[str, object] = {}
        if arm is SelectorArm.OFFICIAL_UNIFORM:
            # This is the literal released runtime path, without an override.
            prepared = self.mem_buffer.prepare_frame_sampling(
                self.step_idx,
                token_budget,
                token_per_image,
                history_feats_gather_fn,
            )
        else:
            def selector(step_idx, selected_budget, selected_token_per_image):
                if selected_budget != 512 or selected_token_per_image != 16:
                    raise RuntimeError("Runtime frame budget changed after selector configuration")
                if arm in {
                    SelectorArm.ORACLE_NEIGHBORHOOD_3,
                    SelectorArm.ORACLE_NEIGHBORHOOD_5,
                }:
                    selected_indices, decision = oracle_neighborhood_coverage_decision(
                        step_idx,
                        visible_boundaries,
                        neighborhood_frames=(
                            3 if arm is SelectorArm.ORACLE_NEIGHBORHOOD_3 else 5
                        ),
                    )
                    selector_metadata.update(decision)
                    return selected_indices
                return select_indices(
                    arm,
                    step_idx,
                    boundary_indices=visible_boundaries,
                    random_seed=selector_seed,
                )

            with self.mem_buffer.temporary_frame_sampling_selector(selector):
                prepared = self.mem_buffer.prepare_frame_sampling(
                    self.step_idx,
                    token_budget,
                    token_per_image,
                    history_feats_gather_fn,
                )

        selector_decision_latency_ms = (
            self.mem_buffer._last_frame_sampling_selector_latency_ms
        )
        if selector_decision_latency_ms is None:  # pragma: no cover - defensive
            raise RuntimeError("MemoryBuffer did not record selector-only latency")
        bookkeeping_started = time.monotonic()
        selected = list(self.mem_buffer._last_frame_sampling_indices)
        image_emb, pos_emb, state_emb, mask = prepared
        mask_array = np.asarray(mask)
        expected_valid_tokens = 16 * len(selected)
        if mask_array.shape != (512,):
            raise RuntimeError(
                f"Perceptual-memory mask shape mismatch: {mask_array.shape} != (512,)"
            )
        if mask_array.dtype != np.bool_:
            raise RuntimeError(
                f"Perceptual-memory mask must be boolean, got {mask_array.dtype}"
            )
        if (
            int(mask_array.sum()) != expected_valid_tokens
            or not bool(mask_array[:expected_valid_tokens].all())
            or not bool((~mask_array[expected_valid_tokens:]).all())
        ):
            raise RuntimeError(
                "Perceptual-memory mask must be one valid prefix followed by false padding"
            )
        selected_hash = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        mask_digest = self._array_digest(mask_array)
        ages = [self.step_idx - index for index in selected]
        gaps = [right - left for left, right in zip(selected, selected[1:], strict=False)]
        selected_boundaries = len(set(selected) & set(visible_boundaries))
        boundary_recall = (
            selected_boundaries / len(visible_boundaries) if visible_boundaries else 1.0
        )
        self._pending_selector_trace = {
            "schema_version": 1,
            "task": config["task"],
            "episode_id": config["episode_id"],
            "selector_name": arm.value,
            "selector_seed": selector_seed,
            "seed_table_sha256": config["seed_table_sha256"],
            "seed_table_scope": config["seed_table_scope"],
            "seed_table_dataset": config["seed_table_dataset"],
            "policy_call_index": call_index,
            "history_length": self.step_idx + 1,
            "current_history_index": self.step_idx,
            "selected_frame_indices": selected,
            "selected_indices_sha256": selected_hash,
            "visible_boundary_indices": visible_boundaries,
            "valid_frame_count": len(selected),
            "padding_frame_count": 32 - len(selected),
            "valid_memory_token_count": int(mask_array.sum()),
            "mask_shape": list(mask_array.shape),
            "mask_dtype": str(mask_array.dtype),
            "mask_valid_prefix_all_true": bool(
                mask_array[:expected_valid_tokens].all()
            ),
            "mask_padding_all_false": bool(
                (~mask_array[expected_valid_tokens:]).all()
            ),
            "mask_sha256": mask_digest,
            "image_tensor_shape": list(np.shape(image_emb)),
            "image_tensor_dtype": str(np.asarray(image_emb).dtype),
            "image_tensor_sha256": self._array_digest(image_emb),
            "position_tensor_shape": list(np.shape(pos_emb)),
            "position_tensor_dtype": str(np.asarray(pos_emb).dtype),
            "position_tensor_sha256": self._array_digest(pos_emb),
            "state_tensor_shape": list(np.shape(state_emb)),
            "state_tensor_dtype": str(np.asarray(state_emb).dtype),
            "state_tensor_sha256": self._array_digest(state_emb),
            "prepared_memory_component_shapes": [
                list(np.shape(image_emb)),
                list(np.shape(pos_emb)),
                list(np.shape(state_emb)),
                list(np.shape(mask)),
            ],
            "prepared_memory_components_sha256": self._memory_digest(
                image_emb, pos_emb, state_emb, mask
            ),
            "age_distribution": ages,
            "maximum_temporal_gap": max(gaps, default=0),
            "boundary_recall": boundary_recall,
            **selector_metadata,
        }
        selector_bookkeeping_latency_ms = (
            time.monotonic() - bookkeeping_started
        ) * 1000
        self._pending_selector_trace.update(
            {
                "boundary_lookup_latency_ms": boundary_lookup_latency_ms,
                "selector_decision_latency_ms": selector_decision_latency_ms,
                "selector_bookkeeping_latency_ms": selector_bookkeeping_latency_ms,
                "selector_latency_ms": boundary_lookup_latency_ms
                + selector_decision_latency_ms
                + selector_bookkeeping_latency_ms,
            }
        )
        self._selector_call_index += 1
        return prepared

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs
        
        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = \
                self.mem_buffer.prepare_token_recurrent(
                    self.step_idx, self.exec_start_idx, history_feats_gather_fn)
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget
            
            if self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_token_dropping(
                        self.step_idx, token_budget, history_feats_gather_fn)
            else:
                token_per_image = self.config.token_per_image
                if self._keyframe_selector_config is None:
                    static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                        self.mem_buffer.prepare_frame_sampling(
                            self.step_idx, token_budget, token_per_image, history_feats_gather_fn)
                else:
                    static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                        self._prepare_experiment_frame_sampling(
                            history_feats_gather_fn, token_budget, token_per_image)
            
            inputs["static_image_emb"] = static_image_emb
            inputs["static_pos_emb"] = static_pos_emb
            normalized_static_state = self._normalize_state(static_state_emb)
            inputs["static_state_emb"] = normalized_static_state
            inputs["static_mask"] = static_mask
            if self._pending_selector_trace is not None:
                bookkeeping_started = time.monotonic()
                prepared_memory_input = np.concatenate(
                    [static_image_emb, static_pos_emb, normalized_static_state], axis=-1
                )
                self._pending_selector_trace.update(
                    {
                        "state_tensor_shape": list(np.shape(normalized_static_state)),
                        "state_tensor_sha256": self._array_digest(normalized_static_state),
                        "prepared_memory_input_shape": list(
                            np.shape(prepared_memory_input)
                        ),
                        "prepared_memory_input_dtype": str(
                            np.asarray(prepared_memory_input).dtype
                        ),
                        "prepared_memory_input_sha256": self._array_digest(
                            prepared_memory_input
                        ),
                        "prepared_memory_components_sha256": self._memory_digest(
                            static_image_emb,
                            static_pos_emb,
                            normalized_static_state,
                            static_mask,
                        ),
                    }
                )
                extra_bookkeeping_latency_ms = (
                    time.monotonic() - bookkeeping_started
                ) * 1000
                self._pending_selector_trace["selector_bookkeeping_latency_ms"] += (
                    extra_bookkeeping_latency_ms
                )
                self._pending_selector_trace["selector_latency_ms"] += (
                    extra_bookkeeping_latency_ms
                )
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")
        
    
        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
