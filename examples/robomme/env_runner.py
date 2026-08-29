"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.
"""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any
import numpy as np

from robomme.robomme_env import *  # noqa: F401, F403 - env registration
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

from utils import TASK_NAME_LIST
from causal_stage_instrumentation import (
    install_current_task_index_instrumentation,
    validate_aligned_stages,
)

np.set_printoptions(precision=4, suppress=True)


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)

class EnvRunner:
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API, and optional subgoal oracles.
    """

    def __init__(
        self,
        env_id: str,
        video_save_dir: str,
        max_steps: int = 1300,
        dataset: str = "test",
        require_current_task_index: bool = False,
    ) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        self.env_id = env_id
        self.video_save_dir = video_save_dir

        resolved_dataset = "val" if dataset == "validation" else dataset
        if resolved_dataset not in {"train", "test", "val"}:
            raise ValueError(f"Unsupported benchmark dataset: {dataset}")
        self.dataset = resolved_dataset
        self.require_current_task_index = require_current_task_index
        if require_current_task_index:
            # Install before make_env/reset so generated conditioning
            # demonstrations are labeled frame-by-frame before batching.
            install_current_task_index_instrumentation(DemonstrationWrapper)
        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset=resolved_dataset,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )

        # Set after make_env()
        self.env: Any = None
        self.episode_id: int | None = None
        self.task_goal: str = ""
        self.current_task_index: int | None = None
        self.initial_condition_hashes: dict[str, str] | None = None
        self.resolved_environment_seed: int | None = None
        self.resolved_difficulty_hint: str | None = None

    @staticmethod
    def _digest_value(value: Any) -> str:
        digest = hashlib.sha256()

        def update(item: Any) -> None:
            if hasattr(item, "detach") and hasattr(item, "cpu"):
                item = item.detach().cpu().numpy()
            if isinstance(item, np.ndarray):
                array = np.ascontiguousarray(item)
                digest.update(b"array:")
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(json.dumps(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            elif isinstance(item, dict):
                digest.update(b"dict{")
                for key in sorted(item, key=str):
                    update(str(key))
                    update(item[key])
                digest.update(b"}")
            elif isinstance(item, (list, tuple)):
                digest.update(b"list[")
                for child in item:
                    update(child)
                digest.update(b"]")
            elif isinstance(item, np.generic):
                update(item.item())
            elif isinstance(item, (str, int, float, bool)) or item is None:
                digest.update(repr(item).encode("utf-8"))
            else:
                raise TypeError(f"Cannot hash initial task-state value {type(item).__name__}")

        update(value)
        return digest.hexdigest()

    _HIGHLIGHT_ACTOR_NAME = re.compile(r"^highlight_disk_\d+_(\d+)$")

    @classmethod
    def _canonicalize_task_state_for_hashing(
        cls,
        value: Any,
        path: tuple[str, ...] = (),
    ) -> Any:
        """Remove only the runtime object address from highlight actor names.

        RoboMME names visual-only highlight actors with ``id(obj)``, which is a
        process-local memory address rather than simulator state.  The actor's
        complete state remains in the digest under a stable, value-sorted name.
        This also preserves multiple highlighted actors without relying on
        their process-local addresses.  Collisions with ordinary actor names
        fail closed instead of silently dropping or merging state.
        """
        if isinstance(value, dict):
            canonical: dict[Any, Any] = {}
            highlight_actors: list[tuple[int, str, Any]] = []
            for key, child in value.items():
                canonical_child = cls._canonicalize_task_state_for_hashing(
                    child,
                    path + (str(key),),
                )
                if path == ("actors",) and isinstance(key, str):
                    match = cls._HIGHLIGHT_ACTOR_NAME.fullmatch(key)
                    if match is not None:
                        highlight_actors.append(
                            (
                                int(match.group(1)),
                                cls._digest_value(canonical_child),
                                canonical_child,
                            )
                        )
                        continue
                if key in canonical:
                    raise RuntimeError(
                        "Initial task-state canonicalization produced a duplicate "
                        f"key at {'/'.join(path) or '<root>'}: {key}"
                    )
                canonical[key] = canonical_child
            for ordinal, (suffix, _digest, child) in enumerate(
                sorted(highlight_actors, key=lambda entry: (entry[0], entry[1]))
            ):
                canonical_key = f"highlight_disk_<runtime-id-{ordinal}>_{suffix}"
                if canonical_key in canonical:
                    raise RuntimeError(
                        "Initial task-state canonicalization produced a duplicate "
                        f"key at {'/'.join(path) or '<root>'}: {canonical_key}"
                    )
                canonical[canonical_key] = child
            return canonical
        if isinstance(value, list):
            return [
                cls._canonicalize_task_state_for_hashing(
                    child,
                    path + (str(index),),
                )
                for index, child in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                cls._canonicalize_task_state_for_hashing(
                    child,
                    path + (str(index),),
                )
                for index, child in enumerate(value)
            )
        return value

    def _aligned_current_task_indices(self, obs: dict[str, Any]) -> list[int] | None:
        stages = obs.get("current_task_index")
        frame_count = len(obs["front_rgb_list"])
        if stages is None:
            if self.require_current_task_index:
                raise RuntimeError(
                    "Global blocker: the live benchmark returned front frames without "
                    "aligned current_task_index metadata"
                )
            return None
        normalized = validate_aligned_stages(obs["front_rgb_list"], stages)
        if len(normalized) != frame_count:  # pragma: no cover - defensive
            raise RuntimeError("Global blocker: stage alignment validation changed length")
        return normalized

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        """Build and set the active env for the given episode."""
        if self.require_current_task_index:
            seed, difficulty_hint = self.env_builder.resolve_episode(episode_id)
            self.resolved_environment_seed = seed
            self.resolved_difficulty_hint = difficulty_hint
        self.env = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty

    def get_init_obs(self) -> dict[str, Any]:
        """Reset env and return initial observation dict (images, wrist_images, states, task_goal)."""
        obs, self.info = self.env.reset()
        if isinstance(self.info["task_goal"], list):
            self.task_goal = self.info["task_goal"][0]
        else:
            self.task_goal = self.info["task_goal"]
        images = obs["front_rgb_list"]
        wrist_images = obs["wrist_rgb_list"]
        current_task_indices = self._aligned_current_task_indices(obs)
        states = [pack_state(joint_state, gripper_state) for joint_state, gripper_state in 
                  zip(obs["joint_state_list"], obs["gripper_state_list"])]
        if self.require_current_task_index:
            task_state_getter = getattr(self.env.unwrapped, "get_state_dict", None)
            if not callable(task_state_getter):
                raise RuntimeError(
                    "Global blocker: live benchmark cannot expose reset task state for fairness hashing"
                )
            task_state = self._canonicalize_task_state_for_hashing(
                task_state_getter()
            )
            self.initial_condition_hashes = {
                "front_observations_sha256": self._digest_value(images),
                "wrist_observations_sha256": self._digest_value(wrist_images),
                "robot_states_sha256": self._digest_value(states),
                "task_state_sha256": self._digest_value(task_state),
                "task_instruction_sha256": self._digest_value(self.task_goal),
            }
        else:
            self.initial_condition_hashes = None

        return {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "current_task_indices": current_task_indices,
            "task_goal": self.task_goal,
        }

    def step(self, action: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool, str]:
        """
        Execute one step.
        Returns ( (img, wrist_img, state), stop_flag, success_flag ).
        success_flag is one of "success", "fail", "timeout", "unknown".
        """
        try:
            obs, _, terminated, truncated, self.info = self.env.step(action)
        except Exception as exc:
            if self.require_current_task_index:
                # A Python exception is not an official benchmark "error"
                # outcome. Preserve it for the experiment failure classifier.
                raise
            # Preserve the released evaluator behavior outside this experiment.
            print(f"Error: {exc}")
            return (None, None, None), True, "error"

        outcome = self.info.get("status", "unknown")
        stop = terminated or truncated
        if stop and outcome == "error":
            # FailAwareWrapper intentionally returns ``obs=None`` for a caught
            # benchmark execution error.  Preserve that official outcome; the
            # evaluator must terminate without trying to consume a new frame.
            return (None, None, None), True, outcome
        if obs is None:
            raise RuntimeError(
                "Benchmark returned obs=None without an official error terminal"
            )

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        joint_state = obs["joint_state_list"][-1]
        gripper_state = obs["gripper_state_list"][-1]
        state = pack_state(joint_state, gripper_state)
        current_task_indices = self._aligned_current_task_indices(obs)
        self.current_task_index = (
            current_task_indices[-1] if current_task_indices is not None else None
        )

        return (img, wrist_img, state), stop, outcome
    
    @property
    def simple_subgoal_oracle(self) -> str:
        return self.info["simple_subgoal_online"]
    
    @property
    def grounded_subgoal_oracle(self) -> str:
        return self.info["grounded_subgoal_online"]
    
    def close_env(self) -> None:
        """Close and clear the current env."""
        if self.env is not None:
            self.env.close()
            del self.env
            self.env = None
            self.episode_id = None
            self.task_goal = None
            self.current_task_index = None
            self.initial_condition_hashes = None
            self.resolved_environment_seed = None
            self.resolved_difficulty_hint = None
