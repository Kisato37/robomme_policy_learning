"""Live, read-only ``current_task_index`` observation instrumentation.

RoboMME's reset demonstration is assembled from the same per-step augmentation
method as online observations.  Wrapping that method is the only point where a
stage value can be captured for each exact frame before the reset batch is
concatenated, filtered, and its info dictionary flattened.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _scalar_stage(value: Any) -> int:
    if value is None:
        raise RuntimeError(
            "Global blocker: live current_task_index is unavailable for a returned front-view frame"
        )
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError("Global blocker: live current_task_index must be scalar per frame")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, np.integer)):
        raise RuntimeError(
            f"Global blocker: live current_task_index must be integer, got {scalar!r}"
        )
    return int(scalar)


def instrument_augment_method(original):
    """Return a wrapper that attaches the exact live stage to one augmented frame."""
    if getattr(original, "_keyframe_stage_instrumented", False):
        return original

    def augmented_with_stage(self, obs, info, action):
        new_obs, new_info = original(self, obs, info, action)
        stage = _scalar_stage(getattr(self.unwrapped, "current_task_index", None))
        new_obs = dict(new_obs)
        new_info = dict(new_info)
        new_obs["current_task_index"] = stage
        new_info["current_task_index"] = stage
        return new_obs, new_info

    augmented_with_stage._keyframe_stage_instrumented = True
    augmented_with_stage._keyframe_original = original
    return augmented_with_stage


def install_current_task_index_instrumentation(demonstration_wrapper_class) -> None:
    """Idempotently instrument the benchmark class before an environment reset."""
    current = demonstration_wrapper_class._augment_obs_and_info
    demonstration_wrapper_class._augment_obs_and_info = instrument_augment_method(current)


def validate_aligned_stages(front_frames: Sequence[Any], stages: Any) -> list[int]:
    if stages is None:
        raise RuntimeError(
            "Global blocker: the live benchmark returned front frames without "
            "aligned current_task_index metadata"
        )
    if not isinstance(stages, (list, tuple, np.ndarray)):
        stages = [stages]
    if len(stages) != len(front_frames):
        raise RuntimeError(
            "Global blocker: live current_task_index/front-frame alignment mismatch: "
            f"{len(stages)} labels for {len(front_frames)} frames"
        )
    return [_scalar_stage(value) for value in stages]
