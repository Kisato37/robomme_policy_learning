"""Strict, opt-in 32 -> 48 frame inference configuration (no weight updates)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json


EXPANSION_FAMILY = "uniform_keyframe_expansion-v1"
RELEASED_HISTORY_CONFIG = {
    "budget": 512,
    "num_views": 1,
    "token_per_image": 16,
    "streaming_obs_horizon": 16,
    "pool_type": "mean",
    "use_pos_emb": True,
    "use_state_emb": False,
    "memory_feature": {
        "img": {"net": "identity", "input_dim": 2048},
        "pos": {"input_dim": 768, "hidden_dim": 768},
        "state": {"input_dim": 8, "hidden_dim": 512},
    },
    "integration_type": "modulation",
    "memory_token_dim": 1024,
    "representation_type": "perceptual",
    "perceptual_memory": {"type": "frame_sampling"},
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def payload_digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def expanded_history_mapping(source: dict) -> tuple[dict, dict]:
    """Copy the exact released mapping; change only memory token budget."""
    if canonical_json(source) != canonical_json(RELEASED_HISTORY_CONFIG):
        raise ValueError("Expansion requires the exact released FrameSamp+Modulator history configuration")
    expanded = deepcopy(source)
    expanded["budget"] = 768
    evidence = {
        "family": EXPANSION_FAMILY,
        "source_history_config": deepcopy(source),
        "effective_history_config": deepcopy(expanded),
        "source_history_config_sha256": payload_digest(source),
        "effective_history_config_sha256": payload_digest(expanded),
        "changed_fields": {"budget": {"from": 512, "to": 768}},
        "base_frame_capacity": 32,
        "frame_capacity": 48,
        "tokens_per_frame": 16,
        "strict_weight_tree_load": True,
        "rope_implementation_changed": False,
        "rope_query_offset_changes_with_memory_length": True,
    }
    return expanded, evidence


def validate_expanded_history_mapping(config: dict) -> None:
    expected, _ = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
    if canonical_json(config) != canonical_json(expected):
        raise ValueError("Expanded runtime history configuration must exactly match the frozen 48/768 contract")
