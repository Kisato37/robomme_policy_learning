import json

import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

from examples.robomme.subgoal_prediction.qwenvl.qwen_cache import (
    ContentAddressedQwenCache,
    append_jsonl,
    hash_array,
)


def test_cache_key_changes_for_every_required_input(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"weights")
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "model.bin").write_bytes(b"base-a")
    cache = ContentAddressedQwenCache(tmp_path / "cache", checkpoint, base_model)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    base = {
        "current_image": image,
        "task_instruction": "insert peg",
        "subgoal_history": ["grasp"],
        "prompt_version": "v1",
        "generation": {"temperature": 0},
        "video_hash": None,
    }
    key, payload = cache.key(**base)
    assert payload["current_front_image_sha256"] == hash_array(image)
    assert payload["predictor_checkpoint_sha256"]
    assert payload["base_model_checkpoint_sha256"]

    changes = [
        {"current_image": np.ones_like(image)},
        {"task_instruction": "move cube"},
        {"subgoal_history": ["align"]},
        {"prompt_version": "v2"},
        {"generation": {"temperature": 0, "max_tokens": 128}},
        {"video_hash": "different"},
    ]
    for change in changes:
        candidate = {**base, **change}
        candidate_key, _ = cache.key(**candidate)
        assert candidate_key != key

    other_base = tmp_path / "other_base"
    other_base.mkdir()
    (other_base / "model.bin").write_bytes(b"base-b")
    other_cache = ContentAddressedQwenCache(tmp_path / "other_cache", checkpoint, other_base)
    other_key, _ = other_cache.key(**base)
    assert other_key != key


def test_cache_is_write_once_and_detects_non_determinism(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    cache = ContentAddressedQwenCache(tmp_path / "cache", checkpoint)
    cache.put("abc", {"raw_response": "first"})
    cache.put("abc", {"raw_response": "first"})
    assert cache.get("abc") == {"raw_response": "first"}
    with pytest.raises(RuntimeError, match="non-deterministic"):
        cache.put("abc", {"raw_response": "second"})


def test_jsonl_call_log_appends_complete_records(tmp_path):
    path = tmp_path / "calls.jsonl"
    append_jsonl(path, {"cache_hit": False, "output_tokens": 10})
    append_jsonl(path, {"cache_hit": True, "output_tokens": 10})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["cache_hit"] for record in records] == [False, True]


def test_concurrent_identical_cache_writes_are_serialized(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    cache = ContentAddressedQwenCache(tmp_path / "cache", checkpoint)
    value = {"raw_response": "same deterministic response"}
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: cache.put("shared", value), range(32)))
    assert cache.get("shared") == value
