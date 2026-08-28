from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.keyframe_oracle_sampling.architecture_smoke import (
    EXPECTED_CHECKPOINT_METADATA_SHA256 as ARCHITECTURE_CHECKPOINT_METADATA_SHA256,
)
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_CONTENT_TREE_ALGORITHM
from experiments.keyframe_oracle_sampling.prepare_smoke import (
    EXPECTED_CHECKPOINT_METADATA_SHA256 as PREPARE_CHECKPOINT_METADATA_SHA256,
)
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity

FROZEN_CHECKPOINT_INVENTORY = (
    ("_CHECKPOINT_METADATA", 327),
    ("assets/robomme/norm_stats.json", 2_073),
    ("params/_METADATA", 28_136),
    ("params/_sharding", 18_332),
    ("params/array_metadatas/process_0", 10_863),
    ("params/d/752ca7a506429a869bf6564d00065319", 27_253),
    ("params/manifest.ocdbt", 120),
    ("params/ocdbt.process_0/d/066523fb731284524447207c4a356a67", 5_853_184),
    ("params/ocdbt.process_0/d/2590f713aa3b79ba96fb14011406cef2", 643_678_208),
    ("params/ocdbt.process_0/d/4d08fa7ea220fea3766d5c8823716949", 2_239_651_840),
    ("params/ocdbt.process_0/d/5878cd22780c25ad2b068813135140a1", 27_238),
    ("params/ocdbt.process_0/d/63f926a0b36250293f84e95963778283", 2_236_559_360),
    ("params/ocdbt.process_0/d/8798615c3cf67a4c7386bd0e54f559e4", 1_276),
    ("params/ocdbt.process_0/d/b6117379ff617c0709a34e86a1c11d9f", 2_239_844_352),
    ("params/ocdbt.process_0/d/b8a280a4ed11b10130bced1c73d043ec", 220),
    ("params/ocdbt.process_0/d/da7b3993fa08df1f808653e316c007e6", 2_271_801_344),
    ("params/ocdbt.process_0/d/dd618c57506684fd63822bc0394634ba", 2_239_647_744),
    ("params/ocdbt.process_0/manifest.ocdbt", 368),
)


def test_checkpoint_metadata_digest_matches_frozen_inventory():
    entries = [{"path": path, "size": size} for path, size in FROZEN_CHECKPOINT_INVENTORY]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    assert digest == PREPARE_CHECKPOINT_METADATA_SHA256
    assert digest == ARCHITECTURE_CHECKPOINT_METADATA_SHA256


def test_content_tree_digest_is_deterministic_and_binds_file_bytes(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"abcd")
    (checkpoint / "nested" / "config.json").write_bytes(b"{}")

    first = checkpoint_content_tree_identity(checkpoint)
    repeat = checkpoint_content_tree_identity(checkpoint)
    assert repeat == first
    assert first["algorithm"] == CHECKPOINT_CONTENT_TREE_ALGORITHM
    assert first["file_count"] == 2
    assert first["total_bytes"] == 6

    # Preserve path and size while changing actual bytes.  A path-and-size-only
    # identity cannot detect this mutation, but the content-tree gate must.
    (checkpoint / "weights.bin").write_bytes(b"abce")
    changed = checkpoint_content_tree_identity(checkpoint)
    assert changed["file_count"] == first["file_count"]
    assert changed["total_bytes"] == first["total_bytes"]
    assert changed["content_tree_sha256"] != first["content_tree_sha256"]


def test_content_tree_digest_binds_relative_paths(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    (second / "nested").mkdir(parents=True)
    (first / "weights.bin").write_bytes(b"same")
    (second / "nested" / "weights.bin").write_bytes(b"same")

    assert (
        checkpoint_content_tree_identity(first)["content_tree_sha256"]
        != checkpoint_content_tree_identity(second)["content_tree_sha256"]
    )


def test_content_tree_digest_rejects_empty_tree_and_symlinks(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="empty"):
        checkpoint_content_tree_identity(empty)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    target = checkpoint / "target.bin"
    target.write_bytes(b"weights")
    (checkpoint / "alias.bin").symlink_to(target)
    with pytest.raises(RuntimeError, match="symbolic link"):
        checkpoint_content_tree_identity(checkpoint)


def test_large_content_hash_is_not_invoked_by_each_smoke_row():
    repo = Path(__file__).resolve().parents[2]
    prepare = (repo / "experiments/keyframe_oracle_sampling/prepare_smoke.py").read_text()
    architecture = (
        repo / "experiments/keyframe_oracle_sampling/architecture_smoke.py"
    ).read_text()
    submit = (
        repo / "experiments/keyframe_oracle_sampling/submit_smoke.py"
    ).read_text()
    row_preflight = (
        repo / "experiments/keyframe_oracle_sampling/preflight_smoke_row.py"
    ).read_text()
    row_launcher = (
        repo / "experiments/keyframe_oracle_sampling/run_smoke.sbatch"
    ).read_text()

    assert "checkpoint_content_tree_identity(checkpoint_dir)" in prepare
    assert "_checkpoint_content_tree_identity(checkpoint)" in architecture
    assert "checkpoint_content_tree_identity(REPO / CHECKPOINT_RELATIVE)" in submit
    assert "checkpoint_content_tree_identity" not in row_preflight
    assert "checkpoint_content_tree_identity" not in row_launcher
