from __future__ import annotations

from pathlib import Path

import pytest

from experiments.keyframe_oracle_sampling.prepare_smoke import (
    CHECKPOINT_CONTENT_TREE_ALGORITHM,
    checkpoint_content_tree_identity,
)


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
