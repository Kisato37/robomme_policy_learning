"""No simulator, GPU, server, mutable run artifact, or global RNG required."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from experiments.uniform_keyframe_expansion import contract as c
from mme_vla_suite.shared.uniform_keyframe_expansion import derive_expansion_seed


def test_formal_matrix_preserves_canonical_parent_population():
    from experiments.keyframe_oracle_sampling.formal_matrix import build_formal_matrix as parent
    from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix as neighborhood

    matrix = c.build_formal_matrix()
    assert matrix["trajectory_count"] == len(matrix["rows"]) == 2400
    assert matrix["tasks"] == parent()["tasks"] == neighborhood()["tasks"]
    assert matrix["episode_ids"] == parent()["episode_ids"] == neighborhood()["episode_ids"] == list(range(50))
    assert Counter(row["arm"] for row in matrix["rows"]) == {"U": 800, "UK48": 800, "UN48": 800}
    assert all(row["dataset"] == "test" and row["max_steps"] == 1300 for row in matrix["rows"])
    assert [row["row_id"] for row in matrix["rows"]] == list(range(2400))
    assert len({(r["task"], r["episode_id"], r["arm"]) for r in matrix["rows"]}) == 2400
    c.validate_matrix(matrix, "formal")
    for row in matrix["rows"]:
        assert c.validate_row(row, "formal") == row

    # Existing population contracts may not silently inherit the new arms.
    assert parent()["trajectory_count"] == 3200
    assert parent()["arms"] == ["U", "O", "OC", "R"]
    assert neighborhood()["trajectory_count"] == 1600
    assert neighborhood()["arms"] == ["OC3", "OC5"]


def test_smoke_is_48_short_and_16_terminal_val_only():
    matrix = c.build_smoke_matrix()
    assert matrix["trajectory_count"] == len(matrix["rows"]) == 64
    assert matrix["short_stop_on_official_terminal"] is True
    assert Counter(r["trajectory_kind"] for r in matrix["rows"]) == {"short": 48, "terminal": 16}
    for index, task in enumerate(c.FORMAL_TASKS):
        rows = matrix["rows"][index * 4:index * 4 + 4]
        assert [r["task"] for r in rows] == [task] * 4
        assert [r["arm"] for r in rows] == ["U", "UK48", "UN48", c.FORMAL_ARMS[index % 3]]
        assert [r["max_steps"] for r in rows] == [64, 64, 64, 1300]
        for row in rows:
            assert row["episode_id"] == 0 and row["dataset"] == "val"
            assert c.validate_row(row, "smoke") == row
    assert len({(r["task"], r["episode_id"], r["arm"], r["trajectory_kind"]) for r in matrix["rows"]}) == 64
    c.validate_matrix(matrix, "smoke")
    formal_contexts = {(r["dataset"], r["task"], r["episode_id"]) for r in c.build_formal_matrix()["rows"]}
    smoke_contexts = {(r["dataset"], r["task"], r["episode_id"]) for r in matrix["rows"]}
    assert formal_contexts.isdisjoint(smoke_contexts)


@pytest.mark.parametrize("stage", ["smoke", "formal"])
@pytest.mark.parametrize("mutation", ["extra_field", "bool_id", "missing_row", "duplicate_row", "wrong_order", "wrong_budget"])
def test_matrix_rejects_any_semantic_mutation(stage, mutation):
    matrix = c.build_formal_matrix() if stage == "formal" else c.build_smoke_matrix()
    if mutation == "extra_field":
        matrix["approved"] = True
    elif mutation == "bool_id":
        matrix["rows"][0]["episode_id"] = False
    elif mutation == "missing_row":
        matrix["rows"].pop()
    elif mutation == "duplicate_row":
        matrix["rows"][1] = deepcopy(matrix["rows"][0])
    elif mutation == "wrong_order":
        matrix["rows"].reverse()
    elif mutation == "wrong_budget":
        matrix["rows"][0]["max_steps"] = 1301
    with pytest.raises(c.ExpansionContractError):
        c.validate_matrix(matrix, stage)


@pytest.mark.parametrize("field,value", [
    ("row_id", False), ("row_id", 0.0), ("row_id", -1), ("row_id", 2400),
    ("episode_id", False), ("episode_id", 50), ("task", "UnknownTask"),
    ("arm", "OC"), ("dataset", "val"), ("trajectory_kind", "terminal"),
    ("max_steps", 64), ("policy_seed", 7),
])
def test_row_binding_rejects_invalid_or_noncanonical_fields(field, value):
    row = c.build_formal_matrix()["rows"][0]
    row[field] = value
    with pytest.raises(c.ExpansionContractError):
        c.validate_row(row)


def test_selector_config_is_independent_reproducible_and_exact():
    rows = c.build_formal_matrix()["rows"]
    u, uk, un = (c.build_selector_config(rows[index]) for index in range(3))
    assert u == {"arm": "U", "split": "test", "task": "BinFill", "episode_id": 0}
    assert uk["arm"] == "UK48" and un["arm"] == "UN48"
    assert uk["random_seeds"] == un["random_seeds"]
    assert len(un["random_seeds"]) == 82
    payload = [2026091001, "uniform_keyframe_expansion-v1", "test", "BinFill", 0, 0, "UN48"]
    compact = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    assert un["random_seeds"][0] == int.from_bytes(hashlib.sha256(compact).digest()[:8], "big")
    assert un["random_seeds"][81] == derive_expansion_seed("test", "BinFill", 0, 81)
    assert un["seed_table_sha256"] == hashlib.sha256(json.dumps(un["random_seeds"], separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert un == c.build_selector_config(rows[2])
    c.validate_selector_config(u, rows[0])
    c.validate_selector_config(un, rows[2])
    for mutation in ("seed", "digest", "arm"):
        bad = deepcopy(un)
        if mutation == "seed":
            bad["random_seeds"][0] += 1
            bad["seed_table_sha256"] = c.canonical_sha256(bad["random_seeds"])
        elif mutation == "digest":
            bad["seed_table_sha256"] = "f" * 64
        else:
            bad["arm"] = "UK48"
        with pytest.raises(c.ExpansionContractError):
            c.validate_selector_config(bad, rows[2])


def test_seed_manifests_are_split_disjoint_unique_and_hash_bound():
    formal = c.build_seed_manifest("formal")
    smoke = c.build_seed_manifest("smoke")
    assert formal["context_count"] == 800
    assert smoke["context_count"] == 16  # Short/full smoke intentionally share context.
    formal_seeds = [seed for record in formal["records"] for seed in record["random_seeds"]]
    smoke_seeds = [seed for record in smoke["records"] for seed in record["random_seeds"]]
    assert len(set(formal_seeds)) == len(formal_seeds) == 65600
    assert len(set(smoke_seeds)) == len(smoke_seeds) == 1312
    assert set(formal_seeds).isdisjoint(smoke_seeds)
    for manifest in (formal, smoke):
        raw = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        assert c.canonical_sha256(raw) == manifest["manifest_sha256"]


def test_frozen_inference_keeps_weights_horizons_and_padding():
    expected = c.frozen_inference_contract()
    assert (expected["base_frame_capacity"], expected["base_token_capacity"]) == (32, 512)
    assert (expected["frame_capacity"], expected["token_capacity"], expected["tokens_per_frame"]) == (48, 768, 16)
    assert (expected["policy_seed"], expected["action_horizon"], expected["executed_action_horizon"], expected["max_steps"]) == (7, 20, 16, 1300)
    assert expected["checkpoint_archive_sha256"] == "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
    assert expected["training_enabled"] is False and expected["extra_uniform_fill_enabled"] is False
    assert expected["padding_values"] == "zero" and expected["padding_mask"] is False
    c.validate_inference_contract(expected)
    for key, value in (("base_frame_capacity", 48), ("token_capacity", 512), ("policy_seed", 8),
                       ("checkpoint_id", 80000), ("training_enabled", True), ("padding_mask", True)):
        bad = {**expected, key: value}
        with pytest.raises(c.ExpansionContractError):
            c.validate_inference_contract(bad)


@pytest.fixture(scope="module")
def readiness_template():
    return c.build_readiness_template()


def _attested_manifest(template):
    """Synthetic structural fixture only; this is never a run authorization."""
    evidence = deepcopy(template)
    evidence.update({
        "explicit_formal_user_approval": True, "protocol_frozen": True, "git_clean": True,
        "code_commit": "a" * 40, "protocol_sha256": "b" * 64,
        "environment_manifest_sha256": "c" * 64,
        "cpu_gate": {"status": "passed", "report_sha256": "d" * 64},
        "gpu_smoke_gate": {"status": "passed", "report_sha256": "e" * 64},
    })
    return evidence


def test_readiness_defaults_blocked_and_is_not_a_launch_authorization(readiness_template):
    assert readiness_template["fresh_u_arm_in_formal_matrix"] is True
    assert readiness_template["same_run_initial_pairing_required"] is True
    assert readiness_template["explicit_formal_user_approval"] is False
    assert readiness_template["readiness_scope"] == "planning-manifest-only-not-launch-authorization"
    with pytest.raises(c.ExpansionContractError):
        c.validate_formal_readiness(readiness_template)
    # Returns None, not a token/permission/job ID.  External artifact verification
    # and stage-specific human authority remain requirements for a future runner.
    assert c.validate_formal_readiness(_attested_manifest(readiness_template)) is None


@pytest.mark.parametrize("field", ["explicit_formal_user_approval", "protocol_frozen", "git_clean"])
@pytest.mark.parametrize("value", [False, 1, "true"])
def test_formal_needs_exact_stage_flags_not_truthy_values(readiness_template, field, value):
    evidence = _attested_manifest(readiness_template)
    evidence[field] = value
    with pytest.raises(c.ExpansionContractError):
        c.validate_formal_readiness(evidence)


@pytest.mark.parametrize("gate", ["cpu_gate", "gpu_smoke_gate"])
def test_formal_needs_gate_evidence_not_only_a_pass_flag(readiness_template, gate):
    evidence = _attested_manifest(readiness_template)
    evidence[gate] = {"status": "passed", "report_sha256": None}
    with pytest.raises(c.ExpansionContractError):
        c.validate_formal_readiness(evidence)


@pytest.mark.parametrize("field", ["fresh_u_arm_in_formal_matrix", "same_run_initial_pairing_required"])
def test_formal_requires_fresh_same_run_u_controls(readiness_template, field):
    evidence = _attested_manifest(readiness_template)
    evidence[field] = False
    with pytest.raises(c.ExpansionContractError):
        c.validate_formal_readiness(evidence)


@pytest.mark.parametrize("field", ["formal_matrix_sha256", "formal_seed_manifest_sha256"])
def test_readiness_bound_to_exact_matrix_and_seed_manifest(readiness_template, field):
    evidence = _attested_manifest(readiness_template)
    evidence[field] = "0" * 64
    with pytest.raises(c.ExpansionContractError):
        c.validate_formal_readiness(evidence)


def test_contract_rejects_unknown_stage_and_nonfinite_json():
    with pytest.raises(c.ExpansionContractError):
        c.build_seed_manifest("train")
    with pytest.raises(c.ExpansionContractError):
        c.canonical_json({"value": float("nan")})
