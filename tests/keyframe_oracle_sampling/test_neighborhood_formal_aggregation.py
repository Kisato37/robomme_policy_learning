from __future__ import annotations

import hashlib
from itertools import pairwise
import json
from pathlib import Path

import pytest

from experiments.keyframe_neighborhood_sampling import aggregate_formal
from experiments.keyframe_neighborhood_sampling.aggregate_formal import AGGREGATE_FILENAMES
from experiments.keyframe_neighborhood_sampling.aggregate_formal import REFERENCE_COMPLETENESS_SHA256
from experiments.keyframe_neighborhood_sampling.aggregate_formal import REFERENCE_CROSS_RUN_VERIFICATION
from experiments.keyframe_neighborhood_sampling.aggregate_formal import REFERENCE_RUN_ID
from experiments.keyframe_neighborhood_sampling.aggregate_formal import REFERENCE_SUMMARY_SHA256
from experiments.keyframe_neighborhood_sampling.aggregate_formal import SELECTOR_SEED_TABLE_ROLE
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _empty_secondary_diagnostic_accumulator
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _expected_formal_rows
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _finalize_secondary_diagnostics
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _merge_secondary_diagnostics
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _repository_provenance
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _validate_reference_alignment
from experiments.keyframe_neighborhood_sampling.aggregate_formal import audit_extension_attempt
from experiments.keyframe_neighborhood_sampling.aggregate_formal import build_aggregate_payloads
from experiments.keyframe_neighborhood_sampling.aggregate_formal import build_per_episode_rows
from experiments.keyframe_neighborhood_sampling.aggregate_formal import build_per_task_rows
from experiments.keyframe_neighborhood_sampling.aggregate_formal import publish_aggregate_directory
from experiments.keyframe_neighborhood_sampling.analysis import load_published_reference_oc
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_MAX_STEPS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_CHECKPOINT_ID
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_CHECKPOINT_PATH
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import released_prepared_component_dtypes
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import oracle_neighborhood_coverage_decision

REPO = Path(__file__).resolve().parents[2]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _extension_attempt(
    tmp_path,
    *,
    arm: str,
    wrong_diagnostic: bool = False,
    wrong_replay_diagnostic: str | None = None,
    dense_final_boundaries: bool = False,
):
    task = "InsertPeg"
    episode_id = 0
    key = ScientificKey(task, episode_id, arm, "formal")
    seed_payload = build_seed_table([task], [episode_id])
    launch = {
        "protocol_version": "v1.0",
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "dataset": "test",
        "trajectory_count": FORMAL_TRAJECTORY_COUNT,
        "max_steps": FORMAL_MAX_STEPS,
        "executed_action_horizon": 16,
        "evaluation_policy_seed": 7,
        "checkpoint_path": SMOKE_CHECKPOINT_PATH,
        "seed_table_scope": FORMAL_SEED_SCOPE,
        "seed_table_dataset": FORMAL_SEED_DATASET,
        "seed_table_entries_sha256": seed_payload["entries_sha256"],
    }
    row = {
        "row_id": 0,
        "task": task,
        "episode_id": episode_id,
        "arm": arm,
        "trajectory_kind": "formal",
        "max_steps": FORMAL_MAX_STEPS,
        "dataset": "test",
    }
    fixed = {
        "dataset": "test",
        "max_steps": FORMAL_MAX_STEPS,
        "executed_action_horizon": 16,
        "evaluation_policy_seed": 7,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
    }
    writer = EpisodeAttemptWriter(tmp_path / f"{arm}_attempt", key, 0)
    writer.create(
        {
            **fixed,
            "protocol_version": "v1.0",
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "seed_table_sha256": seed_payload["entries_sha256"],
            "resolved_environment_seed": 123,
            "resolved_difficulty_hint": "fixture",
            "difficulty": "fixture",
        }
    )
    writer.record_initial_conditions(
        {
            "front_observations_sha256": _sha("front"),
            "wrist_observations_sha256": _sha("wrist"),
            "robot_states_sha256": _sha("robot"),
            "task_state_sha256": _sha("task-state"),
            "task_instruction_sha256": _sha("instruction"),
        }
    )

    policy_latency = []
    model_latency = []
    history_lengths = []
    for call_index in range(4):
        current_index = call_index * 16
        boundaries = ([0], [0, 8], [0, 8, 24], [0, 8, 24, 40])[call_index]
        if dense_final_boundaries and call_index == 3:
            boundaries = list(range(0, 46, 5))
        requested = 3 if arm == "OC3" else 5
        selected, decision = oracle_neighborhood_coverage_decision(
            current_index, boundaries, neighborhood_frames=requested
        )
        if wrong_diagnostic and call_index == 2:
            decision = {**decision, "effective_neighborhood_frames": 99}
        ages = [current_index - index for index in selected]
        gaps = [right - left for left, right in pairwise(selected)]
        replay_diagnostics = {
            "age_distribution": ages,
            "maximum_temporal_gap": max(gaps, default=0),
            "boundary_recall": len(set(selected).intersection(boundaries)) / len(boundaries),
        }
        if wrong_replay_diagnostic is not None and call_index == 2:
            wrong_values = {
                "age_distribution": [999],
                "maximum_temporal_gap": 999,
                "boundary_recall": 0.123,
            }
            replay_diagnostics[wrong_replay_diagnostic] = wrong_values[wrong_replay_diagnostic]
        dtypes = released_prepared_component_dtypes(len(selected))
        end_to_end = 2.0 + call_index
        model = 1.0 + call_index
        writer.append_trace(
            {
                "schema_version": 1,
                "task": task,
                "episode_id": episode_id,
                "selector_name": arm,
                "selector_seed": None,
                "seed_table_sha256": seed_payload["entries_sha256"],
                "seed_table_scope": FORMAL_SEED_SCOPE,
                "seed_table_dataset": FORMAL_SEED_DATASET,
                "policy_call_index": call_index,
                "environment_step": call_index * 16,
                "history_length": current_index + 1,
                "current_history_index": current_index,
                "selected_frame_indices": selected,
                "selected_indices_sha256": hashlib.sha256(
                    json.dumps(selected, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "visible_boundary_indices": list(boundaries),
                **decision,
                **replay_diagnostics,
                "valid_frame_count": len(selected),
                "padding_frame_count": 32 - len(selected),
                "valid_memory_token_count": 16 * len(selected),
                "mask_shape": [512],
                "mask_dtype": "bool",
                "mask_valid_prefix_all_true": True,
                "mask_padding_all_false": True,
                "mask_sha256": _sha(f"mask-{call_index}"),
                "image_tensor_dtype": dtypes[0],
                "image_tensor_sha256": _sha(f"image-{call_index}"),
                "position_tensor_dtype": dtypes[1],
                "position_tensor_sha256": _sha(f"position-{call_index}"),
                "state_tensor_dtype": dtypes[2],
                "state_tensor_sha256": _sha(f"state-{call_index}"),
                "prepared_memory_component_shapes": [
                    [512, 2048],
                    [512, 768],
                    [512, 8],
                    [512],
                ],
                "prepared_memory_components_sha256": _sha(f"components-{call_index}"),
                "prepared_memory_input_shape": [512, 2824],
                "prepared_memory_input_sha256": _sha(f"input-{call_index}"),
                "final_memory_tensor_shape": [1, 512, 1024],
                "final_memory_tensor_dtype": "bfloat16",
                "final_memory_tensor_is_floating": True,
                "final_memory_tensor_finite": True,
                "final_memory_tensor_sha256": _sha(f"memory-{call_index}"),
                "boundary_lookup_latency_ms": 0.1,
                "selector_decision_latency_ms": 0.2,
                "selector_bookkeeping_latency_ms": 0.3,
                "selector_latency_ms": 0.6,
                "model_latency_ms": model,
                "end_to_end_request_latency_ms": end_to_end,
            }
        )
        policy_latency.append(end_to_end)
        model_latency.append(model)
        history_lengths.append(current_index + 1)
    writer.finalize(
        {
            **fixed,
            "task": task,
            "episode_id": episode_id,
            "selector_arm": arm,
            "steps": 49,
            "success": False,
            "terminal_reason": "fail",
            "timeout": False,
            "collision": False,
            "policy_latency_ms": policy_latency,
            "policy_model_latency_ms": model_latency,
            "history_lengths_at_policy_calls": history_lengths,
        }
    )
    return writer, key, row, launch, seed_payload


@pytest.mark.parametrize("arm", EXTENSION_ARMS)
def test_extension_attempt_replays_neighborhood_selector_and_diagnostics(tmp_path, arm):
    writer, key, row, launch, seed_payload = _extension_attempt(tmp_path, arm=arm)
    report, latencies, _, _ = audit_extension_attempt(
        writer,
        expected_key=key,
        expected_row=row,
        launch=launch,
        seed_payload=seed_payload,
    )
    assert report["strict_extension_contract"] is True
    assert report["policy_call_count"] == 4
    assert latencies["selector"] == [0.6] * 4
    secondary = report["secondary_diagnostics"]
    assert secondary["episode_count"] == 1
    assert secondary["policy_call_count"] == 4
    assert secondary["boundary_recall_numerator"] == secondary["boundary_recall_denominator"]
    assert secondary["final_observed_boundary_count_values"] == [4]


def test_extension_attempt_rejects_falsified_fallback_metadata(tmp_path):
    fixture = _extension_attempt(tmp_path, arm="OC5", wrong_diagnostic=True)
    with pytest.raises(ArtifactContractError, match="wrong selector diagnostic"):
        audit_extension_attempt(
            fixture[0],
            expected_key=fixture[1],
            expected_row=fixture[2],
            launch=fixture[3],
            seed_payload=fixture[4],
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("age_distribution", "age distribution was not replay-derived"),
        ("maximum_temporal_gap", "maximum temporal gap was not replay-derived"),
        ("boundary_recall", "boundary recall was not replay-derived"),
    ],
)
def test_extension_attempt_rejects_falsified_replay_diagnostics(tmp_path, field, message):
    fixture = _extension_attempt(tmp_path, arm="OC5", wrong_replay_diagnostic=field)
    with pytest.raises(ArtifactContractError, match=message):
        audit_extension_attempt(
            fixture[0],
            expected_key=fixture[1],
            expected_row=fixture[2],
            launch=fixture[3],
            seed_payload=fixture[4],
        )


def test_secondary_diagnostics_aggregate_replay_sufficient_statistics(tmp_path):
    accumulator = _empty_secondary_diagnostic_accumulator()
    for arm in EXTENSION_ARMS:
        writer, key, row, launch, seed_payload = _extension_attempt(tmp_path, arm=arm)
        report, _, _, _ = audit_extension_attempt(
            writer,
            expected_key=key,
            expected_row=row,
            launch=launch,
            seed_payload=seed_payload,
        )
        _merge_secondary_diagnostics(accumulator, report["secondary_diagnostics"])

    diagnostics = _finalize_secondary_diagnostics(accumulator)
    assert diagnostics["episode_count"] == 2
    assert diagnostics["policy_call_count"] == 8
    assert diagnostics["boundary_recall"] == {
        "numerator": 20,
        "denominator": 20,
        "rate": 1.0,
    }
    assert diagnostics["requested_neighborhood_candidate_retention"]["rate"] == 1.0
    assert diagnostics["effective_neighborhood_candidate_retention"]["rate"] == 1.0
    assert diagnostics["memory_age"]["count"] > 0
    assert diagnostics["selected_index_spacing"]["count"] > 0
    assert diagnostics["maximum_temporal_gap_per_policy_call"]["count"] == 8
    assert diagnostics["final_observed_boundary_count_progress_proxy"]["mean"] == 4.0


def test_oc5_requested_and_effective_retention_distinguish_wholesale_fallback(tmp_path):
    writer, key, row, launch, seed_payload = _extension_attempt(
        tmp_path,
        arm="OC5",
        dense_final_boundaries=True,
    )
    report, _, _, _ = audit_extension_attempt(
        writer,
        expected_key=key,
        expected_row=row,
        launch=launch,
        seed_payload=seed_payload,
    )
    accumulator = _empty_secondary_diagnostic_accumulator()
    _merge_secondary_diagnostics(accumulator, report["secondary_diagnostics"])
    diagnostics = _finalize_secondary_diagnostics(accumulator)
    requested = diagnostics["requested_neighborhood_candidate_retention"]
    effective = diagnostics["effective_neighborhood_candidate_retention"]
    assert requested["rate"] < 1.0
    assert effective["rate"] == 1.0
    assert diagnostics["final_observed_boundary_count_progress_proxy"]["mean"] == 10.0


def _outcome_records():
    records = {}
    reference = []
    for task in FORMAL_TASKS:
        for episode_id in FORMAL_EPISODE_IDS:
            reference.append(
                {
                    "task": task,
                    "episode_id": episode_id,
                    "arm": "OC",
                    "success": episode_id < 20,
                }
            )
            for arm, cutoff in (("OC3", 25), ("OC5", 30)):
                key = ScientificKey(task, episode_id, arm, "formal")
                success = episode_id < cutoff
                records[key] = {
                    "scientific_key": key.as_dict(),
                    "dataset": "test",
                    "success": success,
                    "terminal_reason": "success" if success else "fail",
                    "steps": 20 + episode_id,
                    "collision": False,
                    "attempt_id": 0,
                }
    return records, reference


def _secondary_diagnostics_fixture():
    accumulator = _empty_secondary_diagnostic_accumulator()
    _merge_secondary_diagnostics(
        accumulator,
        {
            "episode_count": 1,
            "policy_call_count": 2,
            "boundary_recall_numerator": 3,
            "boundary_recall_denominator": 4,
            "requested_candidate_retention_numerator": 5,
            "requested_candidate_retention_denominator": 6,
            "effective_candidate_retention_numerator": 5,
            "effective_candidate_retention_denominator": 5,
            "memory_age_values": [0, 4, 8],
            "selected_index_gap_values": [2, 6],
            "maximum_temporal_gap_values": [0, 6],
            "final_observed_boundary_count_values": [3],
        },
    )
    diagnostics = _finalize_secondary_diagnostics(accumulator)
    return {
        "scope": "exploratory_secondary_diagnostics",
        "definitions": {},
        "limitations": [
            "Final observed boundary count is an exploratory within-task progress proxy, not true stage completion."
        ],
        "by_arm": dict.fromkeys(EXTENSION_ARMS, diagnostics),
        "by_task": {
            task: dict.fromkeys(EXTENSION_ARMS, diagnostics) for task in FORMAL_TASKS
        },
    }


def test_extension_tables_are_exact_800_pairs_and_16_tasks():
    records, reference = _outcome_records()
    assert len(_expected_formal_rows(build_formal_matrix())) == 1600
    per_episode = build_per_episode_rows(records, reference)
    per_task = build_per_task_rows(records, reference)
    assert len(per_episode) == 800
    assert len(per_task) == 16
    assert per_task[0]["OC_success_count"] == 20
    assert per_task[0]["OC3_minus_OC_pp"] == pytest.approx(10.0)
    assert per_task[0]["OC5_minus_OC3_pp"] == pytest.approx(10.0)


def test_extension_payload_builder_allows_small_replicates_only_for_tests():
    records, reference = _outcome_records()
    completeness = {
        "passed": True,
        "audited_utc": "fixture",
        "protocol_provenance": {"fixture": True},
        "result_set_sha256": _sha("results"),
        "selector_trace_set_sha256": _sha("traces"),
        "infrastructure_failure_count": 0,
        "protocol_deviations": [],
        "unresolved_risks": [],
        "latency_by_arm": {},
        "selector_diagnostics_by_arm": {},
        "secondary_diagnostics": _secondary_diagnostics_fixture(),
        "outcome_diagnostics": {},
    }
    payloads, summary = build_aggregate_payloads(
        completeness,
        records,
        reference,
        bootstrap_replicates=100,
        randomization_replicates=100,
        allow_nonconfirmatory_replicate_override=True,
    )
    assert tuple(payloads) == AGGREGATE_FILENAMES
    assert summary["extension_cell_count"] == 1600
    assert summary["analysis_status"] == "complete_nonconfirmatory_test_override"
    assert summary["experiment_type"] == "paired_post_hoc_follow_up"
    assert summary["arm_outcomes"]["OC"]["pooled_success_count"] == 320
    assert summary["arm_outcomes"]["OC3"]["pooled_success_count"] == 400
    assert summary["arm_outcomes"]["OC5"]["pooled_success_count"] == 480
    assert len(
        summary["resampling_reproducibility"]["bootstrap"][
            "common_schedule_sha256"
        ]
    ) == 64
    rendered = payloads["analysis.md"].decode("utf-8")
    assert "paired post-hoc follow-up" in rendered
    assert "RNG implementation: numpy.random.Generator(PCG64)" in rendered
    assert "Bootstrap common-schedule SHA-256:" in rendered
    assert "Randomization common-schedule SHA-256:" in rendered
    assert "Exploratory selector and progress diagnostics" in rendered
    assert "3/4 (75.000%)" in rendered
    assert "not true stage completion" in rendered
    assert "| OC | 320 | 800 | 40.000% | 40.000% |" in rendered
    assert "| OC3 | 400 | 800 | 50.000% | 50.000% |" in rendered
    assert "| OC5 | 480 | 800 | 60.000% | 60.000% |" in rendered


def test_extension_aggregate_directory_is_atomic_and_write_once(tmp_path):
    payloads = {filename: f"{filename}\n".encode() for filename in AGGREGATE_FILENAMES}
    target = publish_aggregate_directory(tmp_path, payloads)
    assert sorted(path.name for path in target.iterdir()) == sorted(AGGREGATE_FILENAMES)
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_aggregate_directory(tmp_path, payloads)


def test_reference_oc_loader_rejects_changed_bytes(tmp_path):
    path = tmp_path / "per_episode.csv"
    rows = ["task,episode_id,OC_success"]
    for task in FORMAL_TASKS:
        rows.extend(f"{task},{episode_id},1" for episode_id in FORMAL_EPISODE_IDS)
    path.write_text("\n".join(rows) + "\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(load_published_reference_oc(path, expected_sha256=expected)) == 800
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ArtifactContractError, match="digest mismatch"):
        load_published_reference_oc(path, expected_sha256=expected)


def test_reference_alignment_discloses_selector_seed_and_cross_run_limits():
    summary_path = REPO / aggregate_formal.REFERENCE_RESULTS_RELATIVE / "aggregate/summary.json"
    reference_provenance = json.loads(summary_path.read_text())["artifact_provenance"]
    launch = {
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_results_relative": str(aggregate_formal.REFERENCE_RESULTS_RELATIVE),
        "reference_per_episode_sha256": aggregate_formal.REFERENCE_PER_EPISODE_SHA256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA256,
        "reference_completeness_sha256": REFERENCE_COMPLETENESS_SHA256,
        "selector_seed_table_role": SELECTOR_SEED_TABLE_ROLE,
        "reference_cross_run_verification": REFERENCE_CROSS_RUN_VERIFICATION,
        "checkpoint_unpacked_metadata_sha256": reference_provenance[
            "checkpoint_unpacked_metadata_sha256"
        ],
        "checkpoint_content_tree_algorithm": reference_provenance[
            "checkpoint_content_tree_algorithm"
        ],
        "checkpoint_unpacked_content_tree_sha256": reference_provenance[
            "checkpoint_unpacked_content_tree_sha256"
        ],
    }
    alignment = _validate_reference_alignment(
        REPO,
        launch=launch,
        seed_entries_sha256=reference_provenance["formal_seed_table_entries_sha256"],
    )
    assert alignment["selector_seed_table"]["is_environment_seed_or_initial_state_evidence"] is False
    assert alignment["cross_run_pairing"]["environment_seed_directly_verified"] is False
    assert alignment["cross_run_pairing"]["difficulty_directly_verified"] is False
    assert alignment["cross_run_pairing"]["raw_initial_condition_hashes_directly_verified"] is False


def test_aggregation_requires_exact_clean_formal_commit_and_frozen_sources(tmp_path, monkeypatch):
    analysis_path = tmp_path / "experiments/keyframe_neighborhood_sampling/analysis.py"
    aggregator_path = tmp_path / "experiments/keyframe_neighborhood_sampling/aggregate_formal.py"
    analysis_path.parent.mkdir(parents=True)
    analysis_path.write_text("# analysis\n")
    aggregator_path.write_text("# aggregator\n")
    launch = {
        "frozen_analysis_source_relative": "experiments/keyframe_neighborhood_sampling/analysis.py",
        "frozen_analysis_source_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "frozen_aggregator_source_relative": "experiments/keyframe_neighborhood_sampling/aggregate_formal.py",
        "frozen_aggregator_source_sha256": hashlib.sha256(aggregator_path.read_bytes()).hexdigest(),
    }

    state = {"commit": "formal-commit", "status": ""}

    def fake_check_output(command, **_kwargs):
        return f"{state['commit']}\n" if command[1:3] == ["rev-parse", "HEAD"] else state["status"]

    monkeypatch.setattr(aggregate_formal.subprocess, "check_output", fake_check_output)
    provenance = _repository_provenance(
        tmp_path,
        formal_commit="formal-commit",
        launch=launch,
    )
    assert provenance["formal_commit_exact_match"] is True
    assert provenance["aggregation_worktree_clean"] is True
    assert provenance["frozen_source_hashes_match"] is True

    state["commit"] = "descendant-commit"
    with pytest.raises(ArtifactContractError, match="exactly equal"):
        _repository_provenance(tmp_path, formal_commit="formal-commit", launch=launch)
    state.update(commit="formal-commit", status=" M analysis.py\n")
    with pytest.raises(ArtifactContractError, match="clean committed worktree"):
        _repository_provenance(tmp_path, formal_commit="formal-commit", launch=launch)
