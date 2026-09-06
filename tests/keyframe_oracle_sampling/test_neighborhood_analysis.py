from __future__ import annotations

import numpy as np
import pytest

from experiments.keyframe_neighborhood_sampling.analysis import AnalysisProtocolError
from experiments.keyframe_neighborhood_sampling.analysis import _common_bootstrap
from experiments.keyframe_neighborhood_sampling.analysis import _common_randomization
from experiments.keyframe_neighborhood_sampling.analysis import build_extension_analysis
from experiments.keyframe_neighborhood_sampling.analysis import validate_combined_records
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError


def _fixture_records():
    reference = []
    extension = []
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
            extension.extend(
                [
                    {
                        "task": task,
                        "episode_id": episode_id,
                        "arm": arm,
                        "trajectory_kind": "formal",
                        "dataset": "test",
                        "success": episode_id < (25 if arm == "OC3" else 30),
                    }
                    for arm in EXTENSION_ARMS
                ]
            )
    return extension, reference


def test_combined_records_require_exact_1600_plus_800_censuses():
    extension, reference = _fixture_records()
    table = validate_combined_records(extension, reference)
    assert table.success("BinFill", 0, "OC5") is True
    assert table.success("RouteStick", 49, "OC") is False
    with pytest.raises(ArtifactContractError, match="1,600-cell census"):
        validate_combined_records(extension[:-1], reference)


@pytest.mark.parametrize("missing_field", ["trajectory_kind", "dataset"])
def test_extension_records_require_explicit_formal_test_metadata(missing_field):
    extension, reference = _fixture_records()
    extension[0] = {key: value for key, value in extension[0].items() if key != missing_field}
    with pytest.raises(ArtifactContractError, match=missing_field):
        validate_combined_records(extension, reference)


def test_extension_analysis_uses_three_paired_comparisons_and_common_draws():
    extension, reference = _fixture_records()
    report = build_extension_analysis(
        extension,
        reference,
        bootstrap_replicates=200,
        randomization_replicates=200,
        allow_nonconfirmatory_replicate_override=True,
    )
    assert set(report["comparisons"]) == {"OC3 - OC", "OC5 - OC", "OC5 - OC3"}
    assert report["comparisons"]["OC3 - OC"]["estimate_pp"] == pytest.approx(10.0)
    assert report["comparisons"]["OC5 - OC"]["estimate_pp"] == pytest.approx(20.0)
    assert report["comparisons"]["OC5 - OC3"]["estimate_pp"] == pytest.approx(10.0)
    assert report["holm_family"]["family_size"] == 3
    assert report["experiment_type"] == "paired_post_hoc_follow_up"
    assert report["post_hoc_follow_up"] is True
    assert report["analysis_plan_frozen_before_new_trajectories"] is True
    assert report["arm_outcomes"]["OC"]["pooled_success_count"] == 320
    assert report["arm_outcomes"]["OC3"]["pooled_success_count"] == 400
    assert report["arm_outcomes"]["OC5"]["pooled_success_count"] == 480
    assert report["arm_outcomes"]["OC5"]["pooled_episode_count"] == 800
    assert report["arm_outcomes"]["OC5"]["pooled_success_rate"] == pytest.approx(0.6)
    bootstrap_digests = set()
    randomization_digests = set()
    for result in report["comparisons"].values():
        interval = result["confidence_interval_95"]
        randomization = result["randomization_test"]
        assert interval["common_resampling_schedule_across_comparisons"] is True
        assert randomization["common_randomization_schedule_across_comparisons"] is True
        assert interval["quantile_method"] == "numpy-linear"
        assert interval["percentile_bounds"] == [2.5, 97.5]
        assert interval["rng_implementation"] == "numpy.random.Generator(PCG64)"
        assert randomization["rng_implementation"] == "numpy.random.Generator(PCG64)"
        assert len(interval["common_resampling_schedule_sha256"]) == 64
        assert len(randomization["common_randomization_schedule_sha256"]) == 64
        bootstrap_digests.add(interval["common_resampling_schedule_sha256"])
        randomization_digests.add(randomization["common_randomization_schedule_sha256"])
        assert randomization["holm_adjusted_p_value"] >= randomization["p_value"]
    assert len(bootstrap_digests) == 1
    assert len(randomization_digests) == 1
    assert report["resampling_reproducibility"]["bootstrap"][
        "common_schedule_sha256"
    ] == next(iter(bootstrap_digests))
    assert report["resampling_reproducibility"]["randomization"][
        "common_schedule_sha256"
    ] == next(iter(randomization_digests))


def test_common_resampling_is_deterministic_and_mapping_order_independent():
    first = np.tile(np.asarray([-1.0, 0.0, 1.0, 1.0, 0.0]), (16, 10))
    second = -first
    forward = {"first": first, "second": second}
    reverse = {"second": second, "first": first}

    bootstrap_forward = _common_bootstrap(forward, replicates=137, seed=17)
    bootstrap_reverse = _common_bootstrap(reverse, replicates=137, seed=17)
    randomization_forward = _common_randomization(forward, replicates=137, seed=17)
    randomization_reverse = _common_randomization(reverse, replicates=137, seed=17)

    for name in forward:
        assert bootstrap_forward[name] == bootstrap_reverse[name]
        assert randomization_forward[name] == randomization_reverse[name]
    assert bootstrap_forward["first"]["common_resampling_schedule_sha256"] == (
        "8a63726fa6057fcf487e988ad439d46838701afed9fe6132988974a6703d1954"
    )
    assert randomization_forward["first"][
        "common_randomization_schedule_sha256"
    ] == "da7ea3009e59e65449ce074a41ccba33f46d63e6f49b914e62fd05a2dffb245d"
    assert bootstrap_forward["first"]["lower"] == pytest.approx(0.1525)
    assert bootstrap_forward["first"]["upper"] == pytest.approx(0.25575)


def test_formal_replicate_counts_are_fail_closed():
    extension, reference = _fixture_records()
    with pytest.raises(AnalysisProtocolError, match="100,000"):
        build_extension_analysis(
            extension,
            reference,
            bootstrap_replicates=10,
            randomization_replicates=10,
        )
