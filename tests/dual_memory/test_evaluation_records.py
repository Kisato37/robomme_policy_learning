import json

import pytest

from examples.robomme.evaluation_records import EpisodeResultWriter
from examples.robomme.utils import EpisodeState, check_args


def _record(episode_id=0):
    return {
        "model_id": "SP",
        "training_seed": 42,
        "symbolic_source": "oracle",
        "task": "InsertPeg",
        "episode_id": episode_id,
        "success": True,
    }


def test_episode_results_are_append_only_and_unique(tmp_path):
    writer = EpisodeResultWriter(tmp_path)
    writer.append_episode(_record())
    writer.append_episode(_record(1))
    records = [
        json.loads(line)
        for line in (tmp_path / "evaluation/per_episode.jsonl").read_text().splitlines()
    ]
    assert [record["episode_id"] for record in records] == [0, 1]
    with pytest.raises(RuntimeError, match="duplicate"):
        writer.append_episode(_record())


def test_infra_failures_are_separate_from_scientific_results(tmp_path):
    writer = EpisodeResultWriter(tmp_path)
    writer.append_infra_failure({"task": "InsertPeg", "error": "socket closed"})
    assert not writer.results_path.exists()
    assert "socket closed" in writer.infra_path.read_text()


def test_segment_reset_does_not_reset_episode_history_counter():
    state = EpisodeState()
    state.total_history_frames_sent = 17
    state.image_buffer.append(object())
    state.state_buffer.append(object())
    state.clear_buffers()
    assert state.total_history_frames_sent == 17


def test_symbolic_source_is_inferred_without_ambiguous_logging():
    class Args:
        subgoal_type = "grounded_subgoal"
        obs_horizon = 16
        use_memer = False
        use_oracle = True
        use_qwenvl = False
        use_gemini = False
        symbolic_source = "none"

    args = Args()
    check_args(args)
    assert args.symbolic_source == "oracle"
