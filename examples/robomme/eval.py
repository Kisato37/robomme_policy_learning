import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Tuple

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import (
    pack_buffer,
    check_args,
    TASK_NAME_LIST,
    TASK_WITH_VIDEO_DEMO,
    SUBGOAL_TYPES,
    EpisodeState,
)
from utils import RolloutRecorder
from env_runner import EnvRunner
from subgoal_predictor import build_subgoal_predictor, SubgoalPredictorBase
from evaluation_records import EpisodeResultWriter

from experiments.keyframe_oracle_sampling.artifacts import (
    PROTOCOL_VERSION,
    RunArtifactStore,
    ScientificKey,
    load_seed_table,
    is_retryable_infrastructure_exception,
    validate_prepared_smoke_root,
)
from experiments.keyframe_oracle_sampling.formal_artifacts import (
    validate_prepared_formal_root,
)
from experiments.keyframe_oracle_sampling.formal_matrix import (
    validate_formal_runtime_row_binding,
)
from experiments.keyframe_oracle_sampling.smoke_matrix import (
    validate_runtime_row_binding,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    MAX_POLICY_CALLS,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
    keyframe_timeout_reached,
    parse_arm,
)

# qwen3-vl environment variables
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'



@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    overwrite: bool = False

    use_history: bool = True
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000
    dataset: str = "test"

    # task control
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma
    episode_ids: str = "" # comma-separated explicit IDs; empty uses the full protocol

    # VLM subgoal predictor
    use_oracle: bool = False
    use_qwenvl: bool = False
    use_memer: bool = False
    use_gemini: bool = False
    subgoal_type: Optional[str] = None  # [simple_subgoal, grounded_subgoal]
    gemini_model_name: str = "gemini-2.5-pro"
    qwenvl_simpleSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
    qwenvl_groundSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
    qwenvl_base_model_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/Qwen3-VL-4B-Instruct"
    memer_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
    subgoal_keep_period: int = 1 # ever subgoal should be kept for this many steps
    qwen_cache_dir: str = "runs/evaluation/qwen_cache"
    dual_memory_run_root: str = ""
    model_id: str = ""
    training_seed: int = -1
    symbolic_source: str = "none"
    evaluation_scope: str = "unspecified"
    # Causal keyframe selector experiment. Empty arm disables all experiment
    # instrumentation and preserves the released evaluation behavior.
    keyframe_selector_arm: str = ""
    keyframe_seed_table: str = ""
    keyframe_run_root: str = ""
    keyframe_attempt_id: int = 0
    keyframe_trajectory_kind: str = "formal"
    keyframe_formal_authorization: str = ""
    # this can accelerate the evaluation process for symbolic memory
    # In our experiments, we just set this to 1



def validate_keyframe_args(args: Args) -> None:
    if not args.keyframe_selector_arm and not args.keyframe_run_root:
        return
    if not args.keyframe_selector_arm or not args.keyframe_run_root:
        raise ValueError("Keyframe selector arm and run root must be configured together")
    parse_arm(args.keyframe_selector_arm)
    frozen = {
        "obs_horizon": 16,
        "model_seed": 7,
        "model_ckpt_id": 79999,
        "use_history": True,
        "overwrite": False,
    }
    observed = {name: getattr(args, name) for name in frozen}
    if observed != frozen:
        raise ValueError(f"Frozen keyframe evaluation arguments changed: {observed} != {frozen}")
    if args.subgoal_type is not None or any(
        (args.use_oracle, args.use_qwenvl, args.use_memer, args.use_gemini)
    ):
        raise ValueError("Keyframe arms may not add symbolic prompts or subgoal predictors")
    expected_by_kind = {
        "short": ({"val", "validation"}, 64),
        "terminal": ({"val", "validation"}, 1300),
        "formal": ({"test"}, 1300),
    }
    try:
        allowed_datasets, expected_steps = expected_by_kind[args.keyframe_trajectory_kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown keyframe trajectory kind: {args.keyframe_trajectory_kind}"
        ) from exc
    if args.dataset not in allowed_datasets or args.max_steps != expected_steps:
        raise ValueError(
            f"{args.keyframe_trajectory_kind} requires dataset {sorted(allowed_datasets)} "
            f"and max_steps={expected_steps}"
        )
    if args.exclude_tasks or args.re_eval_tasks:
        raise ValueError("Keyframe runs forbid exclusion and outcome-conditioned re-evaluation")
    if args.keyframe_trajectory_kind == "formal":
        digest = args.keyframe_formal_authorization
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                "Direct formal keyframe evaluation is hard-disabled: the dedicated "
                "formal launcher must supply its recorded submission digest"
            )
    elif args.keyframe_formal_authorization:
        raise ValueError("Development smoke may not carry a formal authorization digest")


class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir
        self.attempt_writer = None
        self._seed_table_payload = None
        self._seed_lookup = None
        if args.keyframe_selector_arm:
            if not args.keyframe_seed_table:
                raise ValueError("A keyframe selector run requires --args.keyframe-seed-table")
            expected_scope, expected_dataset = (
                (FORMAL_SEED_SCOPE, FORMAL_SEED_DATASET)
                if args.keyframe_trajectory_kind == "formal"
                else (SMOKE_SEED_SCOPE, SMOKE_SEED_DATASET)
            )
            self._seed_table_payload, self._seed_lookup = load_seed_table(
                args.keyframe_seed_table,
                expected_scope=expected_scope,
                expected_dataset=expected_dataset,
            )

    def _selector_config(self, env_runner: EnvRunner) -> dict | None:
        if not self.args.keyframe_selector_arm:
            return None
        arm = parse_arm(self.args.keyframe_selector_arm)
        seeds = []
        for call_index in range(MAX_POLICY_CALLS):
            key = (env_runner.env_id, int(env_runner.episode_id), call_index)
            try:
                seeds.append(self._seed_lookup[key])
            except KeyError as exc:
                raise RuntimeError(f"Seed table has no preregistered entry for {key}") from exc
        return {
            "arm": arm.value,
            "task": env_runner.env_id,
            "episode_id": int(env_runner.episode_id),
            "random_seeds": seeds,
            "seed_table_sha256": self._seed_table_payload["entries_sha256"],
            "seed_table_scope": self._seed_table_payload["scope"],
            "seed_table_dataset": self._seed_table_payload["dataset"],
        }

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        subgoal_predictor: SubgoalPredictorBase,
        video_save_dir: Path,
        pre_traj: dict | None = None,
    ) -> str:
        client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.host, self.args.port
        )
        resp = client.reset(self._selector_config(env_runner))
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(
            env_runner, epstate, video_save_dir, pre_traj=pre_traj
        )
        subgoal_predictor.start_episode(epstate, env_runner)        

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None
        subgoal_sequence = []
        history_lengths = []
        policy_latencies_ms = []
        policy_model_latencies_ms = []

        while True:
            subgoal_predictor.step(epstate)

            if not epstate.action_plan:
                if epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
                    subgoal, has_api_error = subgoal_predictor.get_subgoal(
                        epstate.count,
                        subgoal,
                        last_subgoal,
                    )
                else:
                    subgoal = last_subgoal
                    has_api_error = False

                if has_api_error:
                    break

                action_chunk = self.get_action_chunk(
                    client, epstate, img, wrist_img, robot_state, prompt, subgoal, 
                    exec_horizon=self.args.obs_horizon
                )
                subgoal_sequence.append(subgoal)
                history_lengths.append(epstate.total_history_frames_sent)
                policy_latencies_ms.append(self._last_policy_latency_ms)
                policy_model_latencies_ms.append(self._last_policy_model_latency_ms)

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

                last_subgoal = subgoal

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            epstate.count += 1

            reached_step_limit = (
                keyframe_timeout_reached(
                    epstate.count,
                    self.args.max_steps,
                    stop_flag,
                )
                if self.args.keyframe_selector_arm
                else epstate.count > self.args.max_steps
            )
            if reached_step_limit:
                success_flag = "timeout"
                break

            # RoboMME's FailAwareWrapper represents a caught benchmark error as
            # an official terminal with ``obs=None``.  It is a scientific
            # outcome, not a transport exception, so stop before unpacking the
            # intentionally absent post-action observation.
            if stop_flag and success_flag == "error":
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(
                img,
                wrist_img,
                robot_state,
                current_task_index=(
                    env_runner.current_task_index
                    if self.args.keyframe_selector_arm
                    else None
                ),
            )
            recorder.record(
                image=img.copy(),
                wrist_image=wrist_img.copy(),
                state=robot_state.copy(),
                action=action.copy(),
                subgoal=subgoal,
            )

            if stop_flag:
                break

        if success_flag == "unknown":
            return "unknown"

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        subgoal_predictor.end_episode(epstate, success_flag)
        info = getattr(env_runner, "info", {}) or {}
        self.last_episode_record = {
            "task": env_runner.env_id,
            "episode_id": int(env_runner.episode_id),
            "success": success_flag == "success",
            "terminal_reason": success_flag,
            "terminal_state": str(info.get("status", success_flag)),
            "steps": int(epstate.count),
            "collision": bool(info.get("collision", False)),
            "timeout": success_flag == "timeout",
            "benchmark_error_message": (
                str(info.get("error_message"))
                if success_flag == "error" and info.get("error_message")
                else None
            ),
            "benchmark_exception_type": (
                str(info.get("exception_type"))
                if success_flag == "error" and info.get("exception_type")
                else None
            ),
            "subgoal_sequence": subgoal_sequence,
            "history_lengths_at_policy_calls": history_lengths,
            "policy_latency_ms": policy_latencies_ms,
            "policy_model_latency_ms": policy_model_latencies_ms,
            "video_path": str(video_save_dir / video_filename),
        }
        return success_flag


    def init_episode(
        self,
        env_runner: EnvRunner,
        epstate: EpisodeState,
        video_save_dir: Path,
        pre_traj: dict | None = None,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs() if pre_traj is None else pre_traj
        task_goal = pre_traj["task_goal"]

        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)

        print(f"task_goal: {task_goal}")

        stages = (
            pre_traj.get("current_task_indices")
            if self.args.keyframe_selector_arm
            else None
        )
        if self.args.keyframe_selector_arm and stages is None:
            raise RuntimeError(
                "Global blocker: reset demonstration has no aligned current_task_index values"
            )

        if self.args.keyframe_selector_arm:
            for image, wrist_image, state, stage in zip(
                pre_traj["images"],
                pre_traj["wrist_images"],
                pre_traj["states"],
                stages,
                strict=True,
            ):
                epstate.add_observation(
                    image,
                    wrist_image,
                    state,
                    current_task_index=stage,
                )
        else:
            # Preserve the released buffer initialization path exactly.
            epstate.image_buffer.extend(pre_traj["images"])
            epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
            epstate.state_buffer.extend(pre_traj["states"])

        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
                subgoal=None if self.args.subgoal_type is None else "[initializing...]",
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        print(f"exec_start_idx: {epstate.exec_start_idx}")
        return task_goal, recorder

    def get_action_chunk(
        self,
        client,
        state: EpisodeState,
        img: np.ndarray,
        wrist_img: np.ndarray,
        robot_state: np.ndarray,
        prompt: str,
        subgoal: Optional[str],
        exec_horizon: int,
    ) -> list:
        if self.args.use_history:
            segment_length = len(state.image_buffer)
            resp = client.add_buffer(pack_buffer(
                state.image_buffer,
                state.state_buffer,
                state.exec_start_idx,
                current_task_indices=(
                    state.current_task_index_buffer
                    if self.args.keyframe_selector_arm
                    else None
                ),
            ))
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)
            state.total_history_frames_sent += segment_length

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "prompt": prompt,
        }
        if self.args.keyframe_selector_arm:
            element["keyframe_environment_step"] = int(state.count)

        if subgoal is not None:
            element['simple_subgoal'] = subgoal
            element['grounded_subgoal'] = subgoal

        request_started = time.monotonic()
        response = client.infer(element)
        self._last_policy_latency_ms = (time.monotonic() - request_started) * 1000
        self._last_policy_model_latency_ms = float(response.get("infer_time_ms", float("nan")))
        selector_trace = response.get("selector_trace")
        if self.args.keyframe_selector_arm:
            if selector_trace is None:
                raise RuntimeError("Configured server response omitted selector trace")
            selector_trace = dict(selector_trace)
            selector_trace["end_to_end_request_latency_ms"] = self._last_policy_latency_ms
            if self.attempt_writer is None:
                raise RuntimeError("Selector trace has no active immutable attempt writer")
            self.attempt_writer.append_trace(selector_trace)
        action_chunk = response["actions"]
        if self.args.keyframe_selector_arm and len(action_chunk) != 20:
            raise RuntimeError(
                f"Frozen action proposal horizon changed: {len(action_chunk)} != 20"
            )
        return action_chunk[:exec_horizon]


def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = (
        Path(args.save_dir)
        / args.policy_name
        / f"ckpt{args.model_ckpt_id}"
        / f"seed{args.model_seed}"
    )

    if args.subgoal_type in SUBGOAL_TYPES:
        if args.use_gemini:
            save_dir = save_dir / "gemini"
        elif args.use_qwenvl:
            save_dir = save_dir / "qwenvl"
        elif args.use_memer:
            save_dir = save_dir / "memer"
        else:
            save_dir = save_dir / "oracle"

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json", "r") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json", "r") as f:
            log_dict = json.load(f)
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task_name in log_dict:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error":
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                os.system(f"rm -f {save_dir / 'videos' / f'{task_name}_ep*.mp4'}")

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def _keyframe_attempt_manifest(
    args: Args,
    evaluator: "EpisodeEvaluator",
    env_runner: EnvRunner,
    *,
    environment_setup_completed: bool,
) -> dict[str, Any]:
    """Build the immutable attempt manifest before scientific actions begin.

    A failed simulator construction still needs an attempt directory and a
    failure-ledger entry so the protocol's explicit retry path remains usable.
    Environment fields that are unavailable before construction are recorded
    as null rather than guessed.
    """
    difficulty = getattr(env_runner, "difficulty", None)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": env_runner.dataset,
        "max_steps": args.max_steps,
        "executed_action_horizon": args.obs_horizon,
        "evaluation_policy_seed": args.model_seed,
        "checkpoint_id": args.model_ckpt_id,
        "seed_table_sha256": evaluator._seed_table_payload["entries_sha256"],
        "resolved_environment_seed": env_runner.resolved_environment_seed,
        "resolved_difficulty_hint": env_runner.resolved_difficulty_hint,
        "difficulty": None if difficulty is None else str(difficulty),
        "environment_setup_completed": environment_setup_completed,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "formal_matrix_row_id": os.environ.get("KEYFRAME_FORMAL_ROW_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "policy_port": args.port,
        },
    }


def evaluate(args: Args):
    """Main evaluation function."""
    check_args(args)
    validate_keyframe_args(args)

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"

    log_dict = setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    subgoal_predictor = build_subgoal_predictor(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)
    result_writer = EpisodeResultWriter(args.dual_memory_run_root) if args.dual_memory_run_root else None
    keyframe_store = RunArtifactStore(args.keyframe_run_root) if args.keyframe_run_root else None
    if bool(args.keyframe_selector_arm) != bool(keyframe_store):
        raise ValueError("Keyframe selector arm and keyframe run root must be configured together")
    if keyframe_store is not None:
        if args.keyframe_trajectory_kind == "formal":
            validate_prepared_formal_root(
                args.keyframe_run_root,
                args.keyframe_seed_table,
                Path(__file__).resolve().parents[2],
                attempt_id=args.keyframe_attempt_id,
                authorization_digest=args.keyframe_formal_authorization,
            )
        else:
            validate_prepared_smoke_root(
                args.keyframe_run_root,
                args.keyframe_seed_table,
                Path(__file__).resolve().parents[2],
                attempt_id=args.keyframe_attempt_id,
            )

    while not os.path.exists(save_dir / "log.json"):
        for task_name in task_names:
            if task_name not in log_dict:
                log_dict[task_name] = {}

            env_runner = EnvRunner(
                task_name,
                video_save_dir,
                max_steps=args.max_steps,
                dataset=args.dataset,
                require_current_task_index=bool(args.keyframe_selector_arm),
            )
            num_episodes = env_runner.num_episodes
            episode_ids = (
                [int(value) for value in args.episode_ids.split(",") if value.strip()]
                if args.episode_ids
                else list(range(num_episodes))
            )
            invalid_ids = [value for value in episode_ids if value < 0 or value >= num_episodes]
            if invalid_ids:
                raise ValueError(
                    f"Episode IDs {invalid_ids} are outside the official range 0..{num_episodes - 1}"
                )

            success_flag = "unknown"

            for episode_id in episode_ids:
                if str(episode_id) in log_dict[task_name]:
                    print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                    continue

                episode_exception = None
                attempt_writer = None
                episode_output_dir = video_save_dir
                pre_traj = None
                key = (
                    ScientificKey(
                        task_name,
                        episode_id,
                        args.keyframe_selector_arm,
                        args.keyframe_trajectory_kind,
                    )
                    if keyframe_store is not None
                    else None
                )
                try:
                    if keyframe_store is not None:
                        row_validator = (
                            validate_formal_runtime_row_binding
                            if args.keyframe_trajectory_kind == "formal"
                            else validate_runtime_row_binding
                        )
                        row_validator(
                            args.keyframe_run_root,
                            attempt_id=args.keyframe_attempt_id,
                            row_id=int(
                                os.environ.get("KEYFRAME_FORMAL_ROW_ID", "-1")
                                if args.keyframe_trajectory_kind == "formal"
                                else os.environ.get("SLURM_ARRAY_TASK_ID", "-1")
                            ),
                            task=task_name,
                            episode_id=episode_id,
                            arm=args.keyframe_selector_arm,
                            trajectory_kind=args.keyframe_trajectory_kind,
                            max_steps=args.max_steps,
                            dataset=env_runner.dataset,
                        )
                    env_runner.make_env(episode_id)
                    print(
                        f"\n[robomme] env for task {task_name} episode "
                        f"{episode_id} setup finished"
                    )
                    if keyframe_store is not None:
                        attempt_writer = keyframe_store.new_attempt(
                            key,
                            args.keyframe_attempt_id,
                            _keyframe_attempt_manifest(
                                args,
                                evaluator,
                                env_runner,
                                environment_setup_completed=True,
                            ),
                        )
                        evaluator.attempt_writer = attempt_writer
                        episode_output_dir = attempt_writer.attempt_dir
                        # The reserved attempt now preserves reset failures.
                        # Exact live initial hashes are published separately,
                        # atomically, before the first policy call.
                        pre_traj = env_runner.get_init_obs()
                        attempt_writer.record_initial_conditions(
                            env_runner.initial_condition_hashes
                        )
                    success_flag = evaluator.eval_each_episode(
                        env_runner,
                        subgoal_predictor,
                        episode_output_dir,
                        pre_traj=pre_traj,
                    )
                    if success_flag == "unknown":
                        raise RuntimeError(
                            "Subgoal/policy pipeline returned no official terminal outcome"
                        )
                    else:
                        log_dict[task_name][episode_id] = success_flag == "success"
                        if result_writer is not None:
                            result_writer.append_episode(
                                {
                                    **evaluator.last_episode_record,
                                    "model_id": args.model_id,
                                    "training_seed": args.training_seed,
                                    "evaluation_policy_seed": args.model_seed,
                                    "symbolic_source": args.symbolic_source,
                                    "checkpoint_id": args.model_ckpt_id,
                                    "evaluation_scope": args.evaluation_scope,
                                }
                            )
                        if attempt_writer is not None:
                            attempt_writer.finalize(
                                {
                                    **evaluator.last_episode_record,
                                    "dataset": env_runner.dataset,
                                    "selector_arm": parse_arm(args.keyframe_selector_arm).value,
                                    "max_steps": args.max_steps,
                                    "executed_action_horizon": args.obs_horizon,
                                    "evaluation_policy_seed": args.model_seed,
                                    "checkpoint_id": args.model_ckpt_id,
                                }
                            )
                except Exception as e:
                    print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                    log_dict[task_name][episode_id] = "error"
                    episode_exception = e
                    if result_writer is not None:
                        result_writer.append_infra_failure(
                            {
                                "model_id": args.model_id,
                                "training_seed": args.training_seed,
                                "symbolic_source": args.symbolic_source,
                                "task": task_name,
                                "episode_id": episode_id,
                                "error_type": type(e).__name__,
                                "error": str(e),
                            }
                        )
                    if keyframe_store is not None:
                        if attempt_writer is None:
                            # Reserve a write-once failed attempt even when the
                            # simulator could not be constructed. Without this,
                            # the explicit retry contract would be impossible to
                            # satisfy because no preceding attempt would exist.
                            attempt_writer = keyframe_store.new_attempt(
                                key,
                                args.keyframe_attempt_id,
                                _keyframe_attempt_manifest(
                                    args,
                                    evaluator,
                                    env_runner,
                                    environment_setup_completed=False,
                                ),
                            )
                        retry_allowed = is_retryable_infrastructure_exception(e) and (
                            args.keyframe_attempt_id < 2
                        )
                        keyframe_store.record_failure(
                            {
                                "task": task_name,
                                "episode_id": episode_id,
                                "arm": parse_arm(args.keyframe_selector_arm).value,
                                "trajectory_kind": args.keyframe_trajectory_kind,
                                "attempt_id": args.keyframe_attempt_id,
                                "error_type": type(e).__name__,
                                "error": str(e),
                                "classification": (
                                    "infrastructure" if retry_allowed else "hard_stop"
                                ),
                                "retry_allowed": retry_allowed,
                            }
                        )

                env_runner.close_env()
                evaluator.attempt_writer = None
                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)

                if episode_exception is not None:
                    raise RuntimeError(
                        f"Attempt failed for {task_name}/{episode_id}; inspect the immutable "
                        "failure ledger before any protocol-governed retry"
                    ) from episode_exception

            del env_runner
            time.sleep(1)

        try:
            final_results = {}
            final_results["success_rate"] = {
                task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
                for task_name in log_dict.keys()
            }
            final_results["total_success_rate"] = (
                sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
            )
            with open(save_dir / "log.json", "w") as f:
                json.dump(final_results, f, indent=2)
        except Exception as e:
            print(f"Error saving final results: {e}")
            time.sleep(1)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
