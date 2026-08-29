# Causal Boundary Oracle Test-Time Memory Selection

**Protocol version:** v1.0
**Amendment:** A-001 (incorporated unchanged)
**Status:** frozen after the required smoke gates and pre-formal audit; formal
execution has not started
**Protocol date:** 2026-08-27
**Freeze date:** 2026-08-29
**Experiment type:** fixed-checkpoint, test-time-only, four-arm paired evaluation

## Freeze record v1.0

Version v1.0 freezes the scientific definitions from amended protocol v0.9.1
without changing any arm, boundary rule, checkpoint, memory budget, task or
episode population, seed, metric, exclusion rule, statistical test, or decision
criterion. The required pre-formal evidence was reviewed in
`PREFORMAL_AUDIT.md`; the smoke-tested implementation candidate is commit
`899912b11a379346b4c8f4d6c80f54f07c118ea3`.

The v0.9.1-to-v1.0 repository changes are administrative protocol freezing,
protocol-version propagation, and documentation of already-completed
non-scientific implementation fixes. They do not authorize formal execution.
Every formal launch still requires a new immutable run root and separate,
explicit user authorization.

## Amendment record A-001

This user-approved amendment resolves three pre-smoke ambiguities without
changing the four arms, task or episode populations, checkpoint, model weights,
memory budget, outcomes, practical-effect threshold, or preregistered analysis
replicate counts and seeds:

1. it gives development-smoke RandomSamp a dedicated seed namespace and requires
   its complete seed set to be disjoint from the unchanged formal seed set;
2. it defines a short smoke trajectory as ending at the earlier of a valid
   official terminal state and 64 environment steps; and
3. it freezes the confidence-interval construction, randomization-test tail and
   finite-sample correction, secondary raw p-values, and the meaning of
   “beats.”

The rationale is to prevent smoke/formal RNG reuse, avoid stepping past a valid
benchmark terminal state, and make the preregistered statistical decisions
fully reproducible before any scientific result is opened.

## 1. One-sentence protocol

Using the same released Uniform-trained RoboMME FrameSamp+Modulator checkpoint,
run four full closed-loop evaluation arms that differ only in which causal
front-view history frames occupy the existing 32-frame perceptual-memory budget:
official Uniform FrameSamp, privileged boundary-only Oracle, privileged
boundary-plus-temporal-coverage Oracle, and seeded RandomSamp.

## 2. Scientific question

The primary question is:

> For a fixed RoboMME policy trained with Uniform FrameSamp, does replacing the
> test-time history indices with causal, privileged subgoal-boundary frames plus
> broad temporal coverage improve final episode success?

The experiment separates four possible explanations:

1. **Official Uniform** measures the released policy as intended.
2. **Oracle-only** asks whether known subgoal-boundary frames are sufficient, even
   when they leave unused memory slots.
3. **Oracle+Coverage** asks whether boundary evidence and broad temporal context
   are complementary while matching the official valid-frame count.
4. **RandomSamp** controls for gains caused merely by changing the selected
   indices or injecting sampling variation.

The confirmatory comparison is **Oracle+Coverage versus Official Uniform**.
The other comparisons are mechanistic controls.

## 3. Claim boundary

This experiment may establish test-time memory-selection headroom for this fixed
checkpoint. It does **not** train or validate a deployable keyframe detector.

The Oracle is a **privileged subgoal-boundary Oracle**, not an optimal-frame
Oracle. A boundary label states that the simulator's current subtask changed; it
does not state that the frame is globally optimal for the final task.

A negative result only applies to test-time resampling of a checkpoint trained
with Uniform FrameSamp. It does not rule out matched training with a new selector.
A positive result does not imply that the privileged signal exists on a real
robot; it motivates a later learned or rule-based causal approximation.

No result from the earlier short-horizon memory-candidate Oracle, replication,
or bridge run is pooled into this experiment. Those runs use different
interventions, horizons, and seed schedules and remain separate evidence.

## 4. Frozen system components

All four arms must use the following unchanged components.

| Component | Frozen value |
|---|---|
| Policy | released `Yinpei/perceptual-framesamp-modul`, checkpoint `79999` |
| Server checkpoint path | `runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999` |
| Checkpoint archive SHA-256 | `2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62` |
| Representation | perceptual memory, `frame_sampling` |
| Integration | Modulator |
| Memory camera | historical front view only |
| Memory budget | 512 tokens |
| Tokens per history frame | 16 (`4 x 4`) |
| Maximum history frames | 32 |
| State memory | disabled, as in the released config |
| Current observations | current front view and wrist view, unchanged |
| Task input | original task instruction, unchanged |
| Action proposal horizon | 20 steps |
| Executed action horizon | first 16 steps, then observe and replan |
| Episode limit | official maximum of 1300 environment steps |
| Evaluation policy seed | 7, reset at the beginning of every arm trajectory |
| Training or weight updates | none |

Before smoke and formal runs, the implementation must verify the runtime history
configuration is exactly:

```text
budget=512
num_views=1
token_per_image=16
representation_type=perceptual
perceptual_memory.type=frame_sampling
integration_type=modulation
```

The current SigLIP encoder, feature pooling/projection, positional encoding,
Modulator, VLM Expert, Action Expert, flow-matching solver, action normalization,
and checkpoint weights must not differ across arms.

## 5. Timeline and causal unit

Let a policy call occur after the server has accumulated front-view history
frames with integer indices

```text
H_t = [0, 1, ..., t].
```

The selector runs once per policy call, not once per environment step. The new
16-step observation segment is first appended to the episode-long server memory;
the selector then chooses from the complete causal history `H_t`.

The target valid-frame count is

```text
n_t = min(32, t + 1).
```

The current front and wrist observations still follow the ordinary observation
path. Selecting frame `t` as history does not replace the current observation.

Every selector output must be a unique, strictly increasing list of real history
indices, with every index in `[0, t]` and length at most 32. Fewer than 32 valid
frames use the released right-padding and false-mask behavior. A duplicated frame
must never be represented as new information.

## 6. Strictly causal privileged boundary

### 6.1 Definition

For every actually observed frame `i`, record the simulator's privileged current
subtask identifier `stage_i`, corresponding to the benchmark's
`current_task_index`. Define:

```text
boundary_0 = true
boundary_i = (stage_i != stage_(i-1)), for i > 0.
```

The boundary flag is attached to the post-action observation returned at the
same time as `stage_i`. This mirrors the benchmark's production rule for the H5
field `info/is_subgoal_boundary`.

At policy time `t`, an Oracle arm may read only boundary flags already attached
to frames `0..t`. It must never read a future frame, future stage, future H5 row,
terminal outcome, success flag, future simulator state, or reference trajectory
length.

The privileged stage identifier and boundary flag may influence only the frame
selector. They must not enter the task prompt, current-observation tokens,
perceptual features, VLM Expert, Modulator values, Action Expert, or action
scoring.

### 6.2 Why reference-H5 indices are forbidden

The original H5 labels belong to an expert demonstration trajectory. A policy
rollout can diverge from that trajectory, so the same integer timestamp need not
represent the same subtask transition. Therefore, copying boundary timestamps
from a reference H5 into a closed-loop rollout is a protocol violation.

The H5 field may be used only in offline tests that verify the meaning of the
label. Formal rollouts require the online causal stage transition defined above.

### 6.3 Initial demonstration frames

Some RoboMME tasks return a generated conditioning demonstration during reset.
Those frames remain part of the official history and must remain eligible in all
arms. The Oracle instrumentation must capture a per-frame `current_task_index`
for this live generated pre-trajectory before the reset batch is flattened, then
derive boundaries with the same adjacent-transition rule.

Using reference-H5 timestamps as a substitute, assigning one flattened label to
the whole pre-trajectory, or silently marking all demonstration frames as
non-boundaries is forbidden.

If a task cannot expose a correctly aligned online stage identifier for every
returned history frame, that is a **global launch blocker**. The implementation
must stop before scientific execution; it may not fall back to Uniform,
subgoal-text guessing, or H5 timestamps.

## 7. Exact selector definitions

Let:

```text
K = 32
B_t = {i in H_t: boundary_i}    # causal boundary frames visible by time t
```

Define `EVEN_THIN(S, m)` for an ordered, unique index list `S`:

- if `m <= 0`, return an empty list;
- if `len(S) <= m`, return all of `S`;
- otherwise select `m` positions from `S` using the same positional rule as
  `np.linspace(0, len(S)-1, m, dtype=np.int32)`.

This operation is deterministic and preserves the earliest and latest element
of `S` when `m >= 2`.

Define the boundary core:

```text
core = EVEN_THIN(sorted(B_t), K)
```

The explicit thinning rule prevents the existing padding helper from silently
taking the first 32 boundary frames if a long episode produces more than the
budget.

### 7.1 Arm U: Official Uniform FrameSamp

Call the released `MemoryBuffer.get_frame_sampling_indices()` literally, without
an override or reimplementation.

- If `t + 1 <= 32`, it returns every index `0..t`.
- Otherwise it returns 32 indices from `np.linspace(0, t, 32, dtype=np.int32)`.
- It includes the first and latest frame.

This literal path is the control and must be regression-tested against the
unmodified checkpoint behavior.

### 7.2 Arm O: Oracle-only

Return `sorted(core)`. Because `boundary_0=true`, the first frame is included.
No non-boundary frame, including the latest frame, is added merely as an anchor.
The latest image is still available through the unchanged current-observation
path. If fewer than 32 frames are selected, the remaining memory slots are zero
padded and masked false by the released code.

Consequently, this arm intentionally changes both frame identity and the number
of valid memory tokens. It tests whether sparse boundary evidence is sufficient;
it is not the cardinality-matched primary comparison.

### 7.3 Arm OC: Oracle+Coverage

Start from `core`. Repeatedly add one frame from `H_t - selected` that maximizes
its minimum absolute time distance to the currently selected frames. Break a tie
by choosing the earlier frame. Continue until `len(selected) == n_t`.

Equivalently, each coverage step is:

```text
next_index = argmax over i in (H_t - selected) of
             min over j in selected of abs(i - j)
```

If `selected` were ever empty in a defensive fixture, insert frame 0 before this
procedure; under the formal definition `boundary_0=true`, so that fallback is
never reached. Return the final set in chronological order.

This arm prioritizes every boundary frame that survives the deterministic
overflow rule, then fills all available slots with frames that greedily cover
the largest remaining temporal gaps. It always has exactly `n_t` valid frames.

### 7.4 Arm R: RandomSamp

- If `t + 1 <= 32`, return every index `0..t`.
- Otherwise use the time-stratified procedure below.

RandomSamp is a time-stratified random control, matching the previously audited
random-memory candidate construction but used here as a complete standalone
trajectory rather than an Oracle candidate search:

- retain frame 0 and frame `t`;
- split the interior indices `1..t-1` into 30 consecutive bins with sizes that
  differ by at most one, exactly as `np.array_split(interior, 30)`;
- sample one index uniformly from each bin without replacement;
- sort the 32 selected indices.

Exactly one preregistered random realization is used per policy call. Across the
800 formal task/episode blocks, those realizations estimate the behavior of this
randomized selector; no seed may be selected after observing an outcome.

The RandomSamp seed for a policy call is derived before formal execution from:

```text
SHA256([2026082501, task_name, episode_id, policy_call_index, "RandomSamp"])
```

This is the unchanged **formal** RandomSamp seed formula. Amendment A-001 does
not alter any formal seed.

Map the first 64 digest bits deterministically into the valid NumPy PCG64 seed
range. Precompute entries for policy-call indices `0..81`, covering
`ceil(1300/16)=82` possible calls per trajectory. The complete seed table and its
SHA-256 must be written before the first formal rollout. Attempting an
out-of-table call is a hard stop rather than permission to create a new seed.

## 8. Fairness invariants

Within a `(task, episode)` block, the four trajectories must have:

- the same benchmark split and episode ID;
- the same resolved environment seed and difficulty;
- byte-identical initial front/wrist observations, robot state, task state, and
  task instruction, verified by hashes;
- the same checkpoint and runtime model configuration;
- the same policy RNG seed at reset;
- the same maximum steps and action execution horizon;
- independent fresh simulator and policy-memory resets;
- no model training, parameter update, candidate branching, action Oracle,
  replayed winner, or outcome-conditioned retry.

The only treatment variable is the list of historical front-view frame indices
selected at each policy call. Boundary instrumentation may be logged in all arms
for audit, but only O and OC may use it for selection.

Trajectories are full independent closed-loop rollouts from the initial state.
Do not restore an approximate ManiSkill mid-episode state, intervene only at a
few hand-picked steps, or score a short continuation in place of final success.

## 9. Formal evaluation population

The confirmatory experiment uses the complete official RoboMME evaluation
matrix:

- benchmark dataset argument: `test`;
- tasks: the 16 task names in the released `TASK_NAME_LIST`;
- episode IDs: `0..49` for every task;
- four selector arms per task/episode;
- total scientific trajectories: `16 x 50 x 4 = 3200`.

The exact task list is:

```text
BinFill, StopCube, PickXtimes, SwingXtimes,
ButtonUnmask, VideoUnmask, VideoUnmaskSwap, ButtonUnmaskSwap,
PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder,
MoveCube, InsertPeg, PatternLock, RouteStick
```

Formal work may be sharded by task, episode, and arm for wall-clock efficiency,
but the result is not complete until the exact 3200-cell matrix has one valid
scientific outcome per cell. Because this run may be multi-day, submitting it
requires explicit user authorization after the smoke report and frozen v1.0
protocol are committed.

Before formal execution, write a `prior_exposure_manifest.json` listing every
task/split/episode whose outcome was inspected in earlier local, smoke,
replication, or bridge work. The full official test census remains the primary
population; a prespecified sensitivity analysis may additionally exclude this
fixed exposure set. The exposure set must be frozen without reading results from
the present experiment.

## 10. Development and GPU smoke gate

Smoke is part of this protocol, but smoke outcomes are not scientific evidence.
Development data and seeds must be disjoint from the formal `test` matrix.

### 10.1 CPU and fixture checks

Before GPU use, implement and pass tests for:

1. selector outputs at history lengths `1, 2, 16, 32, 33, 64, 1301`;
2. no-boundary, dense-boundary, duplicate-boundary, and more-than-32-boundary
   fixtures;
3. exact U equivalence with the released method;
4. O padding and false-mask behavior;
5. OC valid-frame cardinality, boundary priority, farthest-time coverage, and
   deterministic tie breaking;
6. R bin construction, uniqueness, endpoints, determinism, and different-seed
   sensitivity;
7. strict causality: changing any future label cannot change selection at `t`;
8. demonstration-frame label/index alignment;
9. temporary selector override restoration after success or exception;
10. structured logging of nested NumPy/JAX values and non-finite numbers;
11. atomic writes, resume validation, duplicate-key rejection, and immutable
    completed artifacts;
12. shell syntax and dry-run validation for every Slurm launcher.

### 10.2 GPU architecture smoke

Use synthetic or development histories, never formal test outcomes. Run at least
one real checkpoint inference for each arm at history lengths below and above 32.
Verify:

- shapes and dtypes match the released path;
- all arms expose 512 memory-token slots;
- U, OC, and R have `16 * n_t` valid tokens;
- O has `16 * len(core)` valid tokens;
- padded tokens are masked;
- the same selected indices yield the same memory tensor digest in the same live
  process;
- resetting clears prior episode frames and RNG state;
- the model completes inference without recompilation or cross-arm pollution
  caused by selector choice.

Byte-exact action equality across separate jobs or different GPU kernels is an
audit metric, not a hard gate. Same-live-process repeatability is the hard gate.

### 10.3 End-to-end development smoke

Use the benchmark `validation` split as resolved by the server (expected alias
`val`), never `test`. Use development episode 0 of every one of the 16 formal
task types so task-specific stage instrumentation is exercised before the full
matrix. For each of the 64 short trajectories, stop at
`min(official terminal, 64 environment steps)`: if a valid official success,
failure, or timeout occurs before step 64, preserve that terminal outcome and
stop immediately; otherwise the trajectory must execute exactly 64 environment
steps. A launcher must never suppress, step beyond, reset past, or otherwise
replace a valid early terminal state merely to reach 64. In addition, run one
preregistered arm per task to a normal terminal state or official timeout;
rotate that arm across tasks without looking at outcomes. This smoke therefore
contains 64 short trajectories plus 16 terminal-path checks, all outside the
formal split.

An actual benchmark `error` must still be preserved immutably as the scientific
terminal outcome defined in Section 11.2, including the wrapper error message
and exception type. It is not retryable infrastructure. However, it is not one
of the three accepted development-smoke PASS terminals above: its presence
must make the smoke audit fail closed and require explicit review before any
new smoke or formal launch. This keeps scientific outcome preservation
separate from the stricter non-scientific readiness gate.

For development smoke only, derive every RandomSamp policy-call seed from the
dedicated namespace:

```text
SHA256([2026082501, "development-smoke-v1", "val", task_name,
        episode_id, policy_call_index, "RandomSamp"])
```

Map the first 64 digest bits into the NumPy PCG64 seed range using the same
deterministic mapping as the formal table. Precompute the complete smoke table
for policy-call indices `0..81` before launching smoke. The preparation gate must
construct the complete smoke and formal RandomSamp seed sets and verify their
intersection is empty; any duplicate within either table or any cross-table
intersection is a hard stop. Record each table's scope, dataset/split,
derivation, entries, and SHA-256 in the launch provenance.

The fixed evaluation-policy seed `7` remains intentionally identical across
arms as a matched control. It is not a RandomSamp selector seed and is therefore
outside the smoke/formal disjointness requirement above.

The smoke must explicitly include `InsertPeg` to exercise conditioning-video
history and `PickXtimes` to exercise repeated-stage history. Its purpose is to
validate online stage instrumentation, boundary/frame alignment, accumulated
history, action-chunk cycling, task-specific wrappers, logging, cleanup, and
artifact finalization.

Do not compare smoke success rates, tune thresholds, choose a selector, change
formal tasks, or remove difficult cases based on smoke outcomes.

### 10.4 Freeze rule

After smoke:

- implementation or environment defects may be fixed and re-smoked;
- clarifying a non-scientific path or command may update v0.9.1;
- changing an arm, boundary definition, split, task/episode set, seed, metric,
  exclusion, or decision rule requires a versioned scientific amendment and a
  complete re-smoke;
- formal launch requires a clean committed v1.0 protocol and explicit user
  approval.

## 11. Outcomes and diagnostics

### 11.1 Primary outcome

Official binary full-episode success.

The reported primary estimand is the equal-task-weighted macro success-rate
difference, in percentage points:

```text
Delta_primary = success(Oracle+Coverage) - success(Official Uniform).
```

### 11.2 Secondary scientific outcomes

- per-task success rate;
- official terminal reason (`success`, `fail`, `timeout`, `error` only when it is
  an actual benchmark outcome);
- collision flag where officially available;
- environment steps to termination;
- official subtask/progress count or stage reached, without inventing new task
  thresholds;
- O versus U, OC versus O, R versus U, and OC versus R paired effects.

### 11.3 Selector and systems diagnostics

At every policy call record:

- total history length and current history index;
- policy-call index and environment step;
- selected frame indices and their hashes;
- visible causal boundary indices;
- selector name and selector seed;
- valid frame count, padding count, valid memory-token count, and mask digest;
- image, position, and final memory tensor shapes/digests;
- age distribution, maximum temporal gap, and boundary recall in memory;
- selector latency, model latency, and end-to-end request latency.

Report selector latency with mean, p50, p95, p99, and maximum. This measures the
cost of Oracle bookkeeping and RandomSamp without confusing it with checkpoint
inference time.

## 12. Statistical analysis

All analysis code, pair definitions, and random seeds must be committed before
formal results are opened.

### 12.1 Confirmatory test

For OC versus U:

1. compute the paired success difference within every task/episode;
2. compute equal task-weighted success rates and `Delta_primary`;
3. obtain a paired hierarchical percentile 95% confidence interval by
   resampling tasks, then paired episodes within each sampled task, using
   analysis seed `2026082502` and 100,000 bootstrap replicates; define its lower
   and upper endpoints as the empirical 2.5th and 97.5th percentiles of the
   bootstrap estimand distribution;
4. compute a two-sided paired randomization p-value by swapping U/OC labels
   within task/episode blocks under the null, preserving equal task weights,
   with 100,000 permutations generated under analysis seed `2026082502`; with
   observed effect `Delta_obs`, report the finite-sample-corrected value
   `(1 + count(|Delta_perm| >= |Delta_obs|)) / (100000 + 1)`;
5. additionally report pooled and per-task discordant counts
   `(OC success, U fail)` and `(OC fail, U success)`;
6. repeat the primary effect-size analysis after excluding only the entries in
   the frozen prior-exposure manifest, and label it a sensitivity analysis rather
   than a replacement primary result.

Do not drop ties or failed tasks when reporting denominators.

### 12.2 Secondary comparisons

Use the same paired estimand and hierarchical bootstrap for:

```text
O  - U
OC - O
R  - U
OC - R
```

For each comparison, use the same 100,000-replicate hierarchical percentile
bootstrap, empirical 2.5th/97.5th percentile interval, and two-sided paired
randomization test with analysis seed `2026082502`, 100,000 permutations, and
the same `+1` finite-sample correction as the confirmatory comparison. The four
resulting two-sided randomization p-values are the raw secondary p-values; apply
Holm's step-down adjustment jointly across exactly these four values. Report
unadjusted estimates, percentile confidence intervals, raw p-values, and
Holm-adjusted p-values. They are mechanistic, not additional confirmatory claims.
Progress and step-count analyses are secondary and must not overrule final
success.

### 12.3 Interpretation and decision rule

The practical-effect threshold is 3.0 macro percentage points.

Throughout the decision rules below, “A beats B” means that the lower endpoint
of the relevant paired hierarchical percentile 95% confidence interval for
`A - B` is strictly greater than zero. A positive point estimate alone does not
count as “beats.” This definition does not replace the separately preregistered
`+3.0 pp` practical-effect requirement where that threshold is stated.

- **Semantic-selection GO:** `OC - U >= 3.0 pp`, the paired 95% CI lower bound is
  above zero, and OC also beats R with a 95% CI lower bound above zero.
- **Sampling-sensitivity only:** OC beats U, but OC does not distinguish itself
  from R. Conclude that changing the sampling distribution matters, not that
  boundary semantics are validated.
- **Coverage-complementarity evidence:** OC beats O. This indicates boundary
  frames alone are too sparse or incomplete and generic history remains useful.
- **Decisive test-time NO-GO:** the upper 95% confidence bound for `OC - U` is
  below `+3.0 pp`.
- **Inconclusive:** all other cases, including a positive point estimate whose CI
  crosses zero or still includes a practically important gain.

Even a Semantic-selection GO authorizes only development of a causal deployable
selector. It does not authorize claiming real-robot Oracle availability or
matched-training performance.

## 13. Failures, retries, and exclusions

A benchmark timeout, collision, or ordinary task failure is a scientific
outcome and must not be retried.

An infrastructure failure is limited to events such as scheduler preemption,
node/GPU failure, server startup failure before scientific action, filesystem
unavailability, or a documented transport failure not caused by an arm's
scientific behavior. It must be written to a separate failure ledger.

An infrastructure retry must:

- restart the complete `(task, episode, arm)` trajectory from a fresh reset;
- reuse exactly the same environment, policy, selector, and analysis seeds;
- write to a new append-only attempt directory;
- preserve the failed attempt and its logs;
- never resume from a reconstructed mid-episode simulator state.

At most two infrastructure retry attempts are allowed without a protocol
amendment. A deterministic selector-specific crash, missing/misaligned Oracle
label, invalid index, configuration mismatch, checkpoint mismatch, history
pollution, or result-serialization defect is a hard stop for the experiment, not
an outcome to exclude or silently fall back from.

No formal aggregate may be published until every one of the 3200 cells has one
and only one valid scientific result. No post-hoc episode or task exclusion is
allowed.

## 14. Provenance and immutable output contract

Use a new run root:

```text
runs/keyframe_oracle_sampling/<run_id>/
```

Recommended layout:

```text
protocol/
  protocol_snapshot.md
  protocol_sha256.txt
  seed_table.json
  prior_exposure_manifest.json
  launch_manifest.json
trajectories/<task>/episode_<NN>/<arm>/attempt_<NN>/
  episode_manifest.json
  selector_trace.jsonl
  episode_result.json
  stdout.log
  stderr.log
  video.mp4
failures/infra_failures.jsonl
aggregate/
  completeness_report.json
  per_episode.csv
  per_task.csv
  summary.json
  analysis.md
```

Before the first rollout, `launch_manifest.json` must record:

- run ID and UTC timestamp;
- protocol version and SHA-256;
- policy repository commit, dirty status, and diff hash if dirty;
- benchmark repository commit;
- checkpoint path, archive SHA-256, and unpacked metadata digest;
- complete command/config and environment lock information;
- task/episode/arm matrix;
- policy seed, master selector seed, and seed-table digest;
- Slurm job IDs, nodes, GPUs, and ports as they become known.

Each completed artifact is write-once. Write temporary files, flush and `fsync`,
then atomically rename. Resume code must validate hashes and reject duplicate
scientific keys. Never overwrite an existing result directory, edit generated
metrics, or aggregate an incomplete matrix.

## 15. Slurm and execution policy

Reuse the proven two-GPU pattern when the simulator and policy require separate
GPUs, with pinned environment and checkpoint, a unique port per shard,
`--no-requeue`, readiness timeout, and trap-based cleanup.

Parallelism may reduce wall time but must not change scientific seeds or permit
cross-arm state sharing. Each shard writes only inside its own directory. A
single final aggregation process validates and combines the exact matrix.

Before submitting any formal shard, verify:

- committed protocol v1.0 and committed implementation;
- clean worktree on a dedicated `exp/*` branch;
- push target is the writable `lab` fork, never `origin`;
- exact server commit and checkpoint identity;
- CPU, fixture, GPU, and end-to-end smoke gates all pass;
- formal run root does not exist;
- user has explicitly authorized the multi-day formal launch.

## 16. Codex implementation contract

The server Codex should implement in this order:

1. Add pure, dependency-light selector functions and exhaustive unit tests.
2. Add read-only online `current_task_index` instrumentation aligned one-to-one
   with every returned front frame, including reset demonstrations.
3. Carry boundary flags with non-overlapping history segments into the server
   memory buffer without exposing them to model inputs.
4. Add a runtime selector switch at
   `MemoryBuffer.get_frame_sampling_indices()` or a narrowly scoped equivalent.
5. Keep U on the literal released path; never emulate it through the new code.
6. Add per-call selector tracing and immutable per-episode manifests.
7. Add fixture, resume, completeness, and analysis tests.
8. Add smoke-only Slurm entry points and run all non-GPU checks.
9. Submit smoke only after reviewing the diff and recording the commit SHA.
10. Produce a smoke audit report. Do not launch formal evaluation.

Codex must stop and report rather than decide independently if any of the
following would need to change:

- checkpoint or model weights;
- boundary semantics or causal availability;
- selector definitions or anchors;
- memory budget or mask behavior;
- formal task/episode matrix;
- any seed or outcome definition;
- failure/exclusion rules;
- statistical tests or decision threshold.

## 17. Required pre-formal audit report

The smoke handoff must contain:

- implementation diff summary and exact commits;
- tests and smoke commands with pass/fail status;
- proof of literal U equivalence;
- proof of online boundary causality and demonstration-frame alignment;
- selector examples for normal and overflow cases;
- frame/token/mask counts and digests for all arms;
- random-seed reproducibility evidence;
- policy reset and cross-arm isolation evidence;
- selector and policy latency summary;
- output/resume/immutability audit;
- known limitations and any v0.9.1-to-v1.0 changes;
- estimated formal trajectory count, policy-call count, wall time, and Slurm
  resources;
- an explicit statement that formal execution has not started.

Only after this report is reviewed should the protocol be frozen as v1.0 and the
user be asked to authorize the complete formal run.

Freeze completion: the report was reviewed on 2026-08-29 and this document was
then frozen as v1.0. Formal execution had not started at freeze time and remains
subject to the separate authorization requirement above.
