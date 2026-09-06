# Causal Keyframe-Neighborhood Memory Sampling Extension

**Protocol version:** v1.0
**Protocol date:** 2026-09-05
**Status:** frozen for implementation and smoke validation; formal launch requires
separate user authorization
**Experiment type:** fixed-checkpoint, test-time-only, paired post-hoc follow-up

## 1. Relationship to the completed experiment

This protocol is a separate follow-up to
`experiments/keyframe_oracle_sampling/EXPERIMENT_PROTOCOL.md`. The completed
four-arm experiment and its results remain immutable. In particular, this
extension does not modify, relabel, rerun, or overwrite U, O, OC, or R.

The idea for this extension was formulated after the completed experiment's
outcomes had been inspected. Therefore, the extension must be described as a
post-hoc follow-up whose selector and analysis plan were frozen before its new
trajectories were launched, not as part of the original preregistered study.

The completed reference run is:

- run ID: `20260829T231425Z_7b594786_formal_v1`;
- per-episode result SHA-256:
  `e1ec5d1273f3ba97b14d252a1a1668c00635a71477acca687f3e47d7e9f3a478`;
- summary result SHA-256:
  `e3f40b7192fc1be9a3b280c30234170b63dfae26d389c9b99ba321e82e3efbe9`.

## 2. Scientific question and hypothesis

The completed OC arm guarantees that every causally observed subtask boundary
is eligible for perceptual memory, then fills unused capacity for temporal
coverage. A single boundary frame may, however, be visually ambiguous: it can
fall just before or just after the physical consequence of a stage transition.

This extension asks:

> Does retaining a small, temporally spaced visual neighborhood around each
> causally observed boundary improve a fixed Uniform-trained RoboMME policy over
> retaining the boundary frame alone, when the remaining memory slots use the
> identical OC temporal-coverage rule?

The mechanism under test is local context around a boundary. It is not a claim
that the Oracle boundary is a deployable keyframe detector, nor that the
neighboring frames are globally optimal memories.

## 3. Frozen treatment arms

Let:

- `t` be the latest causally observed history-frame index at the current policy
  call;
- `B_t` be the set of boundary indices observed at or before `t`, defined by the
  unchanged online `current_task_index` transition rule from the parent
  protocol;
- `K = 32` be the maximum number of history frames;
- `EVEN_THIN(S, m)` be the parent protocol's exact ordered positional thinning
  rule, implemented with
  `np.linspace(0, len(S)-1, m, dtype=np.int32)`;
- `COVERAGE_FILL(S, t, K)` be the parent OC arm's exact maximum-time-gap-first
  fill, including its earlier-index tie break.

Only the following two arms generate new trajectories.

### 3.1 OC3: boundary plus two surrounding frames

For every `k` in `B_t`, form the candidate indices:

```text
{k - 2, k, k + 2} intersected with [0, t]
```

Thus, one unselected observation frame lies between each adjacent requested
index. Take the sorted unique union over all boundaries.

If the union contains at most 32 indices, retain it. If it exceeds 32:

1. construct the boundary core using the parent O arm, including its original
   `EVEN_THIN` behavior when more than 32 boundaries are visible;
2. if the core already has 32 frames, retain exactly that core;
3. otherwise retain the entire core and apply `EVEN_THIN` to the sorted unique
   non-boundary `k-2` and `k+2` candidates to fill the remaining slots.

If fewer than 32 frames have been selected, fill the remaining capacity using
the unchanged `COVERAGE_FILL` procedure.

### 3.2 OC5: boundary plus four surrounding frames

For every `k` in `B_t`, form the candidate indices:

```text
{k - 4, k - 2, k, k + 2, k + 4} intersected with [0, t]
```

Take the sorted unique union over all boundaries. If that union contains more
than 32 indices, discard the five-frame construction wholesale and apply the
complete OC3 procedure at the same policy call. This is a fallback to the
three-frame neighborhood, not a direct thinning of the five-frame union.

If the retained neighborhood contains fewer than 32 frames, fill the remaining
capacity using the unchanged `COVERAGE_FILL` procedure.

### 3.3 Causality, deduplication, and early histories

- A positive-offset neighbor is eligible only after that frame has actually
  been observed. Detecting boundary `k` never permits reading `k+2` or `k+4`
  early, and later observations never retroactively change an earlier action.
- Invalid negative indices and indices greater than `t` are omitted rather than
  clipped to a duplicate endpoint.
- Overlapping neighborhoods are unioned, so every selected frame index is
  unique.
- At any policy call, the target valid-frame count is `min(32, t+1)`. When fewer
  than 32 history frames exist, the unchanged model padding and mask behavior is
  used.
- Every returned list is strictly increasing, contains only indices in `[0,t]`,
  and has at most 32 elements.

## 4. Frozen common system

Every setting below is inherited unchanged from the parent protocol:

| Component | Frozen value |
|---|---|
| Policy | released `Yinpei/perceptual-framesamp-modul` |
| Checkpoint | `79999` |
| Checkpoint path | `runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999` |
| Checkpoint archive SHA-256 | `2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62` |
| Training | none; test-time selector change only |
| Perceptual-memory integration | FrameSamp + Modulator |
| History stream | episode-long front-view observation history |
| Current observation | unchanged front-view and wrist-view inputs |
| Tokens per retained history frame | 16 (`4 x 4`) |
| Memory budget | 512 tokens = at most 32 frames |
| Task input | original task instruction |
| Action proposal horizon | 20 steps |
| Executed horizon | first 16 steps, then observe and replan |
| Episode limit | 1300 environment steps |
| Evaluation policy seed | 7, reset at the start of every arm trajectory |

Model weights, encoders, memory Modulator, action generation, observations,
task prompts, termination rules, and simulator behavior must not differ between
OC3 and OC5 or from the completed reference experiment. The selector is the
only treatment variable.

## 5. Formal evaluation matrix and pairing

Use the same `test` population as the completed experiment:

- tasks, in the same canonical order:
  `BinFill`, `StopCube`, `PickXtimes`, `SwingXtimes`, `ButtonUnmask`,
  `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmaskSwap`, `PickHighlight`,
  `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`, `MoveCube`,
  `InsertPeg`, `PatternLock`, and `RouteStick`;
- episode IDs `0..49` for every task;
- arms `OC3` and `OC5` for every task/episode;
- total new scientific trajectories:
  `16 tasks x 50 episodes x 2 arms = 1,600`.

Do not rerun U, O, OC, or R merely to construct this extension. For comparisons
against OC, read the immutable per-episode records from the completed reference
run and verify their checksums before analysis.

Within each `(task, episode_id)` block, OC3 and OC5 must resolve to the same
environment seed, difficulty, task instruction, initial simulator state, initial
front and wrist observations, robot state, task state, and policy RNG seed as
one another and as the corresponding completed OC trajectory. Preserve the
parent episode-to-environment-seed mapping exactly; do not derive a new mapping.

Where raw reference evidence permits, verify byte-identical initial-condition
hashes. If a particular historic reference field was not recorded, report only
`seed/config matched` for that field rather than claiming hash verification.

## 6. Implementation and logging contract

The implementation must:

1. leave the completed four-arm matrix and aggregators fixed at U/O/OC/R;
2. add OC3 and OC5 as separately named selector arms;
3. make OC, OC3, and OC5 call the same implementation of the frozen
   maximum-time-gap coverage fill;
4. construct a separate 1,600-row extension matrix in task-major,
   episode-major, arm-major order;
5. use a new write-once run root directly under
   `runs/keyframe_neighborhood_sampling/<run_id>/`;
6. record, for every policy call, the visible boundary indices, requested
   neighborhood mode, causal neighborhood candidates, whether OC5 fell back to
   OC3, whether OC3 required secondary thinning, final selected indices, valid
   frame/token counts, and selector latency;
7. record run ID, Git commit, checkpoint and environment identities, task,
   episode, all seeds, Slurm job information, attempts, timestamps, terminal
   state, and output checksums using the parent experiment's provenance rules.

No existing run directory or generated result may be overwritten.

## 7. Smoke gates

Formal execution is prohibited until all applicable parent CPU checks pass and
the extension adds passing tests for:

- exact stride-2 neighborhoods at ordinary and edge indices;
- removal of negative and future indices;
- strict causal eligibility of `k+2` and `k+4`;
- deduplication of overlapping neighborhoods and repeated boundary labels;
- OC5 wholesale fallback when its five-frame union first exceeds 32;
- OC3 secondary overflow behavior and preservation of its boundary core;
- behavior with no boundary and more than 32 boundaries;
- exact regression equivalence of the original OC outputs before and after
  sharing the coverage-fill helper;
- exact 32-frame/512-token budget, masks, shapes, reset behavior, and trace
  logging;
- the original completed formal matrix remaining exactly 3,200 U/O/OC/R rows;
- the new extension matrix being exactly 1,600 OC3/OC5 rows;
- extension task, episode, seed, checkpoint, and initial-condition alignment.

After CPU checks, perform a new GPU architecture smoke using the real checkpoint
and a small end-to-end development smoke on the parent protocol's disjoint `val`
development population. Smoke outcomes are plumbing evidence only and must not
be used to change selectors, tune thresholds, choose an arm, or enter formal
statistics. Smoke trajectories must never be counted as formal trajectories.

Any scientific-rule change after this freeze requires a versioned amendment and
a fresh smoke gate. Pure code or environment bug fixes that restore this frozen
behavior must be documented and re-smoked.

## 8. Outcomes and frozen analysis

The primary outcome remains binary episode success. Report equal-task-weighted
macro success rates, while also reporting pooled counts for transparency.

The three planned paired comparisons are:

1. `OC3 - OC`: whether two spaced neighboring observations improve upon the
   boundary-only OC core;
2. `OC5 - OC`: whether a wider spaced neighborhood improves upon OC;
3. `OC5 - OC3`: whether the additional outer neighbors provide incremental
   value.

For each comparison, use matched `(task, episode_id)` outcomes and the parent
analysis machinery:

- equal-task-weighted paired success-rate difference;
- 95% hierarchical bootstrap confidence interval, resampling tasks and then
  paired episodes within task;
- two-sided paired randomization test;
- 100,000 bootstrap replicates and 100,000 permutations;
- analysis seed `2026082502`;
- paired discordant counts overall and by task.

Use a common deterministic resampling schedule across the three comparisons.
Apply Holm--Bonferroni correction across their three primary p-values. Report
raw and adjusted p-values. Do not change the analysis seed or test after viewing
extension outcomes.

Secondary diagnostics include per-task success, stage/progress, boundary recall,
neighborhood-frame retention, OC5 fallback frequency, OC3 thinning frequency,
selected-index spacing, collision/timeout reasons, and selector/model/end-to-end
latency. Per-task and diagnostic findings are exploratory unless separately
specified before launch.

The runtime does not expose one task-independent ground-truth stage-completion
metric. Therefore, the number of causal boundary indices visible at the final
policy call (including the required initial index zero) is reported as an
explicit exploratory progress proxy. It is not a true stage-completion metric
and must not be interpreted as an absolute progress scale across different
tasks.

## 9. Failures, retries, and exclusions

An ordinary task failure, collision, or benchmark timeout is a valid scientific
outcome and must not be rerun. A trajectory may be retried only for documented
infrastructure failure, corrupted output, or a hard protocol-invariant failure.
Every retry must restart the complete trajectory with exactly the same seeds and
configuration, retain all attempt records, and use the first valid attempt as
the only scientific record. At most two infrastructure retries are allowed
without a versioned amendment.

If any matrix cell has no valid result after the allowed retries, the extension
study is incomplete and aggregation must fail closed. Do not silently exclude
the cell, replace it with a different episode or seed, or report a reduced
paired analysis as the formal result. Continuing after such an exhaustion
requires a versioned amendment that records the affected cell and reason, then
preserves the original task, episode ID, seeds, and configuration in any
authorized recovery run.

## 10. Launch authority and completion condition

Local implementation, unit tests, static checks, and preparation of a server
handoff are authorized by this protocol. GPU smoke submission requires a clean,
committed server checkout and an explicit user instruction to begin that smoke.
The 1,600-trajectory formal launch requires a separate explicit user
authorization after the user has reviewed the smoke audit.

The extension is complete only when the exact 1,600-cell matrix has one valid
scientific result per cell, all provenance and checksum audits pass, the
immutable reference OC results are verified, and the frozen paired analysis has
been generated without altering any raw result.
