# Uniform-Preserving Keyframe Expansion

**Protocol version:** v1.1

**Protocol date:** 2026-09-11

**Status:** transport-amended; requires fresh Lighthouse gates before conditional formal launch

**Experiment type:** fixed-checkpoint, test-time-only, paired post-hoc follow-up

## 1. Scientific question and amendment

The original U/O/OC/R study found no reliable benefit from replacing part of a
32-frame Uniform memory with oracle boundary frames. This follow-up asks:

> If all frames selected by the released 32-frame Uniform policy are retained,
> does appending every newly visible causal boundary frame improve full-task
> success?

The earlier v0.9 design proposed reusing U results produced on Athena. A later
input audit found that the old Athena U initial front-view images were not
byte-identical to newly generated Lighthouse inputs. They looked close, but
that is insufficient for a strict paired experiment. This v1.0 amendment
therefore requires **fresh U, UK48, and UN48 trajectories in the same Lighthouse
run family**. Old U outcomes remain immutable prior evidence and are not used in
the v1.0 confirmatory comparisons.

The user approved this amendment and authorized the following execution rule:
run the new smoke; if all frozen gates pass, launch the formal experiment
without another approval pause. This authorization does not permit changing
the scientific definitions below, overwriting earlier runs, hiding failures,
or pushing to any remote other than the approved writable `lab` remote.

This is a post-hoc follow-up: the complete 16-task, 50-episode population and
prior results have already been viewed. It must not be described as an
untouched-test preregistration.

### 1.1 Transport-only v1.1 amendment

The v1.0 real-checkpoint architecture gate and exact 64-trajectory smoke passed,
but the first cell of the first v1.0 formal run stopped before a scientific
outcome was produced. Its first policy request completed in 76.351 seconds;
before the next history append, the WebSocket had closed with code 1011 and
`keepalive ping timeout`. The synchronous cold JAX inference blocked the server
event loop longer than the WebSocket library's default server-side keepalive
tolerance. The failed attempt remains immutable, is not relabeled as a task
failure, and is not reused in the v1.1 formal matrix.

Version v1.1 changes only transport liveness. The experiment server and client
both use a 600-second keepalive timeout, and the server advertises that exact
value in its validated handshake metadata. The existing 600-second application
response bound, controller supervision, fail-closed behavior, and no-hidden-
retry policy remain in force. Selectors, model weights and shapes, tasks,
episodes, seeds, observations, actions, metrics, and analysis are unchanged.

Because source and protocol digests changed, v1.0 gates cannot authorize v1.1.
CPU/static validation, the real-checkpoint architecture gate, and the complete
64-trajectory end-to-end smoke must all be rerun against one new clean commit.
Only after those gates pass may a new 2,400-cell formal run be created. This is
a correction restoring the already frozen scientific behavior, not a new
experimental arm or an infrastructure retry inside the failed v1.0 run.

## 2. Arms and comparisons

| Arm | Policy input | Capacity | Role |
|---|---|---:|---|
| **U** | Released Uniform frame selection | 32 frames / 512 tokens | Fresh same-run baseline |
| **UK48** | Exact U set plus all newly missing causal boundary frames | 48 frames / 768 tokens | Primary intervention |
| **UN48** | Exact U set plus an equal number of random non-boundary, non-U frames | 48 frames / 768 tokens | Auxiliary content control |

The comparisons are frozen as follows:

1. **Primary:** UK48 minus U. This tests the complete expanded-keyframe scheme.
2. **Auxiliary:** UN48 minus U. This tests whether adding ordinary history in an
   expanded input also changes performance.
3. **Auxiliary attribution:** UK48 minus UN48. This asks whether boundary frames
   outperform the matched random additions.

UN48 controls added frame count and the expanded model input shape relative to
UK48. It is not a pure capacity-only control because it contains real extra
observations. UK48 beating UN48 but not U does not validate the primary claim.
No comparison alone proves why the prior OC result failed.

## 3. Frozen common system

| Component | Frozen value |
|---|---|
| Benchmark | RoboMME MME-VLA Suite |
| Policy checkpoint | `Yinpei/perceptual-framesamp-modul`, checkpoint `79999` |
| Checkpoint archive SHA-256 | `2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62` |
| Training | None; no parameter update or matching retraining |
| Memory integration | Perceptual frame sampling plus Modulator |
| History source | Episode-long causal front-view observations |
| Current observation | Current front and wrist views, unchanged |
| Task conditioning | Original task instruction only |
| Symbolic memory | Disabled |
| State memory | Disabled |
| Tokens per history frame | 16 (`4 x 4`) |
| Policy/action seed | 7 |
| Action proposal | 20 steps |
| Executed action prefix | First 16 steps, then re-observe and replan |
| Formal episode limit | 1,300 environment steps |
| Execution host | Lighthouse |
| WebSocket keepalive timeout | 600 seconds on both server and client; server value handshake-validated |
| Application response timeout | 600 seconds |

The model weights must load strictly with no missing, unexpected, or randomly
initialized parameters. U uses the released 32-frame/512-token model shape.
UK48 and UN48 use the same checkpoint weights with an explicit inference-only
48-frame/768-token history-budget override.

The 512-token and 768-token variants cannot alternate in one resident model
process because the effective memory sequence length is static for that loaded
policy. This is an execution constraint, not a scientific split: all 2,400
cells belong to one experiment and one provenance family. U rows run in
U-only resident processes; UK48/UN48 rows run in expanded-only resident
processes. All aggregation and pairing remain across the single frozen matrix.

## 4. Exact selectors

Let the causal history visible at policy call `t` be `H_t = {0, ..., t}`. Let
`U_t` be the literal released Uniform selector output:

```text
get_frame_sampling_indices(t, 512, 16)
```

Let `B_t` be the visible boundary set. A boundary is the first frame or a
causally observed change in `current_task_index`; it is not a future annotation,
success label, manually selected best frame, or oracle action.

Define:

```text
new_keys          = B_t - U_t
nonkey_candidates = H_t - (U_t union B_t)
m                 = len(new_keys)

U     = U_t
UK48  = chronological_sort(U_t union new_keys)
UN48  = chronological_sort(U_t union
         sample_without_replacement(nonkey_candidates, m))
```

The expanded selector must call the released U selector with **512**, never
768. Passing 768 would create Uniform48 rather than “U plus extras.”

UK48 and UN48 must preserve every literal U index. For a common history, both
expanded selectors add exactly `m` unique frames. UN48 excludes every U frame
and every visible boundary. Indices stay absolute, unique, chronological, and
causal. Neither arm fills unused expanded slots with additional Uniform frames;
unused positions are right-padded with zeros and a false validity mask.

Early in an episode, U can already contain the complete history, so `m = 0` and
the expanded arms legitimately have fewer than 48 valid frames. Capacity means
fixed tensor slots, not a requirement to invent 48 observations.

If `len(U_t union B_t) > 48`, or UN48 has fewer than `m` eligible candidates,
the trajectory records a protocol-boundary event and the run stops closed.
The implementation must not drop a boundary, duplicate a frame, use future
information, shrink `m`, or silently count the event as an ordinary task fail.

UN48 uses its own precomputed deterministic selector seed, derived only from
family, split, task, episode, and policy-call index. It must not consume the
policy/action RNG or depend on task ordering, wall time, outcome, host, UK48's
realized future, or a prior failed attempt. UK48 and U do not consume UN48 RNG.

## 5. Formal population and pairing

The formal matrix contains exactly:

```text
16 tasks x 50 test episode IDs x 3 arms = 2,400 trajectories
```

Tasks, in canonical order:

1. BinFill
2. StopCube
3. PickXtimes
4. SwingXtimes
5. ButtonUnmask
6. VideoUnmask
7. VideoUnmaskSwap
8. ButtonUnmaskSwap
9. PickHighlight
10. VideoRepick
11. VideoPlaceButton
12. VideoPlaceOrder
13. MoveCube
14. InsertPeg
15. PatternLock
16. RouteStick

Episode IDs are exactly `0..49` on the `test` split. Matrix order is task,
episode ID, then `[U, UK48, UN48]`. Every three-arm block must share task,
episode ID, environment seed/difficulty mapping, checkpoint identity, policy
seed, reset semantics, task text, and initial observation/state evidence.

Initial equality is verified from stored raw inputs or strong hashes, not
inferred merely from matching seed numbers. Closed-loop observations, boundary
counts, policy-call counts, termination times, and outcomes may diverge after
the initial state; such divergence is part of the intervention, not a pairing
error.

No previously generated Athena U trajectory substitutes for a fresh
Lighthouse U cell. No result may be selected according to success, boundary
count, visual similarity, or agreement with an expectation.

## 6. Validation gates

Formal launch is allowed only after all gates below are recorded against the
same clean commit, protocol digest, checkpoint digest, environment manifest,
and launch family.

### 6.1 CPU and static gate

The complete relevant test suite must pass, including:

- literal U index preservation and unchanged 32/512 behavior;
- exact UK48/UN48 selection, causality, uniqueness, chronology, and equal `m`;
- deterministic independent UN48 seed replay;
- early-history padding and capacity/shortage hard stops;
- exact U 512-token and expanded 768-token trace validation;
- strict weight loading and policy-variant metadata;
- exact 2,400-row formal and 64-row smoke matrices;
- homogeneous resident execution plans;
- write-once artifacts, failure-ledger behavior, ingestion, and statistics.

### 6.2 Real-checkpoint architecture gate

On Lighthouse GPU, load the real checkpoint strictly in the expanded shape and
verify finite `20 x 8` actions, final `[1, 768, 1024]` memory representation,
valid masks, reset behavior, deterministic repeated fixed-input inference, and
recorded latency/peak memory. This synthetic architecture probe covers the
expanded shape. The released U shape is additionally covered end to end by the
smoke below.

### 6.3 End-to-end smoke gate: exactly 64 trajectories

The smoke uses `val` episode 0, never formal `test` outcomes:

- for every task: one short U, one short UK48, and one short UN48 trajectory,
  each capped at 64 steps but allowed to stop on an official terminal;
- for every task: one additional full terminal trajectory, with the arm rotated
  deterministically across U, UK48, and UN48.

Thus the gate is 48 short plus 16 terminal trajectories. U rows execute only
through a U32/512 resident process. UK48/UN48 rows execute only through an
expanded 48/768 process. A smoke pass requires all 64 cells present exactly
once, no unclassified infrastructure failure, strict metadata/trace replay,
initial three-arm pairing for the short cells, finite correctly shaped actions,
valid termination semantics, successful reset isolation, no selector boundary
event, and accepted resource/latency records.

Smoke success rates are not used to tune the selector, choose tasks, change the
capacity, or decide which formal cells to run.

### 6.4 Launch freeze

After the preceding gates pass, freeze and checksum:

- this v1.1 protocol;
- clean Git commit and approved `lab` push;
- environment, checkpoint, hardware, and dependency manifests;
- formal matrix and selector-seed manifest;
- smoke plan, execution records, raw outcomes, completeness report, and gate;
- formal execution plans and the user's conditional authorization record.

Any later scientific change requires a versioned amendment and renewed smoke.
A code correction that only restores this frozen behavior still requires a new
commit, evidence rebinding, and rerunning affected gates.

## 7. Formal execution and resource policy

The formal matrix may be sharded across multiple Lighthouse GPUs. Shards must
be deterministic, disjoint, exhaustive, and homogeneous in policy variant.
U shards contain only U rows; expanded shards contain only UK48/UN48 rows.
Using an otherwise occupied GPU is allowed only when it has sufficient free
memory and useful compute headroom. Throughput and scientific completion have
priority, but a process must not knowingly create memory pressure that corrupts
this or another user's job.

Resident workers should retain the model across consecutive trajectories; they
must not exit and reload after every episode. Every episode still receives a
fresh environment reset, selector reset, RNG reset according to the manifest,
and independently sealed attempt directory.

After formal launch, raw outcome inspection is deferred until the complete
matrix has finished or a genuine operational failure requires intervention.
Monitoring may inspect process liveness, GPU health, progress counts, errors,
and artifact completeness without using scientific success/failure patterns to
alter execution.

## 8. Outcome, retry, and evidence contract

Each policy call records at minimum:

- family, arm, split, task, episode, environment step, and policy-call index;
- history length, latest absolute index, and visible boundary indices;
- literal U indices, missing key indices, non-key candidate count, and `m`;
- UN48 selector seed/payload identity and sampled extras;
- final selected indices, valid/padded frames and tokens, masks, and digests;
- effective capacity and policy variant;
- unchanged policy seed, timing, and reset/config identities.

Each trajectory stores source commit, protocol and matrix hashes, benchmark and
dependency versions, checkpoint identity, host/GPU, commands, timestamps,
initial state/front/wrist/task evidence, terminal metadata, outcome, selector
trace, logs, and checksums.

Use a new write-once root:

```text
runs/uniform_keyframe_expansion/<run_id>/
```

Smoke and formal runs must have distinct roots and scope-qualified cell keys.
Never edit or overwrite raw measurements. Derived artifacts identify every
source result and checksum.

An ordinary collision, benchmark timeout, official failure, or benchmark
`error` is an immutable scientific outcome and is not retried to improve the
result. A documented transport, scheduler, filesystem, node, or GPU
infrastructure failure may receive at most two fresh-reset infrastructure
retries with identical frozen seeds in new attempt directories. Preserve every
attempt and accept only the first valid scientific result for a cell.

Selector/config/checkpoint/serialization defects, mixed policy variants, trace
replay failure, input-pairing failure, and capacity/candidate shortage are hard
stops. No cell may be silently excluded, replaced, resumed from reconstructed
mid-episode state, or relabeled.

## 9. Frozen analysis

The report is valid only after an exact audit finds all 2,400 formal cells,
exactly 800 complete three-arm episode blocks, and no unresolved duplicate,
missing, corrupted, or unclassified attempt.

Primary endpoint: official full-task binary success. Report for every arm:

- overall and per-task success counts/rates;
- episode-block paired differences;
- task-balanced effect estimates;
- stratified paired bootstrap 95% confidence intervals using a frozen analysis
  seed and replicate count;
- paired binary tests and discordant-pair counts;
- task-level heterogeneity.

The primary inferential claim is UK48 minus U. The two auxiliary contrasts are
reported with their prespecified multiplicity handling and are not promoted to
primary after seeing results. Also report latency, peak GPU memory, policy-call
count, observed boundary count, `m`, effective frame count, termination mode,
and protocol-boundary events as diagnostics. A boundary count is observed
progress evidence, not a perfect count of physically completed subtasks.

Interpret point estimates together with uncertainty and practical magnitude:

- UK48 > U and UK48 > UN48 supports the expanded-keyframe method and suggests
  content-specific value relative to this random control;
- UK48 and UN48 both > U without a resolved UK48−UN48 difference supports an
  expanded-history effect but not unique keyframe value;
- UK48 > UN48 but not U does not validate the primary hypothesis;
- no demonstrated UK48 > U is a negative or inconclusive primary result,
  depending on the interval, not proof that all memory methods are ineffective.

## 10. Completion condition

The experiment is complete only when the frozen source is pushed to `lab`, all
gates are bound and passed, the exact 2,400 formal cells are sealed and audited,
the three-arm initial pairing is verified, and the predeclared statistical and
diagnostic artifacts are generated from the immutable outcomes. Starting a job,
finishing a shard, or seeing plausible success rates is not completion.
