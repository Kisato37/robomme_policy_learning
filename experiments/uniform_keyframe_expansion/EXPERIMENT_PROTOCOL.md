# Uniform-Preserving Keyframe Expansion

**Protocol version:** v0.9

**Protocol date:** 2026-09-10

**Status:** scientific definitions frozen for local implementation and CPU validation; not GPU-smoke-ready and not formally launch-frozen

**Experiment type:** fixed-checkpoint, test-time-only, paired post-hoc follow-up

## 1. Scope, authority, and relationship to existing studies

This protocol implements `DESIGN.md` v0.2. The user authorized continuing with
the protocol, local code, and local tests. That authorization does not submit
GPU jobs, authorize a formal batch, authorize additional U trajectories, or
authorize Git commits/pushes. Those stages require explicit instructions.

The original U/O/OC/R study and the OC3/OC5 extension remain immutable. Their
scientific definitions, checkpoint metadata, generated results, and original
32-frame validators must not be relaxed to accommodate this new family.
Applicable parent documents are:

- `experiments/keyframe_oracle_sampling/EXPERIMENT_PROTOCOL.md` v1.0/A-001;
- `experiments/keyframe_neighborhood_sampling/EXPERIMENT_EXTENSION_PROTOCOL.md` v1.0.

This document overrides inherited rules only where it explicitly defines this
new intervention, population, seed namespace, analysis, or launch gates.
Implementation defects may be fixed to restore these definitions. A scientific
change requires a versioned amendment, not a convenient runtime fallback.

The capacity and hypothesis were chosen after inspection of earlier outcomes.
This is a **post-hoc follow-up with definitions fixed before its new runs**, not
an untouched-test preregistration. Prior work exposed the complete 800-block
formal test population. Record that exposure; do not manufacture an unexposed
subset by deleting previously viewed cases or successful/failed cases.

## 2. Primary scientific question

> If every original Uniform-selected frame is retained, does adding all newly
> visible causal boundary frames improve full-task success over original U?

Old OC prioritizes boundaries within 32 slots and then fills temporal gaps. It
does not preserve the exact U32 set. Losing some uniform history might offset
the benefit of boundaries, but the completed results do not prove that cause.

| Comparison | Role |
|---|---|
| **UK48 − U** | **Only primary comparison:** benefit of retaining U and adding keyframes |
| UN48 − U | Auxiliary: benefit of adding ordinary non-keyframe history |
| UK48 − UN48 | Auxiliary attribution: advantage of the keyframe addition over its random-content control |

UK48 beating UN48 without beating U is not a positive primary result. The
random arm is supporting evidence, not a replacement for the U baseline.

The primary intervention includes expanded capacity, added content, and the
unchanged model's response to that larger input. UN48 shares the expanded
shape but adds real non-boundary content; it is not an empty-slot-only capacity
control. It cannot prove that all capacity or positional effects are removed.
Success here does not establish why old OC failed, that boundaries are optimal
memories, or that privileged boundary labels are available on a real robot.

## 3. Frozen common system and capacity

| Component | Value |
|---|---|
| Policy | released `Yinpei/perceptual-framesamp-modul` |
| Checkpoint | `79999` |
| Existing checkpoint-relative path | `runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999` |
| Checkpoint archive SHA-256 | `2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62` |
| Training / parameter updates | none |
| Representation / integration | perceptual `frame_sampling` / Modulator |
| Base Uniform budget | 512 tokens, at most 32 frames |
| New-arm input capacity | **768 token slots, at most 48 frame slots** |
| Tokens per history frame | 16, unchanged `4 x 4` representation |
| History camera | episode-long front view only |
| Current observations | unchanged front and wrist views |
| State memory | disabled |
| Task input | original task instruction; no symbolic subgoal |
| Action proposal / execution | 20 steps / first 16, then replan |
| Episode limit | official 1300 environment steps |
| Policy RNG | seed 7, reset for each arm trajectory |

All weights, image encoding, pooling/projection, Modulator, VLM and Action
Experts, flow-matching solver, action normalization, robot control, benchmark
termination rules, and environment-seed mapping remain unchanged. No Oracle
actions, candidate branches, winner selection, neighbor expansion, or training
are added.

The existing `capacity_audit.json` motivates 48 slots: all 62,492 observed calls
across U/OC/OC3/OC5 require at most 46 frames for the union of U32 and visible
boundaries. This retrospective observation is not a guarantee for new
trajectories. Freeze 48 now; do not tune capacity against new smoke or formal
success rates. A 50% increase in token slots does not imply exactly a 50%
increase in total latency or GPU memory.

## 4. History, boundary, and exact selectors

### 4.1 Causal history and inherited boundary rule

At a policy call, let `H_t = {0,...,t}` be the complete actually received
front-view history of that arm. Append the new observation segment before
selecting. Selection occurs once per policy call, not at every sensor frame.

For correctly aligned online stage values:

```text
boundary_0 = true
boundary_i = (current_task_index_i != current_task_index_(i-1)), i > 0
B_t = indices with boundary=true at or before t
```

Generated reset/conditioning demonstrations remain in history. Capture their
per-frame stage values before flattening, exactly as in the parent pipeline.
Misaligned or missing labels are a global readiness/invariant failure, not a
reason to guess labels, use reference-H5 timestamps, or substitute Uniform.

Only the selector and audit logging may access boundary/stage metadata. Never
feed those values into prompts, observation features, memory feature values,
or action scoring. No future state, frame, label, terminal outcome, reference
demonstration timestamp, or another arm's future trace is eligible.

### 4.2 Common base and two new arms

Obtain `U_t` by calling the literal original
`MemoryBuffer.get_frame_sampling_indices(t, 512, 16)`. Do not pass 768 to this
base call: that would change the base to Uniform48. Existing endpoints,
`np.linspace` rounding, and short-history behavior are preserved.

```text
new_keys = sorted(B_t - U_t)
Q_t = sorted(H_t - (U_t union B_t))
m = len(new_keys)

UK48 = sorted(U_t union new_keys)
UN48 = sorted(U_t union sample_without_replacement(Q_t, m))
```

Every output is unique, strictly chronological, and uses original absolute
history indices. Every U index must remain present. Boundaries already in U
are not added again. Non-key extras cannot belong to U or B; boundaries that
are already in U must not be removed from UN48.

The final memory gather/padding uses 768 slots. Do not fill spare valid-frame
capacity with Uniform, random frames, duplicates, anchors, or coverage. Use
only zero-valued right padding at the pre-normalization feature-gather boundary
and a false mask, explicitly confirmed by the user. Each valid frame contributes
exactly 16 valid tokens; each unused frame slot contributes 16 false-mask
tokens. Early history of at most 32 frames therefore produces zero extras.

The zero requirement applies to the gathered image, positional and raw-state
placeholder arrays before the released normalization/feature-encoder path.
State normalization can map zero placeholders to nonzero values, and learned
encoder biases can produce nonzero final tokens from zero-padded inputs. State
memory remains disabled and the memory-attention mask must continue excluding
every padded slot. Do not require all final encoded padding tokens to be zero
or patch model operations to impose that new behavior.

### 4.3 Equal count, online divergence, and unsupported cases

Equal count means equal extras **for the same causal-history input**. Separate
closed-loop UK48 and UN48 trajectories can visit different stages and have
different `m` at the same call number. UN48 computes m from its own history;
do not use UK48 counts to force across-trajectory equality, pause an arm, or
change episode length. Report their realized valid-frame counts separately.

UN48 is an **Oracle-count-matched random non-keyframe control**. It uses boundary
metadata to define both m and its exclusions; it is not the old RandomSamp and
not a fully non-Oracle deployable policy. Valid-frame count can itself convey
progress; the same-history rules share that signal.

Both selectors must validate the same eligibility conditions before returning:

- `len(U_t union B_t) <= 48`;
- `len(Q_t) >= m`;
- valid initial boundary, aligned history, causal indices, and valid input types.

Violation saves evidence and raises a hard protocol-invariant failure. It is
not an ordinary benchmark failure and not a cell to exclude. Never silently
clip boundaries, reduce m, sample with replacement, borrow future frames, or
increase capacity. Formal completeness remains blocked pending review and a
versioned amendment if the scientific rules must change. The retrospective
audit observed no capacity overflow or non-key candidate shortage.

## 5. Frozen selector randomness

Use a separate NumPy `Generator(PCG64(seed))`; do not consume or reset the policy
RNG. For each UN48 call, serialize this exact list as UTF-8 canonical JSON:

```text
[2026091001, "uniform_keyframe_expansion-v1", split, task_name,
 episode_id, policy_call_index, "UN48"]
```

`split` is the canonical literal `val` or `test`; all integer fields are JSON
integers. Serialization uses `json.dumps(payload, separators=(",", ":"),
ensure_ascii=True).encode("utf-8")`. SHA-256 the bytes; interpret the first
8 digest bytes as an unsigned big-endian integer. That integer is the PCG64
seed. Sample `m` entries uniformly without replacement from the ascending Q
list using that generator; then sort the selected indices. For `m=0`, log the
same derived seed but return no extras. UK48 has no random-selector seed.

Precompute entries for call indices `0..81`. A call outside this table is a hard
stop. Before GPU launch, save the tables and their SHA-256 values, check no seed
collision within either table, and check the val/test seed-set intersection is
empty. There are 65,600 test entries (16×50×82) and 1,312 val entries (16×1×82).
Short and terminal smoke use the same val episode and may intentionally share
the same matching call prefix; do not count those as separate seed-table rows.

Pin the NumPy version in execution provenance. Test reproducibility and the
complete payload/digest mapping. Never choose a random realization after
examining its behavior. Policy seed 7 remains intentionally common across
arms and is not subject to selector-table disjointness.

## 6. Formal population and essential U-baseline gate

Use the canonical task order:

```text
BinFill, StopCube, PickXtimes, SwingXtimes,
ButtonUnmask, VideoUnmask, VideoUnmaskSwap, ButtonUnmaskSwap,
PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder,
MoveCube, InsertPeg, PatternLock, RouteStick
```

The two new arms cover `test`, episode IDs `0..49`, task-major then
episode-major then arm order `[UK48, UN48]`: **1,600 new full trajectories**.
Keep the parent episode-to-environment-seed and difficulty mapping; do not
invent a new mapping from the selector seed.

U is required for the primary question. Prefer reusing immutable original U
records, but freeze the exact U source only after a baseline audit verifies:

- all 800 task/episode scientific outcomes, raw provenance and checksums;
- environment split, resolved seed, difficulty, task text and initial state;
- initial front/wrist, robot-state and task-state evidence, including reset
  demonstration alignment;
- checkpoint identity, unchanged control pipeline, benchmark/software identities
  and a justified compatible hardware/execution environment.

New UK48/UN48 pairs must have byte-identical initial image/state/text hashes and
the same policy reset seed. Comparing new arms to U additionally requires a
reviewed provenance/initial-input alignment record. Same seed, visually similar
video panels, or a matching task name is not proof of byte alignment. Missing
historic evidence must be recorded as missing, not marked PASS.

If old U cannot support the comparison, **formal launch is blocked** until that
issue is resolved. A proposed fresh U run needs independent user authorization;
this protocol does not authorize an extra 800 U trajectories. Do not demote U
or change the primary test to UK48−UN48 to bypass this gate. The earlier approved
OC3/OC5 consistency reruns do not authorize a U rerun for this study.

The baseline run ID, validated artifacts, execution host/stack, and comparison
attestation remain operational launch blockers at v0.9. Fill and review them
before v1.0; do not claim scientific readiness merely because CPU tests pass.

## 7. Runtime integration and positional-effect disclosure

Add a new experiment family with independent 48/768 validation. Keep original
U/O/OC/R and OC3/OC5 defaults, traces, matrices and 32/512 acceptance criteria
unchanged. Reuse their causal instrumentation and immutable-writing machinery
where compatible; do not relax their contracts globally.

The checkpoint loader normally reloads `history_config.txt`. Introduce an
explicit, logged inference-only budget override for this family after resolving
that metadata. Do not edit checkpoint-side metadata, original YAML defaults, or
weights. Record both released and effective config; require strict weight
loading without missing, extra or randomly initialized parameters.

The experimental policy must reject inference until a valid UK48/UN48 selection
configuration has been supplied. It must never silently run Uniform48 while
waiting for the experimental adapter. Strict actual model loading is required;
checking a configuration dictionary alone does not satisfy the checkpoint gate.

Expected prepared single-sample shapes are image `[768,2048]`, positional
features `[768,768]`, disabled-state placeholder `[768,8]`, and bool mask `[768]`.
Expected final perceptual memory is `[1,768,1024]`; action proposal is `[20,8]`.
The placeholder does not activate state memory. Actual real-checkpoint support
remains to be established by GPU smoke, not inferred solely from shape code.

`MemoryAttention` places action queries beginning at `mem_len` and keys beginning
at zero. Consequently 512→768 changes query RoPE positions, even if the same U32
content is used and additional slots are masked. Keep this model behavior
unchanged. UK48 and UN48 share it; UK48−U measures the complete new scheme rather
than isolated added semantic content. Do not silently patch RoPE or relabel an
altered-U model as the original baseline.

A same-input U512-versus-padded-U768 diagnostic is required in GPU smoke. Report
action/velocity differences; nonzero differences alone do not fail the gate.
It is not an extra formal arm and must not be used to choose settings by success.

## 8. CPU, GPU, and pre-formal gates

### 8.1 Authorized local implementation gate

Pass dependency-light unit/fixture tests for:

1. history lengths 1, 2, 16, 32, 33, 64 and 1301; exact literal U preservation;
2. key/U overlap, repeated boundary indices handled by a set, chronological
   uniqueness, early zero extras, and no additional valid-frame filler;
3. same-history equal extras and randomized candidate exclusions;
4. explicit overflow, shortage, missing-initial-boundary and malformed-input
   failures; no silent fallbacks;
5. strict prefix causality and generated-demonstration label alignment;
6. canonical seed derivation, all tables, reproducibility, isolation from policy
   RNG, and out-of-table rejection;
7. 48/768 masks/shapes/counts and unchanged original 32/512 regression behavior;
8. trace completeness, atomic output, immutable attempts, duplicate rejection,
   resume validation, reset and exception cleanup;
9. exact 1,600-row new formal and 48-row smoke manifests, without modifying old
   3,200-row or 1,600-row families;
10. analysis fixtures, paired denominators, confidence intervals, primary/auxiliary
    hierarchy and the two-comparison Holm adjustment.

CPU PASS proves only tested code behavior. Distinguish tests exercising the
real memory-buffer gather from tests using a mocked model loader: the latter
can verify strict-load arguments and control flow but cannot establish that the
actual checkpoint loads. Complete episode artifacts, remote adapter lifecycle,
statistics integration and launch plumbing require their own tests; a pure
selector plus protocol does not mean the runner is ready for GPU submission.

### 8.2 GPU architecture smoke: separate authorization required

With a clean committed server checkout and a fixed real checkpoint, test short
and long causal development histories for both arms. Verify effective768 config,
strict checkpoint load, masks, finite outputs, same-live-process repeatability
for identical inputs/indices/noise, and reset isolation. Same selected indices
in the same live process must yield the same memory digest. Check the separate
padded-U positional diagnostic from Section 7 without asserting old-U equality.

Measure peak allocated GPU memory, warm/cold inference latency, selector latency
and full-request latency. Model residency across episodes and shared GPUs are
allowed when memory/RNG reset and resource checks pass. Do not interfere with
other users or unload solely because one episode completed. Resource choices
must not change scientific settings.

### 8.3 End-to-end development smoke: 48 trajectories

Use canonical `val`, episode 0 of all 16 tasks. Both UK48 and UN48 run a short
trajectory per task (32 total), ending at the earlier of a valid official
terminal and 64 environment steps. Never step past or suppress an early
terminal to reach 64. Also run 16 full terminal-path trajectories: zero-based
even task indices use UK48; odd indices use UN48. Use fresh episode/memory/RNG
resets and distinct artifact keys for short and terminal scopes.

The 16 full checks stop at normal terminal or the official 1300-step limit. They
must exercise the same recording, accumulated-history and finalization paths
as formal execution, including InsertPeg reset video and repeated-stage tasks.
Smoke success/failure/timeout is readiness evidence only. A genuine benchmark
`error` is preserved but makes smoke fail closed for review; it is not converted
to a retryable infrastructure outcome.

The val/test selector tables must be disjoint as specified in Section 5. Do not
use formal test episodes for development, compare smoke success rates, remove
difficult tasks, or choose methods/capacity from smoke outcomes.

### 8.4 Pre-formal freeze

After authorized smoke, publish a readiness report with commits, all tests,
causality/reset proof, examples, exact shapes/masks/seeds, positional diagnostic,
latency/resource measurements, trace/output verification, baseline attestation,
remaining risks and estimated full-run resources. Resolve baseline/environment
blockers and freeze protocol/analysis/manifests as v1.0. Formal launch needs
separate explicit user authorization. No launch command may infer approval
from this v0.9 document or from earlier experiment authorizations.

## 9. Outcomes and statistical plan

Primary outcome: official binary full-episode success. Estimand: equal-task-
weighted macro success-rate difference UK48−U, in percentage points, paired by
task/episode. Failures, timeouts and true benchmark errors are not successes.
Report success numerators/denominators and discordant pairs for transparency.

For each of the three Section 2 comparisons use the parent analysis method:

- 100,000 paired hierarchical bootstrap replicates, resampling tasks and then
  paired episodes within each sampled task;
- percentile 95% CI using empirical 2.5th and 97.5th percentiles;
- 100,000 two-sided paired label-swap permutations, equal task weights, with
  `(1 + count(abs(delta_perm) >= abs(delta_observed))) / 100001`;
- analysis seed `2026082502`, with an explicitly shared deterministic schedule
  across comparisons, and recorded analysis implementation identity.

The sole primary p-value is reported raw. For the two auxiliary comparisons
UN48−U and UK48−UN48, report raw p-values plus Holm step-down adjustment jointly
across exactly those two tests. Report their unadjusted effect sizes and 95% CIs;
do not present those intervals as multiplicity-adjusted. Per-task, four-task-
group, progress, boundary-count, memory-size and latency analyses are exploratory.
Do not promote an attractive exploratory subset to the primary population.

The practical-effect threshold remains **+3.0 macro percentage points**:

- Primary statistically positive: UK48−U CI lower endpoint is strictly above 0.
- Practically positive primary result: the above criterion and point estimate
  at least +3.0 pp. Beating UN48 is **not** required for this primary conclusion.
- Practical benefit ruled out under this setup: upper endpoint is below +3.0 pp.
- Otherwise distinguish small detected effects from imprecision; a CI crossing 0
  is not proof of no effect. Report the raw primary p-value alongside the CI.

These statements concern different questions: a small positive effect can be
detected while a 3 pp benefit is ruled out. Do not force it into a misleading
single winner label. Auxiliary evidence can strengthen key-specific attribution,
but lack of significance against UN48 does not prove equality or that all gain
comes from capacity. UK48 beating UN48 but not U does not validate the main
hypothesis. A negative result does not reject all memory methods or retraining.

Freeze an exposure manifest naming the full prior-viewed test census and its
evidence sources before opening new outcomes. No exposure-based exclusion is
planned: all 800 blocks are already exposed. Do not rerun the original parent's
unexposed-subset sensitivity as an empty or selectively constructed analysis.
No formal statistical report is valid without the complete new matrix and the
audited 800 U baselines. Never silently reduce paired denominators.

## 10. Trace, artifact and failure contract

At each call record at least:

- family, arm, task, split, episode, environment step and policy-call index;
- history length/latest absolute index and visible boundary indices;
- literal base-U indices, newly missing key indices, non-key candidate count;
- m, seed/payload identity for UN48, sampled extras and final indices;
- effective768 budget, valid/padded frames and tokens, masks and tensor digests;
- selected-frame identities, unchanged policy seed, timing and config identity.

Record run/protocol/implementation/benchmark/checkpoint hashes, resolved seeds,
initial inputs/state/text hashes, environment versions, hardware, model reset,
commands, timestamps, terminal metadata and output checksums. A boundary count
at the last policy call is an exploratory observed-progress proxy, not an exact
completed-stage count or final physical failure diagnosis.

Use a new write-once root `runs/uniform_keyframe_expansion/<run_id>/`, separate
from every prior family, with protocol/launch/seed/exposure snapshots, immutable
per-trajectory attempts, selector traces, scientific results, logs, and a failure
ledger. Include scope in smoke keys to distinguish short and terminal runs.
Final aggregate outputs are completeness report, per-episode and per-task
tables, summary JSON and analysis Markdown. Derived outputs must identify every
source result and reference checksum; never edit raw measurements.

Ordinary benchmark failure, collision, timeout, or actual benchmark `error` is
an immutable scientific outcome; never retry it to improve success. Documented
transport, scheduler, filesystem, node or GPU infrastructure failure may use
at most two fresh-reset infrastructure retries with exactly the same seeds in
new attempt directories. Preserve every failed attempt and accept only the
first valid scientific result for a cell. Never resume a reconstructed
mid-episode state or overwrite a result.

Selector/config/checkpoint/label/serialization defects and capacity/shortage
violations are experiment hard stops, not automatically retryable transient
errors. Fix code only if restoring the frozen protocol, validate again, and
seek the appropriate execution authority. A scientific-rule change requires an
amendment. No result cell can be silently excluded or substituted; completeness
fails closed until every required cell has exactly one valid scientific result.

## 11. Current deliverable and next permitted step

At v0.9 the scientific selector, 48-slot capacity, tasks, episodes, RNG and
analysis definitions above are fixed for implementation. Local code/tests and
server handoff documentation may be prepared. The launch host, U-reference
attestation, real-checkpoint768 validation, end-to-end smoke, and frozen v1.0
record are still outstanding. Neither this document nor its presence in Git
means the runner is GPU-smoke-ready or formal execution has started.
