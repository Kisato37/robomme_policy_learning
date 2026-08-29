# Pre-formal audit: causal boundary Oracle sampling

**Review date:** 2026-08-29

**Smoke-tested implementation:** `899912b11a379346b4c8f4d6c80f54f07c118ea3`

**Protocol reviewed:** v0.9.1 plus amendment A-001

**Freeze decision:** PASS; freeze the unchanged scientific definition as v1.0

**Formal execution:** NOT STARTED

This report satisfies Section 17 of `EXPERIMENT_PROTOCOL.md`. It audits
plumbing, causality, determinism, fairness, provenance, and resource planning.
It does not compare smoke success rates and is not evidence that one selector
performs better than another.

## 1. Immutable evidence and verdict

The accepted run root is:

`runs/keyframe_oracle_sampling/20260829T181633Z_899912b1_devsmoke`

The root was created from a clean worktree at the exact candidate commit above.
Its relevant immutable evidence is:

| Evidence | SHA-256 |
|---|---|
| v0.9.1 protocol snapshot | `0008eb9c5704f6477ef79d7b54cb59d57c5129522147baced3398d17dccd13f3` |
| launch manifest | `8788771455b9be314f03068c692007485f2f0cba615e0a32fa3ac3f72d9d3d54` |
| architecture report | `e2ec341ff6bf2f0840baa5ef9707ac7d1ede83a5faa6d229aab15fdde8c00078` |
| development submission record | `e3dfe429d5d40afbd0c5121e0b9256e5b0f8fcc2afa72e2aedacec541d5daf0c` |
| strict smoke audit | `fc1b7c95d0c4b7d7e5353e31a27f31dbef80a6e26c1872b6e5fba6f768d9729e` |

The strict audit reports:

- `passed=true`, `formal_started=false`;
- exactly 80 of 80 expected trajectories, with no missing or unexpected key;
- all 64 short rows ended at an official terminal state or the 64-step cap;
- all 16 four-arm blocks passed initial-condition equality;
- all 16 four-arm blocks matched every declared non-treatment manifest field;
- no failure ledger in the accepted run;
- 1,312 development seeds and 65,600 formal seeds, with intersection size zero.

The architecture smoke was Slurm job `186299`: 8/8 cases passed, job state
`COMPLETED`, exit `0:0`, elapsed `00:01:48`. The development array was
`186303`: all 80 rows completed with exit `0:0`. Four rows ran concurrently,
each with two GPUs. The first row started at 14:19:59 and the final row ended at
14:59:46 local time, a 39 minute 47 second execution window.

## 2. Implementation commits and diff summary

The keyframe-specific commit sequence reviewed for this experiment is:

| Commit | Purpose |
|---|---|
| `f822ae75a8c578e0f8cb6b2d5a5a4888b8f0624e` | Add the authoritative causal keyframe protocol and A-001 clarification. |
| `1b9a4f3a21f4190b3a1817034e24bb56442fb0ee` | Implement selectors, causal stage instrumentation, runtime selection, artifact contracts, analysis, and smoke launchers. |
| `12dfe7f3eae0814fbb7b73e948b3d0055f0662ff` | Correct the frozen unpacked-checkpoint metadata identity. |
| `f70a1ed24347da70f6083fd229375ebb42d67662` | Digest typed JAX PRNG keys through their raw key data. |
| `f30d822ed0e7e7ede11771d52af041b45c9c77f9` | Validate the released padded-memory dtype specialization. |
| `b94083ca79f15631a16c758572337fcfe614c07e` | Audit the two released dtype compile specializations and cache stability. |
| `d3d545d8392f834157f9ecb4041e0e7d0f258fb6` | Bind policy and simulator to the two GPUs actually allocated by Slurm. |
| `feb21e3048df44c1165490c570d9820a7a2d99a5` | Use the available smoke partition while retaining a strict partition allowlist. |
| `899912b11a379346b4c8f4d6c80f54f07c118ea3` | Remove only process-local object addresses from visual highlight actor names before hashing initial task state. |

The last three fixes were the only post-`b94083c` implementation changes.
They corrected GPU binding, scheduling provenance, and a non-state process
address in the fairness hash. They did not change an arm, selector algorithm,
model input, checkpoint, seed, metric, trajectory, or decision rule.

## 3. Commands and validation gates

The final candidate passed the following gates:

```bash
.venv/bin/python -m pytest -q tests/keyframe_oracle_sampling
bash -n experiments/keyframe_oracle_sampling/run_architecture_smoke.sbatch
bash -n experiments/keyframe_oracle_sampling/run_smoke.sbatch
```

The complete keyframe suite reported 162 passed at the smoke-tested candidate.
Scoped Python lint and both launch-script syntax/dry-run checks also passed.
After propagating the frozen v1.0 version and adding the version/guard
regression, the same suite reported 163 passed. The scoped critical-error lint
`ruff check --select E9,F63,F7,F82` passed on every changed Python file.

The accepted GPU and end-to-end commands were:

```bash
.venv/bin/python -m experiments.keyframe_oracle_sampling.prepare_smoke \
  --run-root runs/keyframe_oracle_sampling/20260829T181633Z_899912b1_devsmoke
.venv/bin/python -m experiments.keyframe_oracle_sampling.submit_architecture_smoke \
  --run-root runs/keyframe_oracle_sampling/20260829T181633Z_899912b1_devsmoke
.venv/bin/python -m experiments.keyframe_oracle_sampling.submit_smoke \
  --run-root runs/keyframe_oracle_sampling/20260829T181633Z_899912b1_devsmoke
.venv/bin/python -m experiments.keyframe_oracle_sampling.audit_smoke \
  --run-root runs/keyframe_oracle_sampling/20260829T181633Z_899912b1_devsmoke
```

The actual `sbatch` arguments, array scope `0-79%4`, job IDs, exact repository
commit, paths, checkpoint digests, environments, and timestamps are preserved
in the run root's protocol directory.

## 4. Literal U equivalence

Arm U calls the released
`MemoryBuffer.get_frame_sampling_indices(step_idx, 512, 16)` method directly.
It is not routed through an emulation of the new selector code.

Evidence:

- CPU fixtures compare that literal method against the pinned official uniform
  rule across short and overflow histories.
- Runtime tests spy on the literal method and require the exact call
  `(63, 512, 16)`.
- Both live GPU U cases have
  `literal_uniform_indices_match=true`.
- Repeating each live case in the same process produced identical selected
  indices, memory tensor digest, and action digest.
- The architecture audit also verified the released padded and unpadded dtypes,
  the fixed `[1, 512, 1024]` final memory, and `[20, 8]` action shape.

## 5. Causality and reset-demonstration alignment

The selector can read only stage metadata paired with front frames already in
history. It cannot read future stages, terminal outcomes, or success metadata.
Stage metadata is stored separately from model features and is never packed into
the model input.

Evidence:

- `test_future_boundary_changes_are_causally_invisible` changes only future
  stages and requires the current selection to remain identical.
- Reset demonstration tests require exactly one live stage value for every
  front frame and hard-stop on missing or misaligned labels.
- Instrumentation is installed before environment construction/reset, so
  demonstration frames are labelled online rather than reconstructed later.
- The strict audit recomputed every recorded selection from that call's
  `history_length`, current index, visible boundaries, arm, and preregistered
  seed. All 599 policy calls passed.
- The final InsertPeg trace, which exercises conditioning-video history, records
  only visible boundaries `[0, 115, 242]` at its first call with current
  history index 242; later selections use those already-observed boundaries.

## 6. Selector examples and memory contract

The architecture fixture uses boundaries every seven frames.

For a normal 16-frame history:

- U, OC, and R select `[0, 1, ..., 15]`; all available frames fit.
- O selects only boundaries `[0, 7, 14]` and masks the remaining 29 frame
  slots.

For an overflow 64-frame history:

- U: `[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,63]`
- O: `[0,7,14,21,28,35,42,49,56,63]`
- OC: `[0,1,2,3,4,5,6,7,10,12,14,17,19,21,24,26,28,31,33,35,38,40,42,45,47,49,52,54,56,59,61,63]`
- R: `[0,3,6,8,10,12,13,16,17,19,22,23,26,28,30,32,33,36,37,40,42,43,45,47,49,52,53,55,58,60,61,63]`

Each selected frame contributes 16 tokens. The complete live architecture
counts and digests are:

| Arm | History | Valid / pad frames | Valid tokens | Selected-index SHA-256 | Mask SHA-256 | Final-memory SHA-256 |
|---|---:|---:|---:|---|---|---|
| U | 16 | 16 / 16 | 256 | `4939a292c2f5164ddcf27a07ddc7ef96928baa3e8967453a7294a9ecebf0a5c3` | `e27ea1e253194f2e87100d10e87cc7eae614add3c42c12c8b13ffd97344e79fd` | `96f4f4f7985dcc8b8923622804baab30e0b19e2558cd99adf9878c85955136ea` |
| U | 64 | 32 / 0 | 512 | `29ac73db81982e1c3d1d432fc973082a05d6fb2942d9b04d17c93131e6d754d4` | `9f598ae89556d5d46720691e648e22a611109c89f4c0cea44728848c23972ba8` | `4dd827b26a605ab189f5da964d9d95217e34734838b4aeb301d0b2b719bfcae2` |
| O | 16 | 3 / 29 | 48 | `e7e2d1666d6959106f467158173242240c92b205d1b2c00b8d004fa1b7e96172` | `27f664490a41b71a470b2fef77513413c53884b31327a4c301236146be4da8f0` | `811ba77ed5fb3283b91a75fd00ebfabc50c0f5a5396fc3efd7b5b3eb9d7e4d46` |
| O | 64 | 10 / 22 | 160 | `356f320f61d084ddf2d64e6fe3b05d7360a09d4d96d5a51878440ab90c7f7ebb` | `5e758bc4e8bbdd1007621dbd4e9475f55ba53990f5dd62a969db7da7271dbe41` | `8cffb2e16f6046b6b90b9f1b1b9a16a5181ea9f37799b6c08bb76e04f8fd7a46` |
| OC | 16 | 16 / 16 | 256 | `4939a292c2f5164ddcf27a07ddc7ef96928baa3e8967453a7294a9ecebf0a5c3` | `e27ea1e253194f2e87100d10e87cc7eae614add3c42c12c8b13ffd97344e79fd` | `96f4f4f7985dcc8b8923622804baab30e0b19e2558cd99adf9878c85955136ea` |
| OC | 64 | 32 / 0 | 512 | `7f3e06a44ef2b52eb1e7a06667e078ef7d76f9b22c642335956fb2d7588b2c6c` | `9f598ae89556d5d46720691e648e22a611109c89f4c0cea44728848c23972ba8` | `8aeb0f3d49218861bad1a7c1c000f53e97f23c7e95a3e4a6c20cdbffc9d55784` |
| R | 16 | 16 / 16 | 256 | `4939a292c2f5164ddcf27a07ddc7ef96928baa3e8967453a7294a9ecebf0a5c3` | `e27ea1e253194f2e87100d10e87cc7eae614add3c42c12c8b13ffd97344e79fd` | `96f4f4f7985dcc8b8923622804baab30e0b19e2558cd99adf9878c85955136ea` |
| R | 64 | 32 / 0 | 512 | `b809490e8a8fa84d084e2ca14baf29aa09303e9f51c497f36fc480b585d8f8be` | `9f598ae89556d5d46720691e648e22a611109c89f4c0cea44728848c23972ba8` | `31c9a1b7f1bfd9e31160c325ef4ae88a7afa091960b1e4d12af89c20ca1bdf22` |

All eight cases produced finite bfloat16 final tensors with shape
`[1, 512, 1024]`. Padding masks were a true valid-token prefix followed only
by false padding.

## 7. Random-seed reproducibility

The complete development and formal RandomSamp seed universes were generated
before launch. Their scopes, datasets, derivation strings, entry digests, and
file digests are recorded in the launch manifest.

- development: `development-smoke-v1`, `val`, 1,312 unique seeds;
- formal: `formal-evaluation-v1`, `test`, 65,600 unique seeds;
- intersection: zero;
- development entry digest:
  `a57fd2bf7e8551b2bbaf24707dd3298590d4269c7761a8a2a132640f255a2c5e`;
- formal entry digest:
  `296dbd435906a0affe4239472b0805894aa2e97c01c4f1f093212cc27cdef65b`.

Every R trace recorded its call-specific preregistered seed. The strict audit
re-derived it, recomputed the exact indices, and rejected any mismatch. Repeated
architecture cases produced the same R indices, memory digest, and action
digest in the same live process.

## 8. Policy reset, pairing, and cross-arm isolation

Before every architecture case, reset evidence required empty frame and metadata
history, selector call index zero, cleared selector configuration/RNG/pending
trace, policy RNG reset, and step index -1. The final post-grid reset passed the
same checks.

Every development trajectory was a separate process and fresh simulator reset.
The evaluation policy seed was reset to 7 for each arm. Across the 16 paired
short blocks, all five initial-condition hashes and all non-treatment manifest
fields matched. Those fields include split, step cap, action horizon, checkpoint,
policy seed, RandomSamp table digest, environment seed, and difficulty.

## 9. Selector and policy latency

The accepted 80 trajectories emitted 599 policy-call traces. Selector latency
includes causal-boundary lookup, index selection, and selector bookkeeping, but
excludes feature gathering and padding.

| Arm | Calls | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|
| U | 147 | 39.00 | 43.38 | 50.06 | 60.34 | 76.11 |
| O | 193 | 63.27 | 64.52 | 82.17 | 83.32 | 84.64 |
| OC | 121 | 51.06 | 47.99 | 96.19 | 101.12 | 106.37 |
| R | 138 | 39.32 | 42.13 | 51.15 | 59.51 | 125.73 |

The model-only latency summary is:

| Arm | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| U | 4078.07 | 109.38 | 22415.60 | 25697.32 | 26285.59 |
| O | 2454.98 | 109.23 | 22758.09 | 25653.61 | 27982.11 |
| OC | 4892.80 | 109.24 | 24649.99 | 28186.24 | 29840.42 |
| R | 4246.70 | 109.33 | 23448.23 | 27541.24 | 28456.48 |

The roughly 22-30 second tail is per-process JAX cold compilation; steady calls
are represented by the approximately 109 ms medians. Call counts differ because
the protocol rotates one full terminal-path trajectory per task during smoke.
These figures are systems diagnostics, not selector-performance evidence.

## 10. Output, resume, and immutability

The accepted root contains exactly 80 `attempt_00` directories. Each has one
manifest, initial-condition record, selector trace, result, and video. There are
no later attempts, temporary/partial files, duplicate scientific keys, or
failure ledger.

Artifact tests verify:

- atomic writes refuse overwrite;
- a completed attempt refuses trace append or result replacement;
- resume validation detects trace tampering through its digest;
- duplicate completed scientific keys are rejected;
- retries require the immediately preceding append-only infrastructure failure;
- completeness requires the exact matrix without missing or unexpected cells.

The final strict audit exercised those readers against the complete real smoke
root; generated evidence was not edited or regenerated.

## 11. Failed candidates and v0.9.1-to-v1.0 changes

All failed evidence was preserved under separate run roots:

- `20260829T093544Z_b94083ca_devsmoke`, job `186158`: canceled after Slurm
  GPU allocation was found to be ignored; no scientific action had started in
  the affected attempts. Fixed by `d3d545d`.
- `20260829T164633Z_d3d545d8_devsmoke`, job `186179`: architecture launch
  stopped before inference because a moved job violated a hard-coded partition
  check. Fixed by `feb21e3`.
- `20260829T170511Z_feb21e30_devsmoke`, jobs `186183` and `186185`:
  architecture passed and all 80 rows completed, but the strict fairness audit
  rejected InsertPeg because visual-only highlight actor names contained
  process-local object addresses. Fixed by `899912b`, preserving all actor
  state values and multiplicity.

There is no scientific protocol deviation. Version v1.0 changes only the
administrative status/version and propagates that version into future
manifests. The arm definitions, causal boundary, checkpoint, memory budget,
matrix, seed formulas, outcomes, statistics, and GO/NO-GO rules are unchanged
from amended v0.9.1.

Known limitations:

- architecture smoke is synthetic and development smoke uses only `val`
  episode 0;
- smoke validates correctness and fairness, not comparative success;
- only one rotated full terminal trajectory per task was run;
- a dedicated formal launcher is not yet implemented, and direct formal use of
  `examples/robomme/eval.py` remains hard-disabled.

## 12. Formal scale and resource estimate

The frozen formal matrix is exact:

- 16 tasks x 50 test episodes x 4 arms = 3,200 trajectories;
- at most 82 policy calls per trajectory = 262,400 policy calls;
- the 16 rotated terminal smoke trajectories used 343 calls, which scales to a
  rough point estimate of 68,600 calls over the formal matrix if episode lengths
  are similar.

The 16 terminal smoke trajectories consumed 1,113.28 pair-seconds in total.
Scaling each task's observed runtime by 50 episodes and 4 arms gives 61.85
GPU-pair-hours. With four concurrent two-GPU rows, the idealized wall estimate
is 15.46 hours. A conservative execution envelope is approximately 16-27 hours
before queue delay: 27 hours corresponds to scaling the observed 1,300-step
terminal case across every formal trajectory. Test episodes, filesystem load,
and scheduler contention can extend that envelope, so the launch should reserve
up to roughly two days.

The planned Slurm shape is one node, two GPUs, and 12 CPUs per row; array
`0-3199%4`; at most four concurrent rows, eight GPUs, and 48 CPUs. Unique ports,
no requeue, readiness timeout, trap cleanup, and append-only attempts must match
the proven smoke pattern.

This estimate is planning information only. No formal run root has been created,
no formal shard has been submitted, and no formal test outcome has been opened.

## 13. Review conclusion

All required pre-formal items are present and the evidence supports freezing
the unchanged scientific definition as protocol v1.0. The next safe engineering
step is to implement and dry-run a dedicated formal launcher against the frozen
matrix and provenance gates. Launching the 3,200 trajectories remains a
separate action requiring explicit user authorization.
