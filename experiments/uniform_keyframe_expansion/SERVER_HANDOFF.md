# UK48 / UN48 server preparation handoff

Date: 2026-09-10. Protocol: `EXPERIMENT_PROTOCOL.md` v0.9.

Latest authority update: the user explicitly authorized scoped Git commit/push,
Lighthouse synchronization and server CPU checks. See
`SYNC_AUTHORIZATION_20260910.md`. GPU smoke and formal execution remain separate,
unauthorized stages; no launch permission follows from this repository update.

## Current authority and readiness

The user has authorized local protocol/code/test work and has confirmed that
unused positions in the fixed 48-frame memory capacity are zero padded and
false masked. Zero padding is checked on feature-gather arrays before released
normalization/encoding; normalized state placeholders or final encoded tokens
can be nonzero. State memory stays disabled and padded slots remain attention
masked. Do not change the encoder to force final encoded zeros.

This handoff is preparation guidance, not permission to submit
GPU smoke, run the formal 1,600 trajectories, rerun U, push Git, or change a
shared environment. Do not start those operations without the relevant explicit
instruction. Do not reuse authorization from the older OC3/OC5 experiment.

The new protocol fixes the scientific definitions for implementation. The
independent websocket service/client, episode evaluator and immutable artifact
store have CPU integration tests, including real loopback transport and the
production memory-buffer gather with fixture features. The separate execution
plan, production bootstrap, resident controller and worker-ownership checks are
now implemented and CPU-tested. They are not an already dispatched server run.
The separate architecture diagnostic is now implemented and CPU-tested; actual
real-checkpoint execution, server environment/source binding, server Linux
validation and U-baseline attestation remain outstanding. Lighthouse remains
the intended execution host; Athena is only a read-only old-U evidence source.
Do not infer a usable remote launch command from a manifest or readiness report.

## Read before doing anything

Read repository `AGENTS.md`, then the entire new `EXPERIMENT_PROTOCOL.md`, its
`DESIGN.md` v0.2, and the two parent protocols referenced there. Inspect the
current branch, working-tree changes and exact source revision. Preserve raw
results/checkpoints and unrelated work. Use a dedicated `exp/*` branch; the
writable remote is `lab`, never `origin`, and pushing needs authorization.

The essential study question is **UK48−U**, not UK48−UN48. UN48 provides an
auxiliary control for adding ordinary historical information. Both new arms
retain the literal original U32 selection and add only the specified extras,
then gather into 768 token slots. They must not quietly become Uniform48.

## Interface inspection checklist

The current local implementation provides an isolated
`UniformKeyframeExpansionPolicy` and a loader opt-in named
`experimental_memory_expansion="uniform_keyframe_expansion-v1"`. Confirm these
interfaces and their tests against the actual checkout; their presence is not
a claim that the remote adapter already supports them. The policy is configured
through `configure_uniform_keyframe_expansion(config)` after a fresh reset;
`contract.build_selector_config(row)` supplies the frozen episode/call context.

The new `serving.ExpansionClient` uses an independent operation envelope and
requires the expected resident execution identity. Do not use the old client's
reset or the old OC3/OC5 proxy to transport this new family. The new service
keeps one active episode connection, verifies reset state and latches model or
protocol failures until review; readiness connections do not claim an episode.

`evaluator.evaluate_attempt` accepts a validated store/row, original `EnvRunner`
factory, bound new client factory and `BenchmarkComponents` containing the
unchanged `EpisodeState`, `pack_buffer`, `RolloutRecorder` and demo-task list from
`examples/robomme/utils.py`. It runs exactly one fresh episode, closes its client
and environment, and never retries automatically. `episode_provenance` must
contain `policy_execution_identity` matching the client/server handshake.
Use the original resolver: the local U audit found 72 seeds that cannot be
recreated by a simplified task-base-plus-episode formula.

- Resolve original checkpoint configuration, require released 512/16 settings,
  deep-copy before applying this family's explicit 768-budget override, and
  preserve the original metadata file and old-family defaults.
- Require actual strict weight loading, with no silent removal of extra
  parameters and no missing/random parameters. Current integration uses
  `remove_extra_params=False` for the new family; real GPU loading remains a gate.
- Require a configured UK48/UN48 episode/call context before inference. The new
  policy must fail closed if unconfigured instead of executing Uniform48.
- Derive the base from original `get_frame_sampling_indices(t, 512, 16)`; only
  final gather and padding use 768. Apply the causal labels, overflow/shortage
  checks, isolated selector RNG, logging and reset lifecycle from the protocol.
- Do not change old selection/padding validators to accept 768 globally. Keep
  original U/O/OC/R and OC3/OC5 behavior and artifacts reproducible.

## Next development work before any GPU submission

Local verification is recorded in `LOCAL_VALIDATION.md`: the combined suite
passed 768 tests and 202 subtests, with 24 Linux-specific skips on macOS. The
pure statistical core (`analysis.py`) and selector trace checks
(`trace_validation.py`) are implemented. Their tests do not replace remote
artifact ingestion, source attestation or end-to-end evaluation integration.
The subsequent runtime integration and baseline inventory are recorded in
`RUNTIME_VALIDATION.md` and `BASELINE_EVIDENCE_AUDIT.md`. The earlier test counts
are a historical checkpoint, not the final combined count for these additions.
The final combined CPU run after the runtime/ingestion additions passed
916 tests and 202 subtests, with the same 24 Linux-only skips. See the runtime
report for the exact command, final JUnit checksum and the fixed post-terminal
artifact-error retry regression; no real-checkpoint or simulator claim follows.
The subsequent launcher/bootstrap/ownership/source-recovery stage passed
1,087 tests and 202 subtests, again with 24 Linux-only skips. The latest result,
evidence recovery and remaining gates are in `SERVER_PREPARATION_VALIDATION.md`.
The subsequent architecture implementation passed the combined regression with
1,167 tests and 202 subtests, with 24 Linux-specific skips. See
`ARCHITECTURE_IMPLEMENTATION_VALIDATION.md` for exact evidence and the fresh
read-only Lighthouse inventory. This is not an actual GPU architecture PASS.

1. Review the actual local test report and repeat the relevant CPU tests in the
   pinned server checkout once synchronization is authorized. Inspect the new
   manifest/seed/readiness tools; a read-only readiness validator is not a runner.
   Real-buffer CPU fixtures validate gathering and pre-normalization padding;
   mocked loader tests do not prove actual checkpoint compatibility.
2. Validate the new governed server launcher against the actual clean server
   checkout. Bind actual source revision, run/row/attempt,
   resident execution identity, physical renderer GPU, checkpoint and environment
   evidence before dispatch. Do not loosen the old dispatch types or reuse an old
   family's preflight. Keep the original EnvRunner's causal instrumentation and
   numerical setup. `resident_controller` now supplies the trajectory CLI, but
   rejects unvalidated plans, absent stage authority, CPU-only stages and
   architecture-only stages. The separate `architecture_controller` and
   `architecture_probe` now implement that diagnostic; validate them against the
   actual checkpoint rather than synthesizing a passing report from CPU tests.
3. Audit the existing U source and chosen environment. The original 800 U
   metadata/hash records were recovered read-only from Athena; see
   `ATHENA_U_EVIDENCE_IMPORT.md`. This resolves missing hash values, not comparison
   with the future new runtime. Record remaining missing evidence
   honestly. Initial-state/image mismatch must not be dismissed because images
   look similar. If U cannot be reused, stop and propose a separately authorized
   baseline solution; do not automatically add 800 U runs.
4. Prepare, but do not submit, the protocol's development smoke: real-checkpoint
   architecture checks, same-input 768 determinism, padded-U512 versus U768
   positional diagnostic, and 32 short plus 16 full val trajectories. Maintain
   separate scopes and reset state between episodes; keep model residency where
   validated rather than unloading after every episode.
5. Verify new-family episode-artifact acceptance and statistical
   analysis integration against actual server outputs. The read-only
   `outcome_ingestion.ingest_formal_run(run_root)` now checks the exact 1,600-cell
   new-arm census and the 800 UK48/UN48 initial-condition pairs, returning
   normalized records and per-source checksums. It neither ingests/attests U nor
   executes statistics. Verify exact paired completeness against the audited
   U source, the primary UK48−U estimand, shared resampling schedule and Holm
   adjustment across only the two auxiliary comparisons. Do not reuse an old
   aggregator by loosening its arm/memory rules or report unverified statistics.
   `artifacts.ExpansionRunStore` reserves attempts before environment creation,
   requires original initial observations/state/text attachments, and publishes
   video/results without replacement. Completed scientific outcomes cannot be
   retried, including when later video/artifact writes fail before result
   publication; hard stops block all new attempts in that run. An incomplete
   result/ledger publication requires review, not an outcome-improving rerun.
6. Report the exact source revision, tests, adapter status, unresolved readiness
   blockers, output locations and planned resource use. Request smoke launch
   authority only when a committed clean implementation can actually execute it.

After authorized smoke, produce the full audit and fix only demonstrated code
or environment defects. Do not tune the science from smoke outcomes. Freeze
v1.0 only after all gates and the U comparison attestation pass. The complete
formal batch still needs a separate user authorization.

## Implemented read-only inspection commands

From the repository root, with its existing `.venv` and no environment changes:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.uniform_keyframe_expansion.contract --stage smoke
PYTHONPATH=src .venv/bin/python -m experiments.uniform_keyframe_expansion.contract --stage formal
```

These print matrix/seed/config summaries only and explicitly report launch as
unauthorized and the U baseline as unresolved. `--matrix` additionally prints
the full selected matrix. Neither command launches jobs, writes a run root,
verifies external evidence or authorizes a stage. The formal summary is not a
formal-run command.

## Governed server execution interfaces

The following are implemented, but none independently grants run permission:

- `launch_contract.build_execution_request` binds the new run store, exact
  submitted rows, execution UUID, physical GPU UUID and evidence references.
  `build_execution_plan` adds the separately recorded user authorization;
  `validate_execution_plan` returns the sealed context accepted by the runtime.
- `server_bootstrap.verify_controller_sources` checks live clean Git/source
  bindings before starting wrappers or child processes. Each fresh worker also
  verifies its interpreter, pinned packages, numerical environment and physical
  device. Only the policy worker streams the actual archive and parameter tree,
  once before its resident model load; routine row checks do not reread weights.
- `resident_controller` starts one owned policy process per submitted shard,
  serves its authorized rows sequentially, and creates a fresh evaluator and
  reset state per episode. It does not unload the model between rows. The
  environment record must name a stable, user-owned, host-wide `lock_directory`
  shared by cooperating runs, plus a reviewed free-memory threshold. Sharing
  with other users is allowed; the lock does not control their processes.
- A held loopback listener and inherited GPU lease accompany the owned Linux
  watchdog processes. Worker startup proves live controller/watchdog ancestry
  before loading a backend. Cleanup targets only owned groups. A secondary
  cleanup failure does not hide the original failure. There is no automatic
  retry, no overwrite, and no inference of retry permission from partial output.
- Before a formal controller starts, `deep_validate_runtime_evidence` audits
  the complete raw 48-trajectory smoke evidence once. Routine revalidation checks
  its bound small-file chain, not repeated video/NPZ I/O. Formal source gates also
  bind the smoke execution/bootstrap records and all 800 raw U outcome sources;
  a bare `PASS`, count, or same-seed statement is insufficient.

An already prepared absolute execution plan can be inspected without launch:

```sh
PYTHONPATH=src:. <recorded-python> -m experiments.uniform_keyframe_expansion.resident_controller --plan <absolute-plan.json>
```

`--execute` is the mutating trajectory-entry option and remains unavailable until
all bound evidence and that specific stage's user authorization exist. Do not run
the internal `--role policy` or `--role simulator` entries manually: they require
the recorded owned controller ancestry. The architecture-only diagnostic is
deliberately not implemented by this trajectory controller.

The separate `experiments.uniform_keyframe_expansion.architecture_controller`
also defaults to dry inspection with `--plan`. Its explicit `--execute` needs
an independently authorized architecture plan with no trajectory rows. Never
invoke its internal worker role manually. The controller binds measurements to
the real checkpoint-loaded owned worker, confirms cleanup, then asks the raw
artifact validator to publish `architecture_gate.json` without replacement.
Same-input 512/768 differences are measurements, not a reason to change RoPE or
to introduce another formal arm. See the architecture validation report for
the distinction between true same-path repeatability and an explicit-noise
diagnostic using a different compiled path.

## Non-negotiable stop conditions

Stop for unresolved U-baseline provenance, non-causal or misaligned labels,
checkpoint/config mismatch, extra or missing model weights, unconfigured
experimental inference, over-capacity unions, too few eligible non-key frames,
out-of-table selector calls, reused/overwritten run roots, or unclear launch
authority. Preserve evidence; do not silently fall back, change frame counts,
alter seeds, exclude episodes, or relabel scientific failure as infrastructure.
