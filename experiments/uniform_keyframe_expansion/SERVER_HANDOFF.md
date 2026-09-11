# U / UK48 / UN48 Lighthouse execution handoff

Date: 2026-09-11. Authoritative protocol: `EXPERIMENT_PROTOCOL.md` v1.1.

## Authority and current target

The user approved the fresh same-run design and authorized this sequence:

1. freeze, commit, and push the implementation to the writable `lab` remote;
2. synchronize the exact clean commit to Lighthouse;
3. run the new CPU/static and real-checkpoint GPU gates;
4. execute the exact 64-trajectory `val` smoke;
5. if every gate passes without a scientific change, directly launch the exact
   2,400-trajectory formal matrix.

Never push `origin`. Never overwrite an existing run. Preserve every attempt.
Existing U/O/OC/R, OC3/OC5, and earlier UK48 smoke/result roots are immutable
and are not inputs to the v1.1 formal comparison.

The first v1.0 formal dispatch is also immutable failed infrastructure evidence:
its first cold policy request took 76.351 seconds and the server then closed the
connection under its default keepalive timeout. It produced no scientific
outcome and must not be resumed or reused. Version v1.1 sets and handshake-binds
a 600-second client/server keepalive timeout. All gates and the formal run must
be fresh on the v1.1 commit.

## Scientific matrix

- U: released Uniform, 32 frames / 512 tokens.
- UK48: literal U32 set plus all newly missing causal boundary frames,
  48 frames / 768 tokens.
- UN48: literal U32 set plus an equal number of random non-boundary/non-U
  frames, 48 frames / 768 tokens.
- 16 canonical tasks, `test` episode IDs 0–49, all three arms: 2,400 cells.
- Primary comparison: UK48−U.

All three arms must be newly run on Lighthouse. The old Athena U is excluded
because its initial front-view inputs were not byte-identical to the new
Lighthouse environment.

## Critical execution constraint

One resident model process cannot mix the two static memory shapes. Build
homogeneous plans only:

- `released_u32_512`: U rows only;
- `expanded_uk_un_48_768`: UK48 and/or UN48 rows only.

This does not create separate scientific studies. All plans share the same run
family, matrix, protocol, source, checkpoint, seed mapping, and final audit.
Keep models resident across rows; reset environment, memory, selector, and
policy RNG for each episode.

## Mandatory gates

Read repository `AGENTS.md`, this handoff, and the full v1.1 protocol before
execution. Verify a clean `exp/*` checkout at the exact pushed commit.

1. Run the complete relevant CPU regression suite and seal its JUnit/report.
2. Rebuild source, environment, checkpoint, hardware, matrix, seed, and user
   authorization evidence against that exact commit.
3. Run the real-checkpoint expanded architecture probe. Require strict weight
   load, finite `20 x 8` actions, `[1,768,1024]` final memory, deterministic
   fixed-input repeatability, reset isolation, latency, and peak-memory evidence.
4. Run exactly 64 smoke trajectories: for every val task, short U/UK48/UN48;
   plus one full terminal trajectory whose arm rotates by task.
5. Deep-audit all 64 raw outcomes, trace replay, initial pairing, process
   ownership, termination, and manifests. Do not tune based on smoke outcomes.
6. Only after PASS, freeze formal artifacts and start disjoint exhaustive formal
   shards. U-only and expanded-only workers may use different GPUs.

Sharing a GPU with another user is permitted when measured free memory and
compute headroom are adequate. Experimental throughput has priority; do not
artificially unload after each trajectory. Do not knowingly create memory
pressure that risks either job.

## Runtime interfaces

- `contract.build_smoke_matrix()` and `build_formal_matrix()` are the only
  accepted matrices (64 and 2,400 rows).
- `contract.policy_variant_for_rows(...)` rejects mixed-shape resident plans.
- `serving.load_study_policy(...)` selects the strict 512 or 768 loader.
- `resident_controller` owns one resident policy process and sequentially runs
  only the rows bound into its homogeneous execution plan.
- `trace_validation` independently replays U and expanded selector traces.
- `outcome_ingestion.ingest_formal_run(...)` requires the exact 2,400-cell
  census and verifies 800 same-run three-arm initial-condition blocks.
- `analysis.analyze_matched_outcomes(...)` requires a verified same-run
  attestation bound to the exact ingested records.

Do not manually invoke internal policy/simulator worker roles. Use sealed
execution plans through the controller. Readiness summaries are not launch
commands.

## Monitoring and failure policy

During formal execution, monitor only liveness, GPU health, progress counts,
artifact completeness, and operational errors. Do not inspect success patterns
to change shards, seeds, selectors, capacity, or stopping rules.

Ordinary benchmark failure is a scientific result and is not retried.
Documented infrastructure failures may use at most two fresh-reset retries with
identical frozen inputs in new attempt directories. Selector/config/shape/
checkpoint/trace/pairing failures are hard stops. Never silently fall back to
Uniform48, omit U, reduce the matrix, or reuse old Athena U.
