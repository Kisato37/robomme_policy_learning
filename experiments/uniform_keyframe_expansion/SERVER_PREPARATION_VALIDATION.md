# Server preparation validation — 2026-09-10

Protocol v0.9; implementation and read-only source-recovery stage only.
No GPU job, simulator trajectory, real checkpoint load, training, Git commit or
push was performed in this stage. Repository branch remains
`exp/keyframe-neighborhood-sampling`; HEAD remains
`297cbb2abaf50e7005d40a3ffb7076cdc26ee986`. New implementation files are not yet
committed; this report does not make the current dirty checkout launchable.

## Implemented and reviewed

- `launch_contract.py`: source-bound execution plans and separate stage
  authorization. CPU-only and architecture-only plans cannot submit benchmark
  trajectories; formal execution requires v1.0, raw U source binding and the
  complete validated development smoke. Source hashes do not authenticate a
  user; the corresponding real instruction must exist and be reviewed.
- `server_bootstrap.py`: lazy production bindings to the original environment
  runner and new policy, with live source/interpreter/package/environment/GPU
  checks. Strict real checkpoint hashing/loading occurs once per resident policy
  process, not per episode. No backend is loaded by dry-run inspection.
- `resident_controller.py`: one resident model per authorized shard, fresh
  evaluator/environment and policy reset per row, held loopback listener,
  inherited host-wide GPU lease, owned watchdog lifecycle and immutable output
  records. Existing attempts are not overwritten or automatically retried.
- `worker_ownership.py`: live Linux identity/ancestry checks before child backend
  loading. A valid plan alone is insufficient to run an unowned child manually.
- `baseline_export.py` and `sync_baseline_evidence.py`: fixed-scope read-only U
  collection and exclusive local import. Failed transports preserve partial
  evidence and a failure receipt; validation/import is a separate required step.

Review fixes include a host-wide rather than run-local GPU lock, preserving the
original failure when cleanup also fails, rejecting standalone unowned workers,
binding baseline outcomes to their actual three original metadata files, and
binding smoke results to the actual controller/model execution records. Full raw
smoke files are audited once before a formal controller starts; routine per-row
validation does not reread large videos, NPZ arrays or model weights.

These are execution/provenance changes. They do not change UK48/UN48 selection,
the original Uniform32 base, 48-frame capacity, zero/false-mask unused slots,
checkpoint, seeds, task population, numerical policy settings or statistical
estimands. The previous runtime and original experiment regression suites are
included in the combined validation.

## Local tests

The final combined CPU validation command is:

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest tests/uniform_keyframe_expansion tests/keyframe_oracle_sampling tests/keyframe_neighborhood_sampling tests/dual_memory -q -rs --junitxml=/private/tmp/uk48-server-preparation-final-20260910.xml
```

Final result: **1,087 passed, 24 skipped, 202 subtests passed**, zero failures,
one existing beartype deprecation warning; elapsed time 79.06 seconds.
JUnit SHA-256:
`f72f880e64dfb8903c520f03a0f2a8805d865e680794f428913f2f7794393cbc`.
`git diff --check` also passed.

The Linux-only process and inherited-socket checks must also be run on the
actual server; CPU identity fixtures do not establish real Linux process
cleanup, simulator compatibility or GPU readiness. There is no claim of a
scientifically measured UK48/UN48 result in these tests.

## Actual U source recovery

The shared Athena connection was successfully reused for read-only collection.
All 800 U scientific keys, their actual initial hash values, actual seed/difficulty
mapping and archived source metadata were imported. Independent local inspection
checked all 3,226 imported files against their digests and original bundle bytes
with zero differences. See `ATHENA_U_EVIDENCE_IMPORT.md` for exact scope, digests,
counts and limitations. No old U scientific outcome was altered or rerun.

## Remaining gates / next work

1. Finish the separate real-checkpoint architecture-smoke executable. The
   trajectory controller intentionally does not perform the U512-versus-padded-
   U768 positional diagnostic, strict load, shape/mask, reset/determinism and
   latency/memory measurements by itself. No architecture PASS may be inferred
   from the CPU tests.
2. Under the relevant synchronization authority, commit/synchronize a clean
   source revision, bind the actual server runtime evidence, and execute the
   Linux CPU checks. Preserve unrelated server work and the original environment.
3. With explicit GPU-stage authorization, run architecture validation, then the
   fixed 32-short + 16-full development trajectories. Keep the model resident
   within each authorized shard; shared GPUs are permitted subject to real
   available-memory and ownership safeguards.
4. Compare the chosen runtime/initial inputs/reset history against the recovered
   U sources. Actual hashes are available now, but original raw arrays and
   per-frame stage records are not recreated from those hashes. Resolve or
   explicitly report missing alignment evidence; never silently rerun U.
5. Only after passing the specified gates and baseline review may v1.0 be frozen
   and the separately authorized 1,600-row formal UK48/UN48 study begin.

This stage does not authorize server environment changes, GPU submissions,
additional U runs or pushing the uncommitted implementation.
