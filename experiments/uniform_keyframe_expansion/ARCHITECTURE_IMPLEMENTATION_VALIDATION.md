# UK48 / UN48 architecture implementation validation

Date: 2026-09-10. Scientific protocol remains v0.9; no experimental definition
was changed. This is a **local implementation/CPU validation report**, not a
real-checkpoint architecture PASS or GPU launch authorization.

## Completed implementation

- `architecture_controller.py` supplies the independently authorized architecture
  entry. Its default is dry inspection. Actual execution requires empty
  trajectory rows, a bound source/environment/checkpoint plan and an owned
  Linux worker. One resident model runs all diagnostic cases. Worker failures
  are preserved without automatic retries; cleanup precedes gate publication.
- `architecture_probe.py` measures an actually loaded expansion policy on
  synthetic causal histories of 16 and 64 frames, both UK48 and UN48. InsertPeg,
  val, episode 0 supplies selector context only: no task, dataset or test outcome
  is opened. Repeated cases cross arm/history boundaries in the same process.
- Saved evidence includes gathered features, post-normalization features,
  actual JAX inputs, masks, encoded memory, selected indices, noise and actions.
  Pre-normalization padding must be zero; encoded/modeled padding need not be.
  Normal NumPy-to-JAX dtype conversion is verified rather than mistaken for
  corruption. BF16 evidence is stored as losslessly converted float32 values,
  with original/stored dtypes and hashes recorded; model computation is unchanged.
- The padding diagnostic creates a separate 512-slot model graph using the
  same immutable parameter arrays as the 768-slot graph. It changes no weights,
  RoPE implementation, solver or primary model configuration. Both graphs see
  the same visible short-16 U content and explicit noise. This is 256 valid
  tokens in different static capacities, not a full 32-frame U history.
- Actual action and velocity differences between 512 and 768 are measured;
  nonzero differences alone do not fail the diagnostic. Actual same-path
  inference repeats remain bitwise checks. The separate explicit-noise versus
  implicit-noise 768 replay reports model-space action L-infinity difference;
  different compilation paths can differ numerically, so this additional
  cross-path measurement is not misrepresented as a bitwise-equality gate.
- Timing synchronizes device work. The first request of each case is labeled
  accurately, not called a fresh model load. Peak memory is the measured
  cumulative JAX allocator peak in that process, not an independent per-case
  peak or total GPU occupancy. Missing/zero peak statistics fail closed.
- `architecture_artifacts.py` validates actual execution identity, owned model
  PID, checkpoint bootstrap, completion and cleanup records before write-once
  gate publication. It checks raw-file hashes, array shapes, masks/padding,
  repeated outputs, causal selected indices and measured positional differences.
  Subsequent controller startup repeats the full raw audit once; ordinary
  per-episode plan checks do not repeatedly load large NPZ artifacts.

## Test evidence

Targeted architecture probe/controller/artifact suite: **76 passed**.

Final combined regression:

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/uniform_keyframe_expansion tests/keyframe_oracle_sampling \
  tests/keyframe_neighborhood_sampling tests/dual_memory -q -rs \
  --junitxml=/private/tmp/uk48-architecture-final-20260910.xml
```

Result: **1,167 passed, 24 skipped, 202 subtests passed**, in 83.76 seconds.
The one warning is the existing beartype deprecated-type-hint warning.
The skipped checks require Linux process identity or inherited-listener
semantics and must be exercised on the target server. No test failure occurred.

JUnit SHA-256:
`4f7fc3c8b81f0a1877becf97c2ea8eeeb5b535b91726152a2e5617daea42e0de`.

`git diff --check` passed. The tests use synthetic CPU fixtures or tiny CPU
models; they do not establish real checkpoint compatibility or GPU throughput.

## Lighthouse readiness and remaining work

Lighthouse remains the intended execution host. Athena was used only for
read-only old-U evidence/inventory, not for a newly submitted UK48/UN48 job.
The fresh Lighthouse inventory confirms a clean checkout at the same existing
base revision, unchanged installed package versions relative to its prior
inventory, and the expected checkpoint file census. The new family is **not
yet synchronized there**. The read-only inventory did not rehash/load the large
weight payload, run inference, render a task or reserve a GPU.

Operational inventory and transport receipts are kept separately under
`dual_memory_work/lighthouse_migration/uk48_lighthouse_inventory_vDhTsbpA/`.
Main inventory SHA-256:
`68386fd653b8d531e117252111e1932b9fec2be31294dd34dfb4038efce2a629`.
Path-supplement SHA-256:
`47f6ae23161a59020704ef4ceae7255fa17f7bea3edd158a2b008f09c53dfd4b`.

Before preparing the actual launch evidence, use the real checkpoint data
directory rather than the repository's symlink path. Also avoid applying the
graphics wrapper twice: binding its already expanded environment and then
prepending the same paths again would fail the exact runtime checks. Prefer a
reviewed fully resolved environment with an empty command prefix, or explicitly
model wrapper input/output environments; do not relax environment equality.
GPU availability is a snapshot and must be checked again at authorized startup.

Next stages remain distinct:

1. Obtain authorization to commit/push scoped changes and synchronize the clean
   Lighthouse checkout; rerun server CPU/Linux checks.
2. Bind actual source, environment, physical GPU, checkpoint and separate
   architecture-smoke authorization. Run the real diagnostic and inspect its
   actual raw evidence before accepting a gate.
3. Under the next required authority, run the protocol's 48 val smoke
   trajectories. Audit the original U comparability before formal evaluation.
4. Freeze v1.0 and obtain formal authorization before the 1,600 new trajectories.
   Do not automatically rerun U or change the chosen host to bypass mismatch.

Branch: `exp/keyframe-neighborhood-sampling`.
HEAD: `297cbb2abaf50e7005d40a3ffb7076cdc26ee986`.
No commit, push, server source mutation, dependency installation, GPU experiment
or job submission occurred in this implementation stage.
