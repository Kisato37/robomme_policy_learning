# Single-environment benchmark stop flag

The 48-row validation execution at revision `7df2a39` stopped in the first
environment step of `BinFill/val/0/UK48/short`. Initial evidence and one verified
policy-call trace were saved; no scientific episode result was completed.
This is an adapter hard stop, not a benchmark task failure. The prior run and
its failure records remain immutable.

## Interface mismatch

The pinned benchmark's `planner_denseStep.to_step_batch` returns `torch.bool`
termination/truncation tensors. `DemonstrationWrapper.step` extracts their final
entries, and the existing `EnvRunner.step` returns `terminated or truncated`.
Python's `or` returns an operand, not necessarily a Python `bool`. Thus a
single-value boolean tensor reaches the expansion evaluator, whose original
type check accepted only `bool` and `numpy.bool_`.

The earlier CPU episode fixture returned only Python booleans and missed this
production interface. New regression tests use the actual benchmark batch
producer, without GPU simulation, to reproduce that mismatch across success,
failure, timeout, error, and the 64/1300-step cutoff paths. All six cases fail
against the old check and pass after the adapter correction.

## Narrow correction

Only the expansion evaluator converts an already-boolean scalar or
single-environment container to a Python `bool`. It accepts Python/NumPy boolean
scalars, boolean NumPy arrays of shape `()` or `(1,)`, and dense Torch boolean
tensors of those shapes. Torch is accessed only if already loaded by the
simulator, preserving validated backend initialization order.

Numbers, strings, lists, empty/multi-value arrays, non-boolean dtypes, and
unreadable tensors still fail closed. There is no generic truthiness conversion,
threshold, `any`/`all` reduction, status override, or automatic retry. The
original benchmark, EnvRunner, other experiment families, outcome definitions,
step limits, actions, seeds, and memory selectors remain unchanged.

Local CPU PASS is not server or GPU readiness. A reviewed new source revision
requires synchronization, server validation, and fresh source-bound GPU
evidence under new run IDs before the smoke can be accepted. Do not attach the
previous architecture PASS to changed source or resume the failed trajectory.
