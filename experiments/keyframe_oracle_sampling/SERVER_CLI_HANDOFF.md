# Server Codex CLI handoff: keyframe Oracle sampling smoke

This handoff covers only the committed development-smoke stage. It never
authorizes formal evaluation.

## Authoritative inputs

Read completely, in this order:

1. repository-root `AGENTS.md`;
2. `experiments/keyframe_oracle_sampling/EXPERIMENT_PROTOCOL.md`;
3. this handoff;
4. the implementation diff at the exact reviewed commit supplied by the user.

The protocol is authoritative. Do not change an arm, causal boundary rule,
checkpoint, task/episode/split, seed, outcome, retry/exclusion rule, statistical
test, or decision threshold. Stop and report if one of those definitions is
ambiguous or cannot be implemented exactly.

## Server-owned work

1. Fetch the writable `lab` remote and fast-forward the dedicated
   `exp/keyframe-oracle-sampling` branch to the exact reviewed commit. Never
   merge, rebase, force-push, or push to `origin`.
2. Verify the clean worktree, both pinned virtual environments, both lockfiles,
   frozen checkpoint archive SHA-256, unpacked checkpoint content-tree digest,
   benchmark commit, GPU visibility, Slurm partitions, and write permissions.
3. Run the full non-GPU test and dry-run gates recorded by the implementation.
4. Prepare one new immutable smoke run root. Never reuse or delete an existing
   run root.
5. Submit and audit the one-GPU architecture smoke. Continue only after the
   strict eight-case report is a PASS and is bound to this run root and job ID.
6. Submit the frozen 80-row validation smoke, preserving exact row/seed binding,
   append-only attempts, and the protocol retry limit.
7. Produce the smoke audit report and return exact commands, commit SHA, job
   IDs, output paths, failures/retries, and pass/fail evidence. Do not summarize
   smoke success rates or use smoke outcomes to tune a selector.

## If a code or environment problem appears

- First preserve the failed attempt, logs, and failure ledger.
- A clearly non-scientific implementation defect may receive the smallest
  possible fix on the same experiment branch. Run the relevant CPU tests and
  dry runs, commit only scoped source/test changes, push only to `lab`, report
  the new SHA, and stop for desktop-side review before creating a new smoke run.
- Do not edit generated results or reuse a prior run root.
- Missing/misaligned Oracle labels, checkpoint mismatch, selector mismatch,
  deterministic arm-specific crashes, or any requested scientific-definition
  change are hard stops, not retryable infrastructure failures.
- Formal 3,200-trajectory execution remains disabled until a reviewed smoke
  report, a committed protocol v1.0, and separate explicit user authorization.

## Suggested initial CLI prompt

> Work from the repository root. Read `AGENTS.md`, then read
> `experiments/keyframe_oracle_sampling/EXPERIMENT_PROTOCOL.md` and
> `experiments/keyframe_oracle_sampling/SERVER_CLI_HANDOFF.md` completely.
> Confirm that the current clean branch is `exp/keyframe-oracle-sampling` at the
> exact reviewed commit I provide. Execute only the development-smoke handoff,
> using the repository's validated entry points. Preserve immutable evidence,
> stop on every scientific ambiguity, and do not launch formal evaluation.
