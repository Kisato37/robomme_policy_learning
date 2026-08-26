# Repository Working Agreements

These instructions apply to the entire repository. Keep task-specific scientific
details in the version-controlled experiment protocol; do not duplicate the full
protocol in every prompt.

## Priorities

- Optimize for scientific validity, complete execution, and token efficiency.
- GPU usage is generally available and is not a reason to weaken or prematurely
  downscale an experiment. Small runs are appropriate when they validate the
  design or implementation before a complete run.
- Obtain explicit user approval before launching an unusually large or multi-day
  job.

## Preserve User Work and Provenance

- Treat existing changes, results, manifests, checkpoints, and run artifacts as
  user-owned. Do not overwrite, delete, rewrite, or silently regenerate them.
- Keep unrelated changes untouched. If a requested change overlaps dirty files,
  inspect the diff first and preserve the user's work.
- Use a new run directory or unique run ID for every execution. Never reuse an
  existing result directory in a way that can overwrite prior evidence.
- Never edit generated metrics, trajectories, or result manifests to make a run
  appear successful. Fix code and rerun under a new run ID.

## Git Safety

- `origin` is the upstream RoboMME repository and is read-only for this project.
  Never push to `origin`.
- `lab` is the writable fork. Push only to `lab`, and only when the current task
  explicitly authorizes a push.
- Perform experimental work on a dedicated `exp/*` feature branch. Do not commit
  directly to `main`.
- Before changing branches, pulling, committing, or pushing, inspect the current
  branch, `git status`, and relevant diff.
- For synchronization, prefer `git fetch` followed by an explicit fast-forward
  update. Do not create an automatic merge or rebase when the worktree is dirty
  or branches have diverged; stop and report the state.
- Never use force-push, `git reset --hard`, `git clean`, destructive checkout,
  history rewriting, or remote branch/tag deletion unless the user explicitly
  requests the exact operation and target.
- Do not commit secrets, credentials, `.env` files, `.DS_Store`, checkpoints,
  H5 datasets, archives, large rollout videos, large logs, or cluster caches.
- A commit must contain only scoped changes and should follow relevant tests.
  Report the commit SHA after committing.

## Scientific Protocol Is Authoritative

- Read the applicable version-controlled experiment protocol completely before
  implementing or running an experiment.
- Do not independently change hypotheses, experiment arms, selector definitions,
  checkpoints, memory budgets, tasks, episodes, seeds, metrics, exclusion rules,
  timeouts, statistical tests, or GO/NO-GO criteria.
- If an ambiguity would alter the scientific comparison, stop and ask instead of
  choosing a convenient interpretation.
- Keep all arms matched on every variable not explicitly designated as the
  treatment. Do not add hidden training, filtering, retries, or per-arm tuning.
- For test-time-only experiments, do not retrain or alter model weights unless the
  protocol explicitly requires it.
- Any Oracle signal must be strictly causal at evaluation time: a decision at time
  `t` may use only information available at or before `t`. Never read future
  labels, future frames, future states, terminal outcomes, or success metadata.
- Randomized methods must use recorded deterministic seeds and emit the selected
  indices so the selection can be reproduced exactly.

## Smoke and Formal Runs

- A smoke run validates plumbing, shapes, causality, budgets, determinism,
  logging, and job completion. It is not evidence of comparative performance and
  must not be used to select a winner or tune a method on formal evaluation data.
- Use protocol-designated development episodes and seeds for smoke runs. Keep
  them disjoint from formal evaluation episodes and seeds.
- Code and environment defects found by smoke runs may be fixed and rerun. A
  change to the scientific protocol must be documented, versioned, and re-smoked
  before formal evaluation.
- Do not launch a formal evaluation, formal training run, or multi-day batch until
  the user explicitly authorizes that stage and the protocol is frozen.
- Do not cancel or replace another active cluster job unless the user explicitly
  authorizes it or immediate action is required to prevent material damage; in
  the latter case, report the action and evidence promptly.

## Validation and Evidence

- Start with read-only inspection and reuse the repository's existing pipeline,
  tests, configuration system, and Slurm infrastructure where possible.
- Run the most relevant non-destructive tests after code changes. For GPU work,
  pass CPU/unit/fixture checks before submitting a smoke job unless the check
  itself requires GPU execution.
- Before a formal launch, verify the exact checkpoint, environment, selector,
  memory budget, task set, episode set, and seed manifest against the protocol.
- Every experimental run should record, where applicable: run ID, Git commit SHA,
  command/config, checkpoint identity or hash, environment, task and episode,
  all seeds, selected frame indices, memory-token count, Slurm job ID, timestamps,
  exit status, timeout status, and primary metrics.
- Report failures as failures. Preserve logs and distinguish code/environment
  failures from scientifically valid negative results.

## External and Destructive Actions

- Local inspection, scoped edits, and non-destructive tests are allowed when they
  are part of an implementation task.
- Creating repositories or pull requests, pushing commits, submitting formal or
  large cluster jobs, publishing artifacts, changing shared environments, and
  other external writes require explicit task authorization.
- Never expose credentials or private infrastructure details in prompts, logs,
  commits, or final reports.

## Completion Report

When handing off work, state:

- what changed and why;
- tests and commands run, with results;
- current branch, commit SHA, and whether anything was pushed;
- submitted job IDs and output locations;
- protocol deviations or unresolved risks;
- the next safe action, especially when a formal run still needs approval.
