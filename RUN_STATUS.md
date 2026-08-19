# Dual-memory experiment status

Run ID: `20260818-214308_ecf086c`

## Current phase

Phase 0/1 asset preflight and data audit are in progress. Phase 2 implementation is complete and its GPU-only gates are queued behind immutable asset checks.

## Completed

- Created an independent Athena repository and branch; the previous test-time scaling repository is untouched.
- Pinned policy commit `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` and benchmark commit `856bc3a189d4172f3f47dbee4424d585f8d78db3`.
- Created independent locked policy and simulator environments.
- Verified Athena exposes 8 RTX A5000 24 GB GPUs per training node.
- Implemented the minimal parallel GroundSG plus FrameSamp-Modulator input path.
- Passed 15 dual-memory compatibility, cache, evaluation-record, and statistics tests locally and on Athena.
- Preregistered the matched 2x2 model matrix, seeds, checkpoint rule, and GO/NO-GO threshold.
- Added append-only per-episode results, exact-input Qwen caching, cumulative-history logging, and deterministic Qwen token/cost records.
- Added the 8-GPU gradient/causal/checkpoint-key gate and N/S/P/SP save/reload smoke gates.
- Queued the complete gated chain: released-reference preflight, 100-sample audit, SP architecture gate, four matched 100-step smokes, Oracle/Qwen rollout smoke, paired pilot, 12 formal 80k trainings, complete Oracle evaluation, hierarchical statistics, Oracle-controlled Qwen evaluation, failure taxonomy, and final report.

## In progress

- Downloading and hashing the pinned full preprocessed dataset and the minimal required model assets.
- Base/released checkpoint extraction and official preflight rollouts.
- Independent pinned Qwen3-VL environment/model setup on `athena-small`.

## Blockers

- No scientific blocker at present.
- The official 4xA40-40GB recommendation is unavailable. An 8xA5000-24GB smoke must pass memory and throughput gates before formal training.

## Next

1. Complete downloads/unzip, released-reference episodes, and the 100-sample/20-visual audit.
2. Pass the SP 8-GPU gradient, causal-intervention, shape, and checkpoint compatibility gate.
3. Pass all four 100-step save/reload smokes and SP Oracle/Qwen fixed rollouts.
4. Complete the paired pilot; only its engineering PASS can release formal training.
