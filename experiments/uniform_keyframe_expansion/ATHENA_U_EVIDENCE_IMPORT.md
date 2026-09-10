# Original U evidence recovered — 2026-09-10

This is a read-only source recovery report, not a new evaluation or a baseline
comparability attestation. The earlier `BASELINE_EVIDENCE_AUDIT.md` describes the
local evidence available before this recovery; its missing-hash finding is now
resolved only to the extent stated below.

## Collection and checks

The user re-established the shared Athena connection. The fixed-scope collector
was supplied on standard input to the source host's Python interpreter. It
created no remote file, loaded no checkpoint, launched no GPU process, and did
not rerun or modify an episode. The original run was
`20260829T231425Z_7b594786_formal_v1`.

The received bundle was validated locally before an exclusive import into a
new directory. Original JSON bytes were retained, not rewritten as reconstructed
records. Both collection and import completed successfully.

| Check | Verified result |
|---|---:|
| U scientific episodes | 800 |
| Tasks × episodes per task | 16 × 50 |
| Success / fail / timeout | 369 / 404 / 27 |
| Episodes with all five actual initial-input hash values | 800 |
| Episodes with first-policy-call trace evidence | 800 |
| Reset history already longer than one frame | 450 |
| Maximum initial history length | 1,140 |
| Selected U infrastructure-failure ledger records | 14 |
| Original evidence files imported | 3,226 |

The three source files checked for each scientific outcome are
`episode_result.json`, `episode_manifest.json`, and
`initial_condition_hashes.json`. Their keys, terminal outcomes, actual seeds,
difficulty, policy seed, cadence and available checksum links were checked.
The actual seed/difficulty mapping matches the earlier local U export:
`ed5fef94211e17c6ef4b2efadd91e5446df780e931f6cdce2222a35ea4f8d483`.
Do not replace it with a simplified arithmetic seed formula.

All five initial hashes are now available for comparison: front observations,
wrist observations, robot states, task state, and task instruction. They are
hash values from the original run, **not recovered image/state arrays**.

## Evidence location and digests

Relative to this repository's parent directory, the new local evidence is in
`lighthouse_migration/athena_u_raw_coqcqe7c/`:

- `bundle.json`: 20,763,463 bytes, SHA-256
  `ebdeeae6304913662970902e109126e2f17f1ec882ad63fe67c4ab778e946568`.
- `transport_receipt.json`: records exact collector and transport identity;
  remote exit status is zero and stderr is empty.
- `imported/IMPORT_REPORT.json`: SHA-256
  `f553b790960e321da8d65578df66faaa26918939f612b47c84c647522b70cc95`.
- `imported/source/`: original metadata, selected retry evidence and original
  first trace lines, with individual digests in the import report.

The collector source used for this transfer has SHA-256
`ccbf8b8be108b62660588e04e39e22b7ad19903cafa15cabbf3fc6d97ce425c1`.
The report's `source_bundle_sha256` is the collector's canonical payload digest,
not the transport file-byte digest above; these are intentionally different
hash scopes.

## Remaining limitations

- Entire selector traces were scanned and hashed on Athena. Only their first
  original lines were transferred; this is not a local full-trace replay.
- The selected 14 U retry-ledger lines are locally rehashed. Four legacy lines
  lack an original manifest-digest field; this absence is retained, not invented
  or silently treated as a verified link. The entire source ledger was not
  transferred.
- Neither optional `environment_manifest.json` location existed. The archived
  launch manifest does contain Python-environment and environment-lock fields;
  these still require comparison with the proposed new runtime, rather than
  treating the missing optional filenames as absence of all environment evidence.
- Original image/state arrays, exact task text, videos and weights were not
  transferred. Hashes cannot reconstruct them.
- UK48/UN48 have not been run. Their initial inputs, reset-history alignment and
  numerical environment have not been compared to U.

Therefore the baseline status remains **unresolved for formal reuse**. This
recovery removes the specific obstacle of missing U hash values; it does not
authorize or establish a matched new experiment, and it does not justify
automatically adding a U rerun.
