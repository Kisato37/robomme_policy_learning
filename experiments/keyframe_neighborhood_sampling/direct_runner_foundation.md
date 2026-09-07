# Direct execution foundation — historical initial increment

This file records the state before production wiring. The subsequent adapter
is described in [LIGHTHOUSE_EXECUTION.md](LIGHTHOUSE_EXECUTION.md). Statements
below about unimplemented wiring refer to that initial increment, not the
current tree. Neither CPU tests nor implementation work count as GPU smoke.

This increment adds `runner_contract.py` and `direct_runtime.py` as isolated
building blocks. No production launcher, evaluator, preflight, smoke audit,
formal audit, selector, matrix, protocol, or previous result imports or consumes
them. They do not make direct execution eligible for either smoke or formal
acceptance, grant user permission, or supersede the current Slurm rules.

## Implemented interfaces

`DirectDispatch` records explicit `backend=direct` and its own execution schema
(`keyframe-neighborhood-direct-process-v1`). It binds execution UUID, absolute
new run root, repository commit, launch-manifest/plan/matrix digests, stage,
row/shard/attempt, ordered physical GPU UUIDs, host and boot identity, and policy
port. One GPU is required for architecture inference; a trajectory requires
distinct policy and simulator GPUs, in that order. UUIDs and digests are evidence
identifiers, not cryptographic credentials. An identical self-authored record
does not prove launch authority.

`validate_runtime_binding` compares every observed dispatch field and the
canonical recorded digest. It does not inspect a matrix, resolve environment
seeds, verify checkpoint bytes, enforce live resource allocation, or claim
scientific completion. Those responsibilities belong to existing gates and
their eventual explicitly routed direct counterparts.

`process_start_record` and `process_exit_record` bind process facts to the
dispatch and each other, including timestamps, command, working directory,
boot/PID/start-time identity, exit status, and wall-clock-limit status. They do
not declare success/failure of an episode or permit a retry. `write_once_record`
publishes fsynced complete JSON through exclusive hard-link creation and cannot
replace a previous artifact, including under concurrent writers.

`GpuLease` takes nonblocking advisory locks on full physical NVIDIA GPU UUIDs.
It initializes no CUDA runtime and uses no GPU memory or compute. The caller
must supply one existing stable local directory shared by every cooperating
runner on that host, including other checkouts. Lockfiles persist; never unlink
or replace them to "clear" a lease. Acquire a row's complete allocation just
before actual work and close it once owned GPU processes exit. Failed partial
acquisition releases its acquired locks. No queue or idle worker should hold a
lease while waiting for unrelated work or authorization.

Pass `GpuLease.pass_fds` through `subprocess.Popen(pass_fds=...)` to **all** GPU
children (and propagate to any GPU descendants). Closing the supervisor's copy
does not unlock an inherited child's copy, so an ordinary supervisor crash
cannot immediately hand its live child's GPU to another cooperating runner.
Advisory locks do not detect or prevent unrelated/noncooperating CUDA programs.
Future preflight must resolve a fresh device inventory, inspect occupancy, and
check capacity. Do not silently choose another GPU or evict another process.

`PortReservation` holds a live loopback listening socket; its descriptor can be
adopted by the server without closing/rebinding the port. The current policy
server does **not** support descriptor adoption. Do not close this reservation
then pass its port to the current server and claim race-free ownership. A bare
TCP readiness probe also cannot identify which server accepted a connection.

`LinuxProcessOperations` reads Linux `/proc` boot ID, PID, parent, process group,
session, start ticks, and UID; it fails closed without Linux process evidence.
`OwnedProcessGroup` accepts only this supervisor's unreaped child in a dedicated
session (`Popen(start_new_session=True)`). Keep its child handle private: do not
call `Popen.poll`, `Popen.wait`, `communicate`, or another reaper elsewhere.
An exited, unreaped leader pins its PID/group identity while descendant cleanup
continues. `reap_if_finished` only completes after no live group members remain.
`stop` verifies identity before signaling the group, forwards TERM, waits for a
caller-supplied bounded grace period, escalates to KILL if necessary, and reaps
only after group exit. If identity changes or cleanup cannot finish, completion
remains unconfirmed and resources must remain leased.

These process utilities assume children keep their inherited process group and
do not daemonize, call `setsid`, or auto-reap the leader. Escaped descendants and
host reboots need a stronger service/cgroup supervisor and explicit recovery.
No global signal handler, retry policy, autonomous scheduler, or experiment
launch entry point is installed by this increment.

## Remaining wiring before direct smoke can be accepted

1. Document and version the execution/provenance substitution without changing
   scientific definitions; retain original Slurm support and schemas. Freeze a
   clean reviewed implementation and preserve existing launch-authority gates.
2. Add an explicit backend selection to extension preparation, submission,
   runtime authorization, architecture evidence, attempt/failure evidence,
   development audit, and formal audit. Validate exact canonical matrix rows,
   hashes, source files, environment and checkpoint at every current boundary.
   Do not fabricate `SLURM_*` variables or weaken Slurm checks.
3. Implement a controller that publishes a complete immutable dispatch plan
   before work, records start/exit events, leases each row only when scheduled,
   keeps a strict global concurrency cap and declared distinct GPU pairs, and
   preserves first valid results and the existing retry limits. Keep the
   48-row development and 1,600-row formal censuses unchanged.
4. Make policy serving adopt the reserved socket and expose an execution-bound
   readiness identity. Validate that identity, not just TCP connectivity, before
   any evaluator can connect. Create row/attempt logs exclusively.
5. Wire signal and deadline handling to process ownership and immutable failure
   reconciliation. Preserve the existing 240-second readiness, six-hour
   architecture, twelve-hour row wall-clock limits, and benchmark step limits.
   Complete results remain scientific results even if later cleanup fails;
   missing exit evidence or controller disappearance is not permission to rerun.
6. Resolve crash/reboot/orphan cases without freeing resources or starting
   duplicates based on a stale PID. A written plan with uncertain process start
   must remain pending until owned process and artifact evidence are reconciled.
7. Rebuild and verify both Linux environments, real checkpoint and benchmark
   assets, rendering, and inherited reference evidence. Test Linux process/FD
   behavior on the actual target before authorized GPU smoke. Run a fresh real
   checkpoint architecture smoke and the full development smoke, then review
   its audit. Formal execution still needs separate authorization.

Natural integration points identified by the read-only migration audit:
`submit_{architecture_smoke,smoke,formal}.py`, the three `run_*.sbatch` wrappers
or separate direct wrappers, `architecture_smoke.py`, `formal_artifacts.py`,
`smoke_matrix.py`, `formal_matrix.py`, both `preflight_*_row.py`,
`prepare_{smoke,formal}.py`, `record_launcher_failure.py`, `audit_smoke.py`,
`aggregate_formal.py`, and the extension routing in `examples/robomme/eval.py`.
Socket adoption additionally needs `scripts/serve_policy.py` and its serving
implementation. None of that wiring is included here.

## CPU verification

Run `python -m pytest -q tests/keyframe_oracle_sampling/test_runner_contract.py
tests/keyframe_oracle_sampling/test_direct_runtime.py
tests/keyframe_oracle_sampling/test_direct_runtime_linux.py` from the repository root.
Tests use fake GPU UUIDs, temporary lockfiles, a short CPU child process, local
loopback sockets, synthetic `/proc` files, and deterministic fake process groups.
No model, simulator, CUDA context, or experimental run is started by these tests.
The real-socket test needs local loopback bind permission. Fake lifecycle tests
prove refusal on PID reuse/reboot/foreign ownership, descendant-aware cleanup,
bounded escalation, and unconfirmed cleanup behavior. A separate Linux-only test
starts and terminates one actual owned CPU process, checks its real `/proc`
identity, and verifies it was reaped. It skips on non-Linux systems. A regression
test covers a process disappearing with `ESRCH` during the `/proc` scan.

On Lighthouse, the complete parent-plus-extension CPU suite passed with 290
tests on 2026-09-07 (242 existing tests plus 48 foundation/lifecycle checks).
That is CPU evidence only, not proof of graphics rendering, GPU policy inference,
or a completed direct execution backend. The production wiring above remains
unimplemented.
