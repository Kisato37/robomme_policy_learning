# Lighthouse direct execution

This is an operational adapter, not a change to the frozen experiment. OC3/OC5
selectors, checkpoint, seeds, 32-frame/512-token budget, tasks, episodes,
termination, retry rules and statistics remain unchanged. Existing U/O/OC/R
results and the pinned benchmark checkout are not modified.

## Authorization gates

CPU tests and a Vulkan capability query are not GPU smoke evidence. Live work
still requires clean committed source and explicit stage authorization. Run
the four real-checkpoint architecture cases, then all 48 development rows;
review the strict smoke audit before separately authorizing the 1,600 formal
rows. No training occurs. Formal preparation requires smoke evidence from the
same backend. A prepared root cannot be converted between backends.

## Resource and process ownership

`submit_direct` runs in a foreground persistent terminal session without
Slurm. It writes complete immutable submission plans before starting work.
By default each row declares distinct physical policy/simulator GPU UUIDs, checks fresh
occupancy and uses persistent advisory locks shared across this user's
checkouts. No other user's job is canceled, no alternative GPU is silently
chosen, and queued workers do not hold GPU memory or leases.

### Explicit single-GPU placement

The user authorized implementation and smoke validation of single-GPU sharing
on 2026-09-07. This changes resource placement, not the scientific protocol.
`--gpu-layout colocated` explicitly assigns both policy and simulator roles to
the same full physical UUID. A trajectory still has two separate processes;
its recorded role pair is `[GPU, GPU]`, while its resource lock and CUDA-visible
list each contain that physical device only once. Different concurrent rows
must still use disjoint physical devices. Legacy/default `separate` mode keeps
its two-GPU requirement.

The chosen layout is frozen in the prepared launch manifest, submission,
runtime profile and dispatch. Architecture, development smoke and a future
formal run must agree; existing roots cannot silently switch placement.
For colocated policy inference only, `XLA_PYTHON_CLIENT_PREALLOCATE=false`
avoids reserving most of the shared device before the renderer starts. It does
not cap the combined GPU memory footprint or change model precision, weights,
solver, seeds, simulator physics, observations, or memory budgets. No unrecorded
allocator or numerical override may be used as an ad hoc workaround.

Every direct execution records selected-device memory/utilization samples in
`gpu_telemetry.jsonl`. Report sampled peaks, not exact instantaneous peaks;
telemetry errors are diagnostic and never scientific outcomes. Single-GPU
feasibility remains unproven until real checkpoint inference and actual
front/wrist rendering complete together. By default a populated GPU is not an
available device merely because some VRAM is free. No GPU is reserved while waiting.

### Explicit sharing with other users' jobs

On 2026-09-07 the user additionally authorized sharing a populated GPU,
preferentially choosing low existing utilization and ample free memory. Add
`--allow-shared-gpu --shared-min-free-mib 49152 --shared-max-utilization 80`
to every live or dry-run direct submission, together with `--gpu-layout colocated`.
Without the sharing flag, the original strict idle-device rule is unchanged.
The resource-only admission thresholds are recorded in the hashed runtime
profile and reconstructed command; they are not model or selector parameters.
They are checked before submission and again before each execution, both before
and after acquiring our own advisory lease. Sharing is not supported with the
preallocating separate-GPU policy mode.

The launch evidence records observed free memory/utilization, and each execution
adds `gpu_admission.json` plus ongoing GPU telemetry. These readings describe
the whole shared device, not solely our process. A 48-GiB free-memory admission
threshold is a conservative starting check, not a reservation or guarantee of
future capacity; other jobs can grow. No other user's processes, priorities,
GPU settings, or memory are changed. Our cleanup remains limited to our recorded
process groups. Admission failures stop new work without an implicit GPU switch
or retry. Preserve any partial evidence and review recovery. Sharing-induced
latency is not a clean exclusive-device performance measurement.

Each row uses twelve nonoverlapping allowed CPU IDs. A combined 96-GiB RSS
guard plus a child-group watchdog limit bounds memory usage operationally.
This is **not** a kernel Slurm/cgroup memory reservation; brief overshoot is
possible, and shared host capacity must be checked before launch. Unreviewed
numerical/thread environment overrides are rejected. The architecture gate
retains its original explicit JAX settings.

Every child has recorded Linux boot/PID/parent/start-time/group identity and a
fixed deadline. A watchdog blocks payload startup until ownership and its
immutable start record are confirmed; parent disappearance, timeout or signals
trigger bounded group cleanup. GPU lease descriptors are inherited. Children
must not daemonize or escape the inherited process group/session.

The policy adopts a reserved loopback listening socket without a close/rebind
gap. Its WebSocket metadata must match execution ID and dispatch SHA before
an evaluator may connect. The evaluator verifies the renderer's actual
PCI-to-physical-GPU UUID before reset/actions. Rendering and physics settings
themselves remain unchanged. Readiness is 240 seconds, architecture six hours,
and a trajectory twelve hours, as in the original wrappers.

## Storage and environment

Use the two original locked Python environments. Environment/cache paths and
the private graphics wrapper are process-local scripts whose hashes are frozen
in the direct submission and rechecked at startup. Do not change system
drivers, shell profiles, model weights or the benchmark pin.

The designated `runs/keyframe_neighborhood_sampling` may link to the dedicated
data volume; validators accept only a direct child of that exact resolved
destination, never an arbitrary outside path.

## Usage after review

From the repository root, prepare a **new** smoke root:

```bash
.venv/bin/python -m experiments.keyframe_neighborhood_sampling.prepare_smoke \
  --run-root <new-smoke-root> --checkpoint-archive <verified-79999.zip> \
  --runner-backend direct
```

Validate the architecture launch without running it:

```bash
.venv/bin/python -m experiments.keyframe_neighborhood_sampling.submit_direct \
  --run-root <new-smoke-root> --stage architecture_smoke \
  --gpu-allocation <physical-policy-GPU-UUID> \
  --environment-sh <process-local-environment.sh> \
  --graphics-wrapper <verified-private-graphics-wrapper.sh> \
  --lock-directory <stable-own-lock-directory> --dry-run
```

Replace placeholders with reviewed paths/UUIDs. Only after explicit approval,
replace `--dry-run` with `--confirm-authorized-extension-smoke-v1`. After the
architecture passes, use `--stage development_smoke` and one or more
`--gpu-allocation <policy-UUID>,<simulator-UUID>` pairs (up to four disjoint
pairs). The initial submission always includes all 48 rows. Then run the
existing strict `audit_smoke` entry point.

For the authorized single-GPU check, add `--gpu-layout colocated` both when
preparing the new smoke root and on every direct submission. Architecture still
uses `--gpu-allocation <GPU-UUID>`; development smoke uses
`--gpu-allocation <GPU-UUID>,<same-GPU-UUID>`. The unchanged 48-row development
matrix runs serially when only this one slot is declared. This is not a reduced
scientific population and does not authorize formal trajectories.

After smoke review and separate formal authorization, prepare with
`prepare_formal --runner-backend direct` and the existing evidence/authorization
arguments. Submit through `submit_direct --stage formal` with
`--confirm-authorized-extension-v1`. The direct backend records sequential
scheduling of the unchanged 1,000/600-shard plan and permits a single execution
slot. Each shard is bounded by the recorded global concurrency cap; unused
slots stay unleased. Original concurrent Slurm scheduling remains unchanged.
Controller success is not a substitute for formal aggregation.

## Evidence and failures

Each row writes `direct/attempt_XX/row_XXXX/dispatch.json`, exclusive logs,
role start/exit records and linked `completion.json` after confirmed cleanup.
Architecture uses `direct/architecture/`. Existing scientific artifact paths
remain unchanged. Runtime and offline validators verify explicit direct
bindings instead of fabricated `SLURM_*` variables. Legacy Slurm schemas remain
supported, and live Slurm entry points reject a direct root.

Audited infrastructure failures allow other authorized rows to finish, but
yield a nonzero batch outcome. Only an explicit retry submission may rerun
exact failed rows under the original attempt limits. Scientific failures are
valid results, not retry candidates. Unknown/controller/preflight failures stop
new work. Missing start/exit/completion, partial publication or stale PIDs
remain pending; existing plans/dispatches cannot be silently overwritten or
resumed. Preserve evidence and review recovery before doing more work.

A direct evaluator lifecycle ended by its validated wall-clock deadline is a hard stop requiring review, never a retry authorization, regardless of whether the watchdog or controller delivered the termination; valid completed results retain precedence.

GPU-host migration does not prove bitwise equivalence. Published OC aggregates
alone do not establish equality of raw initial-condition hashes across hosts.
Keep that limitation explicit; do not secretly add same-host OC runs or claim
matching raw states without the original evidence.
