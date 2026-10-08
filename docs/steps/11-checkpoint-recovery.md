# Continue a numerical experiment after worker exit

An expensive run should not need to repeat every completed integration step
after its worker stops. The checkpoint option keeps the solver's internal state
and the saved trajectory prefix on local disk. A later invocation validates that
evidence, rebuilds its private field store, and continues from the completed step.

## Run and recover

```sh
uv run --no-sync flowstate --lake outputs/checkpoint-lake run examples/burgers.json --checkpoint-every 20
```

Repeat the identical command after interruption. Keep the original source,
commit, dependencies, hardware, configuration, parent, and attempt. These define
the experiment identity. Editing code or changing an attempt creates a new
identity and cannot silently consume the earlier run's checkpoint.

The option also applies to `sweep` and `queue submit`. Submission validates every
configuration before creating a queue database. A recovered queue worker first
waits for the abandoned job's lease to expire. The queue stores its checkpoint
interval and reports the restored step in both terminal events and job status.
The default execution path remains checkpoint-free. Steady Darcy has no time
trajectory and rejects the option.

Two output fields distinguish the possible successful paths:

| Output | Meaning |
| --- | --- |
| `resumed: true` | An existing finalized experiment was verified and reused. |
| `checkpoint_resumed_from: <step>` | This invocation continued an unfinished solver from that step. |
| Both false/null | This invocation started integration from the initial condition. |

The recovery interval affects execution overhead, not the numerical experiment
identity. Changing the interval does not require recomputing a finalized result.
Use a consistent interval during recovery to keep the retained history bounded.

## Extraction, transformation, and loading

`numerics.solve` advances the existing RK4 method inside its step loop. At each
scheduled saved time, the frame callback writes the physical fields to a private
Zarr store and records an immutable frame blob. At a checkpoint boundary, the
solver copies its internal state, completed step, saved times, and diagnostic
history into `SolverCheckpoint`.

For Burgers, that state is the float64 velocity vector. For Navier–Stokes, it is
the complex128 Fourier vorticity array after the existing de-aliasing and zero-mode
operations. Preserving the spectral array avoids introducing a Fourier round trip
when restoring state. Randomness is used only to construct the initial condition;
the subsequent integration has no random-number state to restore.

`CheckpointStore.append_frame` packages each physical frame as NPZ. SHA-256 of
the bytes becomes its blob filename. `CheckpointStore.save` writes a state blob
and then publishes a JSON commit marker referencing the complete frame prefix.
The marker binds configuration and provenance to the experiment identity and
includes its own payload hash.

```text
lake/
  checkpoints/<experiment-id>/
    blobs/<sha256>.npz        saved frames and internal solver states
    step-<completed-step>.json
  experiments/<experiment-id>/
    record.json              finalized status, metrics and recovery receipt
    metadata.parquet         queryable experiment record
    fields.zarr/             complete scientific arrays
    manifest.json            checksums of finalized artifacts
```

Publication flushes a private file and uses an exclusive atomic hard link to make
the blob or marker visible. A competing writer may reuse identical bytes; a
conflicting committed artifact raises an error. No worker edits another worker's
checkpoint arrays. This matters because a lost queue lease can allow duplicate
execution even though lease tokens prevent duplicate acknowledgments.

`load_latest` selects the latest published marker and verifies hashes, identity,
array types, shapes, finite values, and the saved-time schedule. An incomplete
private file has no published marker and is ignored. Corruption of a published
checkpoint stops recovery; it does not silently fall back to an older state or
become a numerical failure record.

`replay` loads the earlier frame blobs into a new private Zarr sink. The solver
continues its loop at the saved step plus one, emitting only later frames. A
checkpoint at the final step requires no further integration. After success,
the normal experiment-lake path publishes the complete Zarr, Parquet, JSON, and
manifest artifacts. Ordinary SQL and verification operate on that final record.

The final record's `recovery` object includes the requested interval, restored
step, restored marker hash, and completed integration steps in this invocation.
For a numerical failure, the latter is null because the engine has no exact
counter for partial failed integration. `runtime_seconds` measures this invocation,
including replay and checkpoint I/O; it is not cumulative runtime across crashes.

## Reproduce the controlled recovery proof

Commit source and use a new output directory. The command refuses dirty source
for measured execution and retains its plan, worker requests/logs, final arrays,
checkpoint blobs, per-case reports, and aggregate report.

```sh
uv run --no-sync python -m flowstate.recovery_study --output outputs/checkpoint-proof-01
```

The fixed protocol runs an uninterrupted reference, starts a separate worker,
pauses it after a committed checkpoint between saved frames, kills that owned
process tree, and starts a fresh interpreter to recover. Each worker counts
actual completed RK4 calls. Equality alone would not prove that earlier work was
skipped, so both the count and the restored step must agree with the plan.

Verification compares every field, time, coordinate, and diagnostic array's
dtype, shape, and SHA-256 of its bytes. Timing fields and creation timestamps
naturally differ and are excluded from scientific-array equality. Any failed
case stays in the report and prevents a successful aggregate result.

## Executed results

The [retained report](../reports/checkpoint-recovery.json) comes from clean source
commit `d8ef527ea887853a7c1169d194a72925d030691d` on Windows. The command above was
run with output `outputs/checkpoint-proof-20261007`. Both equations used grid 32,
37 integration steps, and a saved-frame interval of 8. The worker was killed after
committing step 13, between saved frames, and resumed in a new interpreter.

| Equation | Reference RK4 calls | Before kill | After resume | Arrays with identical hashes |
| --- | ---: | ---: | ---: | ---: |
| Burgers | 37 | 13 | 24 | 6 |
| Navier–Stokes | 37 | 13 | 24 | 11 |

Both cases completed with zero failures. All scientific arrays matched by dtype,
shape, and byte hash; all finalized manifests passed verification. Each recovered
record reported the restored step and counted remaining work. The full protocol
took 20.03 seconds, including separate worker processes and verification.

This is a single correctness observation, not a performance benchmark. Recorded
engine time for Burgers was 0.885 seconds uninterrupted and 1.829 seconds during
recovery. Navier–Stokes was 2.035 and 1.712 seconds respectively. The reference
has no checkpoint writes; recovery includes prefix validation/replay and later
checkpoint writes while integrating fewer steps. The slower Burgers result
remains visible, and these timings establish no general speedup.

The [validation receipt](../reports/checkpoint-recovery-validation.json) records
an independent audit reconciling raw worker/case receipts, re-hashing all 17
arrays, validating checkpoint state and prefix blobs, checking terminated worker
ownership, and re-verifying final experiments. Post-study DuckDB queries found
the completed record in every reference/recovered lake. The retained report is
an exact byte copy of the raw aggregate report.

The full local test suite passed 540 tests with four Windows symbolic-link
permission skips; its retained source-file hashes match the measured package.
Ruff passed after the study, and Windows/Linux CI passed the measured source
commit. The earlier [handoff validation](../reports/handoff-validation-20261007.json)
provides the full suite receipt. Mac validation remains a separate migration check.

## Scope and costs

The implementation bounds the planned checkpoint generations, saved-frame count,
and estimated uncompressed history size before execution; the limits are defined
in `checkpoints.py`. Checkpoints add extra disk writes, retained state snapshots,
and prefix verification/replay work. They are useful for recoverable longer runs,
but this mechanism makes no throughput or speedup claim.

This is local process-termination recovery on a compatible runtime. Power loss,
network filesystems, remote workers, and equality across different hardware are
not established. The controlled kill occurs after a committed checkpoint; tests
also exercise ignored partial files, corrupt committed blobs, and competing
writers, without claiming exhaustive interruption-point coverage.

The checkpoint directory is separate from immutable final experiments and is not
included in the current object-storage mirror. Checkpoints and abandoned private
files remain on disk; there is no automatic garbage collection. Repeated crashes
or changed checkpoint intervals can accumulate additional files beyond a single
planned execution's estimate. Finalized experiment verification remains the gate
for reusing or mirroring results.
