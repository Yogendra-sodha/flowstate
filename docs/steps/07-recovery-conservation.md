# Durable work and conservation-aware predictions

This release closes two concrete local-engine gaps: a running worker used to have
no durable job claim to recover, and a learned Burgers update could change a
trajectory's mean velocity even though the modeled periodic PDE preserves it.

## One experiment queue that survives worker exit

```sh
uv run --no-sync flowstate --lake outputs/queued-lake queue submit examples/sweep.json outputs/jobs.sqlite
uv run --no-sync flowstate queue status outputs/jobs.sqlite
uv run --no-sync flowstate queue work outputs/jobs.sqlite --max-jobs 8
```

The first command records normalized jobs and code/environment identities in
SQLite. Repeated submission deduplicates identical work. `work` claims a job,
renews its lease while solving, and records its result. A worker that stops leaves
an expiring lease; a later worker can reclaim it. A unique claim token prevents
an obsolete worker from acknowledging a job now owned by someone else.

The existing immutable experiment lake is the result store. If a process stops
after publishing a result but before acknowledging its job, a recovered worker
verifies and reuses that result. Execution can occur more than once after a lost
lease, but finalized results are never intentionally overwritten. Numerical or
execution failures become terminal failed jobs with inspectable errors. Resubmit
with a new `--attempt` for a deliberate retry. Changing code or environment creates
new submission identities; a worker refuses to execute an old job with new code.
Worker identity is pinned when its module loads and checked again around execution.
Restart worker processes after edits or checkout changes; never change source
files while jobs run. This catches cached imported code that would otherwise be
incorrectly labeled using a newer fingerprint from disk.

SQLite coordinates processes on **one machine and local disk**. Do not put the
queue on a network share or operate the same synced copy from multiple machines.
This provides local recovery, not distributed scheduling. Solver runs still
restart from time zero when no finalized artifact exists. Model training has its
own optimizer/RNG checkpoint mechanism.

## Preserve the quantity the PDE preserves

For periodic unforced Burgers on a uniform grid, the integral of velocity is
constant: the integrals of the flux derivative and diffusion derivative vanish
over the periodic domain. With a fixed grid, this is the spatial arithmetic mean.

The FNO predicts an update to its input. With `--conserve-mean`, the program
subtracts that update's spatial mean before adding it to the input:

```text
learned_update = network(current_field)
zero_mean_update = learned_update - mean(learned_update, over space)
next_field = current_field + zero_mean_update
```

This preserves each sample's own mean, including a nonzero mean. Normalization
uses a single training-set mean and standard deviation, so the property also
holds in physical units, up to floating-point error. Projection changes the
model's function class; it is recorded in training configuration and checkpoints.
The default remains the prior unconstrained model, and old checkpoints without
the flag evaluate with their original behavior. Resuming training still requires
identical configuration, implementation, and PyTorch version.

```sh
uv run --no-sync flowstate train-fno outputs/public-study-20260928/dataset outputs/mean-fno --conserve-mean
uv run --no-sync flowstate conservation-study outputs/public-study-20260928/dataset outputs/conservation-01
```

The paired study freezes its plan before acquisition. It samples 24 new complete
trajectories with seed 20260929, excludes the previous source rows, and rejects
any repeated initial-field hash. Each of three training seeds runs both baseline
and projected models with identical initialization, minibatch order, and budgets.
Checkpoint selection uses validation MSE independently for each run. All six
test results are retained; no winning seed is chosen. This is a fresh cohort from
the same public file and numerical solver, not validation against an independent
physical source or a full PDEBench reproduction.

## Measure drift separately from reference error

`mean_velocity_rmse` measures disagreement with the stored reference mean. The
reference itself may drift numerically, so that metric alone does not establish
conservation. New diagnostics report:

- One-step predicted mean minus the preceding reference frame's mean.
- Rollout predicted mean minus that trajectory's initial mean.
- Reference mean minus its own initial mean.

Each includes RMS, maximum absolute drift, and per-saved-time RMS, in physical
velocity units. A lower drift does not guarantee lower field error, dissipating
energy, stable rollouts, or a converged PDE solution. Those outcomes remain
separate measured quantities.
