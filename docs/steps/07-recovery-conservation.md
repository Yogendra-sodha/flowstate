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
This provides local recovery, not distributed scheduling. The original queue
restarted an unfinished solver from time zero. Optional
[mid-trajectory checkpoints](11-checkpoint-recovery.md) now allow continuation
when submission includes `--checkpoint-every`. Model training has its own
optimizer/RNG checkpoint mechanism.

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

## Executed results on 29 September 2026

The [local recovery report](../reports/local-recovery-0.5.json) records two fresh
worker processes completing four jobs each. All eight Navier–Stokes artifacts
passed verification. Resubmission created no new jobs. A separate process then
exited with code 23 after publishing a result but before acknowledging its job;
the next worker reclaimed the expired lease and reused that exact verified result.
This tests process exit on one Windows machine, not a power failure or distributed
storage recovery. The 13.546-second concurrent run is a smoke check, not a scale
benchmark.

The [paired conservation report](../reports/conservation-0.5.json) uses 24 new
trajectories, 201 saved frames, and 1,024 spatial points from the same public source.
It transferred 46,735,488 bytes, split 16/4/4 by whole trajectory families, and ran
all six planned ten-epoch models. Source rows and initial-field hashes are disjoint
from the earlier study. These are held-out test metrics; lower is better:

| Seed | Variant | One-step field RMSE | Rollout field RMSE | RMS rollout mean drift |
| --- | --- | ---: | ---: | ---: |
| 0 | Baseline | 0.011787 | 0.224106 | 0.186900 |
| 0 | Mean preserving | 0.012931 | 0.135179 | 1.35e-7 |
| 1 | Baseline | 0.012026 | 0.235638 | 0.154972 |
| 1 | Mean preserving | 0.015164 | 0.187749 | 1.35e-7 |
| 2 | Baseline | 0.011865 | 5.300248 | 5.297188 |
| 2 | Mean preserving | 0.011441 | 5.510174 | 9.21e-8 |

Persistence rollout RMSE is 0.394423. Reference mean drift RMS is 5.71e-5.
Projection holds the predicted mean near floating-point rounding levels in all
three seeds, but seed 2 has large finite long-rollout errors in both variants and
worse aggregate field RMSE after projection. One-step errors also worsen for
projected seeds 0 and 1. These results support the conservation property, not a
general claim of stable or more accurate predictions. No seed was discarded.
Checkpoint selection used validation error; the test results were not used to
retune this completed study.

Both studies ran on clean commit `eee73e4`. A complete synthetic demo also passed
its 19 validation cases and six refinement studies, reused all 12 generated
trajectories, and completed two research proposals. Re-evaluating an original
0.4 FNO checkpoint preserved its previous error metrics exactly while adding the
new conservation diagnostics with `conserve_mean=false`.
