# Milestone 1: measure the local experiment loop

The question is how quickly this computer can solve, store, and verify a fixed
workload, and whether additional workers help. The workload deliberately uses
short trajectories. It measures data infrastructure and execution overhead; it
does not establish long-time physical accuracy or distributed capacity.

## Reproduce the study

Prepare the optional plotting dependency and the locked environment:

```sh
uv sync --locked --all-extras --group dev
```

Use a clean checkout of the **source commit recorded in the retained report**.
After preparation, this one command produces both the report and chart:

```sh
uv run --no-sync flowstate scaling-study outputs/scaling-large-01
```

The directory must not exist. Another run needs a new name. A nonsynchronized
local disk directory is preferable if the repository lives in OneDrive. The
report records the actual output path. Different hardware, dependencies, storage,
or background load can produce different timings; repeatability means executing
the same recorded protocol, not promising identical timing numbers.

The command requires clean committed code and checks free disk space before
starting. It reserves estimated output space plus scratch and metadata headroom,
then checks a minimum remaining reserve between trials. It never deletes old
results. It uses local computation and does not access cloud credentials.

## Workload and interpretation

The exact normalized configurations and execution order are in `plan.json` and
the final report. The fixed protocol uses grids 64, 128, and 256, workers 1, 2, 4,
and 8, and three repeats of each combination. There are 72 configurations per grid:
36 initial-condition seeds crossed with two viscosities. That is 216 distinct
configurations, not 216 per grid or 216 independent trajectory families.

Each configuration solves periodic, unforced, two-dimensional Navier–Stokes with
the existing dealiased vorticity solver, amplitude 0.5, viscosities 0.02 and 0.05,
timestep 0.00025, and 32 integration steps. Streaming saves the initial frame,
step 16, and step 32. The physical interval is short; these are throughput inputs,
not a turbulence study. The protocol performs 2,592 fresh executions and then
2,592 verified reuse requests. These are protocol counts, not measured successes.

All grid/worker/repeat combinations are shuffled together with a fixed order
seed. Each trial has its own interpreter and fresh lake. The child environment
sets the common BLAS/OpenMP thread limits to one; Zarr and other I/O can still use
threads. OS caches are not flushed.

## What the program does

1. `cli.main()` dispatches `scaling-study` directly to
   `scaling_study.run_scaling_study()` without constructing an unrelated lake.
2. `study_plan()` builds each parameter grid with `expand_sweep()` through the
   bounded planner. Its loops enumerate grids, seeds, viscosities, workers, and
   repetitions before any solver starts. Existing small-benchmark limits remain
   unchanged; the new protocol explicitly opts into the larger bounded planner.
3. The orchestrator writes the plan, checks the source identity, and writes one
   request file per trial. `_isolated_trial()` launches a fresh Python process.
4. `_execute_case()` times `run_sweep()`. The engine normalizes each configuration,
   computes its content identity, solves it, writes chunked Zarr fields, derives
   diagnostics, writes Parquet/JSON metadata, hashes the artifacts, and publishes
   the immutable experiment. A sampler observes the parent and descendant memory.
5. The audit loop verifies every manifest and computes scientific-array hashes.
   The same sweep runs again against that lake; this timing measures verified
   reuse rather than another solve. Failed numerical runs are also verified and
   reused, with their error preserved and no nonexistent fields read.
6. For each grid, the orchestrator compares scientific values, identities, and
   review flags across every worker count and repeat. Disagreement or storage
   failure prevents a completed-study claim. Infrastructure errors retain a
   failed report, event log, partial artifacts, and available child diagnostics.
7. `_summary()` groups trials by grid and worker count. It calculates medians,
   ranges, serial-relative speedups, and counts without mixing different grids.
   `render_scaling_chart()` plots only these retained measurements.

This is an ETL workflow: **extract** run records, manifests, fields, timing, memory,
and filesystem sizes; **transform** them into verified fingerprints and grouped
statistics; **load** immutable JSON reports and a PNG chart. Raw fields remain in
their per-trial lakes, outside Git. Compact reports and the chart are retained in
`docs/reports/` after the study finishes.

## Reading the evidence

- **Wall time:** includes worker startup, solving, writing, hashing, and publication.
  Extra audit reads, reuse, and the outer interpreter startup are excluded.
- **Runs/hour:** counts successful experiments. Attempted throughput is separate;
  fast numerical failures never masquerade as successful solves.
- **Memory:** sampled sum of process-tree resident memory. Shared pages may be
  counted repeatedly and short peaks may be missed. Inspect sampling errors too.
- **Stored bytes:** logical lengths of committed artifacts, excluding scratch,
  filesystem allocation overhead, and study reports.
- **Verified reuse:** immediate warm-cache replay, including pool startup and
  checksum work. Reuse is not guaranteed to be faster than these short solves.
- **Failures:** numerical-failure counts and errors remain visible. Infrastructure
  faults abort the study and preserve a failed report rather than hiding trials.

The graph shows median wall/reuse times with observed min/max ranges. Three
repeats describe local variability; they do not establish statistical significance.
The study keeps slowdowns and other negative results. Long trajectories, chunk
layout tuning, other equations, remote storage, and distributed workers are not
measured by this protocol.
