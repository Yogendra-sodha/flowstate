# Flowstate implementation stepbook

This is the working engineering notebook for Flowstate. It records the starting
point, design rationale, implementation, data transformations, functions and loops,
commands, validation evidence, and limitations. It explains the program at a level
that a new contributor can follow. Decisions below are engineering summaries, not
a claim that the scientific questions are already answered.

## 0. Starting point and the goal

The repository originally contained an MIT license. The first implementation was
committed on `develop` as `717de85` (`feat: add reproducible PDE experiment engine`).
It supplied periodic Burgers and 2D Navier–Stokes solvers, an immutable local lake,
parameter sweeps, provenance, SQL queries, and experiment lineage. Its 70 tests
passed locally and in Windows/Linux CI. Eleven example runs completed; the
eight-run sweep reused all eight verified results on a second invocation.

The next development phase begins from that commit. The objective is to connect
the numerical evidence to curated datasets, learned baselines, a research graph,
and a bounded next-experiment policy. The Navier–Stokes Millennium problem remains
outside the project's claims. We measure computational evidence, including failure.

### Progress ledger

| Step | Deliverable | Status at this entry |
| --- | --- | --- |
| 1 | Local environment, solvers, experiment lake, catalog | Complete in `717de85` |
| 2 | Darcy adapter and numerical/resource validation report | Implemented; focused tests pass |
| 3 | Dataset ETL, PDEBench HDF5 import, S3 artifact mirror | Implemented; fixture/emulator tests pass |
| 4 | Burgers FNO, checkpoint/evaluation, physics-informed baseline | Implemented; training/resume tests pass |
| 5 | Streaming saved fields and research graph | Implemented; identity/integrity tests pass |
| 6 | Bounded proposal/execution loop with evidence links | Implemented; budget tests pass |
| 7 | Integrated demonstration, scientific review, documentation, CI | 140 local tests pass; integrated study and CI next |

Later entries update this ledger with commands and measured results. A local S3
emulator can validate the storage protocol; it does not establish that a user's
cloud account, bucket, network, or IAM policy has been deployed.

## 1. Setting up a reproducible Python project

`pyproject.toml` defines the installable package, dependencies, the `flowstate`
console entry point, and development tools. Code lives in `src/flowstate/`; tests
live in `tests/`. The source layout helps catch accidental imports from the wrong
working directory. `uv.lock` pins the resolved dependency graph. `.venv/`, data,
outputs, caches, credentials in `.env`, and build artifacts are excluded from Git.

The base stack is NumPy for arrays/FFT, SciPy for sparse Darcy systems, Zarr for
chunked fields, PyArrow for Parquet, DuckDB for SQL, and h5py for HDF5 ingestion.
PyTorch is an optional `ml` extra; boto3 is an optional `s3` extra. PyTorch's explicit
CPU package index keeps this prototype independent of a CUDA installation. The
test environment includes Moto to exercise S3 without a cloud account.

```sh
uv sync --locked --all-extras --group dev
uv run --no-sync ruff check .
uv run --no-sync pytest
```

Run commands from the repository root. `flowstate --lake PATH ...` selects an
artifact lake; the global option precedes the subcommand. The CLI reads JSON
configuration and prints JSON results so shell scripts can consume them without
scraping prose. Exceptions for invalid configuration, files, or SQL produce a
nonzero exit status and a short error on stderr.

## 2. Why fields and metadata have different storage paths

A velocity field is a multidimensional numeric array. A catalog record is a small
row describing an experiment. Combining both in a CSV would lose shape information,
make partial spatial/time access expensive, and repeat metadata needlessly.

```text
JSON configuration -> canonical parameters -> solver
                                         -> fields/diagnostics -> Zarr
                                         -> scalar summaries   -> Parquet
                                         -> provenance          -> JSON
Parquet records -> DuckDB experiments table -> SQL analysis
Experiment records -> typed links -> research graph -> next proposal
```

The initial lake layout is:

```text
data/lake/experiments/<experiment-id>/
  record.json
  metadata.parquet
  fields.zarr/
  manifest.json
```

`record.json` retains nested configuration and provenance. `metadata.parquet`
contains flattened query columns and nested JSON strings for less common fields.
`fields.zarr` contains time, spatial coordinates, physical fields, and aligned
diagnostics. `manifest.json` maps relative artifact paths to SHA-256 hashes.

## 3. Extraction: what enters an experiment

The first source is a numerical configuration. `normalize_config()` fills defaults
and rejects unknown keys, nonfinite values, invalid integer sizes, and unsupported
equations. This is schema validation before computation. The canonical object is
the one recorded and hashed; omitted defaults cannot secretly change run identity.

`capture_provenance()` walks the package's Python source files in sorted order,
hashing each relative path and its bytes. It records Git commit and dirty state,
Python/package versions, precision, and machine description. It deliberately does
not collect arbitrary environment variables, access keys, or credentials. The
source fingerprint identifies code content; it cannot reconstruct uncommitted code.
Commit source and retain the lockfile for reproducible studies.

`experiment_id()` serializes configuration, source/environment identity, parent,
and attempt using stable JSON key ordering, then hashes it. Timestamps are excluded:
the same requested computation should be recognizable tomorrow. Changed code,
configuration, environment, parent, or attempt gives a different identity.

## 4. Transformation: numerical fields into scientific diagnostics

### Functions and loops inside the solvers

The Burgers solver represents `u` on a uniform periodic one-dimensional grid. Its
right-hand-side function uses a centered difference of `u²/2` for the conservative
flux and a centered second derivative for viscosity. `np.roll` supplies periodic
neighbors without a Python loop over individual grid cells. Conservation of the
spatial mean follows from cancellation of the periodic flux differences; this does
not make the method a shock-capturing scheme.

The Navier–Stokes solver evolves scalar vorticity on `(y,x)`. FFTs transform arrays
to Fourier coefficients; multiplication by physical wavenumbers differentiates
them. Inverting the nonzero Laplacian modes reconstructs the streamfunction and
zero-mean velocity. The nonlinear product is formed in physical space, transformed
back, and truncated with a strict two-thirds mask. The zero mode and domain-length
factors are explicit. Pressure is not synthesized.

Both time-dependent solvers have an outer loop over integration steps. RK4 calls
the right-hand side four times per step, using intermediate states; each stage
checks an advective/diffusive timestep bound. NumPy executes the spatial array
operations. A saved-frame condition checks `step % save_every == 0`, with an extra
condition for the final step, so the initial and final states are always retained.
This distinguishes integration steps from output frames.

At a saved frame, diagnostic functions reduce arrays to scalar means, integrals,
or maxima. Energy is a spatial mean of half squared speed, not a per-cell value.
Viscous unforced energy should decay. Burgers mass, NS circulation, enstrophy, and
Fourier divergence use documented conventions. Stability checks run at RK stages;
saved diagnostic samples can miss events between output times.

`summarize()` transforms aligned diagnostic histories into queryable scalars. It
computes initial/final/max energy, drift, physical final time, and the first sampled
energy increase. `np.diff` detects changes between consecutive saved frames;
`np.flatnonzero` finds violating indices. The record preserves flag thresholds.
`needs_review` is an observation for review, not a scientific conclusion. Reynolds
number uses a declared speed/length convention and is null at zero viscosity.

### Parameter-grid processing

`expand_sweep()` validates the entire grid before any solver runs. One loop checks
that each parameter has a nonempty list and that the product stays within the run
budget. `itertools.product` generates Cartesian combinations. Each combination is
merged into the base config, normalized, serialized, and deduplicated with a set.

`run_sweep()` dispatches those independent jobs serially or through a process pool.
Each worker invokes `run_experiment()` and writes its own directory. A numerical
failure is recorded and does not erase successful siblings. Unexpected storage
errors propagate rather than being misclassified as a physical anomaly.

## 5. Loading: atomic publication and immutable evidence

`Lake.write()` creates a hidden staging directory on the same filesystem as the
destination. It writes JSON, Parquet, and arrays, then hashes artifacts. Only after
all steps succeed does it publish the directory with a rename. The catalog ignores
staging directories, so readers do not discover half-written runs. Failed staging
is cleaned up within the lake. A machine crash can leave staging behind; it remains
invisible and the computation can restart from time zero.

Every field array has time first. Burgers uses `(time,x)` and NS uses `(time,y,x)`.
The original storage writer loops over named collections and arrays. It validates
numeric types and the leading time dimension, then chooses chunks: one frame per
time chunk and up to 64 points per spatial dimension. This favors reading a frame
or a local portion without retrieving an entire trajectory.

Overwrite is refused. Concurrent writers for the same identity may compute twice,
but only one publishes. Both Unix `ENOTEMPTY` and `EEXIST` conflicts are normalized
so the losing worker can verify and reuse the winner. This is application-level
immutability, not tamper-proof storage. An unsigned checksum manifest detects
accidental damage but cannot authenticate coordinated malicious changes.

## 6. Querying and resuming the evidence

`Lake.query()` lists only finalized experiment directories and creates the DuckDB
`experiments` table from their Parquet files using `union_by_name`. This permits
equation-specific metric columns while aligning common columns. Explicit common
columns keep an empty or failed-only lake queryable. The database is reconstructible;
there is no second writable catalog that can drift away from artifacts.

The connection ingests internal metadata, disables external file/network access,
locks configuration, and accepts one SELECT statement. The result is converted
from row tuples to named JSON objects. This is a local analysis interface; it is not
an authenticated multi-user SQL service.

Before reuse, `run_experiment()` checks every stored artifact against the manifest.
The result includes `resumed=true` when an existing completed or failed attempt is
reused. A new attempt is a new record; `--parent` links it to earlier evidence.
Resumption at this stage means skipping finalized work, not restarting a solver
mid-trajectory. Dataset exports and model checkpoints add their own identities
and validation rules in subsequent chapters.

## 7. Detailed implementation chapters

- [Darcy and validation](docs/steps/02-darcy-validation.md)
- [Dataset ETL and object storage](docs/steps/03-data-pipelines.md)
- [Learning baselines](docs/steps/04-learning-baselines.md)

The remaining sections explain the research graph, proposal policy, streaming,
review corrections, and measured integrated execution.

## 8. Incremental field loading: reducing retained simulation memory

The first solver accumulated saved frames in a list and called `np.stack` at the
end. That is easy to test, but memory scales with `frames × spatial cells × fields`;
stacking can temporarily hold both the list of arrays and the combined array.

The new `ZarrFrameSink` in `streaming.py` is a synchronous callback. The solver calls
it inside the saved-frame branch of the time loop. On the first frame, the sink
creates arrays with the known final shape and one-frame chunks. On each call it
writes `array[frame_index] = current_field` before the solver mutates its state.
`retain_fields=False` prevents the historical field list from growing. The result
still returns physical coordinates, times, and small scalar diagnostic histories.

`finish()` checks that the number and physical times of saved frames agree with
the diagnostic history, then writes coordinates, diagnostics, and metadata. The
engine passes the completed store to `Lake.write()`, which copies chunk files into
the ordinary staging/publish path without loading the trajectory into RAM. This
uses additional temporary disk space and copies bytes once; it is not a zero-copy
remote streaming writer. Failed integration discards its partial field store and
publishes the usual failure record.

Memory now includes RK/FFT working arrays plus a current frame, Zarr buffers, and
`O(saved frames)` scalar diagnostics. It no longer retains every saved spatial
field. The original estimated 256 MiB output guard remains in place. `--stream`
selects this path for `run` and `sweep`; steady Darcy already has only one field
snapshot. Tests compare every streamed field value to buffered solver output and
check that numerical failures leave no published partial trajectory.

## 9. Research objects, links, and an auditable next-experiment loop

`ResearchGraph.build()` iterates over verified experiment records. It creates typed
Equation, Solver, Experiment, InitialCondition, BoundaryCondition, Dataset, and
Metric nodes. It connects each metric to the experiment it measures. Failure or
sampled diagnostic flags create Anomaly objects that explicitly describe numerical
evidence requiring review. The graph is reconstructed from persisted evidence; it
does not depend on a running graph database.

Explicit Hypothesis, Finding, Proposal, Model, and Checkpoint records are stored in
`lake/research/entities/<id>/`. Their identity hashes type, properties, and links.
The small entity directory is atomically published with a record checksum. Registering
the same assertion reuses it; changed assertions receive new identities. A finding
must cite evidence. Imported or missing link targets appear as unresolved nodes,
so a broken evidence link is visible instead of silently disappearing.

`register_model_artifact()` verifies a model bundle and links a Model to its curated
Dataset and its Checkpoint. The dataset identity is its manifest hash, not just a
human filename. Checkpoint hashes identify the saved model state. Paths remain
useful locators, while hashes distinguish content.

`propose()` is a deterministic policy. It sorts failed runs first, then flagged runs,
then stable runs, with stable ID tie-breaking. For time-dependent equations it halves
`dt`, doubles `steps`, and doubles `save_every`. This preserves final physical time
and saved output times, making comparison meaningful. For steady Darcy it doubles
the number of grid intervals. A normalization call rejects proposals outside the
solver's permitted parameter range. Already executed parent/config combinations
are skipped. Run-count and integration-step budgets stop expansion.

The proposal includes its parent evidence, a before/after parameter delta, rationale,
and hypothesis. `research_cycle(execute=False)` records proposals without running
them. With `execute=True`, a loop calls the existing engine once per approved
budgeted proposal and records an observational Finding. It does not automatically
assert that a hypothesis is supported, contradicted, or proved. Matrix resolution,
output-size guards, run count, and step count are bounded; this is not a wall-clock
or cloud-spend scheduler.

## 10. Integrating the commands and recording pipeline execution

The CLI keeps the base numerical imports light. Training, import, validation, and
S3 modules are imported when their subcommand runs. `dataset export` reads verified
experiment artifacts, `train-fno` consumes the curated dataset, and `evaluate-fno`
consumes a dataset and a matching checkpoint. Each stage validates its input
contract rather than trusting that an earlier process used the right parameters.

`run_demo()` records a sequential study in a new output directory. It writes an
append-only `events.jsonl`: one JSON object per stage transition with UTC time,
stage, status, and relevant result details. The stages are numerical validation,
trajectory generation/reuse, dataset ETL, FNO, PINN, and a research cycle. Validation
failure stops learning. On an exception, the failed stage and error are recorded;
completed evidence remains available. The orchestration log is separate from
immutable datasets/checkpoints and does not masquerade as scientific data.

The demo trains a small model on twelve 32-point Burgers trajectories with different
random initial-condition seeds. It evaluates FNO one-step and rollout predictions,
fits a PINN to one held-out initial-value problem using physics, runs a manufactured
Darcy case, and deliberately creates a timestep failure for the proposal policy.
The final report includes catalog counts and research graph sizes. This is a
reproducible systems demonstration, not a published scientific-ML benchmark.

### Development environment note

Upgrading the editable package during this phase exposed incomplete package metadata
in the local OneDrive-backed environment. The obsolete metadata directory was
preserved under a disabled name; the package manager restored the affected dependency.
No experimental data was changed. In a shared environment, install extras once with
`uv sync --all-extras --group dev`, then use `uv run --no-sync` for concurrent tests.
An ordinary sync without an optional extra may remove that extra, so ML commands
in the instructions specify the necessary extra or use the prepared environment.

### Validation log so far

- Original foundation: 70 tests, clean Windows/Linux CI, eleven demo experiments.
- Darcy/resource addition: 22 focused tests; 19 numerical cases across six studies.
- Core integration after streaming/Darcy/research changes: 51 focused tests passed.
- Complete prototype suite: 140 tests passed in 90.55 seconds on local Windows.
- Ruff passed; `uv build` produced the version 0.2.0 wheel and source archive.
- Full integrated counts and measured ML/storage results are recorded below after
  the end-to-end run, not inferred from code completion.

## 11. Review-driven corrections and storage measurement

The PINN derivative test uses a linear physical-coordinate model whose spatial
gradient is constant. That gradient still depends on trainable weights, so PyTorch
marks it differentiable even though it no longer depends on the input coordinates.
Requesting its second derivative without allowing an unused coordinate tensor
raises an error. The derivative helper now represents that mathematically zero
Hessian correctly, while retaining normal autograd paths for nonlinear models.

The research graph originally identified an initial condition from its name, seed,
amplitude, and domain length. Review found that the same values could describe a
1D Burgers field and a 2D Navier–Stokes field. The identity now includes equation,
grid size, and source fingerprint. Grid matters because the random generator's
retained modes can change on small grids. A regression test covers the collision.

`benchmark_storage()` runs the same NS configuration in buffered and streamed
modes, measures solve/write elapsed time, traced Python peak allocations, actual
stored bytes, and a warm local tile read. It hashes one frame at a time to verify
identical physical values without rebuilding the full trajectory. The read loop
repeats a 32×32-or-smaller tile ten times after warming it, then reports an average.
This includes Zarr decoding and local-cache behavior; it is not cold object-store
latency. `tracemalloc` is explicitly not process RSS and may omit native allocations.
The benchmark records evidence rather than asserting a universal speedup.

Streamed writer errors are wrapped as storage errors. They propagate out of the
engine instead of being logged as numerical instability. A regression injects a
codec failure and verifies that no scientific failure record is published.

```sh
uv run --no-sync flowstate storage-benchmark outputs/storage-study --grid-size 64 --steps 32
```
