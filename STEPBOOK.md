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
| 2 | Darcy adapter and numerical/resource validation report | Complete locally; six studies pass |
| 3 | Dataset ETL, PDEBench HDF5 import, S3 artifact mirror | Complete locally; fixture/emulator tests pass |
| 4 | Burgers FNO, checkpoint/evaluation, physics-informed baseline | Complete locally; measured results in section 12 |
| 5 | Streaming saved fields and research graph | Complete locally; fields agree and graph builds |
| 6 | Bounded proposal/execution loop with evidence links | Complete locally; two refinements executed |
| 7 | Integrated demonstration, scientific review, documentation, CI | 148 local tests and complete study pass; CI runs on Windows/Linux |

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
- [Local worker and storage scaling](docs/steps/05-local-scaling.md)

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
  the end-to-end run, not inferred from code completion; see section 12.

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

### Integration exposed a Windows publication failure

The first integrated study ran from clean commit `3051618`. Numerical validation,
twelve streamed trajectories and their reuse, dataset export, FNO training, and
PINN training completed. During the research cycle, Windows returned `WinError 5`
while renaming a fully staged experiment directory into its final location. No
partial experiment became visible. A separate execution of the pending refinement
succeeded without changing its numerical configuration or the filesystem permissions.

That is evidence of a transient publication failure. The workspace is on OneDrive,
but the exact locking process was not identified. The failure is retained in the
first study's `events.jsonl`; it is not relabeled as numerical instability or erased.
The interrupted study is separate from the final complete demonstration.

All five local artifact publishers now use one bounded rename helper. Only Windows
access/sharing/lock errors (`winerror` 5, 32, or 33) qualify for retry. Delays of
0.01, 0.05, 0.2, 0.5, and 1 second permit six attempts, then the original filesystem
error propagates. Every attempt still uses the same atomic rename operation.
Existing-destination collisions retain their original handling. Ordinary permission
errors on other systems are not retried, and no permissions are widened.

Regression tests inject transient and persistent Windows errors, non-Windows
permission errors, and publication collisions. They check eventual publication,
bounded exhaustion, cleanup, and preservation of existing results. The retry applies
to experiment, dataset, model, research-entity, and downloaded-S3 artifact publication.

After this repair, all 148 tests passed in 75.45 seconds on local Windows. Ruff and
the source/wheel build also passed. The final study uses a new directory and a clean
commit containing this repair; no artifacts from the interrupted run are overwritten.

## 12. Measured complete study and release evidence

The complete study ran on September 24, 2026 in New York (September 25 UTC), from
clean commit `0259a3262bebb7e9c0f98a6d067c69e3d4ab694c`. It began at
03:19:41 UTC and completed at 03:20:20 UTC, approximately 39.2 seconds. The environment
was Windows 11, Python 3.12.9, and CPU PyTorch 2.14.0. The machine-readable
[prototype report](docs/reports/prototype-0.2.json) retains exact measurements,
source fingerprint, dependency versions, configuration, run IDs, and report hashes.
Later documentation-only commits do not change the measured implementation.

```sh
uv sync --locked --all-extras --group dev --python 3.12
uv run --no-sync ruff check .
uv run --no-sync pytest -q
uv build
uv run --no-sync flowstate demo outputs/research-20260924-verified --epochs 20 --pinn-epochs 200
uv run --no-sync flowstate storage-benchmark outputs/storage-20260924-64 --grid-size 64 --steps 32
```

Choose new output names when reproducing the study. The raw study artifacts remain
under ignored `outputs/`; the compact report and this explanation are committed.
Local validation passed 148 tests. Ruff passed, and both wheel and source archives
built successfully. After execution, all sixteen experiment manifests, the curated
dataset, and both training bundles passed integrity verification. GitHub's
[Tests workflow](https://github.com/Yogendra-sodha/flowstate/actions/workflows/ci.yml)
repeats the locked environment setup, lint, and tests on Windows and Linux.

### Numerical accuracy before learning

Nineteen solver cases formed six studies; all passed their stated checks:

| Study | Measured evidence |
| --- | --- |
| Burgers spatial refinement | Orders 1.987 and 1.997, consistent with second order |
| Burgers temporal refinement | Orders 4.096 and 4.047 against a fine-step reference |
| Taylor–Green spatial check | Velocity RMS errors around 3.2e-15; no spatial order inferred from a single resolved Fourier mode |
| Taylor–Green temporal refinement | Orders 4.039 and 4.019 |
| Constant-permeability Darcy | Spatial orders 2.002 and 2.001 |
| Smooth-permeability Darcy | Spatial orders 2.004 and 2.001 |

These serial studies took 8.35 seconds including reference calculations and artifact
persistence before report writing. Their NPZ artifacts used 144,509 bytes. Maximum
traced solver allocations for one case were 1,177,450 bytes. This is a bounded
manufactured/known-solution check, not a proof for arbitrary turbulent flows.

### Dataset and learning results, including the weaker baseline

Twelve Burgers trajectories produced a `[12, 11, 32]` dataset: twelve initial-value
problems, eleven saved times, and thirty-two periodic grid points. Repeating the
sweep reused all twelve verified experiments. Family assignment produced eight
training trajectories, two validation trajectories, and two test trajectories.
Normalization used only the 2,816 training values. Viscosity was 0.1, integration
step 0.002, saved-time interval 0.02, and final physical time 0.2.

FNO trained for twenty epochs; validation selected epoch twenty. The model used
width sixteen, eight Fourier modes, three blocks, batch size sixteen, learning
rate 0.001, and seed zero. Training took 1.66 seconds in this run. Error below is
physical-velocity RMSE against the stored finite-difference trajectories:

| Evaluation | Learned model | Persistence baseline |
| --- | ---: | ---: |
| FNO one saved step, two test trajectories | 0.0002723 | 0.0022445 |
| FNO rollout through ten future frames | 0.0016054 | 0.0141163 |
| PINN future trajectory, one test initial condition | 0.0767211 | 0.0116183 |

Persistence means copying the previous reference field for a one-step prediction,
or retaining the initial field throughout a rollout. FNO reduced the field error
in this small study. Its rollout mean-velocity RMSE was nevertheless 0.0002223,
compared with about 1.45e-9 for persistence: lower field error does not imply exact
mass conservation. This FNO does not enforce conservation by construction.

The PINN used width thirty-two, three hidden layers, 128 sampled collocation points
per epoch, and 200 epochs, taking 2.31 seconds. Its fresh-point physical PDE residual
RMS was 0.47149, and its trajectory error exceeded persistence. This short training
budget did not produce a competitive solution. That negative result is retained;
it is not hidden behind the fact that the training command completed. FNO and PINN
also solve different learning tasks, so these are not equal-budget rankings.

Two FNO test trajectories and one PINN instance are too few for broad claims about
generalization. A next learning study should use more independent families, finer
numerical references, declared budgets, and validation-based tuning without looking
at test outcomes to select settings.

### Streaming trades time for lower retained memory in this run

The separate local storage study used a 64×64 Navier–Stokes grid with thirty-two
integration steps and thirty-three saved frames. It ran the same configuration
serially in both modes:

| Measurement | Buffered | Streamed |
| --- | ---: | ---: |
| Peak traced allocation bytes | 7,867,179 | 1,555,890 |
| Solve plus Zarr-write seconds | 0.720 | 1.551 |
| Stored bytes | 3,108,310 | 3,108,310 |
| Warm 32×32 tile read, average seconds | 0.000918 | 0.003199 |

The three saved field arrays had identical hashes in both modes. Streaming reduced
traced peak allocations by approximately 80% here while increasing elapsed write
time. Python tracing is not whole-process RSS; native libraries and OS caches may
allocate untraced memory. These are single local measurements without confidence
intervals. They do not establish cold-cloud latency or concurrent-worker throughput.

### Evidence-driven follow-up completed

The final lake contained fourteen completed Burgers experiments, one deliberately
failed Burgers experiment, and one completed Darcy experiment. The researcher used
two runs and 208 integration steps within its declared 500-step budget. First it
halved the failed case's timestep from 0.08 to 0.04 while preserving final time;
the refined case completed. It also refined one stable trajectory from timestep
0.002 to 0.001. Both outcomes have parent links, proposal records, and cited findings.

The resulting graph has 309 nodes and 341 links, including datasets, models,
checkpoints, metrics, anomalies, hypotheses, proposals, and findings. Completing a
refinement is an observation; no edge automatically declares a hypothesis proved.

## 13. What remains after this prototype

The next infrastructure steps, in execution order, are:

1. Measure concurrent local execution and storage with representative workloads;
   choose scheduling and retention policies from those results.
2. Validate the existing S3 protocol against an actual configured bucket, including
   permissions, interruptions, network behavior, and storage costs.
3. Import a declared public PDEBench dataset and run larger, reproducible learning
   studies with stronger numerical references and explicit compute budgets.
4. Add durable distributed work queues, remote workers, cancellation, and
   mid-trajectory solver recovery before attempting very large sweeps.
5. Build a dashboard for the existing evidence and a constrained LLM planner whose
   proposals still pass the engine's validation and budget checks.

These are future capabilities and measurements. The delivered work establishes the
local experiment-to-dataset-to-model-to-evidence loop and documents its current limits.

## 14. Continue with measured local concurrency

This phase starts from `aa69bb5`, the verified version 0.2 prototype. The next
question is practical: for a fixed workload on this machine, do extra workers
reduce elapsed time, and how much resident process memory do they add?

Version 0.3 adds `scaling.py` to organize isolated trials and `resources.py` to sample
process-tree RSS with psutil. The [new chapter](docs/steps/05-local-scaling.md) explains
the loops and the benchmark's own extract-transform-load pipeline. The declared
example uses four random seeds for each of Burgers and Navier–Stokes, thirty-two
grid points per dimension, sixty-four integration steps, and a saved frame every
two steps. One, two, and four workers are compared in both storage modes.

The engine now explicitly starts workers with `spawn`. Starting a pool with fork
while a sampler or Zarr runtime has active threads can inherit unsafe runtime
state. Spawn also makes the process-start contract explicit across Windows and Linux.
Its startup cost is included in the measurement.

Review caught two reporting problems before the final study. First, writing an
unignored output directory can change Git's dirty annotation without changing code.
Runtime comparisons now use the actual commit, source fingerprint, dependencies,
Python, hardware, and precision while retaining dirty state as an annotation.
Second, an I/O exception after partial publication must not erase elapsed time or
memory evidence. Those measurements are saved in `finally`, and failed reports list
the published experiment IDs. They never enter a successful speedup summary.

The initial focused verification passed forty-nine engine, scaling, and sampler
tests. Resource tests use controlled process values to prove that the aggregate
peak comes from one sample, rather than adding unrelated parent and child peaks.
An actual subprocess integration test compares serial/parallel and buffered/streamed
results, checks reuse, and confirms that equivalent arrays have identical hashes.
Measured workload results are appended after running from a clean source commit.

The complete local suite passed 186 tests in 88.28 seconds. Ruff passed, and
`uv build` produced the version 0.3.0 source archive and wheel. The larger measured
study is kept separate from these regression checks so its timings are collected
without a concurrent test workload.

## 15. Results of the isolated local scaling study

The study ran from clean source commit `9646962` on the same Windows computer as
the version 0.2 demonstration. The exact configuration, execution order, trial
values, provenance, and report hash are committed in the
[compact scale report](docs/reports/local-scaling-0.3.json). Eighteen fresh trials
each ran eight experiments: four random Burgers and four random 2D Navier–Stokes
problems at grid size 32, viscosity 0.05, 64 integration steps, and a saved frame
every two steps. There were three trials for each worker/mode combination, totaling
144 fresh experiments and 9,216 integration steps. Every trial subsequently reused
all eight verified artifacts. Fields, coordinates, saved times, and diagnostic
values matched across every condition.

The table reports the median of three complete fresh-sweep wall times. The sampled
RSS column is the *largest observed parent-plus-child RSS sum* across those three
trials, not a median or actual physical-memory peak.

| Storage | Workers | Median wall time | Speedup within mode | Maximum sampled RSS sum |
| --- | ---: | ---: | ---: | ---: |
| Buffered | 1 | 11.87 s | 1.00× | 104 MB |
| Buffered | 2 | 9.12 s | 1.30× | 281 MB |
| Buffered | 4 | 9.15 s | 1.30× | 463 MB |
| Streamed | 1 | 19.92 s | 1.00× | 104 MB |
| Streamed | 2 | 15.34 s | 1.30× | 279 MB |
| Streamed | 4 | 10.77 s | 1.85× | 450 MB |

For this small mixed workload, **two buffered workers** gave the lowest median
time. Four buffered workers gave no meaningful further improvement but a much
larger sampled RSS sum. Streamed mode gained more from four workers but remained
slower than two buffered workers in absolute median time. Each trial stored about
3.43 MB of finalized artifacts. The reuse medians were about 2.3–3.5 seconds,
including checksum verification and pool startup.

This does not contradict the earlier 64×64 single-run result showing much lower
traced *Python allocations* with streaming. This 32-grid study has short saved
histories and samples *resident process memory*, including libraries and worker
interpreters. At this size, the sampled one-worker RSS was almost unchanged by
storage mode. The metrics have different scopes and should stay separate.

The local output directory was OneDrive-backed; OS caches, synchronization,
background work, and CPU scheduling were not controlled. At most three repeats per
condition give a useful first comparison, not a universal scaling law. A larger
grid or longer saved history may change the best setting. We therefore retain the
default of one worker for safety and let users choose a measured worker count for
their workload. More representative throughput and an actual cloud bucket remain
future measurements.

## 16. Learning the project from one real experiment

The [plain-English start guide](docs/start-here.md) explains the project as one
literal data journey. It gives a 45–60 minute practice route: run Burgers twice,
see verified reuse, inspect stored fields, query metrics, and locate the functions
and tests responsible. It then introduces dataset ETL, FNO versus PINN, the
research graph, and the scaling study in dependency order. Its videos and articles
are optional support for concepts that appear in this repository.

## 17. Taking real public data through the same pipeline

The next gap was evidence from external data: earlier import tests used synthetic
HDF5 fixtures, while learned-model demonstrations used our own solver. Version
0.4 adds a bounded route from the official PDEBench Burgers release to our existing
training pipeline. The [public-data walkthrough](docs/steps/06-public-data.md)
documents the commands, functions, loops, extraction, transformation, publication,
and scientific limits in detail.

I first checked the official dataset version, license, size, checksum declaration,
and diffusion convention. The smallest Burgers file is 8.23 GB, so downloading a
whole file merely to train on a few rows would be wasteful. The server supports
HTTP byte ranges. `HTTPRangeReader` now exposes a bounded seekable file to h5py,
with 1 MiB blocks, four cached blocks, strict response checks, and stable ETags.
TLS verification remains enabled. The default transfer cap is 64 MiB, and there
is no fallback to downloading the entire file.

The extraction loop selects 24 rows using a recorded random seed before seeing
their values. It preserves all 201 saved frames and all 1,024 spatial points.
The source's unpaired final time coordinate is recorded and omitted. A local HDF5
subset, an attribution/selection/transfer receipt, and their manifest publish
together. Failed staging is removed; previously published artifacts are immutable.

The transformation loop checks finite fields and coordinate spacing and copies
one trajectory at a time into canonical chunked Zarr. Related initial fields stay
in one split. Normalization updates only on training trajectories, before any
learning windows are generated. Public source row numbers remain separate from
row numbers in the smaller local file. The SHA-256 explicitly covers that local
file; the full publisher MD5 is advertised but not verified by a partial download.

`public-study` writes a plan before acquisition and iterates over FNO seeds 0, 1,
and 2 with a fixed split and training budget. Validation chooses checkpoints;
held-out results never choose hyperparameters or a preferred seed. It then fits
one PINN using only a held-out initial frame and physics. Every completed stage
keeps its artifacts, and failures record their stage and error. This adds a
traceable external-data demonstration, not a claim to reproduce the full benchmark.

### Measured public-data result

The complete study ran on 2026-09-28 local time (2026-09-29 UTC) from clean commit
`29377f3f7f9b1a97e72c2570596651ce0e52725b`, version 0.4.0. The
[measured report](docs/reports/public-burgers-0.4.json) preserves the plan, source
attribution, transfer receipts, split membership, every seed's results, per-time
errors, hashes, and execution events. Full local fields, predictions, and
checkpoints are in ignored `outputs/public-study-20260928/`.

The download used 49 range requests and received **49,881,216 bytes**, below the
64 MiB cap. It selected 24 of 10,000 source trajectories, preserving all 201 frames
over physical time 0–2 and all 1,024 spatial points: 4,939,776 field values. The
frozen split contains 16 training, 4 validation, and 4 test trajectories. The local
subset SHA-256 is
`0cff51358bd57ace1fcd879a7de68fdff1207dfef14c62635cce2461c6939cf5`.
The publisher's full-file MD5 remains unverified, as the receipt states.

All FNO runs used ten epochs, with checkpoint selection using validation data.
The table reports physical-field RMSE on the same four held-out trajectories.
One-step prediction receives the true preceding frame; rollout starts from the
initial frame and feeds each prediction into the next step. Lower is better.

| Predictor | Selected epoch | One-step RMSE | 200-step rollout RMSE |
| --- | ---: | ---: | ---: |
| Persistence | — | 0.0309762 | 0.6878275 |
| FNO seed 0 | 10 | 0.0113663 | 0.3021139 |
| FNO seed 1 | 9 | 0.0110455 | 0.1872345 |
| FNO seed 2 | 10 | 0.0124395 | 0.2348113 |

These prediction improvements do **not** establish satisfactory conservation.
Rollout mean-velocity RMSE was 0.1734, 0.06567, and 0.09272 for seeds 0/1/2,
versus 0.00004540 for persistence. Seed 0's rollout energy RMSE was also worse
than persistence (0.2372 versus 0.2225), despite lower field RMSE. Mean preservation
and longer-rollout behavior are concrete targets for the next model improvement.
No seed has been discarded or promoted based on its test results.

The 200-epoch PINN fit the initial-value problem from public source row 4729,
using its initial frame and physics. Its subsequent-trajectory RMSE was 0.26106,
versus 0.82236 for persistence on that **single** trajectory. Its physical residual
RMS was 0.31026, and its mean-velocity error was worse than persistence. This is
not a head-to-head comparison with the FNO's four-trajectory average or a claim
of a converged physics solution. It demonstrates that the same evidence pipeline
can retain both improvements and weaknesses on external numerical data.

The final local suite passed **289 tests in 71.89 seconds**, Ruff passed, and the
0.4.0 source archive and wheel built successfully. GitHub Actions passed on
[Windows and Linux](https://github.com/Yogendra-sodha/flowstate/actions/runs/36513158601).
Cloud-bucket validation, distributed recovery, and broader independent scientific
studies remain open. This milestone completes the first real public-data study.

## 18. Recoverable local jobs and a conservation-aware model

The public-data measurements identified mean drift in FNO rollouts. A separate
operational gap was the lack of a durable job claim between submission and result
publication. Version 0.5 addresses both; the
[implementation walkthrough](docs/steps/07-recovery-conservation.md) explains the
commands, update equation, queue transactions, worker loop, and limitations.

For the model, the new optional `conserve_mean` flag subtracts the spatial mean
of the learned increment before adding it to the input. It preserves each
trajectory's own mean, not the mean of the training dataset. The training and
evaluation constructors record and restore the flag. Older checkpoints retain
their original unconstrained behavior. Tests apply nonzero learned updates over
200 steps on odd and even grids, check gradients, and verify exact legacy behavior
and training resumption. New diagnostics separate invariant drift from error
against a numerical reference that can itself drift.

The fresh-cohort study writes its plan before reading new values. It selects 24
different public source rows, checks initial-field hashes against the old cohort,
then fits both variants for each of three fixed seeds. Paired runs use the same
initial weights, minibatch order, hyperparameters, and budget. Six results are
retained, including any regression. Test metrics never select a preferred seed
or change the predeclared configuration. The earlier public cohort supplies only
exclusion information, not labels for this new study.

For the queue, `submit_sweep` expands and normalizes the input, stores immutable
requests with code/environment identities, and deduplicates identical jobs.
`work_queue` loops over at most `max_jobs`: claim in an immediate SQLite
transaction, renew ownership with a heartbeat, execute through the normal engine,
and acknowledge only if its claim token and lease are still valid. Expired claims
become available to a later worker. If output publication succeeded before a
worker stopped, recovery verifies and reuses the lake artifact. Otherwise it
restarts the solver. Queue events record submissions, claims, lease expiry,
completion, and failure.

The queue uses SQLite on one machine's local disk. It does not make the system a
distributed cluster, prevent duplicate computation after a lost lease, or add
mid-trajectory solver checkpoints. Its token check prevents stale acknowledgments;
the lake's existing publication rules protect finalized results. Changed code or
runtime refuses an old queued job rather than silently producing different work.

Review found another provenance trap: Python retains imported code while files
on disk can change. Comparing a submitted job only with the latest disk hash
could mislabel an old worker as new code. Queue workers now pin their imported
runtime identity and check it before and after execution. Regression tests cover
both matching-new-submission/old-worker and edits during a solve. Operators must
restart workers after changing source or Git checkouts.

Before the measured runs, the complete local suite passed **356 tests in 89.72
seconds** and Ruff passed. The fresh-data and two-process recovery measurements
are recorded below only after they actually run.

### Measured recovery and conservation results

Both measurements ran with clean source at commit `eee73e4`, package 0.5.0. The
[queue report](docs/reports/local-recovery-0.5.json) records eight Navier–Stokes runs
across two fresh worker processes, four jobs each, with all artifacts verified.
Repeating submission found eight existing jobs; another worker processed zero.
A child process deliberately called `os._exit(23)` after `run_experiment()` had
published a complete artifact but before queue acknowledgement. The next process
reclaimed its expired lease and returned `resumed=true` with the same result ID.
The event sequence is submitted, claimed, lease_expired, claimed, completed. The
claim count is two. This demonstrates local recovery at that publication boundary;
it does not claim mid-trajectory checkpoints, power-loss durability, or cloud scale.

The [conservation report](docs/reports/conservation-0.5.json) preserves the frozen
plan, receipt, hashes, cohort checks, and all six model results. The raw report and
its SHA-256 are identified; only per-time evaluation curves are omitted from the
compact committed copy. The new cohort has 24 complete trajectories, 201 times,
1,024 cells, and 16/4/4 train/validation/test families. Acquisition received
46,735,488 bytes. Both old/new source-row and initial-field-hash overlap checks
passed. Each seed pairs the same initialization and minibatch order across the
two model variants; validation chooses each run's checkpoint independently.

| Seed | Baseline rollout RMSE | Mean-preserving rollout RMSE | Baseline RMS mean drift | Mean-preserving RMS mean drift |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.224106 | 0.135179 | 0.186900 | 1.35e-7 |
| 1 | 0.235638 | 0.187749 | 0.154972 | 1.35e-7 |
| 2 | 5.300248 | 5.510174 | 5.297188 | 9.21e-8 |

Persistence rollout RMSE was 0.394423; the stored reference's RMS mean drift was
5.71e-5. The projection preserves mean, but seed 2 still has large finite errors
and its aggregate rollout RMSE worsens. Projected one-step errors worsen for seeds
0 and 1 as well. The engineering conclusion is specific: keeping one invariant
does not ensure accurate fields or reliable long predictions. We retain this
negative evidence; this completed test set was not used for more tuning.

The full synthetic release demo also passed 19 validation cases and six refinement
studies, generated and reused twelve trajectories, and executed two proposals.
An actual pre-flag 0.4 FNO checkpoint re-evaluated with identical previous metrics;
new drift diagnostics were added without changing its predictions. The 0.5 build,
356 local tests, and Windows/Linux CI passed before these measurements.

## 19. Make saved evidence readable in an offline viewer

The next local gap was inspection: the records existed, but a new user had to
read JSON or write plotting code to see a run. `dashboard.py` now turns a verified
lake into one browser file; `dashboard.html` holds its inline UI. The CLI command
is `flowstate --lake PATH dashboard OUTPUT.html --max-experiments 50`.

The extraction loop visits selected experiment directories in sorted ID order.
It calls `Lake.verify()`, reads `record.json`, and records both manifest and record
hashes. `_arrays()` selects an energy history of at most 201 saved samples with
both endpoints, plus only the final scalar field. Spatial stride is
`max(1, ceil(size / limit))`: 256 points for 1D or 64 along each 2D axis. The axis
loop slices coordinates with exactly those same strides. This keeps decoded plot
data small; integrity verification still scans every selected file. Limits bound
the export to 200 experiments and 32 MiB of JSON rather than pretending every lake
fits inside a browser page.

Transformation converts NumPy data into JSON-compatible lists, replacing
nonfinite values with nulls. `_safe_json()` escapes HTML/script delimiters and
Unicode line separators. The browser uses `textContent` for record strings and
creates SVG nodes for plots. `render()` filters the embedded array, `show()` builds
record details, `lineChart()` preserves gaps between nonfinite samples, and nested
row/column loops draw heatmap cells. Each heatmap has its own explicit value range.
No fetch, server database, or live connection is involved.

Loading means publishing a fresh HTML artifact. The exporter rejects destinations
inside the lake and existing files or symlinks. It writes and fsyncs a temporary
sibling, then publishes with `os.link()`, which cannot overwrite a concurrent
winner. Hard-link support is required. The CLI dispatches this command before
constructing `Lake`, so a mistyped source path does not create an empty lake.
The snapshot includes full run provenance and local paths; it should be inspected
before external sharing.

Scientific review corrected the Darcy preview: `t=0` is a storage marker for a
steady solution, not physical evolution time. Nonfinite heatmap ranges now say
unavailable rather than inventing zero. The
[viewer chapter](docs/steps/08-research-viewer.md) documents the commands, functions,
ETL path, sampling limits, and why a visualization is not proof of solver accuracy.
Independent tests exercise corruption rejection, exact sampled values, unsafe
output paths, and script escaping; browser checks exercise the actual filters,
failure records, and line/heatmap rendering. Final validation results follow.

Final local validation on 30 September passed **388 tests with four skips in
151.88 seconds**. The four skipped cases require Windows symbolic-link privileges
not available to this account; the Linux CI job exercises those cases. Ruff and
the source/wheel builds passed. Every packaged Python file and the HTML template
were checked against the current source. The
[viewer validation report](docs/reports/viewer-0.6.json) retains output hashes,
test evidence, browser checks, and their limits.

The exported release lake has sixteen experiments, including the intentional
failure and Darcy case; a second export contains eight Navier–Stokes runs. Browser
checks through localhost showed the Burgers plots, Darcy heatmap, corrected steady
caption, status filter, and no-match states without captured console errors. The
last small template edit narrows the no-energy message to mention Darcy only for
Darcy runs. Direct-file navigation could not be tested because the browser tool
blocks the file protocol; no workaround was attempted. The artifacts contain no
external dependencies, and their HTML template is included in the built wheel.

## 20. Add Google Cloud Storage with verified transfers

The chosen cloud project is `flowstate-510320`, with bucket `flowstate-codex`
and object prefix `flowstate/`. The service account is
`flowstate@flowstate-510320.iam.gserviceaccount.com`. Local authentication uses
Application Default Credentials with service-account impersonation; credentials
stay outside the repository and experiment artifacts. The worker has bucket-level
Storage Object Creator and Storage Object Viewer access. A live CLI check already
confirmed token creation, object creation, reading, and listing. This established
access, but the Python application still required its own GCS adapter.

`gcs_store.py` implements that adapter, and `cli.py` exposes `gcs upload` and
`gcs download`. The optional `gcs` dependency installs Google's storage SDK.
`artifact_mirror.py` now owns the shared publication and restoration workflow for
both GCS and S3, so the two providers follow the same evidence rules. The S3
commands and object layout retain their existing behavior.

The extraction step verifies the experiment and reads its manifest. The
transformation maps each relative file path to an object key under the experiment
and manifest hash; the numerical arrays and metadata retain their bytes. In the
load loop, a private copy of each artifact is hashed before upload. This closes a
race found during review: editing an original file after preflight verification
must not poison an immutable cloud object. The loop uses only one artifact copy
at a time, and raises a clear error if its hash differs from the manifest.

GCS writes use `if_generation_match=0` so an existing object cannot be overwritten.
If it already exists, the adapter reads and hashes its bytes before reporting
reuse. Each upload requests a CRC32C transport checksum, and the completion
manifest is uploaded last. If a transfer stops partway, the already uploaded
objects can be reused by the next attempt. A download pins each object's
generation, reads 1 MiB blocks, hashes them while writing to a temporary local
directory, and only publishes the experiment after all checks pass.

The [GCS chapter](docs/steps/09-gcs-storage.md) follows the functions, loops,
extraction/transformation/loading path, directory structure, authentication, and
terminal commands. Offline checks cover corruption, missing chunks, interruptions,
conflicts, local source edits during upload, unsafe paths, and CLI failures.

On 4 October 2026 the application-level GCS check passed from clean commit
`9e6fee0`. It ran the standard 64-point Burgers example for 200 steps and saved
21 frames. The experiment is `186ee23716212de6260e3be2838b668a`. Its 38 artifacts
and one manifest total 36,507 bytes. Upload took 15.027 seconds; a repeated upload
took 22.262 seconds, read/verified the existing content, reused all 38 artifacts,
and created none. Restoring into a fresh lake took 12.535 seconds. Every one of
the 39 local files matched byte for byte, the record retained its original
provenance, SQL found the restored experiment, and an offline viewer was exported
from it. These single small-transfer timings include CLI startup and credential
work; they are not a cloud throughput benchmark.

The [measured report](docs/reports/gcs-20261004.json) retains commands, per-stage
timings, source identity, object counts, and validation scope. It contains no
credentials. The new GCS tests include 39 offline cases; the full suite passed
427 tests with four Windows symbolic-link permission skips in 267.18 seconds.
Ruff, source/wheel builds, and comparison of packaged source files passed. Live
verification uses the native Python SDK with local ADC impersonation, rather
than shelling out to `gcloud` for transfers. Cloud compute, distributed scheduling,
model/dataset mirroring, and large-transfer measurements remain separate work.

## 21. Freeze and measure a larger local workload

The revised roadmap makes measured data infrastructure the next gate. The
existing isolated benchmark already checked array equality and verified reuse,
but its deliberately small limits could not accommodate the requested scale.
The [new chapter](docs/steps/10-scale-measurement.md) follows the configuration
loops, process boundaries, artifact extraction, aggregation, and report loading.

`scaling_study.py` freezes the workload before execution and shuffles all grid,
worker-count, and repetition combinations together. It requires a clean source
commit, checks disk headroom, records the lockfile hash, and passes explicit
numerical-library thread settings to isolated trials. The older configurable
benchmark retains its original safety bounds; larger runs are an explicit mode.

The trial loop distinguishes completed experiments from numerical failures and
reports successful throughput separately from attempted throughput. Both types
of finalized record must pass manifest verification and verified reuse. Scientific
array hashes and review flags must match across worker counts for each grid.
An infrastructure error keeps partial evidence and prevents a success summary.

The aggregation loop groups equivalent workloads, derives medians and ranges,
and passes the retained measurements to a static chart renderer. The renderer
does not run simulations or invent missing memory values. Short trajectories,
warm-cache reuse, sampled rather than exact peak RSS, and uncontrolled background
load are explicit limitations. Results and validation are recorded after running
the committed implementation; no benchmark success is assumed here.

The completed [larger study](docs/reports/scaling-large.json), run from clean
commit `7ce7c07`, now provides that evidence. It tested 216 distinct configurations
across the three grids, with 36 initial-condition families. Every worker/grid
setting received three repeats of 72 configurations, yielding 2,592 fresh
executions and 2,592 verified reuses. All completed, with zero numerical failures
and matching scientific-array hashes within each grid. A later SQL query found
all 72 completed records in a largest-grid trial; query results and checks are
retained in the [validation receipt](docs/reports/scaling-large-validation.json).

The complete study took 154.23 minutes and retained 4.83 GiB of experiment
artifacts outside Git. The report and [chart](docs/reports/scaling-large.png) are
exact retained copies. Eight workers achieved 3.03–3.62 times the single-worker
throughput across the measured grids. The increase from four to eight workers
was smaller: 13.0–21.9% more throughput, with 72.3–81.9% more sampled peak RSS.
Small-grid verified reuse was slightly slower with eight workers, and overlapping
timing ranges prevent a significance claim. These limits remain visible in the
[results chapter](docs/steps/10-scale-measurement.md#executed-results).

The code added failure-aware counters, larger explicit bounds, thread-setting
records, retained child diagnostics, the fixed study command, and chart tests.
Ruff and the full suite passed: 451 tests passed with four Windows symbolic-link
permission skips. The measured source commit also passed Windows and Linux CI.
An independent report audit found matching per-trial evidence, recomputed totals
and summaries, and a matching committed dependency lockfile. Memory sampling
recorded 117 process-disappearance errors, so the sampled RSS figures remain
estimates. Source fingerprints reflect checkout bytes; line-ending changes can
change identities even when program logic stays the same.

The phase used local computation and no cloud credentials. The earlier unfinished
cloud-worker draft remains preserved in Git stash
`84b667b6307a8971aeb3ed8d75e10aa90c3be664`; it was excluded from the measured source.
The milestone is complete, and work stops here for review before adding
mid-trajectory crash recovery.

## 22. Exact solver checkpoints and recovery

The next change extends recovery inside an unfinished simulation. Previously the
queue could reclaim a job and reuse a finalized result, but an unfinished solver
started at its initial condition again. The optional `--checkpoint-every` argument
now flows from CLI parsing through `run_experiment`, or through queue request JSON
and `work_queue`, into the numerical step loop. It remains an execution option,
outside the scientific configuration and final experiment identity.

The implementation begins at the numerical state, because saved physical fields
alone do not fully represent the spectral integrator. Burgers copies its velocity
vector; Navier–Stokes copies complex spectral vorticity after each completed RK4
step's filtering. Both snapshots include the completed step, saved times, and
diagnostic prefix. On restoration, the loop begins at the following step without
regenerating the initial condition or repeating earlier RK4 work.

The data path has separate extraction, transformation, and loading operations.
`solve` extracts fields and checkpoint snapshots through callbacks.
`CheckpointStore.append_frame` and `save` transform those arrays into deterministic
NPZ blobs and a JSON reference graph. SHA-256 names deduplicate equal blobs.
Private file writes are flushed before an exclusive hard link publishes a blob;
the commit marker is published after its dependencies. Competing workers can
share identical committed evidence without mutating each other's partial stores.

Recovery loads the latest marker, checks provenance and every referenced hash,
validates array shapes/types and the saved-time schedule, and replays earlier
physical frames into a new streamed Zarr sink. The resumed solver emits later
frames into that sink. Final publication follows the existing JSON/Parquet/Zarr
and checksum-manifest path, so the recovered result uses the same query and
verification interfaces. Checkpoint I/O errors propagate as infrastructure
failures; only numerical errors produce numerical failure records.

Tests cover exact restored arrays, off-schedule and final-step snapshots,
corruption, competing checkpoint writers, pending files, queue lease expiry,
and invalid requests before filesystem creation. The separate recovery-study
module launches real workers, counts their RK4 calls, kills an owned worker after
commit, and compares all scientific-array hashes after recovery. The
[checkpoint chapter](docs/steps/11-checkpoint-recovery.md) explains the command,
storage layout, functions, loops, and limitations. Measured evidence will be
recorded after committing and running this implementation.

## 23. Preserve work when changing computers

The user is moving from Windows to a MacBook Pro. `HANDOFF.md` records the actual
code state, unfinished milestone gates, Mac environment commands, and a prompt
for a new Codex chat. Source and the dependency lockfile transfer through Git;
the virtual environment is recreated on the Mac. Raw experiment fields, model
outputs, credentials, and caches stay outside Git and require separate transfer
if the user wants them. A changed hardware/platform identity means new experiment
identities; old provenance is preserved rather than rewritten to claim reuse.

An older inactive cloud-worker draft existed only in a Git stash. A full binary
patch including its untracked files is now retained under `docs/handoff/`, with
its original base recorded in the handoff. This preserves the draft across a
normal clone without applying it to the current engine or activating remote
workers. The final recovery study remains an explicit next action on committed
source, and the migration validation receipt records what was checked on Windows.

## 24. Execute and audit the committed recovery proof

Windows work continued after the handoff. The recovery protocol was executed from
clean commit `d8ef527`, using a fresh ignored output directory. Each equation's
loop launches an uninterrupted reference, a worker paused after checkpoint
publication, and a new worker after process termination. Instrumentation counts
actual completed RK4 calls in each interpreter, separately from the engine's
recovery metadata. The measured counts were 37 reference steps, 13 before the
kill, and 24 after restoration. Thus the proof checks skipped work as well as
correct final data.

The report-retention step copies the raw aggregate JSON byte for byte into
`docs/reports/checkpoint-recovery.json`. Its validation receipt records that
content hash, source-file equality with the full-suite validation, CI, and
post-study DuckDB queries. An independent audit separately reconciled raw receipts,
loaded the exact state and prefix, re-hashed all 17 scientific arrays, verified
final manifests, and checked the controlled worker/launcher termination receipts.
Both cases completed with exact arrays and no failures; the complete protocol
took 20.03 seconds. The [results chapter](docs/steps/11-checkpoint-recovery.md)
retains measured timings, including slower Burgers recovery, and limitations.

The full local suite passed 540 tests with four Windows symbolic-link skips.
The recorded package hashes match the measured source, Ruff passed after the
study, and Windows/Linux CI passed the measured commit. No source edits were
needed between the successful implementation tests and the measured proof.
Only evidence and phase documentation were added. `HANDOFF.md` now marks this
milestone complete and directs the Mac chat to environment validation followed
by the approved cloud-proof milestone. Work stops at this phase boundary.

## 25. Prepare a bounded cloud recovery proof and viewer package

The next phase makes cloud approval concrete. `cloud_proof.prepare_proof` runs a
small fixed local experiment from committed source and generates a plan, full
file inventory, and offline viewer. It creates no cloud client. Approval later
names the plan's SHA-256 and exact generated removal target. Configuration,
runtime and source checks prevent an old plan from silently authorizing different
work; documentation-only commits may advance while the package remains identical.

`execute_proof` checks both approval flags before any client can be constructed.
It uses the existing GCS mirror for upload. A second upload attempts create-only
writes; the adapter verifies existing remote bytes before reporting reuse. Only
after the full mirror is verified does removal begin. The loop removes the exact
inventoried files and then empty directories, deepest first, inside the owned
generated experiment. Reparse points, links, changed inventory and unexpected
output files prevent this path. Partial removal is recorded in the final receipt.

The download loads files into a fresh lake through the existing staging and
manifest-verification protocol. The proof compares every restored file to its
original hash, queries the completed Parquet record with DuckDB, and exports a
new viewer. Plan, removal records and failure evidence remain outside the removed
run. There is no remote delete operation and no retry that hides failed evidence.

The viewer container serves one exported HTML document loaded at startup. Its
HTTP routes never map request paths onto the filesystem; it has no dataset,
simulation or credential access. Tests exercise actual HTTP behavior and compare
an exported snapshot with the response. Linux CI builds and starts the rootless
container against an actual export, retaining base/image/hash details in logs.
The [phase guide](docs/steps/12-cloud-proof-viewer.md) documents the data path,
failure recovery, container commands, and a private Cloud Run proposal. A real
cloud receipt and deployment are not implied by these local implementations.

Preparation was then executed from clean commit `ec565a1`. Its retained
[plan](docs/reports/gcs-proof-plan-20261009.json) describes the exact file hashes,
source identity, GCS prefix, and local removal target. The
[validation receipt](docs/reports/cloud-proof-preparation-validation-20261009.json)
keeps local test results and the initial Docker Hub timeout visible. The plan is
pending authorization; creating it does not authorize credentials or removal.

Review also found a Unix permission mismatch: the exporter deliberately creates
private HTML files, while the container runs as a different user. The container
check now copies reviewed bytes into a private temporary directory, makes only
that copy readable to the container, and removes that generated copy afterward.
The original export's permissions and lake remain unchanged. A baked snapshot
explicitly assigns ownership to the container user. Linux CI exercises the real
image and retries a failed public base-image pull within a fixed attempt bound.
Docker Hub still throttled the CI runner, so the workflow now pulls Docker's
official public mirror and records its resolved digest. The
[retained container validation](docs/reports/viewer-container-validation-20261009.json)
preserves those failures alongside the successful build and HTTP check.
On `d50378d`, Windows and Linux each passed all 608 tests and Ruff; Linux served
the 20,530-byte exported snapshot exactly from the container. This closes local
package validation while the approved live GCS recovery receipt remains pending.

## 26. Complete the approved GCS recovery proof

The user approved the exact prepared test and a total US$50 GCP budget. The
existing Application Default Credentials refreshed successfully; no token or
credential file entered the lake or Git. `execute_proof` ran from clean `bb8daf2`,
while the original experiment kept its `ec565a1` generation provenance.

The upload loop sent 22 artifacts, then published their completion manifest.
Repeating the upload hit the immutable-create checks and read back existing
objects for hashing. Once every remote artifact matched, the removal loop
deleted only the inventoried generated source files and their empty directories.
The before/after records stayed outside that subtree. The download loop wrote a
fresh local lake and published it only after artifact verification.

All 23 restored files, totaling 28,422 bytes, matched the prepared inventory.
DuckDB loaded the restored Parquet metadata and returned the completed Burgers
record. The viewer exporter read the restored fields and produced a new HTML
snapshot. The complete execution took 27.67 seconds. These are observed results
from the [retained raw receipt](docs/reports/gcs-recovery-20261009.json).

An independent audit recomputed local hashes and reconciled provenance, manifest,
query, and removal evidence. The [validation receipt](docs/reports/gcs-recovery-validation-20261009.json)
retains that audit and the equality of executed code with the Windows/Linux CI
source. The [budget ledger](docs/reports/gcp-budget-ledger.json) records the user
authorization and observed storage workload; actual billed charges are unknown.
This completes the cloud recovery, viewer packaging and deployment-proposal gate.
The Mac continuation should validate its environment and begin the larger model
study; it should not repeat this completed proof solely because the machine changed.

## 27. Validate the Mac continuation

The clean develop checkout fast-forwarded to `1d6c3d6`. uv was absent and was
installed from its official installer. The locked all-extras/dev sync created
a native Apple Silicon Python 3.12.15 environment without changing uv.lock.
Ruff passed; all 608 tests passed with no skips. The retained receipt is
[Mac validation](docs/reports/mac-validation-20261010.json). This validates
local execution, not Mac Docker or cross-hardware bitwise equality.

## 28. Freeze a larger trustworthy-model study

The existing dataset exporter already groups initial-condition families and
checks duplicate initial fields before splitting; its online statistics read
training trajectories only. FNO training already selects weights on validation
MSE and saves resumable optimizer/RNG state. Evaluation already separates true
preceding-frame prediction from feedback rollout and compares persistence.
Those paths are reused. An optional `evaluate_test=False` now lets a study defer
test inference without changing existing callers or checkpoint resumption.

`model_study.py` freezes 180 fresh random initial conditions, five training seeds,
and a fixed mean-preserving FNO budget. It verifies at least 100 independent
training families. Twelve fixed finer-grid/time runs audit reference sensitivity.
The training loop completes every model and validation evaluation, then writes
and hashes ensemble calibration before the test loop. Family-level paired
bootstrap summaries retain losses and ties. Physical reductions measure mean,
mass and energy; uncertainty measures empirical held-out coverage and width.
Validation also selects checkpoints, so calibration has no formal coverage
guarantee. The [protocol](docs/steps/13-trustworthy-model.md) explains the loops,
data transformations, scope and limits. Measurement follows committed source;
no successful study result is asserted by this implementation entry.

Implementation validation passed Ruff and all 621 tests with no skips.
Evidence: `docs/reports/model-study-code-validation-20261010.json` binds
tested source hashes to the subsequent committed measurement.

## 29. Measure and audit the frozen five-seed study

The fixed protocol ran from clean `4b18252` into a new ignored directory,
`outputs/model-study-20261010-01/`, taking 159.07 seconds. It verified 180 original
and twelve finer-reference experiments. Export produced 108/36/36 independent
train/validation/test families. Five model bundles and ten evaluation bundles
retain weights and predictions locally. The retained report is an exact copy:
[model study](docs/reports/model-study-20261010.json).

The extraction audit re-hashed 248 recorded evidence files and all numerical,
dataset, model and prediction manifests. It checked tested-source equality,
family disjointness, unique initial hashes and event ordering. A separate NumPy
calculation loaded saved predictions, recalculated validation multipliers, test
errors, seed spread, family losses, mass drift, coverage and all twelve reference
comparisons. The [audit receipt](docs/reports/model-study-audit-20261010.json)
records these checks; it does not claim an independent retraining run.

All seeds beat persistence for every test family's field RMSE. One-step seed
mean/std were 5.79654e-5/1.44430e-6 versus persistence 0.00360433; rollout
mean/std were 0.000469861/0.000017397 versus persistence 0.0857394. This measures
one smooth generator and reproduction of a coarse solver. Finer-reference RMSE
of 0.00029146–0.00067897 is comparable to rollout model error.

Negative results remain explicit: one-step simultaneous interval coverage was
30/36 (below nominal 90%); rollout covered 33/36. All models lose to persistence
on rounding-scale invariant drift, even with mean projection. The
[negative-case report](docs/reports/model-study-negative-cases-20261010.json)
lists family IDs and values; no family field-error losses were invented. No test
threshold or hyperparameter was repaired. The
[results chapter](docs/steps/13-trustworthy-model.md) explains interpretation.
Milestone 4's local evidence is complete; pause for review before the router.

GitHub publication confirmed on 10 October 2026 after Mac authentication
setup: commits `9f96a94`, `4b18252` and `1e6e22d` are on origin/develop.
Source and measurements are unchanged. Paused for Milestone 4 review.
