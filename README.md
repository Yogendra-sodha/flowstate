# Flowstate

**A reproducible experiment engine for computational PDE research.**

Run numerical experiments, retain fields and diagnostics, query the results, and
trace how experiments relate. Flowstate is infrastructure for building computational
evidence. It does not claim to solve the 3D Navier–Stokes existence and smoothness
problem.

## Working research prototype

- **Numerics:** periodic viscous Burgers in 1D using conservative finite differences;
  unforced incompressible Navier–Stokes in 2D using a dealiased vorticity spectral
  method. Both use RK4 and check timestep stability at every stage. A steady 2D
  Darcy adapter uses harmonic face coefficients and a sparse direct solve.
- **Evidence:** velocity/vorticity fields, physical time coordinates, energy,
  mass/circulation, enstrophy, divergence, and sampled numerical review flags.
- **Experiment lake:** chunked Zarr arrays, per-run Parquet metadata, DuckDB queries,
  JSON provenance, and SHA-256 artifact verification.
- **Execution:** parameter grids, process-based parallel workers, verified reuse of
  finalized runs, immutable failure records, explicit retries, and parent lineage.
- **Data engineering:** streamed saved fields, trajectory-family dataset splits,
  training-only normalization, provenance-aware PDEBench Burgers HDF5 import,
  and checksum-verified S3 upload/download with a manifest published last.
- **Learning:** a small CPU Burgers FNO, one-step/rollout evaluation against persistence,
  resumable immutable checkpoints, and a per-instance physics-informed Burgers baseline.
- **Research:** typed evidence objects, hypotheses, findings, and deterministic
  budgeted refinement proposals that can execute through the same engine.
- **Validation:** known-solution comparisons, convergence tests, conservation and
  dissipation checks, and storage/orchestration integration tests.

Read [the implementation stepbook](STEPBOOK.md) for the design decisions, functions,
loops, setup, extraction/transformation/loading logic, and measured validation.
The [roadmap](docs/roadmap.md) distinguishes the implemented local prototype from
production distributed execution, a dashboard, and an LLM research planner.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Commands work from this repository on Windows, macOS, and Linux; CI tests Windows
and Linux.

```sh
uv sync --locked --all-extras --group dev
uv run --no-sync flowstate run examples/burgers.json --stream
uv run --no-sync flowstate run examples/navier_stokes.json --stream
uv run --no-sync flowstate run examples/darcy.json
uv run --no-sync flowstate sweep examples/sweep.json --workers 2 --stream
```

The example sweep runs eight combinations of viscosity, spatial resolution, and
seed, all at the same final physical time. Repeat it to reuse verified results.
The JSON output reports each experiment ID, status, metrics, and whether it was
resumed. Default artifacts live in `data/lake/`, excluded from Git.

The `ml` and `s3` extras are optional for base numerical work. `--all-extras` prepares
the complete prototype. `--no-sync` preserves that prepared environment while
running commands; using a sync without an optional extra may remove it.

## Complete a small research study

```sh
uv run --no-sync flowstate demo outputs/study-01 --epochs 20 --pinn-epochs 200
```

Use a new output directory for each demonstration. It runs numerical validation,
generates twelve streamed Burgers trajectories, checks resumption, exports a frozen
dataset, trains/evaluates FNO and PINN baselines, registers their evidence, and
executes two bounded refinement proposals. Inspect `report.json`, `events.jsonl`,
`research_graph.json`, and the model/dataset manifests inside that directory.

To run individual stages:

```sh
uv run --no-sync flowstate --lake outputs/training-lake sweep examples/training_sweep.json --stream
uv run --no-sync flowstate --lake outputs/training-lake dataset export outputs/burgers-dataset
uv run --no-sync flowstate train-fno outputs/burgers-dataset outputs/fno --epochs 20
uv run --no-sync flowstate evaluate-fno outputs/burgers-dataset outputs/fno --output outputs/evaluation
uv run --no-sync flowstate train-pinn outputs/burgers-dataset outputs/pinn --epochs 200
uv run --no-sync flowstate --lake outputs/training-lake model register outputs/fno
uv run --no-sync flowstate --lake outputs/training-lake research --max-runs 2 --max-total-steps 500
uv run --no-sync flowstate --lake outputs/training-lake research --execute --max-runs 2 --max-total-steps 500
uv run --no-sync flowstate --lake outputs/training-lake graph --ontology
```

Train/validation/test splits group related initial conditions before extracting
time windows. FNO checkpoints are selected using validation data; evaluation reports
both one-step and autoregressive errors at physical times. The PINN fits one
held-out initial-value problem using its initial frame plus PDE and periodic
boundary losses. These are different learning settings, not an equal-budget claim.
Numerical reference data are not exact truth, and beating persistence is measured,
not assumed. Resume training into a **new** directory with `--resume outputs/fno`
and matching original hyperparameters; `--epochs` then means additional epochs.

## Import data and mirror artifacts

```sh
uv run --no-sync flowstate dataset import-pdebench SOURCE.hdf5 outputs/imported --source-url SOURCE_URL --source-version SOURCE_VERSION --license LICENSE_NAME --viscosity EFFECTIVE_NU
uv run --no-sync flowstate dataset verify outputs/imported
uv run --no-sync flowstate s3 upload EXPERIMENT_ID BUCKET --prefix flowstate
uv run --no-sync flowstate --lake outputs/restored s3 download EXPERIMENT_ID BUCKET --prefix flowstate
```

The importer supports the documented periodic 1D Burgers HDF5 layout, with explicit
source/version/license and effective viscosity. PDEBench's Burgers generator uses
`epsilon / pi` in its diffusion term: do not blindly copy the filename's `Nu` value
into `--viscosity`. No public dataset is downloaded automatically. The S3 adapter
uses the normal AWS credential chain and optional `--endpoint-url`; it does not
provision buckets or IAM. Protocol tests use an emulator, not a deployed cloud account.

```sh
uv run --no-sync flowstate list
uv run --no-sync flowstate query "SELECT id, viscosity, grid_size, final_energy FROM experiments WHERE status = 'completed' ORDER BY final_energy"
uv run --no-sync flowstate graph
```

Use an ID returned by `run` or `list` in these commands:

```sh
uv run --no-sync flowstate show EXPERIMENT_ID
uv run --no-sync flowstate verify EXPERIMENT_ID
uv run --no-sync flowstate run examples/burgers.json --parent EXPERIMENT_ID --attempt 1
```

Choose a different lake with the global argument **before** the command:

```sh
uv run --no-sync flowstate --lake outputs/my-study sweep examples/sweep.json --workers 2
```

## Configure an experiment

Configuration files are JSON objects; missing values take the documented defaults.
Unknown keys and invalid values fail before integration.

| Key | Default | Meaning |
| --- | --- | --- |
| `equation` | `burgers1d` | `burgers1d` or `navier_stokes2d` |
| `viscosity` | `0.05` | Nonnegative kinematic viscosity |
| `grid_size` | `32` | Points per periodic spatial dimension, at least 8 |
| `domain_length` | `2π` | Positive domain length per dimension |
| `dt` | `0.001` | Fixed physical timestep |
| `steps` | `100` | Number of integration steps |
| `save_every` | `10` | Saved-frame interval; initial and final frames always included |
| `seed` | `0` | Nonnegative random initial-condition seed |
| `amplitude` | `1` | Nonnegative initial speed scale |
| `initial_condition` | equation-specific | `sine` for Burgers; `taylor_green` for NS; `random` for either |

Darcy has a separate steady-state schema: `equation: darcy2d`, `grid_size` (default
33 boundary-inclusive nodes), `domain_length` (default 1), and `contrast` (default
0.5, absolute value below 0.9). It uses a manufactured pressure and corresponding
forcing to validate the solver; this family is not an operator-learning dataset.

Reynolds number uses the explicit convention `amplitude × domain_length / viscosity`.
It is stored as null for zero viscosity. Smooth inviscid tests are supported, but the
Burgers scheme does not capture shocks. Navier–Stokes arrays contain `u`, `v`, and
`vorticity`; pressure is not computed.

Saved output is capped at an estimated 256 MiB per time-dependent simulation.
Buffered execution retains fields; `--stream` writes saved fields incrementally
and retains only working fields plus scalar diagnostic histories. RK/FFT buffers
and temporary disk copies still consume resources. Learning is separately bounded
to small datasets and explicit training budgets.
Use modest grids and worker counts. This is a validated local foundation, not yet
a distributed engine for thousands of 512×512 trajectories.

## Reproduction, failures, and interpretation

Every run records its complete config, code commit, package source fingerprint,
dirty state, dependency versions, Python, machine description, precision, runtime,
and parent. Commit code before running a study and retain `uv.lock`: the fingerprint
does not archive uncommitted source. Machine-to-machine bitwise equality is not
promised.

An identical run under the same recorded code and environment reuses its immutable
result after checksum verification. This includes recorded failures. Use a new
`--attempt` or change the configuration for another attempt; use `--parent` to link
it to earlier evidence. A process interrupted before publication restarts from time
zero. Mid-trajectory checkpoints are not implemented. Numerical failures retain an
error record, but no partial field trajectory or structured last-valid time.

`completed` means integration finished. Check `metrics.needs_review` and the recorded
tolerances before interpreting the result. Unforced viscous energy should dissipate;
an observed increase flags a numerical question, not physical blow-up. Diagnostics
are sampled at saved frames and can miss intervening events. PDE residuals and
dissipation-balance residuals are future work.

`verify` detects missing, modified, or unexpected artifacts. Application writes are
append-only; the manifest is unsigned and does not prevent external modification.
SQL permits one read-only SELECT over `experiments` with external access disabled.
Exit codes are `0` for successful commands, `1` for recorded numerical failures or
failed verification, and `2` for configuration, I/O, or query errors.

## Development

```sh
uv run --no-sync ruff check .
uv run --no-sync pytest
uv run --no-sync flowstate validate outputs/numerical-validation.json
```

Read [architecture](docs/architecture.md), [scientific validation](docs/scientific-validation.md),
the [stepbook](STEPBOOK.md), and the [roadmap](docs/roadmap.md). Changes for this project
are developed on `develop`.

Scientific references: [PDEBench](https://arxiv.org/abs/2210.07182),
[Fourier Neural Operator](https://arxiv.org/abs/2010.08895), and the
[official Navier–Stokes problem statement](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf).
These inform the direction; Flowstate does not claim to reproduce their full
benchmarks or introduce a new neural operator.

MIT licensed. See [LICENSE](LICENSE).
