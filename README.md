# Flowstate

**A reproducible experiment engine for computational PDE research.**

Run numerical experiments, retain fields and diagnostics, query the results, and
trace how experiments relate. Flowstate is infrastructure for building computational
evidence. It does not claim to solve the 3D Navier–Stokes existence and smoothness
problem.

## Working first milestone

- **Numerics:** periodic viscous Burgers in 1D using conservative finite differences;
  unforced incompressible Navier–Stokes in 2D using a dealiased vorticity spectral
  method. Both use RK4 and check timestep stability at every stage.
- **Evidence:** velocity/vorticity fields, physical time coordinates, energy,
  mass/circulation, enstrophy, divergence, and sampled numerical review flags.
- **Experiment lake:** chunked Zarr arrays, per-run Parquet metadata, DuckDB queries,
  JSON provenance, and SHA-256 artifact verification.
- **Execution:** parameter grids, process-based parallel workers, verified reuse of
  finalized runs, immutable failure records, explicit retries, and parent lineage.
- **Validation:** known-solution comparisons, convergence tests, conservation and
  dissipation checks, and storage/orchestration integration tests.

Darcy, FNO/PINN training, object storage, a full research ontology, autonomous
hypothesis generation, and a dashboard are planned extensions. See the
[15-day roadmap](docs/roadmap.md).

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Commands work from this repository on Windows, macOS, and Linux; CI tests Windows
and Linux.

```sh
uv sync --locked --group dev
uv run flowstate run examples/burgers.json
uv run flowstate run examples/navier_stokes.json
uv run flowstate sweep examples/sweep.json --workers 2
```

The example sweep runs eight combinations of viscosity, spatial resolution, and
seed, all at the same final physical time. Repeat it to reuse verified results.
The JSON output reports each experiment ID, status, metrics, and whether it was
resumed. Default artifacts live in `data/lake/`, excluded from Git.

```sh
uv run flowstate list
uv run flowstate query "SELECT id, viscosity, grid_size, final_energy FROM experiments WHERE status = 'completed' ORDER BY final_energy"
uv run flowstate graph
```

Use an ID returned by `run` or `list` in these commands:

```sh
uv run flowstate show EXPERIMENT_ID
uv run flowstate verify EXPERIMENT_ID
uv run flowstate run examples/burgers.json --parent EXPERIMENT_ID --attempt 1
```

Choose a different lake with the global argument **before** the command:

```sh
uv run flowstate --lake outputs/my-study sweep examples/sweep.json --workers 2
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

Reynolds number uses the explicit convention `amplitude × domain_length / viscosity`.
It is stored as null for zero viscosity. Smooth inviscid tests are supported, but the
Burgers scheme does not capture shocks. Navier–Stokes arrays contain `u`, `v`, and
`vorticity`; pressure is not computed.

Saved output is capped at an estimated 256 MiB per simulation. The current solver
retains saved frames in memory before writing, so actual peak memory is higher.
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
uv run ruff check .
uv run pytest
```

Read [architecture](docs/architecture.md), [scientific validation](docs/scientific-validation.md),
and the [roadmap](docs/roadmap.md). Changes for this project are developed on `develop`.

Scientific references: [PDEBench](https://arxiv.org/abs/2210.07182),
[Fourier Neural Operator](https://arxiv.org/abs/2010.08895), and the
[official Navier–Stokes problem statement](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf).
These inform the direction; Flowstate does not claim to reproduce their full
benchmarks or introduce a new neural operator.

MIT licensed. See [LICENSE](LICENSE).
