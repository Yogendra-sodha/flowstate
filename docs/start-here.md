# Start here: understand Flowstate by following one experiment

Flowstate is a local research engine that runs PDE experiments, preserves their evidence, and uses that evidence to organize further work. Your first goal is to explain one complete experiment. You do not need to understand every numerical method or neural network before doing that.

This guide is a reading and practice route, not a promise of expertise within a fixed number of days. Familiarity means you can find, run, and explain the code. Expertise means you can challenge its assumptions, design useful experiments, and recognize when the results are wrong.

## The idea in one picture

~~~mermaid
flowchart LR
    C[JSON configuration] --> E[Engine]
    E --> S[Numerical solver]
    S --> L[Experiment lake]
    L --> Q[SQL and inspection]
    L --> D[Dataset and learning]
    L --> R[Research graph]
    R --> P[Next proposal]
~~~

The numerical solver generates reference trajectories. The lake preserves fields, metrics, and provenance. Dataset curation selects compatible trajectories and prevents training/test leakage. Models are evaluated against stored references. The research layer records relationships and proposes bounded follow-ups.

Here is one literal example. `examples/burgers.json` asks: *If this simulated
velocity starts as a sine wave, how does it change when viscosity is 0.05?* The
solver represents the wave at 64 spatial points. It advances time by 0.002 for 200
steps, reaching physical time 0.4. It saves the initial state and every tenth step:
21 frames, each containing 64 velocity values. Think of these frames as a very
small movie of a calculated flow.

The frames go to `fields.zarr`; calculated summaries such as final energy and mass
drift go into the run record and its queryable Parquet row. The record also says
which config, code version, and machine produced them. A checksum list makes later
accidental damage visible. DuckDB reads the Parquet rows so you can ask which runs
completed or had an unusual metric. To train a model, Flowstate combines several
compatible verified movies, separates related starting conditions across train,
validation, and test, then saves a curated dataset. The model tries to predict a
future frame; its error is compared with the saved numerical frame. Finally, the
research graph connects the run, data, model, observation, and a proposed follow-up.

That first path generates data by running the included equations. A second path can
import a compatible Burgers HDF5 file using the PDEBench adapter. Uploading an
experiment to S3 is an optional copy of verified artifacts; no live cloud bucket
is part of the local run above.

Flowstate does not solve the 3D Navier–Stokes existence and smoothness problem. Numerical evidence can reveal behavior or defects without establishing a mathematical theorem.

## What exists, and what remains

| Available in this repository | Work still requiring its own implementation or evidence |
| --- | --- |
| Local Burgers, 2D Navier–Stokes, and a manufactured Darcy solver | Broader PDE families and stronger application-specific references |
| Immutable local lake, SQL queries, checksums, streamed output | Distributed work queues, remote workers, and mid-simulation recovery |
| Dataset export and a PDEBench Burgers import adapter | Declared public-dataset studies and larger benchmark comparisons |
| CPU FNO and PINN baselines with reproducible artifacts | Broader generalization studies, tuning, and uncertainty evaluation |
| S3-compatible mirroring with emulator tests | Validation against an actual configured cloud account and workload |
| Local worker/storage benchmark tooling | Representative scale measurements and operational scheduling decisions |
| Typed research evidence and a deterministic proposal policy | A visual dashboard and constrained LLM research planning |

For planning, estimate about **90% of the local prototype** and roughly **40% of the
larger vision**. These are judgment calls, not measured percentages. The local
experiment-to-evidence loop works; each larger deployment or scientific claim needs
a separate acceptance criterion. See the [roadmap](roadmap.md) and
[implementation stepbook](../STEPBOOK.md).

## Follow one run through the code

Start with [examples/burgers.json](../examples/burgers.json). Its parameters describe a periodic one-dimensional velocity field evolving over time.

1. **Configuration:** viscosity, grid size, timestep, number of integration steps, and output frequency specify the numerical problem.
2. **CLI:** [cli.py](../src/flowstate/cli.py), main(), parses the command and reads JSON.
3. **Engine:** [engine.py](../src/flowstate/engine.py), run_experiment(), validates configuration, captures provenance, creates an identity, and checks for an existing result.
4. **Numerics:** [numerics.py](../src/flowstate/numerics.py), solve(), selects _burgers(); _rk4() advances the state through four derivative evaluations per step.
5. **Summary:** engine.summarize() turns saved diagnostic histories into queryable metrics and review flags.
6. **Lake:** [lake.py](../src/flowstate/lake.py), Lake.write(), stages arrays and metadata, hashes artifacts, then atomically publishes the experiment directory.
7. **Query:** Lake.query() reconstructs an experiments table from Parquet records and evaluates a read-only SQL SELECT.

If the same request already exists, the engine verifies its artifacts and returns it with resumed=true. It can also reuse a recorded failure. This is reuse of a finalized result, not recovery halfway through a simulation.

A changed configuration, relevant code/environment identity, parent, or attempt can produce a different experiment ID. Timestamps are not the scientific identity. Read capture_provenance() and experiment_id() together to see the exact boundary.

The settings describe different kinds of change:

| Kind | Examples | What changes? |
| --- | --- | --- |
| Physical inputs | viscosity, initial condition | The equation's physical behavior or starting state |
| Numerical choices | grid_size, dt, steps | Spatial/time approximation and simulated duration |
| Storage choices | save_every | Which computed states are retained |
| Execution choices | workers, buffered/streamed | How independent runs and their output are processed |

Final physical time is T = dt × steps. Halving dt without doubling steps changes duration as well as timestep resolution. Increasing save_every stores fewer frames; it does not skip integration steps.

## A first 45–60-minute hands-on session

Treat these times as a suggested study session after installation. Downloading dependencies or learning Python for the first time may take longer. Run commands from the repository root; the global --lake argument belongs before the subcommand.

### Minutes 0–10: prepare and predict

Follow the current environment instructions in the [README](../README.md). If the complete environment is already prepared, use it without resynchronizing:

~~~sh
uv run --no-sync flowstate --help
~~~

Open examples/burgers.json. Before running it, predict the final time from dt × steps and the number of saved frames from save_every. Check whether the initial and final states should both appear.

The following commands write only tutorial artifacts under outputs/tutorial. Keep source code unchanged during the first rerun so provenance remains comparable.

### Minutes 10–20: run twice and inspect the evidence

~~~sh
uv run --no-sync flowstate --lake outputs/tutorial/lake run examples/burgers.json
uv run --no-sync flowstate --lake outputs/tutorial/lake run examples/burgers.json
uv run --no-sync flowstate --lake outputs/tutorial/lake list
~~~

The first command creates a result if absent. The second should reuse that verified result and report resumed=true. If you have already used this tutorial lake, the first command may also reuse it.

Copy the returned experiment ID and replace EXPERIMENT_ID below:

~~~sh
uv run --no-sync flowstate --lake outputs/tutorial/lake show EXPERIMENT_ID
uv run --no-sync flowstate --lake outputs/tutorial/lake verify EXPERIMENT_ID
~~~

Find the configuration, metrics, code provenance, and status. Locate record.json, metadata.parquet, fields.zarr, and manifest.json inside the corresponding experiment directory.

### Minutes 20–30: ask the catalog a question

~~~sh
uv run --no-sync flowstate --lake outputs/tutorial/lake query "SELECT id, status, initial_energy, final_energy, mass_drift, needs_review FROM experiments ORDER BY id"
~~~

Explain why energy should usually decrease for this unforced viscous problem, while total mass should remain approximately constant. A completed status only means the numerical computation finished; inspect review flags before treating it as a useful reference.

Change the SELECT to return only records with needs_review = true. An empty result is a valid answer, not a failed query. Do not deliberately destabilize an expensive sweep just to create a row.

### Minutes 30–45: inspect an actual field

Save the following as outputs/tutorial/inspect_run.py:

~~~python
import zarr
from flowstate.lake import Lake

lake = Lake("outputs/tutorial/lake")
record = next(
    item for item in lake.records()
    if item["status"] == "completed" and item["equation"] == "burgers1d"
)
path = lake.experiments / record["id"] / "fields.zarr"
group = zarr.open_group(str(path), mode="r")
times = group["time"][:]
velocity = group["fields/velocity"]
x = group["coordinates/x"][:]

print("axes: time, x; shape:", velocity.shape)
print("saved time range:", float(times[0]), float(times[-1]))
print("spatial points:", len(x))
print("first/last spatial mean:", velocity[0].mean(), velocity[-1].mean())
print("stored viscosity:", record["config"]["viscosity"])
~~~

Run it:

~~~sh
uv run --no-sync python outputs/tutorial/inspect_run.py
~~~

Compare the saved-time count with your prediction. velocity[0] reads one field; velocity[:] would load the whole trajectory. Explain which axis is time and why a 2D array here represents a one-dimensional spatial PDE.

### Minutes 45–60: connect behavior to a test

~~~sh
uv run --no-sync pytest tests/test_numerics.py tests/test_lake.py -q
~~~

Read one numerical test and one lake test. Identify the input, expected behavior, and assertion. Finish by narrating the seven-step path above using this actual run.

## Read files in this order

| Order | Read | Question to answer |
| --- | --- | --- |
| 1 | examples/burgers.json; cli.main() | How does a command become a configuration dictionary? |
| 2 | engine.run_experiment(), experiment_id(), capture_provenance() | What makes a result new, reusable, or failed? |
| 3 | numerics.normalize_config(), solve(), _burgers(), _rk4() | What is stored, differentiated, and updated at each step? |
| 4 | lake.Lake.write(), verify(), query(); test_lake.py | Why can readers trust publication and detect corruption? |
| 5 | engine.expand_sweep(), run_sweep(); test_engine.py | How are combinations validated and dispatched? |
| 6 | datasets.export_dataset(), _families(), _split() | Why must related trajectories remain in the same split? |
| 7 | ml._load_data(), FNO1d, train_fno(), evaluate_fno() | What information reaches training, validation, and test? |
| 8 | ml.burgers_residual(), train_pinn() | Which observations supervise a per-instance physics fit? |
| 9 | research.ResearchGraph, propose(), research_cycle() | Which conclusions are recorded, and which are not justified? |
| 10 | streaming.py, resources.py, scaling.benchmark_sweep() | What exactly is measured when execution is scaled? |

Names beginning with an underscore are implementation helpers: useful to read, but not the preferred public entry points for your own integrations.
Read one relevant test before its function, predict the assertion, and then follow the implementation. This keeps unfamiliar syntax connected to observable behavior.

## Keep the three later layers distinct

**Dataset ETL:** [datasets.py](../src/flowstate/datasets.py) extracts verified compatible trajectories, groups initial-condition families, assigns splits, computes training-only normalization, and loads a new immutable dataset. It preserves physical times and viscosity. It does not train a model.

**Learning:** [ml.py](../src/flowstate/ml.py) trains FNO across training trajectories to predict the next saved field, then checks held-out families and rollouts. The PINN instead optimizes one held-out initial-value problem using its first field, PDE residual, and periodic boundary losses. These are different learning settings; their costs and access to information must be stated.

**Research:** [research.py](../src/flowstate/research.py) links evidence and proposes bounded refinements. Its current policy is deterministic. A proposal or graph link is not a mathematical proof, and the policy is not an LLM reasoning autonomously about arbitrary science.

Read [data pipelines](steps/03-data-pipelines.md), [learning baselines](steps/04-learning-baselines.md), and [local scaling](steps/05-local-scaling.md) after the first run makes sense.

## Watch and read in this order

Begin with [3Blue1Brown: *But what is a partial differential equation?*](https://www.youtube.com/watch?v=ly4S0oi3Yz8).
It uses heat spreading to show why a value can depend on both position and time.
That is the core idea behind Flowstate's saved field frames. Then follow the
[official NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners)
through array shapes, slices, and basic operations; those are the building blocks
you will see in `numerics.py`.

After your first run, read the [Zarr arrays guide](https://zarr.readthedocs.io/en/main/user-guide/arrays/)
to understand why the movie is stored in chunks, and the [DuckDB Parquet guide](https://duckdb.org/docs/data/parquet)
to see why the small experiment summaries can be searched with SQL. The
[PDEBench project overview](https://github.com/pdebench/PDEBench) is useful once
you understand Flowstate's own data path: it shows an established collection of
PDE data and model baselines that Flowstate can work alongside. Watch
[3Blue1Brown's Fourier transform introduction](https://www.youtube.com/watch?v=spUNpyF58BY)
later, when you reach the Navier–Stokes spectral solver and FNO model. You need
the data flow before that math.

## A one-week practice route

Use the following sequence flexibly; revisit a day when its self-check is unclear.

| Day | Focus | Concrete output |
| --- | --- | --- |
| 1 | Run and trace one Burgers experiment | Explain configuration → fields → metrics → query |
| 2 | Python arrays and numerical updates | Sketch array shapes and hand-calculate a tiny periodic difference |
| 3 | Validation and provenance | Explain one known-solution test and one failure/reuse test |
| 4 | Storage and dataset ETL | Trace one trajectory's source hash, family, split, and normalizer |
| 5 | FNO and PINN distinctions | Explain one-step versus rollout error and what PINN supervision uses |
| 6 | Graphs, proposals, execution budgets | Inspect one proposal's evidence, parameter change, and budget |
| 7 | Make a small reviewed change | Add a meaningful test, run focused checks, explain limitations |

This can build working familiarity when you already know basic Python. Scientific and systems expertise grows through sustained study, debugging, independent validation, and feedback; a calendar alone cannot certify it.

Useful prerequisites are Python functions, dictionaries, classes, exceptions, loops, and tests; NumPy shapes, slicing, broadcasting, reductions, and FFT basics; derivatives, grids, boundary/initial conditions, timesteps, stability, and convergence; SQL, JSON, Parquet, Zarr, checksums, and provenance; and train/validation/test splits, normalization, gradients, and optimizer state. Learn them in the context of the file currently being read.

## Small changes that teach something

- Add a descriptive output-inspection script under outputs/tutorial; assert the saved time and shape match its selected run's configuration.
- Add a numerical regression test showing that changing save_every changes output sampling while preserving the final state for the same integration steps.
- Add a CLI regression test for one mistyped configuration key; require an intelligible failure and no completed experiment.
- Read the existing amplitude-family dataset test, then extend it to a timestep refinement with the same physical saved times; the initial-condition family must still stay together.

Read existing tests first so you extend coverage rather than duplicate it. Use small workloads. After a source edit, a new experiment ID can be correct because code provenance changed. Run the relevant test file, then the repository checks before committing.

## Can you answer these without guessing?

1. Which loop advances physical time, and which loop enumerates independent experiments?
2. Why are integration steps and saved frames different counts?
3. What does a checksum prove, and what mathematical correctness does it not prove?
4. Why must test trajectories not determine normalization or checkpoint selection?
5. Why can a good one-step model produce a poor rollout?
6. Why can adding workers make a small sweep slower?
7. Which current operations are local, and which claimed deployment would need fresh evidence?

If you can answer those using an actual artifact and a specific function or test, you are becoming able to maintain the project rather than merely run its commands.
