# Trustworthy Burgers model evaluation

This Milestone 4 study extends the existing dataset and FNO implementation.
It evaluates a frozen local synthetic task; it does not reproduce PDEBench or
claim reliability for other equations, shocks, grids or viscosities.

```sh
uv sync --locked --all-extras --group dev --python 3.12
uv run --no-sync ruff check .
uv run --no-sync pytest
# Commit source and this protocol first. Choose an unused directory.
uv run --no-sync python -m flowstate.model_study --output outputs/model-study-01
```

`model_study.study_plan()` freezes the protocol. There are 180 random initial
conditions, seeds 20000 through 20179. Each is a fresh draw of the existing
four-mode Fourier generator, normalized to amplitude 0.5. Related trajectories
and exact duplicate initial fields cannot leak across splits. The canonical
family splitter assigns 60% to training and 20% each to validation and test:
108/36/36 when every draw is independent. The harness refuses fewer than 100
training families or duplicate initial fields. Saved time windows are not counted
as independent families.

The periodic Burgers reference uses viscosity 0.1, 64 points, timestep 0.0025,
800 steps and a saved frame every 20 steps. This gives 41 frames through physical
time 2. A fixed audit of the first twelve initial-condition seeds uses 128 points
and half the timestep. Comparing every second fine-grid point at the same saved
times measures combined spatial/time/reference sensitivity. Both grids normalize
the generator using their sampled maximum, so this also includes a small initial
sampling difference. This is not exact-solution validation or a convergence-order
estimate, and it is not used to alter model settings.

All five FNO training seeds, 0–4, use 30 epochs, width 16, eight Fourier modes,
three blocks, batches of 64, learning rate 0.001, and mean-preserving increments.
Normalization fits only training values. Within each seed, minimum validation
MSE chooses the checkpoint, including the initial persistence checkpoint.
`train_fno(evaluate_test=False)` defers its customary test report. All five models
are trained before calibration; no test inference occurs until calibration is
saved and hashed. Settings and seeds cannot be changed through the study CLI.

For one-step prediction the input is the preceding true saved frame. For rollout,
the model starts at the initial frame and feeds back its own predictions for 40
steps. Persistence uses the preceding true frame for one-step and holds the initial
frame throughout rollout. Every seed is compared with that same reference and
baseline. Reports retain physical RMSE, relative L2, per-time errors, mean/mass
drift, mean discrepancy and energy error. Energy increases are counted from the
initial state using an absolute tolerance of 1e-8; these sampled transitions are
not a dissipation-balance proof.

`paired_summary()` first reduces each independent family to RMSE, then subtracts
persistence RMSE for that family. Positive differences identify losses. Its fixed
2,000-draw bootstrap resamples families, never correlated cells or windows.
Intervals describe this cohort conditional on its split. Mean, sample standard
deviation and range across all five aggregate seed errors describe optimizer
variation; five seeds are not five independent datasets. No best test seed is
selected. Full lists identify losses and ties even when average error improves.

Uncertainty uses the ensemble mean and sample standard deviation over all five
models. A fixed physical-velocity standard-deviation floor of 1e-6 avoids division
by zero. Validation supplies one score per family: the largest absolute error
divided by floored spread across all its times and cells. The higher empirical
90th percentile scales an interval around the ensemble mean. One-step and rollout
have separate calibrations. The frozen test report measures cell coverage,
whole-trajectory coverage, interval width, spread/error correlation and uncovered
families. This is empirical calibration, **not a formal conformal guarantee**:
validation also selected model checkpoints. Shared model bias can produce low
spread with large error, and interval coverage says nothing about out-of-domain
reliability. No interval setting is fitted or repaired using test labels.

The processing path is the existing request → solver → immutable lake → manifest
verification → family dataset → training → held-out evaluation → retained report.
Generation streams saved fields to Zarr; export reads one verified trajectory at
a time and fits training statistics. Evaluation writes model predictions into
separate immutable bundles. Aggregation loads those bounded arrays, computes
paired/physical/uncertainty summaries and records hashes. Final checks verify
source manifests, query the original Parquet catalog with DuckDB, and reject
source/runtime or calibration changes during execution. Errors keep the original
plan, completed bundles, events and a failed-stage receipt. Reruns require a new
directory. Raw lakes, datasets and model weights stay under ignored outputs/;
small measured JSON reports and validation receipts are retained in docs/reports/.

The study uses local CPU work and no GCP resources. Existing GCP authorization
remains in force; actual billed spending remains unknown in its budget ledger.
