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

## Executed Mac results — 10 October 2026

The frozen protocol completed from clean commit `4b18252` in 159.07 seconds.
The [exact raw report](../reports/model-study-20261010.json),
[independent audit](../reports/model-study-audit-20261010.json),
[source validation](../reports/model-study-code-validation-20261010.json), and
[negative cases](../reports/model-study-negative-cases-20261010.json) retain evidence.
Raw arrays and weights remain in `outputs/model-study-20261010-01/` on this Mac.

All 180 coarse trajectories and twelve finer audits passed their numerical flags
and final manifests. There are 108 independent training families, 36 validation
families and 36 test families. DuckDB found all 180 completed original runs.
All five model and ten prediction bundles verified. The tested source hashes
match the measured source; Ruff and all 621 Mac tests passed with no skips.

Physical velocity RMSE against the coarse numerical reference:

| Prediction | Persistence | Five-seed mean | Sample standard deviation | Ensemble mean prediction |
| --- | ---: | ---: | ---: | ---: |
| One saved step | 0.00360433 | 0.0000579654 | 0.00000144430 | 0.0000540332 |
| 40-step rollout | 0.0857394 | 0.000469861 | 0.0000173970 | 0.000393172 |

Every seed improved family-level field RMSE on all 36 test families. No family
field-error losses or ties occurred in this cohort; that absence does not
establish performance on harder tasks. Selected epochs were 26, 29, 27, 25 and
30 for seeds 0–4. The paired family-bootstrap 95% intervals for ensemble-minus-
persistence mean family RMSE were [-0.00370333, -0.00331971] for one-step and
[-0.0889552, -0.0794891] for rollout. These intervals remain conditional on the
fixed generator and split.

**Where the model loses:** persistence holds the initial mean exactly during
rollout, whereas every model has nonzero rounding-scale mean drift on all 36
families. Aggregate drift RMS ranges from 3.51e-9 to 4.80e-9 velocity units;
mass drift RMS ranges from 2.21e-8 to 3.01e-8. For example, seed 0 on trajectory
`0225f4908af6733196ed95ebea688d40` has mean drift RMS 3.10e-9 versus zero for
persistence. This is a small floating-point weakness, not a large physical
conservation failure. All cases and their IDs are retained. No model had a
sampled energy-increase transition above the fixed 1e-8 tolerance; energy and
mean errors against the numerical reference are retained separately.

**Uncertainty failure:** the nominal 90% one-step intervals covered all cells
and times for only 30/36 test families (83.33%). Rollout covered 33/36 (91.67%).
Cell coverage was 99.976% and 99.988%, respectively, which hides the more useful
whole-trajectory misses. Mean interval widths were 0.00082523 and 0.0103402
velocity units. The uncovered family IDs are retained; no threshold was adjusted
using these outcomes. This calibration is empirical, uses the checkpoint-selection
validation set, and must not be promoted to a formal 90% guarantee or a general
request-router boundary.

**Reference limit:** the twelve coarse-versus-fine RMSE values range from
0.00029146 to 0.00067897. This is comparable to model rollout error and larger
than one-step model error. The models closely reproduce the coarse solver;
these results cannot establish that their physical predictions are more accurate
than the reference discretization. The audit includes initial sampling differences
and does not measure a convergence order. One smooth Fourier generator at one
viscosity/amplitude, one split and a two-time-unit horizon is a narrow task.

Milestone 4's fixed-study evidence gate is complete, and its code and audited
evidence are published to develop. Stop for review before Milestone 5; no
settings were changed after reading test results, and no GCP resource was used.
