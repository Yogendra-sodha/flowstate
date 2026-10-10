# Paste into the new Mac Codex chat

Continue my Flowstate project on this MacBook Pro. This is a continuation of work
from my Windows computer, not a new project. Work in the existing cloned repo:
https://github.com/Yogendra-sodha/flowstate, branch `develop`.

First inspect Git status and branch. Preserve any local changes, then safely
pull the latest `develop` with `git pull --ff-only`. Read `README.md`, `STEPBOOK.md`,
`docs/roadmap.md`, `HANDOFF.md`, and `docs/progress.md` before editing. Use the
repository's current files and retained reports as the source of truth.

Flowstate is a Python/MIT data engineering project for reproducible physics
experiments. Its core loop is request -> run or verified reuse -> store -> verify
-> query. It has Burgers 1D, Navier–Stokes 2D and Darcy solvers; Zarr fields,
Parquet metadata, DuckDB queries, SHA-256 manifests and provenance; local sweeps,
a durable queue and recovery checkpoints; dataset pipelines, FNO/PINN baselines,
a research graph, offline viewer, and S3/GCS mirrors.

Milestones 1–3 are complete for their documented gates: scale measurement,
mid-trajectory crash recovery, and a real GCS upload -> remove generated local
copy -> download -> verify proof. The cloud proof restored all 23 files exactly;
its receipt and audit are in `docs/reports/gcs-recovery-20261009.json` and
`docs/reports/gcs-recovery-validation-20261009.json`. The viewer container passed
Linux CI, and all 608 tests passed on Windows and Linux. Mac validation is pending.
No hosted viewer was deployed. Do not repeat completed milestones merely because
the computer changed.

Validate this Mac first:

```sh
uv sync --locked --all-extras --group dev --python 3.12
uv run --no-sync ruff check .
uv run --no-sync pytest
```

Then start Milestone 4. Audit existing dataset, splitting, FNO training,
evaluation, conservation and checkpoint code before extending it. Freeze and
commit a reproducible study protocol before running measurements. Train on at
least 100 independent trajectory families with at least five training seeds;
report mean/spread versus persistence, one-step and rollout errors, conservation,
and ensemble uncertainty. Calibrate using validation data only. Keep cases where
the model loses visible. Milestones 5 and 6 later add the measured `flowstate ask`
router, actor provenance and portfolio polish.

I already authorized GCP credential use and spending up to **US$50 total** for
Flowstate. Do not ask for that permission again. Track usage with
`docs/reports/gcp-budget-ledger.json`; actual billed spend and remaining credits
are currently unknown. Project: `flowstate-510320`; bucket: `flowstate-codex`;
prefix: `flowstate/`; service account:
`flowstate@flowstate-510320.iam.gserviceaccount.com`. Authentication may need setup
on this Mac; never commit credentials. Hosted deployment remains outside scope,
and this budget does not authorize unrelated data deletion.

Keep making small commits and pushing to `develop`. Update HANDOFF.md,
STEPBOOK.md and docs/progress.md as you work, explaining functions, loops and
data processing in plain English. Pass Ruff and relevant/full tests for each
phase. Commit code before measurements, use fresh output directories, retain
measured reports in docs/reports/, and never commit outputs/, datasets or secrets.
Raw Windows data and environments are not included in Git; do not assume they
exist here or rewrite old provenance to pretend it came from this Mac.

Do not claim to solve Navier–Stokes existence/smoothness, tune on test data, or
hide negative results. Distributed remote workers, 3D Navier–Stokes, a hosted
dashboard and an LLM planner are outside this scope. At each phase's completion,
record results and limitations, commit/push, then stop for review. Start with
Mac validation and Milestone 4 now.
