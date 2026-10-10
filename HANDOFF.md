# Flowstate computer and Codex handoff

Updated 10 October 2026 for the Mac continuation. Continue on branch **develop**.
Repository: https://github.com/Yogendra-sodha/flowstate

Windows handoff validation: Ruff passed and 540 tests passed, with four
symbolic-link permission skips. The receipt is
`docs/reports/handoff-validation-20261007.json`. A fresh Mac validation is still
required.

Latest Windows continuation: the committed recovery proof has passed for both
equations. Results are retained in `docs/reports/checkpoint-recovery.json` and
`docs/reports/checkpoint-recovery-validation.json`. Raw fields and worker logs
remain in `outputs/checkpoint-proof-20261007/`, outside Git. Source commit:
`d8ef527ea887853a7c1169d194a72925d030691d`.

**Milestone 3 is complete for its stated gate.** The approved real GCS proof ran
from clean commit `bb8daf2`: upload, verify remote bytes, remove only the generated
local experiment, download, verify all hashes, query metadata, and export a viewer.
All 23 files, totaling 28,422 bytes, were restored exactly. The protocol took
27.67 seconds. Original experiment provenance was preserved, and the independent
local audit passed. Evidence: `docs/reports/gcs-recovery-20261009.json` and
`docs/reports/gcs-recovery-validation-20261009.json`.

The same implementation passed Ruff and all 608 tests on both Windows and Linux
CI, with no skips. The viewer container built and ran successfully in Linux CI;
`docs/reports/viewer-container-validation-20261009.json` preserves the results
and earlier registry failures. A private deployment proposal is in
`docs/steps/12-cloud-proof-viewer.md`. No hosted service was deployed. A successful
Windows or Mac Docker build has not been established. **Milestone 4 is next.**

The original run was generated from clean commit
`ec565a1c9edabba9ab2a15775c5f768250e83ae0`. The exact historical plan is
`docs/reports/gcs-proof-plan-20261009.json`, SHA-256
`aa44f296de41d0d3122aef79c4847d69c9c70ddc235a77f785a1bf3db0bc266d`.
The Windows root is `C:/Users/yuvis/AppData/Local/Flowstate/gcs-proof-20261009`.
The plan is already executed; do not rerun or reset its execution marker.
The user approved the described recovery test and **up to US$50 total GCP use**
for Flowstate, explicitly asking not to be asked again for that permission.
`docs/reports/gcp-authorization-20261009.json` records the authorization;
`docs/reports/gcp-budget-ledger.json` tracks known usage. Actual billed charges
and the promotional-credit balance have not been measured.

## Purpose

Flowstate is a Python/MIT data engineering project for reproducible numerical
experiments. A request runs a PDE solver or reuses a verified identical result.
It stores fields, metrics, parameters, and code/environment provenance, verifies
their checksums, and makes the results queryable. Its current solvers cover
periodic Burgers in one dimension, incompressible Navier–Stokes in two dimensions,
and a manufactured steady Darcy problem. It makes no claim about solving the
Navier–Stokes existence and smoothness problem.

The core software is already working: Zarr arrays, Parquet metadata, DuckDB
queries, immutable experiment identities/manifests, sweeps, a local SQLite queue,
dataset ETL, basic FNO/PINN models, a research graph, offline viewer, and S3/GCS
mirrors. The revised milestones measure the reliability and usefulness of this
prototype rather than treating implemented code as completed scientific evidence.

## Read first on the new computer

1. `README.md`: capabilities and runnable commands.
2. `HANDOFF.md`: this migration and continuation context.
3. `docs/progress.md` and `docs/roadmap.md`: current milestone gates and evidence.
4. `STEPBOOK.md`: setup, implementation decisions, functions, loops, and ETL.
5. `docs/steps/12-cloud-proof-viewer.md`: the latest completed proof and limits.

## Mac setup

Install Git, Python tooling through uv, and the Xcode command-line tools if Git
is not available. Install uv using its official installation instructions:
https://docs.astral.sh/uv/getting-started/installation/

Run these in the Mac Terminal after uv and Git are available:

```sh
git clone --branch develop https://github.com/Yogendra-sodha/flowstate.git
cd flowstate
uv sync --locked --all-extras --group dev --python 3.12
uv run --no-sync ruff check .
uv run --no-sync pytest
uv run --no-sync flowstate --lake outputs/mac-smoke run examples/burgers.json --checkpoint-every 20
uv run --no-sync flowstate --lake outputs/mac-smoke list
```

The sync creates a new Mac virtual environment from `uv.lock`. Do not copy the
Windows `.venv`. All extras prepare the existing numerical, ML, storage, and
benchmark tooling; optional platform package availability must be verified on
the Mac. Mac execution has not yet been validated. Existing CI covers Windows
and Linux. If dependency installation fails, retain the lockfile and diagnose
the platform-specific dependency before changing it.

Open the cloned folder as a Codex project. Start a new chat with the continuation
prompt below. A fresh chat will not automatically contain this Windows chat's
history. Commit code before generating new measured results on the Mac.

## Current delivery status

- **Milestone 1 complete:** larger local scale measurement. Evidence is in
  `docs/reports/scaling-large.json`, its chart, and its validation receipt.
- **Milestone 2 complete for the documented local process-crash protocol:**
  both workers were killed after step 13, then completed only the remaining
  24 of 37 steps. All scientific-array hashes matched uninterrupted references,
  with no failed cases. Exact internal state, immutable blobs, checked prefix
  replay, CLI/queue options, and actual worker-kill tests are implemented.
  The independent audit passed and Windows/Linux CI passed the measured source.
  See the retained recovery reports and `docs/progress.md` for the phase boundary.
- **Milestone 3 complete:** approved real GCS upload, verified remote reuse,
  generated-local-copy removal, fresh download, exact hashes and metadata query.
  Viewer container validation and a private deployment proposal are retained.
  The measured receipt is `docs/reports/gcs-recovery-20261009.json`; its independent
  audit is in the corresponding validation report. Deployment is not required
  for this gate and has not been performed.
- **Milestone 4 remaining:** train on at least 100 independent trajectory families
  with at least five training seeds; report mean/spread against persistence,
  one-step and rollout error, mean/mass conservation, ensemble uncertainty, and
  explicit cases where the model loses. Calibrate on validation data, never test.
- **Milestone 5 remaining:** `flowstate ask` routing to verified identical reuse,
  a compatible model within a calibrated uncertainty threshold, or the solver.
  Retain routing, latency, error, and speed-versus-accuracy evidence.
- **Milestone 6 remaining:** actor provenance (human/rule/LLM plus relevant model
  and prompt hash), concise README/architecture, demo, and limitations.

The subscription deadline created a preference for finishing a useful runnable
portfolio project promptly. It did not waive tests, evidence, scientific limits,
or permission requirements, and the full milestone scope must not be declared
complete merely to meet that deadline.

### First unfinished action on the Mac

Validate the Mac environment using the setup commands above. The Windows
recovery proof is already retained; a Mac run is optional local confirmation:

```sh
uv run --no-sync python -m flowstate.recovery_study --output outputs/mac-recovery-proof-01
```

Use a new directory and commit any code changes before a new measurement. Keep
any failed result visible. No large scaling rerun is required merely because
the computer changed. After Mac validation, start **Milestone 4**, using
`docs/progress.md` and `docs/roadmap.md` for its acceptance gate. Audit the existing
Burgers dataset splitting, FNO training/evaluation, persistence baseline,
mean-preserving updates, and training checkpoints before changing them. Freeze
a study protocol with independent trajectory families, training seeds, held-out
evaluation, and uncertainty calibration. Commit study code before measurement.
The GCS proof is already complete; no repeat is needed for migration. Windows
absolute paths and runtime identity cannot be reused as Mac execution plans.

The small Windows proof measured recovery correctness, not general performance.
Burgers recovery was slower than its uninterrupted reference in the retained
single observation; checkpoint I/O and prefix replay add overhead. Mac
performance and cross-hardware bitwise equality have not been established.

## Code map

| File | Responsibility |
| --- | --- |
| `src/flowstate/cli.py` | CLI request parsing and command dispatch. |
| `src/flowstate/numerics.py` | Normalized configuration, Burgers/NS integration, exact solver snapshots. |
| `src/flowstate/darcy.py` | Steady Darcy solver. |
| `src/flowstate/engine.py` | Provenance, experiment identity, execute/reuse, summaries, sweeps. |
| `src/flowstate/checkpoints.py` | Immutable checkpoint blobs/markers, integrity validation, frame replay. |
| `src/flowstate/streaming.py` | Incremental physical fields in a private Zarr sink. |
| `src/flowstate/lake.py` | Atomic final publication, Parquet catalog, verification and queries. |
| `src/flowstate/queue.py` | Local SQLite jobs, leases, fenced acknowledgments and recovery options. |
| `src/flowstate/recovery_study.py` | Independent reference, controlled worker kill, resume, actual RK4 counts and array hashes. |
| `src/flowstate/cloud_proof.py` | Local approval plan, bounded GCS recovery proof, and exact removal/restore receipts. |
| `containers/viewer/` | Single-snapshot HTTP server, container, and Linux container check. |
| `src/flowstate/scaling.py` | Isolated scaling trials, process-tree memory and retained reports. |
| `docs/steps/` | Detailed workflow and reproduction guides. |
| `docs/reports/` | Small retained measurement evidence committed to Git. |

Checkpoint frequency is an execution option, outside scientific run identity.
`resumed` means finalized verified reuse; `checkpoint_resumed_from` identifies an
actual continuation step. Checkpoint mode streams fields automatically. A
published corrupt checkpoint fails closed as infrastructure failure, rather than
becoming a physics failure. Local compatible process-crash recovery is the scope;
power-loss durability and remote/distributed workers are not established.

## What a Git clone will and will not transfer

Git transfers source, tests, configurations, lockfile, documentation, and retained
reports. `data/`, `outputs/`, `.venv/`, credentials, and logs are ignored. If you
want earlier datasets, model checkpoints, and raw studies, transfer them separately
to the Mac before retiring the Windows laptop. No data is deleted by this handoff.

The larger scaling study's raw lake is currently outside the repository:

```text
C:/Users/yuvis/AppData/Local/Flowstate/scaling-large-20261004
```

The new raw recovery proof is under the Windows checkout:

```text
C:/Users/yuvis/OneDrive/Documents/ChatGPT/Flowstate/outputs/checkpoint-proof-20261007
```

Other earlier studies are in the repository's ignored `outputs/` tree. Locate and
copy the actual data you need rather than assuming GitHub contains it. Committed
reports retain the measurements even if the raw fields are not transferred.

The completed cloud proof's restored lake, viewers and raw receipts are in
`C:/Users/yuvis/AppData/Local/Flowstate/gcs-proof-20261009/`. The mirrored experiment
also remains under `gs://flowstate-codex/flowstate/proofs/5ffab66dac614dc486b592b0678b454a/`.
Its experiment ID is `842524f5280649c4aa6b4271d25391e3`. Git contains the reports,
not these raw files. Restoring old artifacts preserves their original provenance;
it does not make them newly computed Mac experiments.

A Mac has different hardware/platform provenance, and Git checkout line endings
may differ from Windows. New experiment IDs are therefore expected. Do not edit
old provenance to force identity reuse or claim cross-machine bitwise equality.
Interrupted Windows solver checkpoints cannot be transparently continued under
a different Mac identity; keep completed artifacts for inspection and start new
Mac runs. Restart workers after code or checkout changes.

### Preserved older cloud-worker draft

An unfinished cloud-worker draft was held in a Windows-only Git stash:
`84b667b6307a8971aeb3ed8d75e10aa90c3be664`, based on
`57cc0fad6d2cb4ba1572523f93018ce5349b109c`.
Stashes do not transfer through a normal clone. Its full tracked/untracked patch
is now retained as `docs/handoff/cloud-worker-draft.patch` for review.

It is **not active implementation, tested deployment, or part of the revised
delivery scope**. Remote distributed workers remain out of scope. Do not apply
it automatically to `develop`; it changes the old CLI and could conflict with
current checkpoint options. If later needed, review it in an isolated worktree
at its recorded base and port only authorized changes. The patch contains no
cloud credentials.

## Cloud context and permissions

- Project: `flowstate-510320`.
- Bucket: `flowstate-codex`.
- Object prefix: `flowstate/`.
- Service account: `flowstate@flowstate-510320.iam.gserviceaccount.com`.
- User-reported region: South Carolina, `us-east1`; verify actual resource
  locations before deployment.
- The latest user instruction authorizes GCP credentials and spending up to
  **US$50 total for Flowstate** without repeated permission requests. Consult
  `docs/reports/gcp-budget-ledger.json`; unknown billed charges do not mean zero.

Cloud authentication was configured on Windows; it is not shipped in Git and
must be set up separately on the Mac when needed. Read
`docs/steps/09-gcs-storage.md`. Do not ask again for credential use or GCP spending
already covered by the authorization. Ask for missing login interaction only if
needed. Review actual billing before material new resource commitments, and stay
within the total budget. The approved deletion was the exact generated proof
copy; unrelated destructive work still needs authorization. Hosted deployment
remains outside the revised scope and needs an explicit deployment instruction.

## Rules the next Codex chat must retain

- Work on `develop`, make small commits, and push authorized finished work.
- Read README, STEPBOOK, and roadmap before implementation.
- Every phase adds meaningful tests and passes `ruff check .` and `pytest`.
- Commit source before a measured study; use a new output directory.
- Never commit `outputs/`, private data, credentials, or environments.
- Every measured number in documentation must come from a retained report in
  `docs/reports/`. Keep negative outcomes visible and do not tune on test data.
- Keep scientific claims unchanged. No claim about solving Navier–Stokes
  existence/smoothness.
- Append phase changes, commands, results, and limits to `docs/progress.md`;
  stop at each phase boundary for user review.
- Keep STEPBOOK updated with actual functions, loops, processing, and ETL.
- Exclude distributed remote workers, three-dimensional NS, hosted dashboard,
  and an LLM planner from the present scope.

## Paste this into the new Codex chat

> Continue Flowstate on branch develop. Read HANDOFF.md, README.md, STEPBOOK.md,
> docs/roadmap.md, and docs/progress.md first. Check the actual Git state and
> retained reports. Validate the Mac environment, then resume the first unfinished
> revised milestone. Keep the trusted request/run-or-reuse/store/verify/query loop
> central. Add tests and pass Ruff/pytest, commit code before measurements, retain
> reports, and update the stepbook and handoff. Milestones 1–3 are complete;
> begin Milestone 4 after validating the Mac environment. GCP use up to US$50
> total is already authorized; do not ask again for that permission. Track costs,
> retain scientific limits, and do not deploy a hosted service or delete unrelated
> data. At each phase boundary record progress and stop for review. Do not declare
> unfinished evidence complete.
