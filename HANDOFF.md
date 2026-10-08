# Flowstate computer and Codex handoff

Updated 7 October 2026. Continue on branch **develop**.
Repository: https://github.com/Yogendra-sodha/flowstate

Windows handoff validation: Ruff passed and 540 tests passed, with four
symbolic-link permission skips. The receipt is
`docs/reports/handoff-validation-20261007.json`. A fresh Mac validation is still
required.

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
5. `docs/steps/11-checkpoint-recovery.md`: the current recovery implementation.

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
- **Milestone 2 implemented; final evidence pending at handoff preparation:**
  exact internal solver checkpoints, immutable state/frame blobs, checked prefix
  replay, CLI and queue options, actual worker-kill proof, and tests. Check
  `docs/progress.md` for any recovery study completed during final Windows validation.
- **Milestone 3 remaining:** the revised GCS upload, approved local removal,
  download, verification receipt, viewer container, and deployment proposal.
  An earlier small live round trip is retained in `docs/reports/gcs-20261004.json`;
  it does not satisfy every revised phase requirement.
- **Milestone 4 remaining:** larger independent-family and multiple-seed model
  evaluation, paired persistence comparisons, mean/mass conservation, rollout
  error, ensemble uncertainty, and explicit negative results.
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

After the Mac test suite passes and the checkout is clean, run the committed
recovery protocol into a new directory:

```sh
uv run --no-sync python -m flowstate.recovery_study --output outputs/mac-recovery-proof-01
```

Inspect its aggregate and case reports. A successful result must verify both
equations, preserve every scientific array hash, and show measured RK4 counts
that skip earlier completed steps. Retain the report under `docs/reports/`, add
the validation receipt and measured results to the checkpoint chapter/stepbook,
update `docs/progress.md`, and commit before marking Milestone 2 complete.
Keep any failed result visible. Milestone 1 does not need another large run
merely because the computer changed.

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

Other earlier studies are in the repository's ignored `outputs/` tree. Locate and
copy the actual data you need rather than assuming GitHub contains it. Committed
reports retain the measurements even if the raw fields are not transferred.

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
- User prefers low costs and previously stated a free-credit budget. A budget
  preference does not itself authorize spending or credential use.

Cloud authentication was configured on Windows; it is not shipped in Git and
must be set up separately on the Mac if cloud work is approved. Read
`docs/steps/09-gcs-storage.md`. **Ask the user before using cloud credentials,
spending money, deleting data, or deploying.** Do not run authentication or cloud
commands merely because this document records the account details.

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
> reports, and update the stepbook. Ask before cloud credential use, spending,
> deletion, or deployment. At each phase boundary record progress and stop for
> review. Do not declare unfinished evidence complete.
