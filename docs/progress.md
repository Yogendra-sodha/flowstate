# Flowstate progress

## Milestone 1 — local scale measurement complete

Study executed on 4 October 2026 from clean commit
`7ce7c07883080d7cf676ed8512727c28bdb19732`. Implementation commits are `ccd1f68`
and `e1ede3c`; the measured commit also includes the protocol documentation.

**Changed:** added a fixed larger scaling command, failure-aware throughput,
retained diagnostics, bounded workload planning, process-tree memory measurements,
verified reuse, per-grid aggregation, and a static chart. Added tests and expanded
the stepbook with the functions, loops, and extraction/transformation/loading path.

**Commands run:**

```sh
uv sync --locked --all-extras --group dev
uv run --no-sync ruff check .
uv run --no-sync pytest --junitxml=outputs/phase1-validation/pytest.xml
uv run --no-sync flowstate scaling-study C:/Users/yuvis/AppData/Local/Flowstate/scaling-large-20261004
```

**Measured results:** all 36 trials completed, covering 216 distinct configurations
(72 per grid) and 36 initial-condition families. All 2,592 fresh executions and
2,592 verified reuse requests succeeded, with zero numerical failures and identical
scientific results across workers/repeats within each grid. The full study took
154.23 minutes and stored 4.83 GiB of logical experiment artifacts. Eight workers
provided 3.03–3.62 times serial throughput. Ruff passed; the suite passed 451 tests
with four Windows symbolic-link permission skips, and Windows/Linux CI passed.

Evidence: [raw measurements](reports/scaling-large.json),
[chart](reports/scaling-large.png), [validation receipt](reports/scaling-large-validation.json),
and [reproduction/results walkthrough](steps/10-scale-measurement.md).

**Remaining limits:** short trajectories on one local host; warm-cache reuse;
sampled RSS with process-disappearance errors; no cloud/distributed measurement or
long-time accuracy claim. Doubling four to eight workers yielded only 13.0–21.9%
more throughput at 72.3–81.9% more sampled peak RSS. Small-grid reuse was slightly
slower at eight workers, with overlapping ranges. Checkout line endings can change
source fingerprints and IDs. Raw fields remain outside Git. The old cloud-worker
draft is preserved in stash `84b667b6307a8971aeb3ed8d75e10aa90c3be664`.

**Next:** Milestone 2, mid-trajectory checkpoint recovery. Stopped at this phase
boundary for user review; no cloud credentials, spending, or data deletion were
used for this phase.

## Computer handoff — Milestone 2 implementation preserved

On 7 October 2026 the user requested migration from Windows to a MacBook Pro.
The current checkpoint implementation and tests are being committed on `develop`
with `HANDOFF.md`, rather than left only in the Windows workspace.

**Changed:** added exact Burgers/NS solver checkpoints, immutable state/frame
blobs and commit markers, verified prefix replay, engine/CLI/queue integration,
and a reproducible process-kill proof harness. Tests cover real killed workers,
skipped integration work, exact array equality, corruption, concurrent publication,
queue recovery, and validation before filesystem effects. The stepbook and
checkpoint walkthrough describe the functions, loops, and storage pipeline.

**Migration:** `HANDOFF.md` records Mac setup, the code map, milestone status,
required permissions, raw-data locations, and a continuation prompt. The old
cloud-worker stash is preserved in `docs/handoff/cloud-worker-draft.patch`, so a
Git clone retains that inactive draft. No dataset, credentials, or environment
is added to Git; ignored raw data needs a separate user-managed transfer.

**Validation commands:**

```sh
uv run --no-sync ruff check .
uv run --no-sync pytest --junitxml=outputs/handoff-validation-20261007/pytest.xml
```

Ruff passed; the full suite passed 540 tests with four Windows symbolic-link
permission skips. Evidence is retained in
`docs/reports/handoff-validation-20261007.json`. The worker-kill protocol has
automated test coverage, but the final
clean-source measured recovery study/report has not been run for this milestone.

**Remaining limits:** Mac dependencies and execution need validation on the new
computer; hardware and checkout bytes change provenance/identities. Recovery is
local and assumes a compatible recorded runtime. No power-loss, remote-worker,
or cross-hardware equality guarantee is established. Milestone 2 remains open
until its committed-source study is retained and documented. The next chat should
run that protocol after Mac checks pass, then stop at the phase boundary.

No cloud credentials, cloud spending, deployment, or user-data deletion occurred
during this handoff.

## Milestone 2 — local checkpoint recovery complete

The user resumed Windows work after the handoff on 7 October 2026. The measured
protocol ran from clean commit `d8ef527ea887853a7c1169d194a72925d030691d`.

**Changed:** completed the recovery proof, retained its raw report and validation
receipt, independently audited checkpoints/final arrays, and updated the
checkpoint chapter, roadmap, stepbook, and Mac handoff. Evidence was committed
and pushed before final phase documentation.

**Commands and validation:**

```sh
uv run --no-sync python -m flowstate.recovery_study --output outputs/checkpoint-proof-20261007
uv run --no-sync ruff check .
```

The same implementation's full suite passed 540 tests with four Windows
symbolic-link permission skips. The validation receipt confirms those source
hashes match the measured package. Windows/Linux CI passed the measured commit.

**Measured results:** both workers stopped after committed step 13 and resumed
with 24 actual RK4 calls, versus 37 for each uninterrupted reference. All 17
scientific arrays matched; all final manifests verified; zero cases failed.
The protocol took 20.03 seconds. Independent audit reconciled raw receipts,
checkpoint blobs/state, final arrays, recovery records, and process ownership.
Post-study queries found the completed records. Evidence:
[report](reports/checkpoint-recovery.json),
[validation/audit](reports/checkpoint-recovery-validation.json), and
[walkthrough](steps/11-checkpoint-recovery.md).

**Remaining limits:** controlled kill after a committed boundary on one local
Windows runtime, short trajectories, and single timing observations. Burgers
recovery took longer than its uninterrupted reference because checkpoint work
adds overhead. This establishes no general speedup, power-loss durability,
remote-worker support, or cross-hardware bitwise equality. Mac validation is
pending; raw arrays/logs remain in ignored `outputs/checkpoint-proof-20261007`.

**Next:** Milestone 3, approved GCS recovery proof plus viewer container and
deployment proposal. Stopped at this phase boundary for review. No cloud
credentials, spending, deployment, or user-data deletion were used. `HANDOFF.md`
now directs the Mac chat to environment validation and the next milestone.

## Milestone 3 — local preparation, cloud approval pending

On 8 October 2026, implemented a two-stage GCS recovery proof and static viewer
package. Preparation creates a fixed local experiment and a hash-bound review
plan without a cloud client. Execution requires explicit approval, verifies the
remote bytes before enumerated local removal, restores into a fresh lake, and
retains exact file/query evidence or a failed-stage receipt. Tests exercise the
full transfer through the in-memory SDK fake, approval and path guards, partial
removal, corruption, and failures before/after removal.

The viewer serves one preloaded HTML export with no lake or credential access.
Actual HTTP tests cover routes and unchanged export bytes. Linux CI now includes
a real rootless container build/start check with its base/image/snapshot details
logged. The Windows Docker client is installed, but its local engine did not
respond; no successful Windows container build is claimed.

The stepbook, handoff and [phase guide](steps/12-cloud-proof-viewer.md) record the
implementation, failure recovery, and a private Cloud Run proposal. Full local
validation and a concrete prepared plan are now retained after committing source.
Real GCS execution, billing, removal of the generated proof run, image publication,
and deployment have not been performed. Milestone 3 remains open.

On 9 October, committed and pushed the proof harness, viewer package and guide,
then prepared the local experiment from clean commit `ec565a1`. The command was
`uv run --no-sync python -m flowstate.cloud_proof prepare C:/Users/yuvis/AppData/Local/Flowstate/gcs-proof-20261009`.
The [retained plan](reports/gcs-proof-plan-20261009.json) inventories 23 files and
28,422 bytes. Its [validation receipt](reports/cloud-proof-preparation-validation-20261009.json)
records the local suite: 600 passed, five Windows symlink permission skips, and
Ruff passed. No cloud client was created by preparation. Requested explicit
approval for credentials, a US$0.10 spending budget, and removal of only the
inventoried generated experiment after remote verification. Approval is pending.

Container review found that Unix exports are private files, unreadable by the
container's separate user. The smoke check stages an independent readable copy
inside a private temporary directory, preserves the original, and cleans up its
own copy after stopping its container. Regression tests exercise success and
failure cleanup. The first CI attempt failed fetching the public base image;
the next encountered Docker Hub throttling despite bounded retries. CI now uses
Docker's official public mirror and runs Windows checks independently of Linux
failures. These changes are committed in `009f58f` and `d50378d`.

**Verified continuation:** the [container and CI receipt](reports/viewer-container-validation-20261009.json)
retains both failed attempts and the successful run on `d50378d`. Ruff and all
608 tests passed on Windows and Linux CI, without skips. Linux built and ran the
viewer as UID 65532 with a read-only filesystem, dropped capabilities, and a
loopback port. The 20,530-byte exported snapshot was served byte for byte;
readiness and closed file routes passed. The report records the resolved base
digest, built image ID, snapshot hash, raw-log hashes, and job URLs.

**Commands:** CI ran `uv run --no-sync ruff check .`, `uv run --no-sync pytest`,
an actual Burgers run/export, and the Docker build followed by
`uv run --no-sync python containers/viewer/smoke.py outputs/viewer-ci.html --image flowstate-viewer:ci --base-image <resolved-digest>`.
Local focused regression tests passed with unchanged source provenance.

**Remaining boundary:** the viewer package and deployment proposal are ready.
Milestone 3 still needs explicit user approval and the actual GCS/removal/restore
receipt. No cloud credentials, generated-experiment removal, image publication,
or deployment occurred. `HANDOFF.md` includes the exact pending Windows plan;
Mac continuation must generate its own plan. Work pauses at the approval gate.

## Milestone 3 — approved cloud recovery complete

**Changed:** the user approved the prepared GCS/removal/restore test and granted
up to US$50 total GCP use for Flowstate without repeated permission requests.
Executed the already committed harness, retained its exact raw receipt and
authorization, independently audited local evidence, and updated the budget
ledger, phase guide, roadmap, stepbook and Mac handoff. No source changes were
needed for the measured run.

**Command:**

```sh
uv run --no-sync python -m flowstate.cloud_proof execute C:/Users/yuvis/AppData/Local/Flowstate/gcs-proof-20261009 --plan-sha256 aa44f296de41d0d3122aef79c4847d69c9c70ddc235a77f785a1bf3db0bc266d --allow-cloud --allow-local-delete
uv run --no-sync ruff check .
```

**Measured results:** clean execution commit `bb8daf2`; 22 artifacts uploaded
plus their manifest. A repeat verified all existing remote artifacts without
new uploads. The generated local experiment was removed, then freshly restored:
all 23 files and 28,422 bytes matched exactly. Manifest verification and the
completed Burgers metadata query passed. The execute call took 27.67 seconds.
Evidence: [raw receipt](reports/gcs-recovery-20261009.json),
[independent validation](reports/gcs-recovery-validation-20261009.json), and
[authorization](reports/gcp-authorization-20261009.json).

The executed source and tests are unchanged from `d50378d`, which passed Ruff
and all 608 tests on both Windows and Linux CI. Ruff also passed after the live
proof. The [viewer container validation](reports/viewer-container-validation-20261009.json)
and private deployment proposal satisfy the remaining phase deliverables.

**Limits:** a single small Burgers experiment is not a cloud throughput,
availability, durability, or cost benchmark. The audit independently verified
local bytes and receipts, not a second cloud read or provider billing. Actual
charges and remaining promotional credits are unknown in the
[budget ledger](reports/gcp-budget-ledger.json). No hosted service was deployed.
Raw restored fields remain outside Git in the Windows proof directory and its
recorded GCS prefix.

**Next:** Milestone 4, the larger trustworthy-model study. The user has cloned the
repository on a MacBook Pro; validate that environment first, then audit existing
dataset/FNO/evaluation code and freeze the study protocol before new measurements.
Milestones 1–3 are complete for their gates. Stopping at this phase boundary for
review; the existing GCP authorization persists into the next chat.

## Mac validation complete — 10 October 2026

Clean checkout; safely fast-forwarded develop to `1d6c3d6`. Installed uv and
ran the locked all-extras/dev Python 3.12 sync, Ruff, and full pytest suite.
All 608 tests passed without skips; Ruff passed.
[Receipt](reports/mac-validation-20261010.json) retains runtime, lockfile and
JUnit hashes. No source or lockfile changes were needed. Milestones 1–3 remain
complete; proceeding to the authorized Milestone 4 study. No cloud use occurred.
