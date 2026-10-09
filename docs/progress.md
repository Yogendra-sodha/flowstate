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
validation and a concrete prepared plan will be retained after source is
committed. Real GCS execution, billing, removal of the generated proof run, image
publication, and deployment have not been performed. Milestone 3 remains open.
