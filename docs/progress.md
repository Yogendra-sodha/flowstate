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
