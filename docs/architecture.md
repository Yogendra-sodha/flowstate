# Engine architecture

Flowstate connects a scientific question to executable configurations and retained
evidence. The first version implements the numerical experiment and data layer.

```mermaid
flowchart TD
    Config[JSON configuration or parameter grid] --> Validate[Canonical validation]
    Validate --> Identity[Code + environment + parameters + parent + attempt]
    Identity --> Reuse{Verified result exists?}
    Reuse -->|yes| Resume[Return immutable result]
    Reuse -->|no| Solve[Burgers or 2D Navier–Stokes]
    Solve --> Diagnostics[Sampled metrics and numerical review flags]
    Solve --> Failure[Recorded numerical failure]
    Diagnostics --> Stage[Stage artifacts locally]
    Failure --> Stage
    Stage --> Publish[Atomic directory publication]
    Publish --> Fields[Zarr fields and diagnostics]
    Publish --> Catalog[Parquet metadata]
    Publish --> Record[JSON provenance + checksums]
    Catalog --> SQL[DuckDB SELECT queries]
    Record --> Graph[Experiment lineage graph]
```

## Modules

| Module | Responsibility |
| --- | --- |
| `numerics.py` | Strict config validation, initial conditions, integration, saved fields and diagnostics |
| `engine.py` | Execution identity, provenance, metric summaries, attempts, process-based sweeps |
| `lake.py` | Atomic publication, Zarr arrays, Parquet catalog, SQL, checksums, lineage |
| `cli.py` | JSON commands, readable failures, machine-readable output and exit status |

There is no central writable metadata database. A DuckDB connection builds the
`experiments` table from the per-run Parquet records, so distinct workers can publish
without competing over a database writer lock. Temporary staging directories are
hidden from discovery. A crash before publication leaves no finalized run; a later
invocation starts that configuration again. Power-loss durability and remote object
storage transactions are outside this local prototype's guarantees.

## On-disk contract

```text
data/lake/experiments/<id>/
  record.json
  metadata.parquet
  manifest.json
  fields.zarr/          # completed numerical runs only
    time
    coordinates/x
    coordinates/y      # 2D only
    fields/...
    diagnostics/...
```

Field arrays use a saved-time leading axis. Burgers uses `(time, x)` and Navier–Stokes
uses `(time, y, x)`. Frames are chunked individually, with spatial chunks capped at
64 cells per dimension. The initial and final states are saved, even when the final
step is not divisible by `save_every`. Diagnostics align with the saved times.

The engine uses 128-bit identifiers derived from SHA-256 over the canonical config,
parent ID, attempt, schema version, source fingerprint, Git commit, dependency/Python
versions, and recorded hardware. Run timestamps are excluded. Changing source,
commit, environment, machine description, or attempt creates a distinct experiment.
An existing result is checksum-verified before reuse, including existing failures.
After numerical failure, use a new attempt or corrected configuration; linking the
new run to its predecessor is explicit with `--parent`.

Git dirty state and a package source fingerprint are recorded. A fingerprint can
identify changed code but cannot reconstruct it. For exact reproduction, commit the
source and retain `uv.lock`; avoid relying on uncommitted code. Hardware information
does not capture every floating-point library or scheduling detail, so portability
does not imply bitwise equality.

The manifest detects accidental damage. It is unsigned, and directory permissions
do not enforce immutability against external modification. `verify` is explicit for
analysis and automatic for resumption; SQL does not rehash every field chunk.

## Scientific entities

Current records implement Experiment, Equation, Solver, configuration/initial
condition, Dataset artifacts, Metric, and a parent relationship. The graph command
exports experiment nodes and `parent_of` edges. It is a lineage graph, not yet a
full ontology of models, checkpoints, hypotheses, and findings. Those entities need
versioned evidence links and action semantics before an autonomous researcher can
operate on them.

The scientific reviewer has a separate role: challenge numerical validity,
provenance, benchmark splits, and interpretation. Review flags are numerical
observations, not accepted hypotheses. Future agents should propose bounded
experiments with compute budgets; they should never silently promote an observation
into a scientific finding.
