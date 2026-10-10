# Flowstate: experiment engine first

## Revised delivery milestones

The current priority is a portfolio-grade data engineering system around the
trusted loop: request, run or verified reuse, store, verify, and query. The earlier
day-by-day prototype plan below is historical context. The current acceptance
gates are sequential:

Milestones 1, 2 and 3 are complete for their documented acceptance gates. The
[scaling report](reports/scaling-large.json), [chart](reports/scaling-large.png),
[recovery report](reports/checkpoint-recovery.json),
[live GCS recovery receipt](reports/gcs-recovery-20261009.json), and
[progress entries](progress.md) retain the evidence and limitations. The
[checkpoint protocol](steps/11-checkpoint-recovery.md) describes the implementation
and executed proof. Milestone 4's fixed local evaluation gate is also complete;
see the [five-seed report](reports/model-study-20261010.json),
[audit](reports/model-study-audit-20261010.json), and
[negative cases and limits](steps/13-trustworthy-model.md#executed-mac-results--10-october-2026).
Milestones 5 and 6 remain open. This local evidence does not establish broad model
reliability; one-step empirical interval coverage missed its nominal target.

1. **Scale measurement:** a larger local workload across the requested grids and
   worker counts, retained timings/memory/storage/reuse/failure evidence, a chart,
   and a single reproduction command. See the
   [measurement protocol](steps/10-scale-measurement.md).
2. **Crash recovery:** checkpoints within a numerical trajectory, tested by killing
   a worker and checking resumed results against an uninterrupted reference.
3. **Cloud proof:** an approved GCS upload, local-removal, restore, and verification
   receipt; a viewer container and deployment proposal. Cloud credentials,
   spending, deletion, and deployment require approval before those actions.
   The [viewer container validation](reports/viewer-container-validation-20261009.json)
   passed on Linux CI. The [approved GCS receipt](reports/gcs-recovery-20261009.json)
   and [independent audit](reports/gcs-recovery-validation-20261009.json) confirm
   exact restoration after generated-local removal. The private deployment
   proposal is documented; deployment is not required for this gate.
4. **Trustworthy model:** enough independent training families and seeds, paired
   baseline comparisons, conservation, rollout behavior, uncertainty, and explicit
   cases where the model loses.
5. **Request router:** verified exact reuse, then a compatible model only inside
   its calibrated uncertainty boundary, otherwise a stored solver result. Measure
   routing decisions, latency, and error against the solver.
6. **Provenance and presentation:** actor provenance, concise architecture/readme,
   a short demo, limitations, and the stepbook as an implementation appendix.

Each phase adds tests and passes lint and the full test suite. Study code must be
committed on `develop` before measurement; new output directories preserve raw
evidence outside Git. Retained reports support measured documentation claims.
Append the phase outcome to `docs/progress.md`, then stop for user review.
Distributed remote workers, three-dimensional Navier–Stokes, a hosted dashboard,
and an LLM planner are outside this delivery scope.

Flowstate's contribution is a reproducible scientific investigation system: generate experiments, execute numerical or learned models, preserve fields and provenance, query failures, and use that evidence to choose the next experiment. The first deliverable is a small working CPU engine that makes this loop inspectable.

## Current prototype boundary

Version 0.2 adds a manufactured Darcy problem, numerical refinement reports, streamed field output, dataset ETL, a PDEBench Burgers import contract, CPU FNO/PINN baselines, an S3 artifact mirror, a research graph, and a deterministic proposal/execution loop to the original numerical engine. The [stepbook](../STEPBOOK.md) explains the implementation, processing loops, and validation. Numerical checks and limits are specified in [scientific-validation.md](scientific-validation.md).

The table below preserves the original 15-day plan as acceptance criteria, not elapsed development time. Most capabilities now have a local prototype and tests. Version 0.3 adds an [isolated local concurrency study](reports/local-scaling-0.3.json) over one, two, and four workers. The scale criterion remains partial: a live cloud deployment, broader workloads, and distributed scheduling have not been measured. The S3 protocol is tested with an emulator. PDEBench ingestion initially used representative HDF5 fixtures; versions 0.4 and 0.5 add the small public-data studies described below. The complete local demonstration passed; its [measured report](reports/prototype-0.2.json) and stepbook distinguish executed evidence from code completion, including the PINN's weak short-budget result.

| Days | Deliverable | Exit criterion |
| --- | --- | --- |
| 1–3 | CPU reference engine, configuration schema, deterministic initial conditions, CLI, and Burgers/2D Navier–Stokes smoke experiments | Known solutions and conservation/incompressibility checks pass; invalid inputs and unstable timesteps fail intelligibly. |
| 4–5 | Local lake, provenance, sweep generation, queryable metrics, failure records, and lineage | An interrupted sweep can be rerun without overwriting complete results; a query retrieves experiments and opens the corresponding fields. |
| 6 | Numerical validation report and resource accounting | Spatial/time refinement, runtime, memory, and storage results are recorded; practical grid/sweep limits are stated. |
| 7–8 | One small Burgers FNO baseline with training, checkpointing, and evaluation | A trajectory-disjoint split, training-only normalization, and paired numerical-reference metrics reproduce from a manifest. This gates expansion to Navier–Stokes learning. |
| 9 | Darcy adapter and PDEBench import contract | One elliptic Darcy case passes a manufactured-solution or validated-reference check; imported datasets retain license, version, units, splits, and source provenance. |
| 10 | A small PINN comparison, conditional on baseline readiness | Its boundary conditions, training budget, residual sampling, reference, and generalization setting are explicit. Postpone if the FNO baseline is not trustworthy. |
| 11 | Storage and execution scale experiment | Measure local chunk access, bounded-memory output, process concurrency, and one object-storage backend before declaring distributed support. No credentials enter artifacts. |
| 12 | Research object and relation schema | Equation, Solver, Experiment, InitialCondition, BoundaryCondition, Dataset, Model, Checkpoint, Metric, Anomaly, Hypothesis, and Finding have stable identities and documented links. |
| 13 | Evidence-based next-experiment proposals | A deterministic policy proposes a bounded sweep near observed failures or uncertainty, citing the run IDs and objective behind each suggestion. |
| 14 | Independent scientific review and targeted repair | Review checks leakage, solver assumptions, reference quality, failure handling, and reproduction on a clean environment. Open issues are recorded as limitations. |
| 15 | Reproducible demonstration and release notes | From a clean checkout: run a small sweep, query an anomaly, inspect lineage, evaluate the available baseline, and generate an auditable next-step proposal. |

Version 0.4 adds bounded acquisition of an attributed subset from the official
PDEBench Burgers release, verified conversion into canonical Zarr, and a fixed
three-seed FNO plus per-instance PINN study. The [workflow](steps/06-public-data.md)
preserves full source trajectories and distinguishes received-range/subset hashes
from the unverified publisher checksum. It remains a small-sample demonstration;
larger benchmark studies, a live cloud deployment, and distributed scheduling
remain beyond the measured local prototype. The
[executed public-data report](reports/public-burgers-0.4.json) includes 24 complete
trajectories, all three FNO seeds, and one PINN case. Lower prediction error came
with weaker mean-velocity conservation, giving the next model work a measured target.

## Investigation model

Version 0.5 adds a durable **local** SQLite job queue and optional mean-preserving
FNO updates. The [recovery check](reports/local-recovery-0.5.json) exercised two
workers and a process exit after result publication. The
[fresh-cohort comparison](reports/conservation-0.5.json) reduced mean drift to about
1e-7, but retained large long-rollout errors for seed 2. Conservation alone does
not resolve model reliability. Version 0.6 adds a
[verified offline experiment viewer](steps/08-research-viewer.md), including
energy curves, final fields, search, filters, and provenance. A hosted service and
model/graph visualizations remain separate work.

The initial parent-experiment link is the seed of a research graph. Extend it with typed, versioned relations rather than overloading free-form notes:

```text
Experiment --solves--> Equation
Experiment --uses--> Solver
Experiment --starts-from--> InitialCondition
Experiment --has-boundary--> BoundaryCondition
Experiment --produces--> Dataset
Experiment --derived-from--> Experiment
Model --trained-on--> Dataset
Checkpoint --belongs-to--> Model
Metric --measures--> Experiment
Anomaly --observed-in--> Experiment
Finding --cites--> Experiment
Finding --supports/contradicts--> Hypothesis
```

A changed timestep and a changed viscosity are different relations with different scientific implications. Store the parameter delta and rationale. An anomaly's detection rule, threshold, physical time, and evidence belong in its record. Supporting or contradicting a hypothesis requires a stated test and interpretation, not just a graph edge inferred by an agent.

This object-and-link approach is inspired by the documented [Palantir Ontology concepts](https://www.palantir.com/docs/foundry/ontology/core-concepts), which map datasets and models into objects, properties, links, and actions. Flowstate applies the pattern to scientific evidence; no Palantir dependency is required.

## Scale, automation, and what comes later

Start by measuring one run. For example, 10,000 runs × 1,000 saved frames × 512² cells × one float32 scalar is approximately 10.49 TB before compression and overhead. Three scalar fields need approximately 31.46 TB. Saving fewer frames, choosing chunks around access patterns, streaming output, and retaining selected derived quantities are experimental-design decisions, not substitutes for validation. Integrator timesteps and saved frames are separate counts.

The S3 and native GCS mirrors implement conditional immutable publication and verified download. A [live GCS round trip](reports/gcs-20261004.json) uploaded one small Burgers experiment, verified reuse, restored every file byte for byte, and queried the restored metadata. S3 remains tested with an emulator. Optional local mid-trajectory solver checkpoints passed the [committed-source process-kill proof](reports/checkpoint-recovery.json), with exact restored scientific arrays and counted skipped integration work. Representative cloud throughput/cost measurements, remote workers, distributed scheduling, a graph database, a hosted multi-user dashboard, and an LLM research planner remain future work. Model training already supports optimizer/RNG checkpoint resumption. A local directory and process pool do not establish distributed support. Local prototype data should stay outside Git; commit source, schemas, configurations, documentation, and small deliberate fixtures.

The implemented researcher uses an auditable policy: prioritize failures and flags, propose timestep or grid refinement, preserve comparison conditions, and attach the supporting experiments. Its budgets bound run count and integration steps, not elapsed time or cloud spend. A later language-model planner can formulate hypotheses and explanations, while execution remains constrained by explicit budgets and validation gates. Repeated observations should earn confidence through independently reproducible evidence.

The [PDEBench benchmark](https://arxiv.org/abs/2210.07182) and [Fourier Neural Operator paper](https://arxiv.org/abs/2010.08895) provide established tasks and methods to integrate and compare. Flowstate's intended novelty is the orchestration, data lineage, queryable evidence, and experiment-selection workflow around those methods.
