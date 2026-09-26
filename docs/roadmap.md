# Flowstate: experiment engine first

Flowstate's contribution is a reproducible scientific investigation system: generate experiments, execute numerical or learned models, preserve fields and provenance, query failures, and use that evidence to choose the next experiment. The first deliverable is a small working CPU engine that makes this loop inspectable.

## Current prototype boundary

Version 0.2 adds a manufactured Darcy problem, numerical refinement reports, streamed field output, dataset ETL, a PDEBench Burgers import contract, CPU FNO/PINN baselines, an S3 artifact mirror, a research graph, and a deterministic proposal/execution loop to the original numerical engine. The [stepbook](../STEPBOOK.md) explains the implementation, processing loops, and validation. Numerical checks and limits are specified in [scientific-validation.md](scientific-validation.md).

The table below preserves the original 15-day plan as acceptance criteria, not elapsed development time. Most capabilities now have a local prototype and tests. Version 0.3 adds an [isolated local concurrency study](reports/local-scaling-0.3.json) over one, two, and four workers. The scale criterion remains partial: a live cloud deployment, broader workloads, and distributed scheduling have not been measured. The S3 protocol is tested with an emulator. PDEBench ingestion is tested with representative HDF5 fixtures; no public benchmark result is claimed. The complete local demonstration passed; its [measured report](reports/prototype-0.2.json) and stepbook distinguish executed evidence from code completion, including the PINN's weak short-budget result.

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

## Investigation model

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

The S3 mirror implements conditional immutable publication and verified download; deploying a live bucket and measuring cloud behavior remain future work. Remote workers, mid-trajectory solver checkpoints, distributed scheduling, a graph database, a visual dashboard, and an LLM research planner are also future capabilities. Model training already supports optimizer/RNG checkpoint resumption. A local directory and process pool do not establish distributed support. Local prototype data should stay outside Git; commit source, schemas, configurations, documentation, and small deliberate fixtures.

The implemented researcher uses an auditable policy: prioritize failures and flags, propose timestep or grid refinement, preserve comparison conditions, and attach the supporting experiments. Its budgets bound run count and integration steps, not elapsed time or cloud spend. A later language-model planner can formulate hypotheses and explanations, while execution remains constrained by explicit budgets and validation gates. Repeated observations should earn confidence through independently reproducible evidence.

The [PDEBench benchmark](https://arxiv.org/abs/2210.07182) and [Fourier Neural Operator paper](https://arxiv.org/abs/2010.08895) provide established tasks and methods to integrate and compare. Flowstate's intended novelty is the orchestration, data lineage, queryable evidence, and experiment-selection workflow around those methods.
