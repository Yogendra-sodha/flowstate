# Scientific validation

Flowstate is an experiment and evidence system. A `completed` run means a particular numerical configuration finished; its diagnostic flags may still require review. Completion is not a certificate of convergence or a theorem. The Clay problem concerns three-dimensional incompressible Navier–Stokes existence and smoothness. This milestone studies a much narrower two-dimensional periodic problem. Numerical instability is not evidence of a mathematical singularity. See [Fefferman's official problem statement](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf).

## Implemented numerical scope

The CPU experiment engine targets viscous, unforced equations on uniform periodic grids. Configurations also accept zero viscosity for smooth inviscid experiments; the Burgers discretization is not suitable for capturing shocks.

| Equation | State and method | Deliberate boundary |
| --- | --- | --- |
| Viscous Burgers, one dimension | Velocity; conservative finite differences and classical explicit RK4 | No walls, forcing, inviscid shock benchmark, or adaptive mesh |
| Incompressible Navier–Stokes, two dimensions | Scalar vorticity; Fourier derivatives, dealiased nonlinear term, explicit RK4; velocity reconstructed from vorticity | Zero mean velocity and compatible zero mean vorticity; no pressure output or three-dimensional flow |
| Steady Darcy, two dimensions | Positive variable permeability, harmonic face fluxes, sparse direct solve | Manufactured forcing and pressure on a square with zero Dirichlet boundaries; a validation family, not a general Darcy dataset |

The vorticity formulation enforces incompressibility through velocity reconstruction. It is not a separately implemented pressure-projection solver. Pressure could later be recovered from a Poisson equation with an explicit gauge, but absent pressure data must never be synthesized or labelled as measured.

The data path is local Zarr fields, Parquet metadata/metrics, DuckDB queries, immutable finalized results, and parent-experiment lineage. Resuming a sweep reuses both completed and recorded failed configurations after integrity verification; a new attempt creates a distinct result. A crash before publication reruns from the initial state. There are no timestep checkpoints. Failed solves retain an error record, but no partial fields or structured last-valid time.

The solvers use float64. Buffered mode retains saved fields; `--stream` writes each saved field during integration and keeps only scalar diagnostic histories alongside solver working arrays. A 256 MiB estimated saved-output limit bounds accepted unsteady configurations, not total process memory. Darcy bounds its sparse grid separately. Neither chunked storage nor streaming establishes distributed scalability. See the [Darcy/validation chapter](steps/02-darcy-validation.md) and [stepbook](../STEPBOOK.md) for measured scope.

## Equations and conventions

Let angle brackets denote spatial averages over a periodic domain of side length `L`. These equations define the intended continuous problem; discrete diagnostics approximate them.

**Burgers:**

```text
u_t + (u² / 2)_x = ν u_xx
M = <u>                        dM/dt = 0
E = <u²> / 2                   dE/dt = -ν <u_x²>
```

Conservation of mean velocity does not by itself guarantee conservation or correct dissipation of energy. In particular, a centered discretization of the conservative flux can conserve the mean while introducing energy error. Interpret results against the actual discrete flux and refinement evidence. A dissipative flux can introduce additional numerical diffusion, which must be separated from physical viscosity when evaluating accuracy.

**Two-dimensional Navier–Stokes:**

```text
ω_t + u ω_x + v ω_y = ν Δω
ω = v_x - u_y
u = ψ_y, v = -ψ_x, -Δψ = ω
E = <u² + v²> / 2             dE/dt = -ν <ω²>
Z = <ω²> / 2                  dZ/dt = -ν <|∇ω|²>
```

Both kinetic energy and enstrophy should decay for these unforced viscous equations, up to documented discretization and roundoff error. “Energy conservation violation” is therefore an ambiguous alarm: report excess energy growth or a dissipation-balance defect instead. A future forced problem needs the power-injection term in its energy budget.

The inverse Laplacian cannot recover the constant velocity mode from vorticity. Set that mode explicitly to zero for this milestone, and remove only documented incompatible initial mean vorticity. Fourier wavenumbers are `2π m/L`; derivatives and decay rates must retain this domain factor. Quadratic dealiasing must exclude the `|m| = N/3` boundary when `N` is divisible by three, and apply consistently to the evolved state and nonlinear evaluation.

Always distinguish step number, saved-frame index, physical time, and wall-clock duration. A failure at step 700 means `t = 700 dt` only for a constant timestep without a shortened terminal step. Compare trajectories at common physical times.

Reynolds number is a derived convention, `Re = U_ref L_ref / ν`, not an independent replacement for viscosity. This engine reports `configured amplitude × domain_length / viscosity` and records that convention; zero viscosity gives a null Reynolds value. A scale based on initial RMS speed and the domain side is not interchangeable with one based on peak speed, forcing scale, or evolving RMS speed. A bare filter `Re > 5000` is meaningful only within the same convention.

## Evidence required for numerical trust

These are validation criteria, not claims that every possible parameter regime is validated. The test suite and individual run records provide the implemented evidence.

| Check | Evidence and interpretation |
| --- | --- |
| Known solution | Constant Burgers data remain constant. A periodic Cole–Hopf solution tests nonlinear Burgers accuracy. A single Laplacian-eigenmode Taylor–Green vorticity field decays as `exp(-2 ν k² t)` and tests signs, units, and time integration. |
| Spatial refinement | Compare the same initial function at several grid sizes at a fixed physical time; use sufficiently small `dt` so spatial error dominates. A first-order dissipative flux is not expected to exhibit second-order convergence. |
| Temporal refinement | Fix a sufficiently resolved grid and halve `dt`; RK4 has fourth-order global accuracy in the smooth asymptotic regime until spatial error or roundoff dominates. A single successful trajectory is not a convergence study. |
| Conservation and dissipation | Track mean velocity for Burgers; track mean vorticity, kinetic energy, and enstrophy for Navier–Stokes. Evaluate finite differences in time against viscous dissipation before inferring a budget defect. |
| Incompressibility | Evaluate Fourier divergence of reconstructed velocity near roundoff. This checks reconstruction, not the independent accuracy of the nonlinear evolution. |
| Stability | Check advective and diffusive timestep restrictions, finite values, and reported energy-growth thresholds. Explicit RK4 still requires a timestep restriction; nonlinear intermediate stages can fail even if initial data satisfy a CFL estimate. |
| Reproducibility | Re-run a configuration with its seed and recorded code/environment, compare fields and metrics within stated tolerances, and verify output identity/provenance. Bitwise equivalence across hardware is not assumed. |

A useful exact Burgers family on a period `L` is `u = -2ν ∂x log φ`, with `φ = 1 + a exp(-ν k²t) cos(kx)`, `k = 2π/L`, and `|a| < 1`. This yields a nontrivial smooth nonlinear test without mistaking a diffusing sine wave for an exact Burgers solution.

The current [numerical tests](../tests/test_numerics.py) exercise domain-scaled Taylor–Green decay, fourth-order temporal refinement, second-order Burgers spatial refinement against an independent Cole–Hopf heat-equation reference, random-flow dissipation, incompressibility, seed reproducibility, zero flow, and unstable-input rejection. Stability bounds are checked at each RK4 stage. These small resolved cases establish a baseline; new parameter regimes still need their own resolution study.

Saved-frame diagnostics may miss short events between outputs. Distinguish a stability check evaluated during integration from a metric recorded every several steps. Current summaries flag sampled energy increases and absolute mass, circulation, or divergence drift. They do not evaluate a full PDE residual or a viscous energy-balance residual. A failed run records its exception; ordinary I/O or process failures are not physical anomalies. Structured failure-time evidence is a future improvement.

## Reproducible experiments and fair comparisons

Each result needs the resolved configuration, initial-condition definition and seed, equation and boundary condition, solver/version, precision, code commit and dirty-state information, dependency versions, machine description, run status, physical time coordinates, and parent relation. The engine records a package-source fingerprint and git dirty state, but does not snapshot source code. Exact reruns therefore require retained source or a clean committed revision. A fingerprint identifies changes; it cannot reconstruct their contents. Seeds alone are also insufficient if the initial-condition algorithm changes.

Completed artifacts must not be silently overwritten by a rerun. Failed attempts remain identifiable through their configuration and attempt number; an explicit parent relation can additionally link a retry to its previous attempt. Catalog rows become visible after atomic publication of the artifact directory. SHA-256 manifests detect accidental artifact changes, but the manifests are unsigned: this is application-level immutability and integrity verification, not tamper-proof storage.

Implemented FNO/PINN baselines and future extensions must apply these rules:

1. Split by entire initial-condition trajectories and related experiment families before extracting windows. Adjacent timesteps from one trajectory must not enter both training and test sets.
2. Fit normalizers and select model hyperparameters using training/validation only. Version the split, preprocessing, model checkpoint, and training dataset hashes.
3. Measure one-step and autoregressive rollout error separately at matching physical times. Use the same initial conditions, boundary conditions, units, and interpolation/restriction rules across methods.
4. Define the reference solution and estimate its error using refinement or an independent validated solver. “FNO error exceeds spectral-solver error” requires both errors against that common reference; the spectral result is not exact ground truth by declaration.
5. Report field norms, conservation/dissipation metrics, failure counts, and cost together. Include training cost and warmup when making speed claims, and separate in-distribution tests from held-out viscosities, resolutions, or domains.
6. Keep numerical anomalies, statistical outliers, and scientific hypotheses distinct. A proposal generated from previous runs is a candidate to test, not an established finding.

The [PDEBench paper](https://arxiv.org/abs/2210.07182) supplies an existing scientific-ML benchmark with simulation data, generation code, and FNO/U-Net/PINN baselines. The [original FNO paper](https://arxiv.org/abs/2010.08895) studies function-space operators on Burgers, Darcy, and Navier–Stokes problems. The prototype implements a small Burgers FNO/PINN workflow and a provenance-aware Burgers HDF5 import contract. It does not reproduce their published benchmarks or claim a novel neural operator. The [learning chapter](steps/04-learning-baselines.md) documents supervision, tensor shapes, optimizer/checkpoint loops, and comparison limits.
