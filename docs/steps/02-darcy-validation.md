# Step 02 — Add Darcy flow and measure numerical evidence

This chapter explains [`darcy.py`](../../src/flowstate/darcy.py) and
[`validation.py`](../../src/flowstate/validation.py). The first adds a real steady
elliptic solve. The second turns convergence checks into reproducible artifacts
with measured time, memory scope, and file sizes. Together they separate three
questions that a research system must not collapse into one: did the linear
solver finish, did the discretization approximate the PDE, and what resources
did this particular computation consume?

## 1. Choose a problem whose answer is independently known

The Darcy benchmark solves

```text
-div(a(x,y) grad(p(x,y))) = f(x,y),       (x,y) in [0,L]²
p(x,y) = 0,                            on all four boundaries.
```

`p` is pressure, `a` is a positive scalar permeability, and `f` is a source term.
This is a steady problem: there is no timestep, viscosity parameter, or initial
condition to evolve. The current adapter intentionally supports a small
manufactured family, rather than pretending to support arbitrary Darcy inputs.

Start from the desired exact pressure and derive the source term:

```text
k = pi/L
p_exact = sin(kx) sin(ky)
a = 1 + c p_exact
```

The contrast `c` satisfies `abs(c) < 0.9`; because the sine product lies between
zero and one on this square, permeability stays above 0.1. Setting `c=0` gives
constant permeability. A nonzero contrast exercises variable face coefficients.

The forcing is obtained by differentiation, not by applying our discrete matrix
to sampled exact pressure. That distinction matters: constructing the forcing
with the same matrix would make a mistaken discretization reproduce its own
mistake exactly.

```text
laplacian(p_exact) = -2 k² p_exact
grad(a) = c grad(p_exact)

f = -a laplacian(p_exact) - grad(a) dot grad(p_exact)
  = 2 k² a p_exact
    - c k² [cos²(kx) sin²(ky) + sin²(kx) cos²(ky)]
```

`_manufactured_fields` evaluates those expressions. It explicitly zeros the
boundary pressure values so that floating-point `sin(pi)` does not turn an exact
Dirichlet boundary into a tiny nonzero value.

One limitation follows directly from the construction: the exact pressure is
the same for every contrast. Permeability and forcing change together. This is
a useful discretization test and a poor operator-learning dataset. A Darcy
learning task needs independent coefficient/source variation and held-out cases
whose solution maps are nontrivial; the current manufactured family does not
establish that capability.

## 2. Validate the actual supported configuration

`normalize_darcy_config` accepts these keys and fills their defaults:

| Key | Default | Meaning |
| --- | --- | --- |
| `equation` | `darcy2d` | The steady benchmark adapter. |
| `grid_size` | `33` | Nodes in each direction, including both boundaries. |
| `domain_length` | `1.0` | Physical side length of the square. |
| `contrast` | `0.5` | Coefficient `c` in the manufactured permeability. |

The function rejects unknown keys. In particular, passing `dt` or `viscosity`
does not silently suggest those parameters affect a steady Darcy result. Grid
sizes must be integers from 5 through 513; booleans are not accepted as integers.
Numerical parameters must be finite. The domain must be positive and its grid
spacing must remain within a supported float64 range. The upper grid bound is a
practical limit on a sparse direct solver, whose factorization can allocate
substantially more memory than the matrix's original five-point stencil.

The returned dictionary uses ordinary Python strings, integers, and floats, so
it can be serialized directly into the experiment record and identity hash.

## 3. Convert local flux balance into a sparse matrix

Let `n = grid_size`, `m = n - 2`, and `h = L/(n-1)`. The full pressure grid is
`n × n`, but only the `m × m` interior values are unknown. Boundary values are
already known to be zero. Flatten the interior in row-major order:

```text
interior index = row * m + column
```

At a face separating neighboring permeabilities `a_left` and `a_right`, use the
harmonic mean:

```text
a_face = 2 a_left a_right / (a_left + a_right)
```

For equal half-cell widths this is the effective coefficient for two scalar
conductances in series. It stays positive for positive coefficients and assigns
the same face coefficient to both neighboring nodes. Smoothness of this
manufactured family supports the expected second-order convergence; this chapter
does not claim the same pointwise accuracy across arbitrary discontinuities.

The discrete equation at one interior node is

```text
[(a_e+a_w+a_n+a_s) p_center
 - a_e p_east - a_w p_west - a_n p_north - a_s p_south] / h²
 = f_center.
```

The diagonal is positive. The neighbor coefficients are negative. An interior
face supplies a pair of equal off-diagonal entries, so the operator is symmetric.
With positive permeability and Dirichlet boundaries it is positive definite.
A face next to the boundary still contributes to the diagonal; its boundary
pressure term contributes zero to the right-hand side. Omitting that diagonal
contribution would impose a different physical boundary condition.

`_assemble_operator` constructs arrays for east/west and north/south face
coefficients, then arrays of matrix rows, columns, and values. It avoids a Python
loop over every node. The coordinate-format sparse array is converted to CSC for
the direct solve. SciPy documents the [sparse coordinate array format](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html)
and [`spsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).

`solve_darcy` solves only for interior pressure. It uses SciPy's SuperLU path
explicitly, rather than changing backends if an optional UMFPACK installation
happens to be present. The interior solution is inserted into a zero-initialized
full grid, preserving the boundary exactly. Non-finite solutions and singular
matrix warnings are treated as failures.

## 4. Return the same experiment result shape without inventing time dynamics

The adapter returns the shared `SimulationResult` dataclass:

| Member | Shape or content |
| --- | --- |
| `times` | `[0.0]`, a steady-state marker. |
| `fields['pressure']` | `(1, n, n)`, axes `(time, y, x)`. |
| `fields['permeability']` | `(1, n, n)`. |
| `fields['forcing']` | `(1, n, n)`. |
| `coordinates['x']`, `coordinates['y']` | `(n,)`, including `0` and `L`. |
| `diagnostics['residual_l2']` | `(1,)`, RMS algebraic residual over interior nodes. |
| `diagnostics['pressure_l2_error']` | `(1,)`, physical grid-quadrature L2 error. |

The coordinates differ from the periodic Burgers and Navier–Stokes solvers,
which exclude the repeated periodic endpoint. Consumers must inspect the
boundary/coordinate metadata rather than assume every solver uses the same grid.

The two error diagnostics answer different questions:

```text
residual_l2 = sqrt(mean((A p_interior - f_interior)²))
pressure_l2_error = h * sqrt(sum((p - p_exact)²))
```

The residual measures whether the computed vector solves the discrete equations.
The pressure error measures disagreement with the exact continuous solution
sampled on the grid. A residual near roundoff can coexist with a visible pressure
error. Grid refinement should reduce the second quantity at the expected rate.

Metadata records the solver, precision, boundary conditions, axis ordering,
manufactured forcing, steady-time convention, norm definitions, matrix size,
number of nonzeros, and modeling limits. It does not label this result as a
time-dependent conservation experiment or manufacture an energy history.

## 5. Test more than one way of being correct

[`tests/test_darcy.py`](../../tests/test_darcy.py) checks three contrasts: `0`,
`0.5`, and `-0.7`. Grids `17`, `33`, and `65` halve `h` at each refinement. If
error behaves as `C h²`, successive errors should fall by roughly a factor of
four. The test also requires a small linear residual and positive permeability.

For constant permeability, the sampled sine pressure is a discrete eigenvector.
Its continuous and discrete eigenvalues are independently known:

```text
lambda_continuous = 2 (pi/L)²
lambda_discrete = 8/h² * sin²(pi/(2(n-1)))
p_discrete = (lambda_continuous/lambda_discrete) * p_exact.
```

That check catches indexing, sign, domain-length, and boundary-assembly mistakes
without merely rerunning the same implementation. A separate domain-rescaling
test verifies that doubling or tripling `L` changes forcing by `1/L²`, leaves the
dimensionless pressure shape unchanged, and scales its physical L2 error by `L`.

The remaining tests check boundary zeros, result shapes, finite JSON metadata,
normalization, and rejection of unsupported configurations. These validate the
manufactured family; they do not validate arbitrary geology, anisotropic
permeability, mixed boundaries, or industrial reservoir models.

## 6. Produce a bounded validation report with retained evidence

`run_validation(output)` takes a JSON **file** path. It creates a sibling
`<report-stem>-artifacts` directory and refuses to overwrite existing output.
For example, from the repository root:

```powershell
uv run python -c "from flowstate.validation import run_validation; print(run_validation('results/validation.json')['passed'])"
```

Each call to the nested `run_case` helper does the following:

1. Normalize one configuration and choose the appropriate solver.
2. Start a fresh allocation trace and wall clock, then solve.
3. Stop the trace and retain the measured solver duration and peak allocation.
4. Store fields, coordinates, times, and diagnostics in a compressed NPZ file.
5. Measure that actual file's size and record its relative path in the report.

The artifact arrays use names such as `field_pressure`, `coordinate_x`, and
`diagnostic_residual_l2`. They can be loaded with `numpy.load(...,
allow_pickle=False)`. Actual compressed bytes are reported separately from the
sum of array `nbytes`, since compressibility depends strongly on the solution.

The report runs 19 small cases across six studies:

| Study | Refinement/reference | Interpretation |
| --- | --- | --- |
| Burgers spatial | `24`, `48`, `96` points against a 1536-point Cole–Hopf heat solution | Approximately second-order spatial error. |
| Burgers temporal | Fixed 16-point grid; `dt=0.08`, `0.04`, `0.02`, against `dt=0.0005` | Approximately fourth-order RK4 error for this smooth case. |
| Taylor–Green spatial | `8`, `16`, `32` points against exact viscous decay | Accuracy check; a single resolved Fourier mode cannot establish spatial order. |
| Taylor–Green temporal | Fixed 8-point grid; `dt=0.08`, `0.04`, `0.02` | Approximately fourth-order RK4 error. |
| Constant Darcy spatial | `17`, `33`, `65` nodes against manufactured pressure | Approximately second-order spatial error. |
| Smooth Darcy spatial | Same grids with contrast `0.5` | Exercises variable harmonic face coefficients. |

For a halved spacing or timestep, observed order is
`log2(coarse_error/fine_error)`. The report compares it with the expected order
within a stated tolerance. It deliberately leaves Taylor–Green spatial order
unset: once its one Fourier mode is resolved, temporal error and roundoff
dominate, so computing a spatial slope would produce an attractive but misleading
number. Taylor–Green also has vanishing nonlinear vorticity advection; the
separate random-flow numerical tests remain necessary to exercise that term.

The Burgers spatial reference uses the Cole–Hopf transformation, which reduces
viscous Burgers with sinusoidal initial velocity to a heat equation. The temporal
study holds the spatial discretization fixed so that spatial error does not mask
time-integration convergence. Its fine-step reference is numerical, not an exact
continuum solution, and is labeled accordingly.

## 7. Interpret the resource numbers within their measured scope

The allocation field is named `peak_traced_allocation_bytes`, not “RAM used.”
Python's [`tracemalloc`](https://docs.python.org/3/library/tracemalloc.html)
traces allocations visible to its hooks. Some extension allocations may be
visible and others may not be. In particular, the measurement is **not process
RSS**, and it can omit native FFT, BLAS, or sparse-factorization allocations.
It also excludes arrays allocated before that case's trace, reference
calculations, compression, report writing, and parallel workers. The API refuses
to replace an already active caller-owned tracing session.

Each case reports solver time and persistence time separately. Overall elapsed
time includes reference calculations but is recorded before writing the report.
Stored-output bytes are measured NPZ file sizes and exclude the JSON report.
These are serial CPU smoke studies, not evidence of distributed throughput or a
machine's maximum sustainable grid size.

The JSON includes configurations, library versions, platform, errors, observed
orders, pass flags, artifact locations, and measurement definitions. Numerical
criterion failures produce `passed: false`; infrastructure or solver exceptions
propagate instead of being converted into fabricated successful results. An
interrupted run may leave partial artifacts, so rerun under a new output name.

Run the chapter's checks with:

```powershell
uv run pytest tests/test_darcy.py tests/test_validation.py -q
uv run ruff check src/flowstate/darcy.py src/flowstate/validation.py tests/test_darcy.py tests/test_validation.py
```

The initial local check passed all 22 tests in these two test files and all Ruff
checks. Its report measured the following observed orders:

| Study | First refinement | Second refinement |
| --- | --- | --- |
| Burgers spatial | 1.987 | 1.997 |
| Burgers temporal | 4.096 | 4.047 |
| Taylor–Green temporal | 4.039 | 4.019 |
| Constant Darcy spatial | 2.002 | 2.001 |
| Smooth Darcy spatial | 2.004 | 2.001 |

Taylor–Green spatial RMS errors were about `3.2e-15`, with no spatial order
reported. That development run wrote 144,509 bytes of NPZ artifacts in about
7.21 seconds including references; its largest per-case traced allocation peak
was 1,177,226 bytes. These are observations from one local run, not fixed
acceptance thresholds or predictions for another machine. Regenerate the report
to obtain the measurements and complete environment record for your checkout.

This gives Flowstate a measured baseline: a researcher can inspect what equation
was solved, compare the errors across refinements, load the retained arrays, and
see precisely which resource claims the measurements support.
