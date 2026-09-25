"""Steady manufactured Darcy benchmarks with conservative harmonic face fluxes.

The square grid includes both Dirichlet boundaries, unlike the endpoint-excluded
periodic grids in ``numerics``. Fields have shape (1, y, x); the single time value
zero denotes a steady solution, not an evolution step.
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral, Real

import numpy as np
from scipy.sparse import coo_array, csc_array
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from flowstate.numerics import SimulationResult

MAX_DARCY_GRID = 513
_KEYS = {"equation", "grid_size", "domain_length", "contrast"}


def normalize_darcy_config(config: dict) -> dict:
    """Return JSON-native benchmark parameters, rejecting unsupported physics."""
    if not isinstance(config, dict):
        raise ValueError("Darcy config must be a dictionary")
    unknown = set(config) - _KEYS
    if unknown:
        raise ValueError(
            f"Unknown Darcy configuration keys: {', '.join(sorted(map(str, unknown)))}"
        )
    equation = config.get("equation", "darcy2d")
    if not isinstance(equation, str) or equation != "darcy2d":
        raise ValueError("Darcy equation must be 'darcy2d'")
    grid = config.get("grid_size", 33)
    if isinstance(grid, (bool, np.bool_)) or not isinstance(grid, Integral):
        raise ValueError("grid_size must be an integer")
    if not 5 <= grid <= MAX_DARCY_GRID:
        raise ValueError(f"grid_size must be between 5 and {MAX_DARCY_GRID}")
    values = {}
    for name, default in (("domain_length", 1.0), ("contrast", 0.5)):
        value = config.get(name, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        try:
            value = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        values[name] = value
    if values["domain_length"] <= 0:
        raise ValueError("domain_length must be positive")
    # Reject domains that make either derivative scaling or error norms overflow.
    length = values["domain_length"]
    spacing = length / (int(grid) - 1)
    if not 1e-140 <= spacing <= 1e140:
        raise ValueError("domain/grid spacing is outside the supported float64 range")
    if abs(values["contrast"]) >= 0.9:
        raise ValueError("contrast must satisfy abs(contrast) < 0.9")
    return {"equation": "darcy2d", "grid_size": int(grid), **values}


def _manufactured_fields(coordinate: np.ndarray, length: float, contrast: float):
    phase_x, phase_y = np.meshgrid(
        math.pi * (coordinate / length), math.pi * (coordinate / length), indexing="xy"
    )
    sx, sy = np.sin(phase_x), np.sin(phase_y)
    cx, cy = np.cos(phase_x), np.cos(phase_y)
    exact = sx * sy
    # Set the analytic Dirichlet value exactly rather than retain sin(pi) noise.
    exact[[0, -1], :] = 0
    exact[:, [0, -1]] = 0
    permeability = 1 + contrast * exact
    k2 = (math.pi / length) ** 2
    forcing = 2 * k2 * permeability * exact - contrast * k2 * (
        cx * cx * sy * sy + sx * sx * cy * cy
    )
    return exact, permeability, forcing


def _assemble_operator(permeability: np.ndarray, spacing: float) -> csc_array:
    """Assemble -div(a grad) on interior nodes with boundary pressure fixed zero."""
    x_faces = (
        2
        * permeability[:, :-1]
        * permeability[:, 1:]
        / (permeability[:, :-1] + permeability[:, 1:])
    )
    y_faces = (
        2
        * permeability[:-1, :]
        * permeability[1:, :]
        / (permeability[:-1, :] + permeability[1:, :])
    )
    west, east = x_faces[1:-1, :-1], x_faces[1:-1, 1:]
    south, north = y_faces[:-1, 1:-1], y_faces[1:, 1:-1]
    interior_width = permeability.shape[0] - 2
    index = np.arange(interior_width**2).reshape(interior_width, interior_width)
    diagonal = (west + east + south + north).ravel()
    # Each interior face contributes two equal off-diagonal entries. Faces next
    # to the boundary contribute only to the diagonal because p_boundary = 0.
    left, right = index[:, :-1].ravel(), index[:, 1:].ravel()
    down, up = index[:-1, :].ravel(), index[1:, :].ravel()
    horizontal, vertical = -east[:, :-1].ravel(), -north[:-1, :].ravel()
    rows = np.concatenate((index.ravel(), left, right, down, up))
    columns = np.concatenate((index.ravel(), right, left, up, down))
    data = np.concatenate((diagonal, horizontal, horizontal, vertical, vertical)) / spacing**2
    return coo_array((data, (rows, columns)), shape=(index.size, index.size)).tocsc()


def solve_darcy(config: dict) -> SimulationResult:
    """Solve a positive, smooth manufactured Darcy problem with sparse direct LU.

    residual_l2 is the interior RMS algebraic residual in physical PDE units.
    pressure_l2_error is h * ||p - p_exact||_2, a grid quadrature approximation
    to the physical L2 norm. A tiny algebraic residual alone is not evidence of
    a small discretization error; use spatial refinement to assess that error.
    """
    config = normalize_darcy_config(config)
    n, length = config["grid_size"], config["domain_length"]
    spacing = length / (n - 1)
    coordinate = np.linspace(0, length, n, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            exact, permeability, forcing = _manufactured_fields(
                coordinate, length, config["contrast"]
            )
            operator = _assemble_operator(permeability, spacing)
            rhs = forcing[1:-1, 1:-1].ravel()
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                interior = spsolve(operator, rhs, permc_spec="MMD_AT_PLUS_A", use_umfpack=False)
            if not np.isfinite(interior).all():
                raise ValueError("Darcy sparse solve produced non-finite pressure")
            pressure = np.zeros((n, n), dtype=np.float64)
            pressure[1:-1, 1:-1] = interior.reshape(n - 2, n - 2)
            residual = operator @ interior - rhs
            # hypot.reduce avoids avoidable overflow from squaring physical residuals.
            residual_l2 = float(np.hypot.reduce(residual) / math.sqrt(residual.size))
            pressure_error = float(spacing * np.linalg.norm(pressure - exact))
        except (FloatingPointError, MatrixRankWarning) as exc:
            raise ValueError("Darcy numerical failure; rescale domain or lower grid_size") from exc
    return SimulationResult(
        times=np.array([0.0]),
        fields={
            "pressure": pressure[None],
            "permeability": permeability[None],
            "forcing": forcing[None],
        },
        diagnostics={
            "residual_l2": np.array([residual_l2]),
            "pressure_l2_error": np.array([pressure_error]),
        },
        coordinates={"x": coordinate, "y": coordinate.copy()},
        metadata={
            "equation": "darcy2d",
            "solver": "harmonic_face_flux_sparse_direct",
            "precision": "float64",
            "boundary_conditions": "zero_dirichlet_pressure",
            "field_axes": ["time", "y", "x"],
            "coordinate_endpoints": "both_included",
            "steady": True,
            "time_convention": "single_zero_steady_state_marker",
            "benchmark": "manufactured_sine_pressure",
            "permeability_convention": "1 + contrast * sin(pi*x/L) * sin(pi*y/L)",
            "forcing_convention": "analytic -div(permeability * grad(exact_pressure))",
            "residual_convention": "interior_rms_discrete_operator_pressure_minus_forcing",
            "pressure_error_convention": "grid_quadrature_L2_h_times_euclidean_norm",
            "spatial_order": 2,
            "linear_solver": "scipy_superlu",
            "interior_unknowns": (n - 2) ** 2,
            "operator_nonzeros": int(operator.nnz),
            "limitations": (
                "Smooth manufactured scalar permeability only; pressure is fixed across contrasts. "
                "This verifies discretization and is not a useful operator-learning dataset."
            ),
        },
    )
