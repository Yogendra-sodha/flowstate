"""Bounded, reproducible refinement studies with explicit measurement scope."""

from __future__ import annotations

import json
import math
import platform
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from flowstate.darcy import normalize_darcy_config, solve_darcy
from flowstate.numerics import normalize_config, solve


def _cole_hopf(n: int, viscosity: float, amplitude: float, final_time: float) -> np.ndarray:
    """Independent periodic heat-equation solution for sinusoidal Burgers data."""
    fine_n = 1536
    if fine_n % n:
        raise ValueError("Reference grid must be divisible by requested Burgers grid")
    phase = np.arange(fine_n) * (2 * math.pi / fine_n)
    wave = np.fft.fftfreq(fine_n) * fine_n
    theta0 = np.exp(amplitude * np.cos(phase) / (2 * viscosity))
    theta_hat = np.fft.fft(theta0) * np.exp(-viscosity * wave**2 * final_time)
    theta = np.fft.ifft(theta_hat).real
    theta_x = np.fft.ifft(1j * wave * theta_hat).real
    return (-2 * viscosity * theta_x / theta)[:: fine_n // n]


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _orders(errors: list[float]) -> list[float]:
    if any(error <= 0 or not math.isfinite(error) for error in errors):
        return []
    return [
        math.log2(first / second) for first, second in zip(errors[:-1], errors[1:], strict=True)
    ]


def _refinement_group(names: list[str], errors: list[float], order: int, reference: str) -> dict:
    observed = _orders(errors)
    return {
        "cases": names,
        "errors": errors,
        "error_convention": "spatial_RMS_except_Darcy_uses_physical_grid_L2",
        "expected_order": order,
        "observed_orders": observed,
        "reference": reference,
        "passed": len(observed) == len(errors) - 1
        and all(order - 0.3 <= value <= order + 0.3 for value in observed),
    }


def run_validation(output: Path | str) -> dict:
    """Write a JSON report plus NPZ artifacts for six small refinement studies.

    ``output`` is the report file, e.g. ``results/validation.json``; arrays are
    stored in a sibling ``validation-artifacts`` directory. Existing outputs
    are refused. Failures of numerical criteria are recorded with passed=False;
    infrastructure or numerical exceptions propagate instead of faking a pass.

    Each solver call gets its own tracemalloc session. Peak traced allocations
    are not process RSS and can omit native BLAS/SuperLU/FFT memory. Persistence,
    analytic references and the report itself lie outside each solver profile.
    """
    output = Path(output)
    if output.suffix.lower() != ".json":
        raise ValueError("Validation output must be a .json file path")
    artifacts = output.with_name(f"{output.stem}-artifacts")
    if output.exists() or artifacts.exists():
        raise FileExistsError("Validation output/artifact path already exists; choose a new path")
    if tracemalloc.is_tracing():
        raise RuntimeError("Run validation outside an existing tracemalloc session")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    started = time.perf_counter()
    runs = []
    studies = {}

    def run_case(name, config):
        is_darcy = config.get("equation") == "darcy2d"
        canonical = normalize_darcy_config(config) if is_darcy else normalize_config(config)
        solver = solve_darcy if is_darcy else solve
        tracemalloc.start()
        case_started = time.perf_counter()
        try:
            result = solver(canonical)
            solver_seconds = time.perf_counter() - case_started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        arrays = {"times": result.times}
        for prefix, values in (
            ("field", result.fields),
            ("diagnostic", result.diagnostics),
            ("coordinate", result.coordinates),
        ):
            arrays.update({f"{prefix}_{key}": value for key, value in values.items()})
        artifact = artifacts / f"{name}.npz"
        persistence_started = time.perf_counter()
        with artifact.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        persistence_seconds = time.perf_counter() - persistence_started
        record = {
            "name": name,
            "config": canonical,
            "solver": result.metadata["solver"],
            "solver_wall_seconds": solver_seconds,
            "persistence_wall_seconds": persistence_seconds,
            "peak_traced_allocation_bytes": int(peak),
            "raw_array_bytes": sum(array.nbytes for array in arrays.values()),
            "stored_output_bytes": artifact.stat().st_size,
            "artifact": artifact.relative_to(output.parent).as_posix(),
            "diagnostics_final": {
                key: float(values[-1]) for key, values in result.diagnostics.items()
            },
        }
        runs.append(record)
        return result

    # Spatial error: independent Cole--Hopf solution avoids treating another
    # instance of our finite-difference implementation as ground truth.
    names, errors = [], []
    for n in (24, 48, 96):
        name = f"burgers-spatial-{n}"
        result = run_case(
            name,
            {
                "grid_size": n,
                "viscosity": 0.2,
                "amplitude": 0.25,
                "dt": 0.00025,
                "steps": 400,
                "save_every": 400,
            },
        )
        reference = _cole_hopf(n, 0.2, 0.25, 0.1)
        names.append(name)
        errors.append(_rms(result.fields["velocity"][-1] - reference))
    studies["burgers_spatial"] = _refinement_group(
        names, errors, 2, "Cole-Hopf heat solution on 1536 periodic points at t=0.1"
    )

    # Temporal error is isolated on a fixed spatial grid. Comparing against the
    # continuum here would confuse the much larger spatial truncation error.
    config = {"grid_size": 16, "viscosity": 0.2, "amplitude": 0.5, "save_every": 10000}
    reference = run_case("burgers-temporal-reference", {**config, "dt": 0.0005, "steps": 800})
    reference_velocity = reference.fields["velocity"][-1].copy()
    names, errors = [], []
    for dt in (0.08, 0.04, 0.02):
        name = f"burgers-temporal-{dt:g}"
        result = run_case(name, {**config, "dt": dt, "steps": round(0.4 / dt)})
        names.append(name)
        errors.append(_rms(result.fields["velocity"][-1] - reference_velocity))
    studies["burgers_temporal"] = _refinement_group(
        names, errors, 4, "Same 16-point spatial discretization with dt=0.0005 at t=0.4"
    )

    # Taylor--Green occupies a single resolved Fourier mode. Grid refinement
    # verifies accuracy, but cannot legitimately estimate a spatial order.
    names, errors = [], []
    for n in (8, 16, 32):
        name = f"taylor-green-spatial-{n}"
        result = run_case(
            name,
            {
                "equation": "navier_stokes2d",
                "grid_size": n,
                "amplitude": 0.5,
                "viscosity": 0.4,
                "dt": 0.002,
                "steps": 200,
                "save_every": 200,
            },
        )
        exact = result.fields["u"][0] * np.exp(-0.8 * 0.4)
        names.append(name)
        errors.append(_rms(result.fields["u"][-1] - exact))
    studies["taylor_green_spatial"] = {
        "cases": names,
        "errors": errors,
        "error_convention": "spatial_RMS_u",
        "observed_orders": None,
        "reference": "Exact Taylor-Green viscous decay at t=0.4",
        "interpretation": (
            "Single Fourier mode is resolved on every grid; no spatial order inferred"
        ),
        "passed": all(error < 1e-10 for error in errors),
    }
    names, errors = [], []
    for dt in (0.08, 0.04, 0.02):
        name = f"taylor-green-temporal-{dt:g}"
        result = run_case(
            name,
            {
                "equation": "navier_stokes2d",
                "grid_size": 8,
                "amplitude": 0.5,
                "viscosity": 0.4,
                "dt": dt,
                "steps": round(0.4 / dt),
                "save_every": 100,
            },
        )
        exact = result.fields["u"][0] * np.exp(-0.8 * 0.4)
        names.append(name)
        errors.append(_rms(result.fields["u"][-1] - exact))
    studies["taylor_green_temporal"] = _refinement_group(
        names, errors, 4, "Exact Taylor-Green viscous decay at t=0.4"
    )

    for label, contrast in (("constant", 0.0), ("smooth", 0.5)):
        names, errors = [], []
        for n in (17, 33, 65):
            name = f"darcy-{label}-{n}"
            result = run_case(name, {"equation": "darcy2d", "grid_size": n, "contrast": contrast})
            names.append(name)
            errors.append(float(result.diagnostics["pressure_l2_error"][0]))
        studies[f"darcy_{label}_spatial"] = _refinement_group(
            names, errors, 2, "Manufactured sin(pi*x/L) sin(pi*y/L) pressure with analytic forcing"
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "passed": all(study["passed"] for study in studies.values()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "measurement_scope": {
            "wall_time": (
                "Each solver call, separately from NPZ persistence; total includes references"
            ),
            "memory": (
                "Peak allocations traced by Python tracemalloc per solver call; NOT process RSS. "
                "Native FFT/BLAS/SuperLU allocations may be omitted. Prior arrays, reference "
                "calculations, persistence, report writing and parallel workers are excluded."
            ),
            "stored_output_bytes": (
                "Actual compressed NPZ artifact file sizes; excludes JSON report"
            ),
            "scale": "Bounded serial CPU smoke studies, not a distributed throughput benchmark",
        },
        "studies": studies,
        "runs": runs,
        "total_wall_seconds_before_report_write": time.perf_counter() - started,
        "peak_traced_allocation_bytes_max_case": max(
            run["peak_traced_allocation_bytes"] for run in runs
        ),
        "stored_output_bytes": sum(run["stored_output_bytes"] for run in runs),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
