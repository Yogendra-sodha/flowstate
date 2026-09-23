"""Small, reproducible periodic PDE reference solvers.

All fields have saved time as the leading dimension. Burgers arrays use
``(time, x)``; Navier--Stokes arrays use ``(time, y, x)``. Coordinates contain
one-dimensional physical ``x`` (and ``y``) arrays on [0, domain_length), without
the repeated periodic endpoint. Diagnostics are spatial means unless their
names explicitly identify integrals (mass/circulation) or maxima.

The 2-D incompressible solver evolves vorticity and reconstructs zero-mean
velocity spectrally. Pressure is not produced. These are numerical experiments,
not evidence establishing existence or regularity for 3-D Navier--Stokes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

MAX_OUTPUT_BYTES = 256 * 1024 * 1024
_CONFIG_KEYS = {
    "equation",
    "viscosity",
    "grid_size",
    "dt",
    "steps",
    "save_every",
    "seed",
    "domain_length",
    "amplitude",
    "initial_condition",
}


@dataclass
class SimulationResult:
    """Saved solution and diagnostics; every time-dependent array aligns with times."""

    times: np.ndarray
    fields: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    coordinates: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _real(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} number")
    return result


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def normalize_config(config: dict) -> dict:
    """Validate a numerical configuration and return complete JSON-native values.

    Viscosity and amplitude may be zero. A zero-viscosity centered Burgers run
    has no shock capturing; it is suitable only while the solution is smooth.
    Unknown keys are rejected to catch misspelled scientific parameters.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(map(str, unknown)))}")
    equation = config.get("equation", "burgers1d")
    if not isinstance(equation, str) or equation not in ("burgers1d", "navier_stokes2d"):
        raise ValueError("equation must be 'burgers1d' or 'navier_stokes2d'")
    default_ic = "sine" if equation == "burgers1d" else "taylor_green"
    initial_condition = config.get("initial_condition", default_ic)
    if not isinstance(initial_condition, str) or initial_condition not in (default_ic, "random"):
        raise ValueError(f"initial_condition for {equation} must be '{default_ic}' or 'random'")
    canonical = {
        "equation": equation,
        "viscosity": _real("viscosity", config.get("viscosity", 0.05)),
        "grid_size": _integer("grid_size", config.get("grid_size", 32), 8),
        "dt": _real("dt", config.get("dt", 0.001), positive=True),
        "steps": _integer("steps", config.get("steps", 100), 1),
        "save_every": _integer("save_every", config.get("save_every", 10), 1),
        "seed": _integer("seed", config.get("seed", 0), 0),
        "domain_length": _real(
            "domain_length", config.get("domain_length", 2 * math.pi), positive=True
        ),
        "amplitude": _real("amplitude", config.get("amplitude", 1.0)),
        "initial_condition": initial_condition,
    }
    try:
        final_time = canonical["dt"] * canonical["steps"]
        spacing = canonical["domain_length"] / canonical["grid_size"]
        fundamental_wavenumber = 2 * math.pi / canonical["domain_length"]
    except OverflowError as exc:
        raise ValueError("Configuration exceeds floating-point range") from exc
    if not math.isfinite(final_time) or spacing <= 0 or not math.isfinite(fundamental_wavenumber):
        raise ValueError("Simulation duration or domain/grid spacing exceeds floating-point range")
    if (
        not 0 < spacing * spacing < math.inf
        or not 0 < fundamental_wavenumber * fundamental_wavenumber < math.inf
    ):
        raise ValueError("Domain/grid spacing is too extreme for float64 derivatives")
    if not math.isfinite(canonical["amplitude"] * canonical["amplitude"]):
        raise ValueError("amplitude is too large for float64 energy diagnostics")
    if _output_size(canonical) > MAX_OUTPUT_BYTES:
        raise ValueError(
            "Saved output exceeds 256 MiB; lower grid_size/steps or increase save_every"
        )
    return canonical


def _output_size(config: dict) -> int:
    frames = (
        1 + config["steps"] // config["save_every"] + bool(config["steps"] % config["save_every"])
    )
    n = config["grid_size"]
    if config["equation"] == "burgers1d":
        return 8 * (frames * (n + 4) + n)
    return 8 * (frames * (3 * n * n + 6) + 2 * n)


def _check_timestep(rate: float, dt: float, step: int) -> None:
    # A deliberately conservative combined advection/diffusion RK4 bound.
    # This is a stability screen, not a guarantee of resolution or accuracy.
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(
            f"Non-finite stability rate at step {step}; "
            "reduce amplitude/grid_size or rescale domain"
        )
    if dt * rate > 0.5:
        raise ValueError(
            f"CFL/diffusion stability bound exceeded at step {step}: dt={dt:g}; "
            f"use dt <= {0.5 / rate:.6g} (or coarsen grid/lower amplitude/viscosity)"
        )


def _rk4(state: np.ndarray, dt: float, rhs) -> np.ndarray:
    a = rhs(state)
    b = rhs(state + 0.5 * dt * a)
    c = rhs(state + 0.5 * dt * b)
    d = rhs(state + dt * c)
    return state + (dt / 6) * (a + 2 * b + 2 * c + d)


def _base_metadata(config: dict, solver: str, axes: list[str]) -> dict:
    return {
        "equation": config["equation"],
        "solver": solver,
        "precision": "float64",
        "boundary_conditions": "periodic",
        "field_axes": axes,
        "energy_convention": "spatial_mean_of_half_speed_squared",
        "forcing": "none",
        "estimated_output_bytes": _output_size(config),
    }


def solve(config: dict) -> SimulationResult:
    """Run a validated config, including initial/final frames and finite checks.

    Explicit conservative stability bounds are checked at every RK stage.
    Passing them does not establish convergence; refine space and time
    and compare diagnostics before drawing scientific conclusions.
    """
    canonical = normalize_config(config)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if canonical["equation"] == "burgers1d":
                return _burgers(canonical)
            return _navier_stokes(canonical)
        except FloatingPointError as exc:
            raise ValueError(
                "Non-finite numerical state; reduce dt/amplitude or rescale domain"
            ) from exc


def _burgers(config: dict) -> SimulationResult:
    n, length, dt = config["grid_size"], config["domain_length"], config["dt"]
    viscosity = config["viscosity"]
    dx = length / n
    x = np.arange(n, dtype=np.float64) * dx
    phase = 2 * math.pi * (x / length)
    velocity = np.sin(phase)
    if config["initial_condition"] == "random":
        rng = np.random.default_rng(config["seed"])
        velocity = np.zeros(n, dtype=np.float64)
        for mode in range(1, min(4, (n - 1) // 3) + 1):
            velocity += (
                rng.normal() * np.sin(mode * phase) + rng.normal() * np.cos(mode * phase)
            ) / mode**2
        velocity /= np.max(np.abs(velocity))
    velocity *= config["amplitude"]

    def rhs(u):
        rate = float(np.max(np.abs(u))) / dx + 2 * viscosity / dx**2
        _check_timestep(rate, dt, step - 1)
        flux = 0.5 * u * u
        return (
            -(np.roll(flux, -1) - np.roll(flux, 1)) / (2 * dx)
            + viscosity * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        )

    frames, times = [], []
    diagnostic_rows = []

    def save(step):
        frames.append(velocity.copy())
        times.append(step * dt)
        mean = float(np.mean(velocity))
        diagnostic_rows.append((mean, mean * length, float(0.5 * np.mean(velocity**2))))

    save(0)
    for step in range(1, config["steps"] + 1):
        velocity = _rk4(velocity, dt, rhs)
        if not np.all(np.isfinite(velocity)):
            raise ValueError(f"Non-finite velocity at step {step}; reduce dt")
        if step % config["save_every"] == 0 or step == config["steps"]:
            save(step)
    diagnostics = np.asarray(diagnostic_rows, dtype=np.float64)
    metadata = _base_metadata(config, "conservative_centered_finite_difference_rk4", ["time", "x"])
    metadata["spatial_order"] = 2
    metadata["limitations"] = (
        "Centered advection is not shock capturing; underresolved steep gradients can oscillate."
    )
    return SimulationResult(
        times=np.asarray(times, dtype=np.float64),
        fields={"velocity": np.stack(frames)},
        diagnostics={
            name: diagnostics[:, index]
            for index, name in enumerate(("mean_velocity", "mass", "energy"))
        },
        coordinates={"x": x},
        metadata=metadata,
    )


def _navier_stokes(config: dict) -> SimulationResult:
    n, length, dt = config["grid_size"], config["domain_length"], config["dt"]
    viscosity = config["viscosity"]
    x = np.arange(n, dtype=np.float64) * (length / n)
    y = x.copy()
    xx, yy = np.meshgrid(x / length * (2 * math.pi), y / length * (2 * math.pi), indexing="xy")
    modes = np.fft.fftfreq(n) * n
    mx, my = np.meshgrid(modes, modes, indexing="xy")
    kx, ky = mx * (2 * math.pi / length), my * (2 * math.pi / length)
    k2 = kx * kx + ky * ky
    inverse_k2 = np.zeros_like(k2)
    np.divide(1.0, k2, out=inverse_k2, where=k2 != 0)
    # Strictly less than N/3 excludes the cutoff when N is divisible by 3.
    # This avoids quadratic aliasing even at the retained boundary modes.
    dealias = (np.abs(mx) < n / 3) & (np.abs(my) < n / 3)

    def velocity_from(w_hat):
        psi_hat = w_hat * inverse_k2
        u_hat, v_hat = 1j * ky * psi_hat, -1j * kx * psi_hat
        return np.fft.ifft2(u_hat).real, np.fft.ifft2(v_hat).real, u_hat, v_hat

    if config["initial_condition"] == "taylor_green":
        omega = 2 * config["amplitude"] * (2 * math.pi / length) * np.sin(xx) * np.sin(yy)
        omega_hat = np.fft.fft2(omega) * dealias
    else:
        rng = np.random.default_rng(config["seed"])
        psi = np.zeros((n, n), dtype=np.float64)
        for ix in range(1, min(3, (n - 1) // 3) + 1):
            for iy in range(1, min(3, (n - 1) // 3) + 1):
                psi += (
                    rng.normal()
                    * np.sin(ix * xx + rng.uniform(0, 2 * math.pi))
                    * np.sin(iy * yy + rng.uniform(0, 2 * math.pi))
                    / (ix * ix + iy * iy) ** 2
                )
        omega_hat = np.fft.fft2(psi) * k2 * dealias
        u, v, _, _ = velocity_from(omega_hat)
        omega_hat *= config["amplitude"] / np.max(np.hypot(u, v))
    omega_hat[0, 0] = 0

    def rhs(w_hat):
        w_hat = w_hat * dealias
        u, v, _, _ = velocity_from(w_hat)
        rate = (float(np.max(np.abs(u))) + float(np.max(np.abs(v)))) * max_k + viscosity * max_k2
        _check_timestep(rate, dt, step - 1)
        omega_x = np.fft.ifft2(1j * kx * w_hat).real
        omega_y = np.fft.ifft2(1j * ky * w_hat).real
        nonlinear = np.fft.fft2(u * omega_x + v * omega_y) * dealias
        result = -nonlinear - viscosity * k2 * w_hat
        result[0, 0] = 0
        return result

    frames = {"u": [], "v": [], "vorticity": []}
    times, diagnostic_rows = [], []

    def save(step):
        u, v, u_hat, v_hat = velocity_from(omega_hat)
        omega = np.fft.ifft2(omega_hat).real
        divergence = np.fft.ifft2(1j * kx * u_hat + 1j * ky * v_hat).real
        for name, field in (("u", u), ("v", v), ("vorticity", omega)):
            frames[name].append(field.copy())
        mean = float(np.mean(omega))
        times.append(step * dt)
        diagnostic_rows.append(
            (
                mean,
                mean * length**2,
                float(0.5 * np.mean(u * u + v * v)),
                float(0.5 * np.mean(omega * omega)),
                float(np.max(np.abs(divergence))),
            )
        )

    save(0)
    max_k = float(np.max(np.abs(kx[dealias])))
    max_k2 = float(np.max(k2[dealias]))
    for step in range(1, config["steps"] + 1):
        omega_hat = _rk4(omega_hat, dt, rhs) * dealias
        omega_hat[0, 0] = 0
        if not np.all(np.isfinite(omega_hat)):
            raise ValueError(f"Non-finite vorticity at step {step}; reduce dt")
        if step % config["save_every"] == 0 or step == config["steps"]:
            save(step)
    diagnostics = np.asarray(diagnostic_rows, dtype=np.float64)
    metadata = _base_metadata(config, "vorticity_pseudospectral_rk4", ["time", "y", "x"])
    metadata.update(
        {
            "dealiasing": "strict_two_thirds_truncation",
            "velocity_convention": (
                "u=d(psi)/dy; v=-d(psi)/dx; vorticity=d(v)/dx-d(u)/dy=-laplacian(psi)"
            ),
            "mean_velocity": [0.0, 0.0],
            "pressure": "not_computed",
            "enstrophy_convention": "spatial_mean_of_half_vorticity_squared",
        }
    )
    return SimulationResult(
        times=np.asarray(times, dtype=np.float64),
        fields={name: np.stack(saved) for name, saved in frames.items()},
        diagnostics={
            name: diagnostics[:, index]
            for index, name in enumerate(
                ("mean_vorticity", "circulation", "energy", "enstrophy", "max_abs_divergence")
            )
        },
        coordinates={"x": x, "y": y},
        metadata=metadata,
    )
