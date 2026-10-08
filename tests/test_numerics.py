"""Physical and analytic checks, rather than snapshots of solver internals."""

import json
import math

import numpy as np
import pytest

from flowstate.numerics import normalize_config, solve


def test_taylor_green_matches_viscous_decay_on_nondefault_domain():
    config = {
        "equation": "navier_stokes2d",
        "grid_size": 24,
        "domain_length": 3.0,
        "amplitude": 0.7,
        "viscosity": 0.3,
        "dt": 0.002,
        "steps": 100,
        "save_every": 23,
    }
    result = solve(config)
    k = 2 * math.pi / config["domain_length"]
    xx, yy = np.meshgrid(result.coordinates["x"], result.coordinates["y"], indexing="xy")
    decay = np.exp(-2 * config["viscosity"] * k * k * result.times)[:, None, None]
    exact_u = config["amplitude"] * np.sin(k * xx) * np.cos(k * yy) * decay
    exact_v = -config["amplitude"] * np.cos(k * xx) * np.sin(k * yy) * decay
    exact_omega = 2 * config["amplitude"] * k * np.sin(k * xx) * np.sin(k * yy) * decay
    np.testing.assert_allclose(result.fields["u"], exact_u, atol=2e-11)
    np.testing.assert_allclose(result.fields["v"], exact_v, atol=2e-11)
    np.testing.assert_allclose(result.fields["vorticity"], exact_omega, atol=8e-11)
    exact_energy = (
        config["amplitude"] ** 2 / 4 * np.exp(-4 * config["viscosity"] * k * k * result.times)
    )
    np.testing.assert_allclose(result.diagnostics["energy"], exact_energy, rtol=2e-10)
    np.testing.assert_allclose(
        result.diagnostics["enstrophy"], 2 * k * k * exact_energy, rtol=2e-10
    )
    assert np.max(result.diagnostics["max_abs_divergence"]) < 1e-13
    assert np.max(np.abs(result.diagnostics["circulation"])) < 1e-13
    np.testing.assert_array_equal(result.times, np.array([0, 23, 46, 69, 92, 100]) * 0.002)
    assert result.fields["u"].shape == (6, 24, 24)


def test_taylor_green_rk4_temporal_refinement():
    errors = []
    for dt in (0.08, 0.04):
        result = solve(
            {
                "equation": "navier_stokes2d",
                "grid_size": 8,
                "amplitude": 0.5,
                "viscosity": 0.4,
                "dt": dt,
                "steps": round(0.4 / dt),
                "save_every": 100,
            }
        )
        exact = result.fields["u"][0] * np.exp(-0.8 * 0.4)
        errors.append(np.max(np.abs(result.fields["u"][-1] - exact)))
    assert 14 < errors[0] / errors[1] < 18


def test_burgers_conserves_mass_and_dissipates_energy_when_resolved():
    result = solve({"grid_size": 64, "amplitude": 0.5, "viscosity": 0.2, "steps": 200})
    assert np.max(np.abs(result.diagnostics["mass"] - result.diagnostics["mass"][0])) < 1e-13
    assert np.all(np.diff(result.diagnostics["energy"]) < 0)
    assert result.diagnostics["energy"][-1] < result.diagnostics["energy"][0]
    assert result.coordinates["x"][-1] < 2 * math.pi


def _burgers_cole_hopf_reference(n, viscosity, amplitude, time):
    """Independent heat-equation Cole--Hopf solution for sinusoidal initial u."""
    fine_n = 1536
    x = np.arange(fine_n) * 2 * math.pi / fine_n
    wave = np.fft.fftfreq(fine_n) * fine_n
    theta_initial = np.exp(amplitude * np.cos(x) / (2 * viscosity))
    theta_hat = np.fft.fft(theta_initial) * np.exp(-viscosity * wave**2 * time)
    theta = np.fft.ifft(theta_hat).real
    theta_x = np.fft.ifft(1j * wave * theta_hat).real
    return (-2 * viscosity * theta_x / theta)[:: fine_n // n]


def test_burgers_spatial_refinement_against_cole_hopf_solution():
    errors = []
    for n in (24, 48, 96):
        result = solve(
            {
                "grid_size": n,
                "viscosity": 0.2,
                "amplitude": 0.25,
                "dt": 0.00025,
                "steps": 400,
                "save_every": 400,
            }
        )
        exact = _burgers_cole_hopf_reference(n, 0.2, 0.25, 0.1)
        errors.append(np.sqrt(np.mean((result.fields["velocity"][-1] - exact) ** 2)))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
def test_random_initial_conditions_are_reproducible_and_seed_dependent(equation):
    config = {
        "equation": equation,
        "initial_condition": "random",
        "seed": 52,
        "steps": 11,
        "save_every": 4,
    }
    first, repeated, different = solve(config), solve(config), solve({**config, "seed": 53})
    for name in first.fields:
        np.testing.assert_array_equal(first.fields[name], repeated.fields[name])
        assert not np.array_equal(first.fields[name][0], different.fields[name][0])
    np.testing.assert_array_equal(first.times, np.array([0, 4, 8, 11]) * 0.001)
    assert all(len(values) == len(first.times) for values in first.diagnostics.values())


def test_random_navier_stokes_dissipates_energy_and_enstrophy():
    result = solve({"equation": "navier_stokes2d", "initial_condition": "random", "viscosity": 0.1})
    assert np.all(np.diff(result.diagnostics["energy"]) < 0)
    assert np.all(np.diff(result.diagnostics["enstrophy"]) < 0)
    assert np.max(result.diagnostics["max_abs_divergence"]) < 1e-13


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
def test_zero_flow_is_stationary(equation):
    result = solve({"equation": equation, "amplitude": 0, "steps": 1, "save_every": 50})
    assert len(result.times) == 2
    for values in result.fields.values():
        np.testing.assert_array_equal(values, 0)


@pytest.mark.parametrize(
    "config, message",
    [
        ({"viscocity": 0.1}, "Unknown"),
        ({"equation": "navier_stokes3d"}, "equation"),
        ({"equation": []}, "equation"),
        ({"initial_condition": "taylor_green"}, "initial_condition"),
        ({"initial_condition": []}, "initial_condition"),
        ({"viscosity": -1}, "viscosity"),
        ({"dt": 0}, "dt"),
        ({"dt": math.inf}, "dt"),
        ({"amplitude": math.nan}, "amplitude"),
        ({"grid_size": 7}, "grid_size"),
        ({"grid_size": 32.0}, "grid_size"),
        ({"steps": True}, "steps"),
        ({"steps": 0}, "steps"),
        ({"save_every": 0}, "save_every"),
        ({"seed": -1}, "seed"),
        ({"domain_length": 0}, "domain_length"),
        ({"domain_length": 1e-200}, "spacing"),
        ({"steps": 10**6, "save_every": 1}, "256 MiB"),
        ({"equation": "navier_stokes2d", "grid_size": 2048}, "256 MiB"),
    ],
)
def test_configuration_validation(config, message):
    with pytest.raises(ValueError, match=message):
        normalize_config(config)


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
def test_unstable_timestep_has_actionable_error(equation):
    with pytest.raises(ValueError, match=r"CFL/diffusion.*use dt <="):
        solve({"equation": equation, "dt": 100, "steps": 1})


def test_canonical_config_is_complete_json_native_and_does_not_mutate_input():
    source = {"grid_size": np.int64(16), "viscosity": np.float64(0.2)}
    normalized = normalize_config(source)
    assert len(normalized) == 10
    assert len(source) == 2
    assert json.loads(json.dumps(normalized)) == normalized
    assert normalize_config(normalized) == normalized
    assert normalize_config({"equation": "navier_stokes2d"})["initial_condition"] == "taylor_green"
