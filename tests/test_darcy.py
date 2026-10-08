"""Manufactured solutions distinguish linear residual from discretization error."""

import json
import math

import numpy as np
import pytest

from flowstate.darcy import normalize_darcy_config, solve_darcy


@pytest.mark.parametrize("contrast", [0.0, 0.5, -0.7])
def test_darcy_second_order_spatial_convergence(contrast):
    errors = []
    for n in (17, 33, 65):
        result = solve_darcy({"grid_size": n, "contrast": contrast})
        errors.append(result.diagnostics["pressure_l2_error"][0])
        assert result.diagnostics["residual_l2"][0] < 2e-11
        assert np.min(result.fields["permeability"]) > 0
    assert 3.7 < errors[0] / errors[1] < 4.3
    assert 3.7 < errors[1] / errors[2] < 4.3


def test_constant_permeability_matches_discrete_sine_eigenfunction():
    n, length = 21, 2.5
    result = solve_darcy({"grid_size": n, "domain_length": length, "contrast": 0})
    coordinate = result.coordinates["x"]
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    exact = np.sin(math.pi * xx / length) * np.sin(math.pi * yy / length)
    continuous_eigenvalue = 2 * (math.pi / length) ** 2
    discrete_eigenvalue = 8 / (length / (n - 1)) ** 2 * np.sin(math.pi / (2 * (n - 1))) ** 2
    np.testing.assert_allclose(
        result.fields["pressure"][0],
        continuous_eigenvalue / discrete_eigenvalue * exact,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result.fields["forcing"][0], continuous_eigenvalue * exact, atol=2e-14
    )


def test_shapes_exact_boundaries_and_json_metadata():
    result = solve_darcy({"grid_size": 13})
    assert result.times.tolist() == [0]
    assert all(field.shape == (1, 13, 13) for field in result.fields.values())
    assert all(diagnostic.shape == (1,) for diagnostic in result.diagnostics.values())
    p = result.fields["pressure"][0]
    assert np.all(p[[0, -1], :] == 0)
    assert np.all(p[:, [0, -1]] == 0)
    assert result.coordinates["x"][[0, -1]].tolist() == [0, 1]
    assert result.metadata["steady"] is True
    assert json.loads(json.dumps(result.metadata)) == result.metadata


def test_nondefault_domain_scales_forcing_and_error_norm():
    unit = solve_darcy({"domain_length": 1})
    scaled = solve_darcy({"domain_length": 3})
    np.testing.assert_allclose(unit.fields["pressure"], scaled.fields["pressure"], atol=1e-13)
    np.testing.assert_allclose(unit.fields["forcing"] / 9, scaled.fields["forcing"], atol=1e-14)
    assert scaled.diagnostics["pressure_l2_error"][0] == pytest.approx(
        3 * unit.diagnostics["pressure_l2_error"][0], rel=1e-10
    )


@pytest.mark.parametrize(
    "config, match",
    [
        ({"viscosity": 0.1}, "Unknown"),
        ({"equation": "burgers1d"}, "equation"),
        ({"grid_size": True}, "grid_size"),
        ({"grid_size": 4}, "grid_size"),
        ({"grid_size": 514}, "grid_size"),
        ({"grid_size": 33.0}, "grid_size"),
        ({"domain_length": 0}, "positive"),
        ({"domain_length": math.inf}, "finite"),
        ({"domain_length": 1e-200}, "range"),
        ({"contrast": math.nan}, "finite"),
        ({"contrast": True}, "finite"),
        ({"contrast": 0.9}, "contrast"),
        ({"contrast": -0.9}, "contrast"),
    ],
)
def test_darcy_rejects_invalid_or_unsupported_configuration(config, match):
    with pytest.raises(ValueError, match=match):
        normalize_darcy_config(config)


def test_darcy_normalization_is_complete_and_json_native():
    source = {"grid_size": np.int64(17), "contrast": np.float64(0.1)}
    normalized = normalize_darcy_config(source)
    assert set(normalized) == {"equation", "grid_size", "domain_length", "contrast"}
    assert normalize_darcy_config(normalized) == normalized
    assert json.loads(json.dumps(normalized)) == normalized
    assert len(source) == 2
