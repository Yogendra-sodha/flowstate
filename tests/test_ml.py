"""Learning, physical gradients, leakage boundaries, and exact epoch resumption."""

import copy
import hashlib
import json
import math
import shutil

import numpy as np
import pytest
import zarr

torch = pytest.importorskip("torch")

from flowstate.datasets import export_dataset  # noqa: E402
from flowstate.engine import run_experiment  # noqa: E402
from flowstate.ml import (  # noqa: E402
    BurgersPINN,
    FNO1d,
    SpectralConv1d,
    _cpu_session,
    _load_data,
    _pinn_loss,
    burgers_residual,
    evaluate_fno,
    train_fno,
    train_pinn,
    verify_model,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml-dataset")
    lake = root / "lake"
    for seed in range(6):
        result = run_experiment(
            {
                "equation": "burgers1d",
                "grid_size": 16,
                "viscosity": 0.2,
                "initial_condition": "random",
                "seed": seed,
                "amplitude": 0.5,
                "steps": 8,
                "save_every": 2,
                "dt": 0.005,
            },
            lake,
        )
        assert result.record["status"] == "completed"
    output = root / "dataset"
    export_dataset(lake, output, seed=42)
    return output


def test_spectral_layer_retains_modes_and_has_complex_weight_gradients():
    n = 17  # Odd length tests explicit irfft(n=...) rather than inferred even size.
    x = torch.arange(n) * (2 * math.pi / n)
    low = torch.cos(2 * x)
    value = (low + 0.3 * torch.cos(6 * x))[None, None].requires_grad_(True)
    layer = SpectralConv1d(1, 1, modes=4)
    with torch.no_grad():
        layer.weights.fill_(1)
    output = layer(value)
    torch.testing.assert_close(output[0, 0], low, atol=1e-6, rtol=1e-5)
    (output.square().mean()).backward()
    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()
    assert abs(layer.weights.grad[0, 0, 2]) > 0.1
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_fno_can_learn_a_smooth_decay_operator():
    with _cpu_session(7):
        x = torch.arange(32) * (2 * math.pi / 32)
        phase = torch.arange(4)[:, None] * 0.4
        initial = torch.sin(x[None] + phase)
        target = 0.7 * initial
        model = FNO1d(width=8, modes=5, depth=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_error = float(torch.mean((model(initial) - target) ** 2).detach())
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.mean((model(initial) - target) ** 2)
            loss.backward()
            optimizer.step()
        final_error = float(torch.mean((model(initial) - target) ** 2).detach())
    assert final_error < initial_error * 0.02


def test_checkpoint_resume_matches_uninterrupted_training(dataset, tmp_path):
    options = {
        "seed": 8,
        "width": 4,
        "modes": 3,
        "depth": 1,
        "batch_size": 4,
        "learning_rate": 0.002,
    }
    complete = train_fno(dataset, tmp_path / "complete", epochs=2, **options)
    train_fno(dataset, tmp_path / "first", epochs=1, **options)
    resumed = train_fno(
        dataset, tmp_path / "resumed", epochs=1, resume=tmp_path / "first", **options
    )
    a = torch.load(tmp_path / "complete" / "checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=True)
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    assert complete["history"] == resumed["history"]
    assert complete["epochs_completed"] == resumed["epochs_completed"] == 2
    assert resumed["parent_checkpoint_sha256"]
    assert verify_model(tmp_path / "resumed") == []
    with pytest.raises(ValueError, match="hyperparameters"):
        train_fno(
            dataset,
            tmp_path / "invalid",
            epochs=1,
            resume=tmp_path / "first",
            **{**options, "learning_rate": 0.003},
        )
    assert not (tmp_path / "invalid").exists()


def test_evaluation_matches_training_report_and_persistence(dataset, tmp_path):
    output = tmp_path / "fno"
    report = train_fno(dataset, output, epochs=1, width=4, modes=3, depth=1)
    evaluation = evaluate_fno(dataset, output, output=tmp_path / "evaluation")
    assert evaluation["one_step"] == report["evaluation"]["one_step"]
    assert evaluation["rollout"] == report["evaluation"]["rollout"]
    assert evaluation["split"] == "test"
    data = _load_data(dataset)
    test = data["u"][data["splits"]["test"]].astype(np.float64)
    expected_rmse = np.sqrt(np.mean((test[:, 1:] - test[:, :-1]) ** 2))
    assert evaluation["persistence_one_step"]["rmse"] == pytest.approx(expected_rmse)
    assert np.load(tmp_path / "evaluation" / "rollout.npy").shape == test[:, 1:].shape
    with pytest.raises(FileExistsError):
        train_fno(dataset, output, epochs=1)
    with (output / "checkpoint.pt").open("ab") as stream:
        stream.write(b"corruption")
    assert verify_model(output)
    with pytest.raises(ValueError, match="verification"):
        evaluate_fno(dataset, output)


def test_burgers_residual_matches_an_exact_nonlinear_solution():
    class RationalSolution(torch.nn.Module):
        def forward(self, coordinates):
            return coordinates[:, 1] / (coordinates[:, 0] + 2)

    points = torch.tensor([[0.2, 0.3], [0.7, -0.4], [1.3, 0.8]], dtype=torch.float64)
    residual = burgers_residual(RationalSolution(), points, viscosity=0.123)
    torch.testing.assert_close(residual, torch.zeros(3, dtype=torch.float64), atol=1e-15, rtol=0)


def test_pinn_input_output_scaling_preserves_physical_derivatives():
    contract = {
        "time_start": 2.0,
        "time_end": 5.0,
        "x_origin": -1.0,
        "domain_length": 7.0,
        "normalization": {"mean": 0.4, "std": 2.0},
    }
    model = BurgersPINN(contract, width=4, depth=1).double()
    linear = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.3, -0.7]], dtype=torch.float64))
    model.network = linear
    points = torch.tensor([[2.4, 0.2], [4.1, 3.0]], dtype=torch.float64)
    u = model(points)
    exact_ut, exact_ux = 2 * 2 * 1.3 / 3, 2 * 2 * -0.7 / 7
    torch.testing.assert_close(burgers_residual(model, points, 0.6), exact_ut + u * exact_ux)


def test_pinn_loss_never_uses_heldout_interior_targets(dataset):
    data = _load_data(dataset)
    index = int(data["splits"]["test"][0])
    changed = copy.deepcopy(data)
    changed["u"][index, 1:] += 1000
    with _cpu_session(9):
        model = BurgersPINN(data["contract"], width=4, depth=1)
        torch.manual_seed(123)
        first, _ = _pinn_loss(model, data, index, 8)
        first.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        torch.manual_seed(123)
        second, _ = _pinn_loss(model, changed, index, 8)
        second.backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for expected, parameter in zip(gradients, model.parameters(), strict=True):
            torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)


def test_pinn_checkpoint_evaluation_and_heldout_requirement(dataset, tmp_path):
    output = tmp_path / "pinn"
    report = train_pinn(dataset, output, epochs=2, width=4, depth=1, collocation_points=8)
    assert report["trajectory_error"]["finite"]
    assert report["physical_residual_rms"] >= 0
    assert report["epochs_completed"] == 2
    assert "initial frame only" in report["supervision"]
    assert verify_model(output) == []
    data = _load_data(dataset)
    with pytest.raises(ValueError, match="held-out"):
        train_pinn(
            dataset,
            tmp_path / "bad-pinn",
            epochs=1,
            trajectory_index=int(data["splits"]["train"][0]),
        )


def test_training_budgets_and_dataset_are_immutable(dataset, tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        train_fno(dataset, tmp_path / "bad-budget", epochs=0)
    with pytest.raises(ValueError, match="outside"):
        train_fno(dataset, dataset / "model", epochs=1, width=4, modes=3, depth=1)
    with pytest.raises(ValueError, match="frequency"):
        train_fno(dataset, tmp_path / "bad-modes", epochs=1, modes=32)


def test_float32_source_coordinate_rounding_is_accepted(dataset, tmp_path):
    imported = tmp_path / "rounded-coordinates"
    shutil.copytree(dataset, imported)
    group = zarr.open_group(str(imported), mode="r+")
    for name in ("time", "x"):
        group[name][:] = group[name][:].astype(np.float32).astype(np.float64)
    manifest_path = imported / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"] = {
        name: hashlib.sha256((imported / name).read_bytes()).hexdigest()
        for name in manifest["artifacts"]
    }
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    data = _load_data(imported)
    np.testing.assert_array_equal(data["times"], group["time"][:])
    assert data["contract"]["coordinate_uniformity_tolerance"]["rtol"] == 1e-4


def test_pinn_resume_restores_collocation_rng_and_optimizer(dataset, tmp_path):
    options = {"seed": 31, "width": 4, "depth": 1, "collocation_points": 8}
    train_pinn(dataset, tmp_path / "complete-pinn", epochs=2, **options)
    train_pinn(dataset, tmp_path / "first-pinn", epochs=1, **options)
    train_pinn(
        dataset, tmp_path / "resumed-pinn", epochs=1, resume=tmp_path / "first-pinn", **options
    )
    complete = torch.load(tmp_path / "complete-pinn" / "checkpoint.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed-pinn" / "checkpoint.pt", weights_only=True)
    assert complete["history"] == resumed["history"]
    for name, value in complete["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
