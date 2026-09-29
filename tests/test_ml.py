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
    _evaluate,
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


@pytest.mark.parametrize("conserve_mean", [False, True])
def test_checkpoint_resume_matches_uninterrupted_training(dataset, tmp_path, conserve_mean):
    options = {
        "seed": 8,
        "width": 4,
        "modes": 3,
        "depth": 1,
        "batch_size": 4,
        "learning_rate": 0.002,
        "conserve_mean": conserve_mean,
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
    assert complete["config"]["conserve_mean"] is conserve_mean
    assert a["config"]["conserve_mean"] is conserve_mean
    evaluation = evaluate_fno(dataset, tmp_path / "complete")
    assert evaluation["conserve_mean"] is conserve_mean
    assert evaluation["conservation"] == complete["evaluation"]["conservation"]
    if conserve_mean:
        assert evaluation["conservation"]["rollout_mean_drift"]["max_abs"] < 1e-6
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
    with pytest.raises(ValueError, match="hyperparameters"):
        train_fno(
            dataset,
            tmp_path / "changed-constraint",
            epochs=1,
            resume=tmp_path / "first",
            **{**options, "conserve_mean": not conserve_mean},
        )
    assert not (tmp_path / "changed-constraint").exists()


@pytest.mark.parametrize("n", [16, 17])
def test_mean_projection_preserves_nonzero_means_for_200_nontrivial_steps(n):
    with _cpu_session(41):
        model = FNO1d(width=4, modes=4, depth=2, conserve_mean=True)
        with torch.no_grad():
            model.project[-1].weight.normal_(mean=0, std=0.01)
            model.project[-1].bias.fill_(0.15)
        x = torch.arange(n) * (2 * math.pi / n)
        initial = torch.stack((2.0 + 0.6 * torch.sin(x), -0.7 + 0.3 * torch.cos(2 * x)))
        initial_mean = initial.double().mean(dim=-1)
        with torch.no_grad():
            current = model(initial)
            assert float((current - initial).abs().max()) > 1e-5
            max_drift = 0.0
            for _ in range(199):
                current = model(current)
                assert torch.isfinite(current).all()
                max_drift = max(
                    max_drift, float((current.double().mean(-1) - initial_mean).abs().max())
                )
        assert max_drift < 1e-5


@pytest.mark.parametrize("n", [16, 17])
def test_projected_residual_is_trainable_and_preserves_each_batch_mean(n):
    with _cpu_session(6):
        model = FNO1d(width=4, modes=4, depth=2, conserve_mean=True)
        with torch.no_grad():
            model.project[-1].weight.normal_(mean=0, std=0.05)
            model.project[-1].bias.fill_(0.2)
        x = torch.arange(n) * (2 * math.pi / n)
        initial = torch.stack((1 + torch.sin(x), -2 + torch.cos(2 * x))).requires_grad_(True)
        target = initial.detach().mean(-1, keepdim=True) + 0.8 * (
            initial.detach() - initial.detach().mean(-1, keepdim=True)
        )
        output = model(initial)
        assert float((output - initial).abs().max().detach()) > 1e-5
        torch.testing.assert_close(output.mean(-1), initial.mean(-1), rtol=0, atol=3e-7)
        ((output - target) ** 2).mean().backward()
        assert initial.grad is not None and torch.isfinite(initial.grad).all()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert float(model.project[-1].weight.grad.abs().max()) > 1e-7
        assert float(model.spectral[0].weights.grad.abs().max()) > 1e-9


def test_default_forward_is_exactly_the_legacy_unprojected_computation():
    with _cpu_session(25):
        default = FNO1d(width=4, modes=3, depth=2)
        explicit = FNO1d(width=4, modes=3, depth=2, conserve_mean=False)
        with torch.no_grad():
            default.project[-1].weight.normal_(mean=0, std=0.2)
            default.project[-1].bias.fill_(0.7)
        explicit.load_state_dict(default.state_dict())
        value = torch.randn(3, 17)
        latent = default.lift(value[:, None, :])
        for spectral, local in zip(default.spectral, default.local, strict=True):
            latent = torch.nn.functional.gelu(spectral(latent) + local(latent))
        legacy = value + default.project(latent)[:, 0, :]
        torch.testing.assert_close(default(value), legacy, rtol=0, atol=0)
        torch.testing.assert_close(explicit(value), legacy, rtol=0, atol=0)
        assert float((legacy.mean(-1) - value.mean(-1)).abs().max().detach()) > 0.1


@pytest.mark.parametrize("conserve_mean", [False, True])
def test_conservation_diagnostics_distinguish_invariant_drift_from_reference_error(conserve_mean):
    truth = np.repeat(np.array([[[2.0], [2.25], [2.5]]], dtype=np.float32), 8, axis=2)
    data = {
        "u": truth,
        "splits": {"test": np.array([0])},
        "times": np.array([0.0, 0.1, 0.2]),
        "trajectories": [{"id": "drifting-reference"}],
        "contract": {"normalization": {"mean": 1.0, "std": 2.0}},
    }
    with _cpu_session(3):
        model = FNO1d(width=4, modes=3, depth=1, conserve_mean=conserve_mean)
        with torch.no_grad():
            model.project[-1].bias.fill_(0.25)  # A physical +0.5 increment if unconstrained.
        report, _ = _evaluate(model, data, "test")
    conservation = report["conservation"]
    expected_step = [0, 0] if conserve_mean else [0.5, 0.5]
    expected_rollout = [0, 0] if conserve_mean else [0.5, 1.0]
    assert conservation["one_step_mean_change"]["per_saved_time_rms"] == expected_step
    assert conservation["rollout_mean_drift"]["per_saved_time_rms"] == expected_rollout
    assert conservation["reference_mean_drift"]["per_saved_time_rms"] == [0.25, 0.5]
    assert conservation["rollout_mean_drift"]["rms"] == pytest.approx(
        np.sqrt(np.mean(np.square(expected_rollout)))
    )
    assert report["one_step"]["mean_velocity_rmse"] == 0.25
    assert conservation["quantity"] == "spatial_mean_velocity"
    assert conservation["units"] == "physical_velocity"


@pytest.mark.parametrize("invalid", [0, 1, "true", None, np.bool_(True)])
def test_conserve_mean_requires_a_boolean_before_dataset_io(tmp_path, invalid):
    with pytest.raises(ValueError, match="conserve_mean"):
        FNO1d(conserve_mean=invalid)
    with pytest.raises(ValueError, match="conserve_mean"):
        train_fno(tmp_path / "missing-dataset", tmp_path / "model", conserve_mean=invalid)
    assert not (tmp_path / "model").exists()


def test_legacy_checkpoint_without_flag_evaluates_identically_and_cannot_silently_resume(
    dataset, tmp_path
):
    options = {"width": 4, "modes": 3, "depth": 1, "seed": 7}
    current = tmp_path / "current-model"
    train_fno(dataset, current, epochs=1, **options)
    expected = evaluate_fno(dataset, current, output=tmp_path / "current-evaluation")
    legacy = tmp_path / "legacy-format-model"
    shutil.copytree(current, legacy)
    checkpoint = torch.load(legacy / "checkpoint.pt", weights_only=True)
    del checkpoint["config"]["conserve_mean"]
    torch.save(checkpoint, legacy / "checkpoint.pt")
    legacy_report = json.loads((legacy / "report.json").read_text("utf-8"))
    del legacy_report["config"]["conserve_mean"]
    legacy_report["checkpoint_sha256"] = hashlib.sha256(
        (legacy / "checkpoint.pt").read_bytes()
    ).hexdigest()
    (legacy / "report.json").write_text(json.dumps(legacy_report), "utf-8")
    manifest = json.loads((legacy / "manifest.json").read_text("utf-8"))
    manifest["artifacts"] = {
        name: hashlib.sha256((legacy / name).read_bytes()).hexdigest()
        for name in manifest["artifacts"]
    }
    (legacy / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    assert verify_model(legacy) == []
    actual = evaluate_fno(dataset, legacy, output=tmp_path / "legacy-evaluation")
    assert actual["conserve_mean"] is False
    for key in ("one_step", "rollout", "conservation"):
        assert actual[key] == expected[key]
    for name in ("one_step.npy", "rollout.npy"):
        np.testing.assert_array_equal(
            np.load(tmp_path / "current-evaluation" / name),
            np.load(tmp_path / "legacy-evaluation" / name),
        )
    with pytest.raises(ValueError, match="hyperparameters"):
        train_fno(dataset, tmp_path / "legacy-resume", epochs=1, resume=legacy, **options)


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
