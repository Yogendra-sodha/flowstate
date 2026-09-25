"""Bounded CPU learning baselines for the canonical periodic Burgers dataset.

FNO learns one saved-time map across training trajectories. The PINN solves one
held-out initial-value problem using its initial state and physics only. Neither
model promotes the stored numerical reference to exact mathematical ground truth.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import zarr

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("Learning requires PyTorch: run uv sync --extra ml") from exc

from flowstate.engine import capture_provenance

MAX_DATA_ELEMENTS = 16_000_000
REFERENCE_LIMITATION = (
    "Errors are measured against the stored numerical simulation, not an exact solution. "
    "This small fixed-grid, fixed-viscosity baseline does not reproduce full PDEBench results."
)


def _integer(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _learning_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("learning_rate must be a finite number in (0, 0.1]")
    if not math.isfinite(value) or not 0 < value <= 0.1:
        raise ValueError("learning_rate must be a finite number in (0, 0.1]")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def _verify_artifacts(directory: Path) -> None:
    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        artifacts = manifest["artifacts"]
        if (
            manifest.get("version") != 1
            or manifest["algorithm"] != "sha256"
            or not isinstance(artifacts, dict)
        ):
            raise ValueError("Invalid model manifest")
        if not {"checkpoint.pt", "report.json"}.issubset(artifacts):
            raise ValueError("Model manifest is missing checkpoint.pt or report.json")
        for relative, digest in artifacts.items():
            part = Path(relative)
            if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
                raise ValueError("Unsafe model manifest path")
            file = directory / part
            if (
                any(
                    (directory / Path(*part.parts[:i])).is_symlink()
                    for i in range(1, len(part.parts) + 1)
                )
                or not file.is_file()
                or _sha256(file) != digest
            ):
                raise ValueError(f"Model artifact verification failed: {relative}")
        actual = {
            file.relative_to(directory).as_posix()
            for file in directory.rglob("*")
            if file.is_file() and file.name != "manifest.json"
        }
        if actual != set(artifacts):
            raise ValueError("Model artifact set differs from its manifest")
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Missing or invalid model artifacts") from exc


def verify_model(directory: str | Path) -> list[str]:
    """Verify an unsigned model artifact manifest; return integrity problems."""
    try:
        _verify_artifacts(Path(directory))
    except (ValueError, OSError) as exc:
        return [str(exc)]
    return []


@contextmanager
def _publication(output: str | Path, dataset: str | Path):
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}; choose a new directory")
    if target.is_relative_to(Path(dataset).resolve()):
        raise ValueError("Model output must be outside the immutable dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield stage
        artifacts = {
            file.relative_to(stage).as_posix(): _sha256(file)
            for file in sorted(stage.rglob("*"))
            if file.is_file()
        }
        _json(
            stage / "manifest.json", {"version": 1, "algorithm": "sha256", "artifacts": artifacts}
        )
        try:
            os.rename(stage, target)
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(f"Output already exists: {target}") from exc
            raise
    finally:
        if stage.exists() and stage.resolve().is_relative_to(target.parent):
            shutil.rmtree(stage)


@contextmanager
def _cpu_session(seed: int):
    """Restore caller settings; training itself uses a single deterministic CPU thread."""
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(rng)
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)


def _load_data(path: str | Path) -> dict:
    from flowstate.datasets import verify_dataset

    path = Path(path)
    if problems := verify_dataset(path):
        raise ValueError(f"Dataset verification failed: {problems}")
    group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    if attrs.get("equation") != "burgers1d" or attrs.get("boundary") != "periodic":
        raise ValueError("Learning requires a periodic Burgers1D dataset")
    array = group["fields/u"]
    if len(array.shape) != 3 or math.prod(array.shape) > MAX_DATA_ELEMENTS:
        raise ValueError("Dataset must be [trajectory,time,x] and contain <=16 million values")
    u = np.asarray(array[:], dtype=np.float32)
    times, x = np.asarray(group["time"][:]), np.asarray(group["x"][:])
    if u.shape[1] < 2 or u.shape[2] < 8 or not np.isfinite(u).all():
        raise ValueError("Dataset requires finite fields, >=2 saved frames and >=8 grid points")
    if times.shape != (u.shape[1],) or x.shape != (u.shape[2],):
        raise ValueError("Dataset coordinates do not match fields")
    for name, values in (("time", times), ("x", x)):
        delta = np.diff(values)
        if not np.isfinite(values).all() or not (delta > 0).all():
            raise ValueError(f"Dataset {name} must increase with finite values")
        # Match the canonical importer: float32 source coordinates can have
        # rounded differences even when their intended sampling is uniform.
        if not np.allclose(delta, delta[0], rtol=1e-4, atol=1e-10):
            raise ValueError(f"Dataset {name} spacing must be uniform")
    splits = {
        name: np.asarray(group[f"splits/{name}"][:]) for name in ("train", "validation", "test")
    }
    for name, indices in splits.items():
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
            raise ValueError(f"Dataset {name} split must contain trajectory indices")
    all_indices = np.concatenate(list(splits.values()))
    if sorted(all_indices.tolist()) != list(range(len(u))):
        raise ValueError("Dataset splits must partition all trajectories without overlap")
    metadata = json.loads((path / "metadata.json").read_text("utf-8"))
    trajectories = metadata["trajectories"]
    if len(trajectories) != len(u):
        raise ValueError("Dataset trajectory metadata does not match fields")
    seen_families = set()
    for indices in splits.values():
        families = {trajectories[int(index)]["family_id"] for index in indices}
        if seen_families & families:
            raise ValueError("Related trajectory families occur in multiple splits")
        seen_families |= families
    norm = attrs.get("normalization", {})
    mean, std = float(norm["mean"]), float(norm["std"])
    if norm.get("fit_split") != "train" or not math.isfinite(mean) or not 0 < std < math.inf:
        raise ValueError("Dataset must supply finite training-only normalization")
    train_values = u[splits["train"]].astype(np.float64)
    if not np.isclose(mean, train_values.mean(), rtol=1e-5, atol=1e-7):
        raise ValueError("Normalizer mean was not fitted on training trajectories")
    actual_std = float(train_values.std())
    if actual_std > 1e-8 and not np.isclose(std, actual_std, rtol=1e-5, atol=1e-7):
        raise ValueError("Normalizer std was not fitted on training trajectories")
    viscosity = float(attrs["viscosity"])
    if not math.isfinite(viscosity) or viscosity < 0:
        raise ValueError("Dataset viscosity must be finite and nonnegative")
    contract = {
        "grid_size": len(x),
        "saved_dt": float(times[1] - times[0]),
        "domain_length": float((x[1] - x[0]) * len(x)),
        "x_origin": float(x[0]),
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "viscosity": viscosity,
        "normalization": {"mean": mean, "std": std, "fit_split": "train"},
        "split_sha256": attrs["split_sha256"],
        "coordinate_uniformity_tolerance": {"rtol": 1e-4, "atol": 1e-10},
    }
    return {
        "u": u,
        "times": times,
        "x": x,
        "splits": splits,
        "contract": contract,
        "trajectories": trajectories,
        "dataset_sha256": _sha256(path / "manifest.json"),
    }


class SpectralConv1d(nn.Module):
    """Complex learned channel mixing for the retained real-FFT modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
            / math.sqrt(in_channels * out_channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(value, dim=-1, norm="ortho")
        count = min(self.modes, spectrum.shape[-1])
        transformed = spectrum.new_zeros(value.shape[0], self.weights.shape[1], spectrum.shape[-1])
        transformed[..., :count] = torch.einsum(
            "bim,iom->bom", spectrum[..., :count], self.weights[..., :count]
        )
        return torch.fft.irfft(transformed, n=value.shape[-1], dim=-1, norm="ortho")


class FNO1d(nn.Module):
    """Periodic, translation-equivariant residual FNO mapping [batch,x] to [batch,x]."""

    def __init__(self, width: int = 16, modes: int = 8, depth: int = 3):
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        latent = self.lift(value[:, None, :])
        for spectral, local in zip(self.spectral, self.local, strict=True):
            latent = torch.nn.functional.gelu(spectral(latent) + local(latent))
        return value + self.project(latent)[:, 0, :]


def _pairs(data: dict, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    trajectories = data["u"][data["splits"][split]]
    norm = data["contract"]["normalization"]
    normalized = (trajectories - norm["mean"]) / norm["std"]
    n = trajectories.shape[-1]
    return (
        torch.from_numpy(normalized[:, :-1].reshape(-1, n).copy()),
        torch.from_numpy(normalized[:, 1:].reshape(-1, n).copy()),
    )


def _loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, batch: int) -> float:
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch):
            prediction = model(inputs[start : start + batch])
            total += float(((prediction - targets[start : start + batch]) ** 2).sum())
    loss = total / targets.numel()
    if not math.isfinite(loss):
        raise RuntimeError("Non-finite model loss; lower the learning rate")
    return loss


def _metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    prediction, reference = prediction.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(prediction).all():
        return {"finite": False, "rmse": None, "relative_l2": None}
    error = prediction - reference
    denominator = float(np.linalg.norm(reference))
    return {
        "finite": True,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "relative_l2": float(np.linalg.norm(error) / denominator) if denominator > 0 else None,
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_velocity_rmse": float(np.sqrt(np.mean(np.mean(error, axis=-1) ** 2))),
        "energy_rmse": float(
            np.sqrt(np.mean((0.5 * np.mean(prediction**2 - reference**2, axis=-1)) ** 2))
        ),
        "per_saved_time_rmse": np.sqrt(np.mean(error**2, axis=(0, 2))).tolist(),
    }


def _evaluate(model: FNO1d, data: dict, split: str) -> tuple[dict, dict]:
    if split not in data["splits"]:
        raise ValueError("split must be train, validation, or test")
    indices = data["splits"][split]
    truth = data["u"][indices]
    norm = data["contract"]["normalization"]
    one_step = np.empty_like(truth[:, 1:])
    rollout = np.empty_like(one_step)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for trajectory in range(len(indices)):
            normalized = torch.from_numpy((truth[trajectory] - norm["mean"]) / norm["std"])
            for first in range(0, len(normalized) - 1, 64):
                stop = min(first + 64, len(normalized) - 1)
                one_step[trajectory, first:stop] = (
                    model(normalized[first:stop]).numpy() * norm["std"] + norm["mean"]
                )
            current = normalized[:1]
            for frame in range(truth.shape[1] - 1):
                current = model(current)
                rollout[trajectory, frame] = current[0].numpy() * norm["std"] + norm["mean"]
    report = {
        "split": split,
        "trajectory_indices": indices.tolist(),
        "trajectory_ids": [data["trajectories"][int(i)]["id"] for i in indices],
        "physical_times": data["times"][1:].tolist(),
        "one_step": _metrics(one_step, truth[:, 1:]),
        "rollout": _metrics(rollout, truth[:, 1:]),
        "persistence_one_step": _metrics(truth[:, :-1], truth[:, 1:]),
        "persistence_rollout": _metrics(
            np.repeat(truth[:, :1], truth.shape[1] - 1, axis=1), truth[:, 1:]
        ),
        "inference_seconds": time.perf_counter() - started,
        "reference_limitation": REFERENCE_LIMITATION,
    }
    return report, {"one_step": one_step, "rollout": rollout}


def _checkpoint(path: str | Path, data: dict, kind: str) -> tuple[dict, str]:
    path = Path(path)
    path = path / "checkpoint.pt" if path.is_dir() else path
    if path.name != "checkpoint.pt":
        raise ValueError("Expected a model directory or its checkpoint.pt file")
    _verify_artifacts(path.parent)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != kind or checkpoint.get("schema_version") != 1:
        raise ValueError(f"Expected a {kind} checkpoint")
    if checkpoint["dataset_sha256"] != data["dataset_sha256"]:
        raise ValueError("Checkpoint dataset hash does not match")
    if checkpoint["contract"] != data["contract"]:
        raise ValueError("Checkpoint grid, time, viscosity, split, or normalization changed")
    return checkpoint, _sha256(path)


def _resume(checkpoint: dict, config: dict, model: nn.Module, optimizer) -> None:
    if checkpoint["config"] != config:
        raise ValueError("Resume hyperparameters must match the original training configuration")
    if checkpoint["implementation_sha256"] != _sha256(Path(__file__)):
        raise ValueError("Resume requires the same learning implementation")
    if checkpoint["torch_version"] != str(torch.__version__):
        raise ValueError("Resume requires the same PyTorch version")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["rng_state"])


def _base_checkpoint(kind: str, data: dict, config: dict, model, optimizer, epoch: int) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "dataset_sha256": data["dataset_sha256"],
        "contract": data["contract"],
        "config": config,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "torch_version": str(torch.__version__),
        "implementation_sha256": _sha256(Path(__file__)),
    }


def train_fno(
    dataset_path: str | Path,
    output: str | Path,
    *,
    epochs: int = 20,
    seed: int = 0,
    width: int = 16,
    modes: int = 8,
    depth: int = 3,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    resume: str | Path | None = None,
) -> dict:
    """Train for additional epochs; publish immutable final/resumable and best-val weights."""
    _integer("epochs", epochs, 1, 2000)
    config = {
        "seed": _integer("seed", seed, 0, 2**32 - 1),
        "width": _integer("width", width, 4, 128),
        "modes": _integer("modes", modes, 1, 64),
        "depth": _integer("depth", depth, 1, 6),
        "batch_size": _integer("batch_size", batch_size, 1, 256),
        "learning_rate": _learning_rate(learning_rate),
    }
    data = _load_data(dataset_path)
    if modes > data["contract"]["grid_size"] // 2 + 1:
        raise ValueError("modes exceeds the real-FFT frequency count for this grid")
    inputs, targets = _pairs(data, "train")
    valid_inputs, valid_targets = _pairs(data, "validation")
    if epochs * math.ceil(len(inputs) / batch_size) > 100_000:
        raise ValueError("Training budget exceeds 100,000 optimizer updates per call")
    parent_hash = None
    with _publication(output, dataset_path) as stage, _cpu_session(seed):
        model = FNO1d(width=width, modes=modes, depth=depth)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history, start_epoch = [], 0
        best_loss = _loss(model, valid_inputs, valid_targets, batch_size)
        best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
        if resume is not None:
            saved, parent_hash = _checkpoint(resume, data, "fno1d")
            _resume(saved, config, model, optimizer)
            history, start_epoch = saved["history"], saved["epoch"]
            best_loss, best_state = saved["best_validation_loss"], saved["best_model_state"]
            best_epoch = saved["best_epoch"]
        started = time.perf_counter()
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            model.train()
            order = torch.randperm(len(inputs))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(inputs[indices]) - targets[indices]) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; lower learning_rate")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
            train_loss = _loss(model, inputs, targets, batch_size)
            validation = _loss(model, valid_inputs, valid_targets, batch_size)
            history.append(
                {
                    "epoch": epoch,
                    "train_mse_normalized": train_loss,
                    "validation_mse_normalized": validation,
                }
            )
            if validation < best_loss:
                best_loss, best_epoch = validation, epoch
                best_state = copy.deepcopy(model.state_dict())
        training_seconds = time.perf_counter() - started
        checkpoint = _base_checkpoint("fno1d", data, config, model, optimizer, epoch)
        checkpoint.update(
            {
                "history": history,
                "best_validation_loss": best_loss,
                "best_model_state": best_state,
                "best_epoch": best_epoch,
            }
        )
        torch.save(checkpoint, stage / "checkpoint.pt")
        model.load_state_dict(best_state)
        evaluation, _ = _evaluate(model, data, "test")
        report = {
            "kind": "fno1d",
            "dataset_sha256": data["dataset_sha256"],
            "dataset_path": str(Path(dataset_path).resolve()),
            "checkpoint_sha256": _sha256(stage / "checkpoint.pt"),
            "parent_checkpoint_sha256": parent_hash,
            "contract": data["contract"],
            "config": config,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "history": history,
            "training_seconds": training_seconds,
            "evaluation": evaluation,
            "provenance": capture_provenance(),
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "checkpoint_selection": "minimum validation MSE, including epoch-0 persistence",
            "reference_limitation": REFERENCE_LIMITATION,
        }
        _json(stage / "report.json", report)
    return report


def evaluate_fno(
    dataset_path: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
    output: str | Path | None = None,
) -> dict:
    """Evaluate best-validation weights with matching physical times and persistence."""
    data = _load_data(dataset_path)
    saved, checkpoint_hash = _checkpoint(checkpoint, data, "fno1d")
    with _cpu_session(saved["config"]["seed"]):
        model = FNO1d(**{key: saved["config"][key] for key in ("width", "modes", "depth")})
        model.load_state_dict(saved["best_model_state"])
        report, predictions = _evaluate(model, data, split)
    report.update(
        {
            "kind": "fno_evaluation",
            "dataset_sha256": data["dataset_sha256"],
            "dataset_path": str(Path(dataset_path).resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "best_epoch": saved["best_epoch"],
            "contract": data["contract"],
        }
    )
    if output is not None:
        with _publication(output, dataset_path) as stage:
            for name, values in predictions.items():
                np.save(stage / f"{name}.npy", values, allow_pickle=False)
            _json(stage / "report.json", report)
    return report


class BurgersPINN(nn.Module):
    """A physical-coordinate model; normalization stays in the autograd graph."""

    def __init__(self, contract: dict, width: int = 32, depth: int = 3):
        super().__init__()
        self.contract = contract
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        c = self.contract
        duration = c["time_end"] - c["time_start"]
        scaled = torch.stack(
            (
                2 * (coordinates[:, 0] - c["time_start"]) / duration - 1,
                2 * (coordinates[:, 1] - c["x_origin"]) / c["domain_length"] - 1,
            ),
            dim=-1,
        )
        return self.network(scaled)[:, 0] * c["normalization"]["std"] + c["normalization"]["mean"]


def _gradient(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    if not value.requires_grad:
        return torch.zeros_like(coordinates)
    gradient = torch.autograd.grad(
        value,
        coordinates,
        torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    # A linear network has u_x depending on trainable weights but not on x.
    # Its exact second spatial derivative is zero, not an autograd error.
    return torch.zeros_like(coordinates) if gradient is None else gradient


def burgers_residual(model: nn.Module, coordinates: torch.Tensor, viscosity: float) -> torch.Tensor:
    """u_t + u*u_x - nu*u_xx, differentiating physical [t,x] coordinates."""
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().clone().requires_grad_(True)
    value = model(coordinates)
    gradient = _gradient(value, coordinates)
    second_x = _gradient(gradient[:, 1], coordinates)[:, 1]
    return gradient[:, 0] + value * gradient[:, 1] - viscosity * second_x


def _collocation(contract: dict, count: int) -> torch.Tensor:
    coordinates = torch.rand(count, 2)
    coordinates[:, 0] *= contract["time_end"] - contract["time_start"]
    coordinates[:, 0] += contract["time_start"]
    coordinates[:, 1] *= contract["domain_length"]
    coordinates[:, 1] += contract["x_origin"]
    return coordinates.requires_grad_(True)


def _pinn_loss(model: BurgersPINN, data: dict, index: int, count: int) -> tuple[torch.Tensor, dict]:
    c = data["contract"]
    std = c["normalization"]["std"]
    initial = torch.stack(
        (
            torch.full((len(data["x"]),), c["time_start"]),
            torch.tensor(data["x"], dtype=torch.float32),
        ),
        dim=-1,
    )
    # This is the only trajectory field used to optimize the PINN.
    initial_target = torch.from_numpy(data["u"][index, 0])
    initial_loss = torch.mean(((model(initial) - initial_target) / std) ** 2)
    points = _collocation(c, count)
    residual = burgers_residual(model, points, c["viscosity"])
    residual_scale = std / (c["time_end"] - c["time_start"])
    physics_loss = torch.mean((residual / residual_scale) ** 2)
    boundary_times = _collocation(c, max(8, count // 4))[:, 0].detach()
    left = torch.stack((boundary_times, torch.full_like(boundary_times, c["x_origin"])), -1)
    right = left.clone()
    right[:, 1] += c["domain_length"]
    left.requires_grad_(True)
    right.requires_grad_(True)
    left_u, right_u = model(left), model(right)
    left_x, right_x = _gradient(left_u, left)[:, 1], _gradient(right_u, right)[:, 1]
    boundary_loss = torch.mean(((left_u - right_u) / std) ** 2) + torch.mean(
        ((left_x - right_x) * c["domain_length"] / std) ** 2
    )
    loss = 10 * initial_loss + physics_loss + boundary_loss
    return loss, {
        "initial": float(initial_loss.detach()),
        "physics": float(physics_loss.detach()),
        "periodic_boundary": float(boundary_loss.detach()),
        "total": float(loss.detach()),
    }


def train_pinn(
    dataset_path: str | Path,
    output: str | Path,
    *,
    trajectory_index: int | None = None,
    epochs: int = 200,
    seed: int = 0,
    width: int = 32,
    depth: int = 3,
    collocation_points: int = 128,
    learning_rate: float = 0.001,
    resume: str | Path | None = None,
) -> dict:
    """Fit one held-out initial-value problem without interior reference supervision."""
    _integer("epochs", epochs, 1, 5000)
    data = _load_data(dataset_path)
    index = int(data["splits"]["test"][0]) if trajectory_index is None else trajectory_index
    _integer("trajectory_index", index, 0, len(data["u"]) - 1)
    if index not in data["splits"]["test"]:
        raise ValueError("PINN comparison must select a held-out test trajectory")
    config = {
        "trajectory_index": index,
        "seed": _integer("seed", seed, 0, 2**32 - 1),
        "width": _integer("width", width, 4, 128),
        "depth": _integer("depth", depth, 1, 6),
        "collocation_points": _integer("collocation_points", collocation_points, 8, 4096),
        "learning_rate": _learning_rate(learning_rate),
    }
    if epochs * collocation_points > 2_000_000:
        raise ValueError("PINN budget exceeds two million residual points per call")
    parent_hash = None
    with _publication(output, dataset_path) as stage, _cpu_session(seed):
        model = BurgersPINN(data["contract"], width=width, depth=depth)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history, start_epoch = [], 0
        if resume is not None:
            saved, parent_hash = _checkpoint(resume, data, "burgers_pinn")
            _resume(saved, config, model, optimizer)
            history, start_epoch = saved["history"], saved["epoch"]
        started = time.perf_counter()
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = _pinn_loss(model, data, index, collocation_points)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite PINN loss; lower learning_rate or rescale the problem"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            history.append({"epoch": epoch, **pieces})
        training_seconds = time.perf_counter() - started
        checkpoint = _base_checkpoint("burgers_pinn", data, config, model, optimizer, epoch)
        checkpoint["history"] = history
        torch.save(checkpoint, stage / "checkpoint.pt")
        tt, xx = np.meshgrid(data["times"], data["x"], indexing="ij")
        coordinates = torch.tensor(np.stack((tt.ravel(), xx.ravel()), axis=-1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            prediction = torch.cat([model(batch) for batch in coordinates.split(1024)]).numpy()
        prediction = prediction.reshape(tt.shape)
        reference = data["u"][index : index + 1, 1:]
        persistence = np.repeat(data["u"][index : index + 1, :1], reference.shape[1], axis=1)
        validation_points = _collocation(data["contract"], 256)
        residual_rms = float(
            torch.sqrt(
                torch.mean(
                    burgers_residual(model, validation_points, data["contract"]["viscosity"]) ** 2
                )
            ).detach()
        )
        report = {
            "kind": "burgers_pinn",
            "dataset_sha256": data["dataset_sha256"],
            "dataset_path": str(Path(dataset_path).resolve()),
            "checkpoint_sha256": _sha256(stage / "checkpoint.pt"),
            "parent_checkpoint_sha256": parent_hash,
            "contract": data["contract"],
            "config": config,
            "epochs_completed": epoch,
            "history": history,
            "training_seconds": training_seconds,
            "physical_residual_rms": residual_rms,
            "trajectory_id": data["trajectories"][index]["id"],
            "physical_times": data["times"][1:].tolist(),
            "trajectory_error": _metrics(prediction[None, 1:], reference),
            "persistence_trajectory_error": _metrics(persistence, reference),
            "supervision": "selected test trajectory initial frame only; PDE and periodic u/u_x",
            "checkpoint_selection": "last epoch; no interior test target used for selection",
            "comparison_scope": "per-instance physics optimization, not an amortized operator",
            "provenance": capture_provenance(),
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "reference_limitation": REFERENCE_LIMITATION,
        }
        np.save(stage / "predictions.npy", prediction, allow_pickle=False)
        _json(stage / "report.json", report)
    return report
