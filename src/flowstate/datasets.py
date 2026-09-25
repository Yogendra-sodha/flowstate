"""Stream verified trajectories into a frozen, leakage-aware Burgers dataset."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import zarr

from flowstate.lake import Lake, _rename_with_retry, _sha256

_SPLITS = ("train", "validation", "test")
_CONVENTION = "u_t + u*u_x = viscosity*u_xx; viscosity is the effective diffusion coefficient"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _coordinate(values: Any, name: str, minimum: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < minimum or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 1-D coordinate with >= {minimum} entries")
    differences = np.diff(result)
    if np.any(differences <= 0) or not np.allclose(
        differences, differences[0], rtol=1e-4, atol=1e-10
    ):
        raise ValueError(f"{name} must be strictly increasing and uniformly spaced")
    return result


def _viscosity(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("viscosity must be a finite nonnegative effective diffusion coefficient")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("viscosity must be a finite nonnegative effective diffusion coefficient")
    return value


def _families(trajectories: list[dict]) -> None:
    """Unite configured IC families and exact duplicate initial fields."""
    parents = list(range(len(trajectories)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, item in enumerate(trajectories):
        for category, value in (
            ("configuration", item["family_key"]),
            ("initial_field", item["initial_field_sha256"]),
        ):
            key = (category, value)
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    members: dict[int, list[str]] = {}
    for index, item in enumerate(trajectories):
        members.setdefault(root(index), []).append(item["family_key"])
    for index, item in enumerate(trajectories):
        item["family_id"] = _json_hash(sorted(set(members[root(index)])))


def _split(trajectories: list[dict], seed: int) -> dict[str, list[int]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a nonnegative integer")
    families = sorted({item["family_id"] for item in trajectories})
    if len(families) < 3:
        raise ValueError("At least 3 distinct initial-condition families are required")
    families = np.random.default_rng(seed).permutation(families).tolist()
    holdout = max(1, len(families) // 5)
    assignments = {
        family: split
        for split, group in zip(
            _SPLITS,
            (families[2 * holdout :], families[:holdout], families[holdout : 2 * holdout]),
            strict=True,
        )
        for family in group
    }
    return {
        split: [
            index
            for index, item in enumerate(trajectories)
            if assignments[item["family_id"]] == split
        ]
        for split in _SPLITS
    }


def _publish(output: Path, writer: Callable[[Path], dict]) -> dict:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Dataset already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{output.name}-", dir=output.parent))
    try:
        metadata = writer(staging)
        (staging / "metadata.json").write_text(_json(metadata) + "\n", encoding="utf-8")
        artifacts = {
            file.relative_to(staging).as_posix(): _sha256(file)
            for file in sorted(staging.rglob("*"))
            if file.is_file()
        }
        (staging / "manifest.json").write_text(
            _json({"version": 1, "algorithm": "sha256", "artifacts": artifacts}) + "\n",
            encoding="utf-8",
        )
        try:
            _rename_with_retry(staging, output)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(f"Dataset already exists: {output}") from error
            raise
        return metadata
    except BaseException:
        if staging.resolve().is_relative_to(output.parent.resolve()):
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_dataset(
    staging: Path,
    trajectories: list[dict],
    loader: Callable[[int], np.ndarray],
    times: np.ndarray,
    x: np.ndarray,
    viscosity: float,
    seed: int,
    provenance: dict,
) -> dict:
    _families(trajectories)
    splits = _split(trajectories, seed)
    split_sha256 = _json_hash(splits)
    group = zarr.open_group(str(staging), mode="w")
    fields = group.create_group("fields")
    shape = (len(trajectories), len(times), len(x))
    u = fields.create_array(
        "u", shape=shape, dtype="float32", chunks=(1, min(16, len(times)), min(256, len(x)))
    )
    group.create_array("time", data=times, chunks=(min(len(times), 1024),))
    group.create_array("x", data=x, chunks=(min(len(x), 1024),))
    split_group = group.create_group("splits")
    for name, indices in splits.items():
        split_group.create_array(name, data=np.asarray(indices, dtype=np.int64))
    training = set(splits["train"])
    count, mean, m2 = 0, 0.0, 0.0
    for index in range(len(trajectories)):
        # Exactly one trajectory is loaded. No stack/list accumulates field arrays.
        values = np.asarray(loader(index))
        if values.shape != shape[1:] or not np.isfinite(values).all():
            raise ValueError(f"Trajectory {index} has an incompatible shape or non-finite values")
        if np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError(f"Trajectory {index} cannot be represented as float32")
        values = values.astype(np.float32)
        u[index] = values
        if index in training:
            block = values.astype(np.float64)
            block_count = block.size
            block_mean = float(block.mean())
            block_m2 = float(np.sum((block - block_mean) ** 2))
            delta = block_mean - mean
            total = count + block_count
            m2 += block_m2 + delta * delta * count * block_count / total
            mean += delta * block_count / total
            count = total
    std = math.sqrt(max(0.0, m2 / count))
    normalization = {
        "mean": mean,
        "std": std if std > 0 else 1.0,
        "fit_split": "train",
        "count": count,
        "constant_training_data": std == 0,
    }
    metadata = {
        "schema_version": 1,
        "equation": "burgers1d",
        "boundary": "periodic",
        "shape": list(shape),
        "axis_order": ["trajectory", "time", "x"],
        "dtype": "float32",
        "stored_values": "raw_physical",
        "viscosity": viscosity,
        "viscosity_convention": _CONVENTION,
        "domain_length": float((x[1] - x[0]) * len(x)),
        "saved_dt": float(times[1] - times[0]),
        "normalization": normalization,
        "split_seed": seed,
        "splits": splits,
        "split_sha256": split_sha256,
        "trajectories": trajectories,
        "provenance": provenance,
        "units": {"x": "source coordinates", "time": "source time", "u": "source velocity"},
    }
    group.attrs.update(
        {
            key: metadata[key]
            for key in (
                "schema_version",
                "equation",
                "boundary",
                "viscosity",
                "viscosity_convention",
                "domain_length",
                "saved_dt",
                "normalization",
                "split_sha256",
                "stored_values",
            )
        }
    )
    return metadata


def export_dataset(
    lake_root: str | Path, output: str | Path, ids: list[str] | None = None, seed: int = 0
) -> dict:
    """Export compatible completed Burgers runs, splitting IC families before windows.

    Viscosity, physical grid, and saved times must match. Configuration families
    depend on initial-condition type and random seed, never viscosity/grid.
    Deterministic sine runs share a family even if their unused seed differs.
    """
    lake = Lake(lake_root)
    if ids is None:
        ids = [
            item["id"]
            for item in lake.records()
            if item.get("status") == "completed" and item.get("equation") == "burgers1d"
        ]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate experiment IDs are not allowed")
    ids = sorted(ids)
    if not ids:
        raise ValueError("No completed Burgers trajectories selected")
    sources, trajectories = [], []
    times = x = None
    viscosity = None
    for experiment_id in ids:
        if problems := lake.verify(experiment_id):
            raise ValueError(f"Source {experiment_id} failed verification: {problems}")
        record = lake.load_record(experiment_id)
        if record.get("status") != "completed" or record.get("equation") != "burgers1d":
            raise ValueError("Only completed burgers1d experiments can be exported")
        if record.get("metrics", {}).get("needs_review", False):
            raise ValueError(f"Source {experiment_id} needs_review; resolve quality flags first")
        config = record["config"]
        directory = lake.experiments / experiment_id
        source = zarr.open_group(str(directory / "fields.zarr"), mode="r")
        if source.attrs.get("simulation_metadata", {}).get("boundary_conditions") != "periodic":
            raise ValueError("Only periodic Burgers trajectories are supported")
        current_times = _coordinate(source["time"][:], "time", 2)
        current_x = _coordinate(source["coordinates/x"][:], "x", 4)
        current_nu = _viscosity(config["viscosity"])
        if not np.isclose(
            (current_x[1] - current_x[0]) * len(current_x),
            config["domain_length"],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError("Periodic grid must omit the repeated endpoint")
        if source["fields/velocity"].shape != (len(current_times), len(current_x)):
            raise ValueError("Burgers field must have shape [time,x]")
        if times is None:
            times, x, viscosity = current_times, current_x, current_nu
        elif (
            not np.array_equal(times, current_times)
            or not np.array_equal(x, current_x)
            or viscosity != current_nu
        ):
            raise ValueError(
                "All trajectories must share the same physical time, grid, and viscosity"
            )
        ic = config["initial_condition"]
        family_key = _json(
            {"initial_condition": ic, "seed": config["seed"] if ic == "random" else None}
        )
        initial = np.asarray(source["fields/velocity"][0], dtype="<f4")
        trajectories.append(
            {
                "id": experiment_id,
                "family_key": family_key,
                "initial_field_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                "config": config,
                "metrics": record.get("metrics", {}),
                "source_manifest_sha256": _sha256(directory / "manifest.json"),
                "source_artifacts": json.loads((directory / "manifest.json").read_text("utf-8"))[
                    "artifacts"
                ],
            }
        )
        sources.append(source["fields/velocity"])
    return _publish(
        Path(output),
        lambda staging: _write_dataset(
            staging,
            trajectories,
            lambda index: sources[index][:],
            times,
            x,
            viscosity,
            seed,
            {"kind": "flowstate_lake", "source_ids": ids},
        ),
    )


def import_pdebench(
    source: str | Path,
    output: str | Path,
    *,
    source_url: str,
    source_version: str,
    license_name: str,
    viscosity: float,
    seed: int = 0,
) -> dict:
    """Import the periodic uniform-viscosity PDEBench Burgers HDF5 contract.

    The input contains tensor[sample,time,x], x-coordinate, t-coordinate.
    viscosity is the effective diffusion coefficient, e.g. filename Nu0.01
    means 0.01/pi in the official PDEBench generator. The caller asserts that
    this coefficient is shared by every trajectory; filenames are not parsed.
    """
    import h5py

    if (
        urlparse(source_url).scheme not in ("http", "https")
        or not urlparse(source_url).netloc
        or not source_version.strip()
        or not license_name.strip()
    ):
        raise ValueError("Explicit source_url, source_version, and license_name are required")
    viscosity = _viscosity(viscosity)
    source = Path(source)
    raw_sha256 = _sha256(source)
    with h5py.File(source, "r") as handle:
        if not {"tensor", "x-coordinate", "t-coordinate"}.issubset(handle.keys()):
            raise ValueError("Expected PDEBench tensor, x-coordinate, and t-coordinate datasets")
        tensor = handle["tensor"]
        if tensor.ndim != 3 or tensor.shape[0] < 3:
            raise ValueError("PDEBench tensor must have shape [sample,time,x] with >= 3 samples")
        x = _coordinate(handle["x-coordinate"][:], "x", 4)
        original_time = _coordinate(handle["t-coordinate"][:], "time", 2)
        nt = tensor.shape[1]
        if len(original_time) not in (nt, nt + 1) or tensor.shape[2] != len(x):
            raise ValueError("PDEBench coordinate lengths do not match tensor shape")
        times = original_time[:nt]
        _coordinate(times, "time", 2)
        for name in ("viscosity", "nu"):
            if name in handle and np.asarray(handle[name]).size != 1:
                raise ValueError("Per-trajectory viscosity arrays are not supported")
        trajectories = []
        for index in range(tensor.shape[0]):
            initial = np.asarray(tensor[index, 0], dtype="<f4")
            if not np.isfinite(initial).all():
                raise ValueError("PDEBench initial fields contain non-finite values")
            initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
            trajectories.append(
                {
                    "id": f"{raw_sha256[:16]}-{index}",
                    "source_index": index,
                    "family_key": initial_hash,
                    "initial_field_sha256": initial_hash,
                    "config": {
                        "viscosity": viscosity,
                        "initial_condition": "pdebench",
                        "domain_length": float((x[1] - x[0]) * len(x)),
                    },
                    "source_sha256": raw_sha256,
                }
            )
        provenance = {
            "kind": "pdebench_hdf5",
            "source_url": source_url,
            "source_version": source_version,
            "license_name": license_name,
            "source_sha256": raw_sha256,
            "uniform_viscosity_asserted_by_caller": viscosity,
            "viscosity_convention": _CONVENTION,
            "discarded_terminal_time_coordinate": (
                float(original_time[-1]) if len(original_time) == nt + 1 else None
            ),
        }
        return _publish(
            Path(output),
            lambda staging: _write_dataset(
                staging,
                trajectories,
                lambda index: tensor[index],
                times,
                x,
                viscosity,
                seed,
                provenance,
            ),
        )


def verify_dataset(dataset_dir: str | Path) -> list[str]:
    """Verify every canonical dataset artifact against its unsigned SHA256 manifest."""
    directory = Path(dataset_dir)
    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        artifacts = manifest["artifacts"]
        if (
            manifest.get("version") != 1
            or manifest.get("algorithm") != "sha256"
            or not isinstance(artifacts, dict)
            or "metadata.json" not in artifacts
            or "zarr.json" not in artifacts
        ):
            return ["Invalid dataset manifest"]
    except (OSError, ValueError, TypeError, KeyError):
        return ["Missing or invalid dataset manifest"]
    problems = []
    for relative, expected in artifacts.items():
        part = Path(relative)
        if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
            problems.append(f"Unsafe artifact path: {relative}")
            continue
        path = directory / part
        if any(
            (directory / Path(*part.parts[:index])).is_symlink()
            for index in range(1, len(part.parts) + 1)
        ):
            problems.append(f"Symbolic link artifact: {relative}")
        elif not path.is_file():
            problems.append(f"Missing artifact: {relative}")
        elif _sha256(path) != expected:
            problems.append(f"Hash mismatch: {relative}")
    actual = {
        item.relative_to(directory).as_posix()
        for item in directory.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    }
    problems.extend(f"Unexpected artifact: {name}" for name in sorted(actual - artifacts.keys()))
    return problems
