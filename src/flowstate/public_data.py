"""Acquire a bounded, attributed subset of the pinned public PDEBench Burgers file."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from flowstate.datasets import _coordinate, _publish
from flowstate.engine import capture_provenance
from flowstate.http_ranges import HTTPRangeReader
from flowstate.lake import _sha256

SOURCE = {
    "source_url": "https://darus.uni-stuttgart.de/api/access/datafile/281363",
    "source_version": "DaRUS doi:10.18419/darus-2986 V8.0 (2024-02-13), file 281363",
    "dataset_doi": "https://doi.org/10.18419/darus-2986",
    "dataset_version_url": (
        "https://darus.uni-stuttgart.de/dataset.xhtml?"
        "persistentId=doi:10.18419/darus-2986&version=8.0"
    ),
    "filename": "1D_Burgers_Sols_Nu0.01.hdf5",
    "license_name": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": (
        "Takamoto et al., PDEBench: An Extensive Benchmark for Scientific Machine Learning"
    ),
    "publisher_checksum": {"algorithm": "MD5", "value": "e6d9a4f62baf9a29121a816b919e2770"},
    "size_bytes": 8232968312,
    "tensor_shape": [10000, 201, 1024],
    "filename_nu": 0.01,
    "effective_viscosity": float(0.01 / np.pi),
    "viscosity_convention": "PDEBench epsilon/pi; u_t + u*u_x = viscosity*u_xx",
}


def acquire_pdebench(
    output: str | Path,
    *,
    samples: int = 24,
    sample_seed: int = 20260926,
    time_stop: int = 201,
    max_bytes: int = 64 * 1024**2,
) -> dict:
    """Publish source.hdf5 + metadata + manifest; partial downloads never publish.

    Draw source rows without replacement before inspecting their values. Preserve
    every spatial point and a declared time prefix. Only received ranges and the
    derived local file are hashed; the publisher's full-file MD5 is not verified.
    """
    for name, value, low, high in (
        ("samples", samples, 3, 64),
        ("sample_seed", sample_seed, 0, 2**32 - 1),
        ("time_stop", time_stop, 2, SOURCE["tensor_shape"][1]),
        ("max_bytes", max_bytes, 8, 256 * 1024**2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if samples * time_stop * SOURCE["tensor_shape"][2] > 16_000_000:
        raise ValueError("Public subset exceeds the 16M-value learning budget")
    indices = sorted(
        np.random.default_rng(sample_seed)
        .choice(SOURCE["tensor_shape"][0], size=samples, replace=False)
        .tolist()
    )

    def write(staging):
        with HTTPRangeReader(
            SOURCE["source_url"], SOURCE["size_bytes"], max_bytes=max_bytes
        ) as remote:
            # h5py must close before its underlying Python file object closes.
            with h5py.File(remote, "r") as handle:
                if not {"tensor", "x-coordinate", "t-coordinate"}.issubset(handle):
                    raise ValueError("Public source is missing required HDF5 datasets")
                tensor = handle["tensor"]
                if list(tensor.shape) != SOURCE["tensor_shape"] or tensor.dtype != np.dtype("f4"):
                    raise ValueError("Public source tensor contract changed")
                source_nt = tensor.shape[1]
                x = _coordinate(handle["x-coordinate"][:], "x", 4)
                original_time = _coordinate(handle["t-coordinate"][:], "time", 2)
                if len(x) != tensor.shape[2] or len(original_time) not in (
                    tensor.shape[1],
                    tensor.shape[1] + 1,
                ):
                    raise ValueError("Public source coordinate lengths changed")
                with h5py.File(staging / "source.hdf5", "w") as local:
                    local.create_dataset("x-coordinate", data=x)
                    local.create_dataset("t-coordinate", data=original_time[:time_stop])
                    target = local.create_dataset(
                        "tensor", (samples, time_stop, len(x)), dtype="f4"
                    )
                    for row, source_index in enumerate(indices):
                        values = tensor[source_index, :time_stop, :]
                        if not np.isfinite(values).all():
                            raise ValueError(f"Non-finite public trajectory {source_index}")
                        target[row] = values
            transfer = remote.receipt()
        return {
            "schema_version": 1,
            "kind": "pdebench_http_subset",
            "source": SOURCE,
            "sample_seed": sample_seed,
            "selection": {
                "sample_indices": indices,
                "time_start": 0,
                "time_stop": time_stop,
                "spatial_stride": 1,
            },
            "shape": [samples, time_stop, len(x)],
            "time_range": [float(original_time[0]), float(original_time[time_stop - 1])],
            "discarded_terminal_time_coordinate": (
                float(original_time[-1]) if len(original_time) == source_nt + 1 else None
            ),
            "transfer": transfer,
            "source_hdf5_sha256": _sha256(staging / "source.hdf5"),
            "provenance": capture_provenance(),
            "transformations": [
                "Select seeded source rows",
                "Keep declared time prefix",
                "Preserve all spatial points and physical field values",
            ],
            "limitations": [
                "Full remote file checksum was not verified",
                "ETag consistency and HTTPS do not prove the publisher checksum",
                "This is a derived subset, not the complete published benchmark",
            ],
        }

    return _publish(Path(output), write)


def verify_acquisition(path: str | Path) -> list[str]:
    root = Path(path)
    try:
        if root.is_symlink():
            raise ValueError("Acquisition must not be a symbolic link")
        manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        if (
            manifest.get("version") != 1
            or manifest.get("algorithm") != "sha256"
            or set(manifest["artifacts"]) != {"metadata.json", "source.hdf5"}
        ):
            raise ValueError("Invalid acquisition manifest")
        if {p.name for p in root.iterdir()} != {"metadata.json", "source.hdf5", "manifest.json"}:
            raise ValueError("Unexpected acquisition artifacts")
        for name, digest in manifest["artifacts"].items():
            file = root / name
            if file.is_symlink() or not file.is_file() or _sha256(file) != digest:
                raise ValueError(f"Acquisition hash mismatch: {name}")
        metadata = json.loads((root / "metadata.json").read_text("utf-8"))
        if (
            metadata.get("kind") != "pdebench_http_subset"
            or metadata.get("schema_version") != 1
            or metadata["source_hdf5_sha256"] != manifest["artifacts"]["source.hdf5"]
        ):
            raise ValueError("Invalid acquisition metadata")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return [str(exc)]
    return []


def import_acquired_pdebench(source: str | Path, output: str | Path, *, seed: int = 17) -> dict:
    from flowstate.datasets import import_pdebench

    root = Path(source)
    if problems := verify_acquisition(root):
        raise ValueError(f"Acquisition verification failed: {problems}")
    metadata = json.loads((root / "metadata.json").read_text("utf-8"))
    return import_pdebench(
        root / "source.hdf5",
        output,
        source_url=metadata["source"]["source_url"],
        source_version=metadata["source"]["source_version"],
        license_name=metadata["source"]["license_name"],
        viscosity=metadata["source"]["effective_viscosity"],
        seed=seed,
        acquisition=root,
    )
