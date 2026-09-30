"""Export an offline, read-only snapshot of verified local experiment artifacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

from flowstate.lake import Lake, _sha256

_FIELDS = {
    "burgers1d": ("velocity", "Velocity (simulation units)"),
    "navier_stokes2d": ("vorticity", "Vorticity (inverse simulation time)"),
    "darcy2d": ("pressure", "Pressure (simulation units)"),
}


def _finite_list(values):
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array), array, None).tolist()


def _diagnostic_indices(count):
    if count <= 201:
        return np.arange(count)
    # Include both endpoints without reading the full saved history.
    return np.unique(np.linspace(0, count - 1, 201, dtype=np.int64))


def _arrays(directory: Path, equation: str) -> dict:
    store = directory / "fields.zarr"
    if not store.exists():
        return {"energy": None, "preview": None, "note": "No saved field artifacts."}
    group = zarr.open_group(str(store), mode="r")
    times = group["time"]
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("Saved physical time must be a nonempty vector")
    count = times.shape[0]
    energy = None
    if "diagnostics/energy" in group:
        values = group["diagnostics/energy"]
        if values.shape != times.shape:
            raise ValueError("Energy diagnostic must align with saved times")
        indices = _diagnostic_indices(count)
        energy = {
            "time": _finite_list(times.oindex[indices]),
            "values": _finite_list(values.oindex[indices]),
            "source_samples": count,
            "display_samples": len(indices),
            "sample_indices": indices.tolist(),
            "label": "Spatial mean kinetic energy: 0.5 × mean(speed²)",
            "units": "simulation velocity squared",
        }
    preview = None
    field = _FIELDS.get(equation)
    if field is not None and f"fields/{field[0]}" in group:
        array = group[f"fields/{field[0]}"]
        if array.ndim not in (2, 3) or array.shape[0] != count or array.dtype.kind not in "biuf":
            raise ValueError("Preview requires a real scalar field aligned with saved time")
        sizes = array.shape[1:]
        limit = 256 if array.ndim == 2 else 64
        strides = [max(1, math.ceil(size / limit)) for size in sizes]
        selection = (count - 1, *(slice(None, None, step) for step in strides))
        values = array[selection]
        coordinates = {}
        for axis, size, step in zip(
            ("x",) if array.ndim == 2 else ("y", "x"), sizes, strides, strict=True
        ):
            coordinate = group[f"coordinates/{axis}"]
            if coordinate.shape != (size,):
                raise ValueError("Field preview coordinates do not match its spatial dimensions")
            coordinates[axis] = _finite_list(coordinate[::step])
        preview = {
            "field": field[0],
            "label": field[1],
            "time": float(times[count - 1]),
            "steady": equation == "darcy2d",
            "source_frame_index": count - 1,
            "source_shape": list(sizes),
            "display_shape": list(values.shape),
            "strides": strides,
            "values": _finite_list(values),
            "coordinates": coordinates,
            "sampling": "Spatial stride sampling without averaging or an anti-alias filter",
        }
    return {
        "energy": energy,
        "preview": preview,
        "note": "Final saved scalar field; diagnostics show at most 201 samples.",
    }


def _safe_json(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char, replacement in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        encoded = encoded.replace(char, replacement)
    if len(encoded.encode("utf-8")) > 32 * 1024**2:
        raise ValueError("Dashboard payload exceeds the 32 MiB export limit")
    return encoded


def export_dashboard(lake_root, output_html, *, max_experiments=50) -> dict:
    """Write one self-contained HTML file exclusively, without modifying the lake.

    Select valid experiment directories in ascending ID order. Every selected
    artifact is checksum-verified (including all field chunks), but only bounded
    diagnostic samples and one spatially sampled final frame are decoded.
    """
    if (
        isinstance(max_experiments, bool)
        or not isinstance(max_experiments, int)
        or not 1 <= max_experiments <= 200
    ):
        raise ValueError("max_experiments must be an integer in [1, 200]")
    source = Path(lake_root).resolve(strict=True)
    experiments = source / "experiments"
    if not experiments.is_dir() or experiments.is_symlink():
        raise ValueError("An existing local lake with a real experiments directory is required")
    requested = Path(output_html).absolute()
    # Resolve only the parent: an existing leaf symlink must never be followed.
    target = requested.parent.resolve() / requested.name
    if target.is_relative_to(source):
        raise ValueError("Dashboard output must be outside the entire immutable lake")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Dashboard already exists: {target}")
    lake = Lake(source)
    directories = lake._run_dirs()
    selected = directories[:max_experiments]
    records = []
    for directory in selected:
        if problems := lake.verify(directory.name):
            raise ValueError(f"Experiment {directory.name} verification failed: {problems}")
        record = lake.load_record(directory.name)
        if record.get("id") != directory.name:
            raise ValueError("Experiment record identity differs from its directory")
        records.append(
            {
                "record": record,
                "manifest_sha256": _sha256(directory / "manifest.json"),
                "record_sha256": _sha256(directory / "record.json"),
                **_arrays(directory, record.get("equation")),
            }
        )
    summary = {
        "displayed": len(records),
        "total": len(directories),
        "truncated": len(selected) < len(directories),
        "max_experiments": max_experiments,
        "selection": "Ascending experiment ID; counts include committed valid-ID directories only",
    }
    payload = {
        "schema_version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "lake_root": str(source),
        "summary": summary,
        "experiments": records,
        "limitations": [
            "Offline snapshot; filtering changes only this page, never the lake.",
            "Completed means the solver finished; flags still require scientific review.",
            "Unsigned manifests detect changed artifacts, not forged replacement data and hashes.",
            "Verification scans all selected files; previews decode no complete field histories.",
            "Stride sampling can hide small features; analyze the original fields.",
            "Axes use stored simulation coordinates and units; no SI conversion is inferred.",
            "A static preview is not a new experiment, validation result, or scientific finding.",
        ],
    }
    html = _HTML.replace("__PAYLOAD__", _safe_json(payload), 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".tmp", dir=target.parent)
    stage = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(html)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the fully written file atomically and refuses an
        # existing destination on both POSIX and Windows, including racing exports.
        os.link(stage, target)
    finally:
        stage.unlink(missing_ok=True)
    return {"output_html": str(target), "html_sha256": _sha256(target), **summary}


_HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
