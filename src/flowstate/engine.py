"""Execution, provenance, restartable sweeps, and diagnostic summaries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve

SCHEMA_VERSION = 1
MAX_SWEEP_RUNS = 10_000


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_provenance() -> dict[str, Any]:
    """Record code and environment without collecting credentials or environment vars."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        digest.update(source.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    # Never accidentally associate an installed wheel with the caller's repository.
    repo = next((p for p in package.parents if (p / ".git").exists()), None)
    commit = _git_value(repo, "rev-parse", "HEAD") if repo else None
    status = _git_value(repo, "status", "--porcelain") if repo else None
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "zarr", "duckdb", "pyarrow", "scipy", "h5py", "flowstate-research")
    }
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "source_sha256": digest.hexdigest(),
        "python": sys.version.split()[0],
        "dependencies": versions,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "precision": "float64",
    }


def experiment_id(config: dict, provenance: dict, parent_id: str | None, attempt: int) -> str:
    """Content identity excludes timestamps; changing code/environment creates a new run."""
    identity = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "parent_id": parent_id,
        "attempt": attempt,
        "code": {
            key: provenance[key]
            for key in ("git_commit", "source_sha256", "python", "dependencies", "hardware")
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


def summarize(result: Any, config: dict) -> dict:
    """Flag sampled numerical diagnostics; flags are not physical conclusions."""
    diagnostic = result.diagnostics
    if config["equation"] == "darcy2d":
        residual = float(diagnostic["residual_l2"][-1])
        error = float(diagnostic["pressure_l2_error"][-1])
        if not np.isfinite([residual, error]).all():
            raise FloatingPointError("Non-finite Darcy diagnostics")
        return {
            "residual_l2": residual,
            "pressure_l2_error": error,
            "saved_frames": 1,
            "final_time": 0.0,
            "needs_review": residual > 1e-8,
            "residual_tolerance": 1e-8,
            "reference": "manufactured analytic pressure",
            "diagnostic_sampling": "steady_state",
        }
    energy = np.asarray(diagnostic["energy"], dtype=float)
    if not all(np.isfinite(values).all() for values in diagnostic.values()):
        raise FloatingPointError("Non-finite diagnostic values")
    tolerance = 1e-8 * max(float(energy[0]), 1.0)
    increases = np.flatnonzero(np.diff(energy) > tolerance) + 1
    first = int(increases[0]) if len(increases) else None
    metrics = {
        "initial_energy": float(energy[0]),
        "final_energy": float(energy[-1]),
        "max_energy": float(energy.max()),
        "energy_change": float(energy[-1] - energy[0]),
        "energy_increase_observed": bool(len(increases)),
        "energy_increase_tolerance": tolerance,
        "invariant_drift_tolerance": 1e-10,
        "divergence_tolerance": 1e-10,
        "diagnostic_sampling": "saved_frames",
        "first_energy_increase_time": float(result.times[first]) if first is not None else None,
        "first_energy_increase_step": (
            int(round(result.times[first] / config["dt"])) if first is not None else None
        ),
        "final_time": float(result.times[-1]),
        "saved_frames": len(result.times),
        "reynolds": (
            config["amplitude"] * config["domain_length"] / config["viscosity"]
            if config["viscosity"] > 0
            else None
        ),
        "reynolds_convention": "configured amplitude * domain_length / viscosity",
    }
    if "mass" in diagnostic:
        metrics["mass_drift"] = float(np.max(np.abs(diagnostic["mass"] - diagnostic["mass"][0])))
    if "circulation" in diagnostic:
        circulation = diagnostic["circulation"]
        metrics["circulation_drift"] = float(np.max(np.abs(circulation - circulation[0])))
    if "max_abs_divergence" in diagnostic:
        metrics["max_divergence"] = float(np.max(diagnostic["max_abs_divergence"]))
    if "enstrophy" in diagnostic:
        metrics["initial_enstrophy"] = float(diagnostic["enstrophy"][0])
        metrics["final_enstrophy"] = float(diagnostic["enstrophy"][-1])
    metrics["needs_review"] = bool(
        metrics["energy_increase_observed"]
        or metrics.get("max_divergence", 0) > 1e-10
        or metrics.get("mass_drift", 0) > 1e-10
        or metrics.get("circulation_drift", 0) > 1e-10
    )
    return metrics


@dataclass(frozen=True)
class RunOutcome:
    record: dict
    resumed: bool


def run_experiment(
    config: dict,
    lake_root: str | Path = "data/lake",
    *,
    parent_id: str | None = None,
    attempt: int = 0,
    stream: bool = False,
) -> RunOutcome:
    """Execute once, or reuse a verified immutable result including a recorded failure.

    A new attempt number creates a distinct record; failed results are never overwritten.
    A crash before atomic publication is rerun from t=0, not from a solver checkpoint.
    """
    config = normalize_config(config)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("attempt must be a nonnegative integer")
    lake = Lake(lake_root)
    if parent_id is not None:
        if not lake.exists(parent_id):
            raise ValueError(f"Parent experiment does not exist: {parent_id}")
        if problems := lake.verify(parent_id):
            raise ValueError(f"Parent artifacts failed verification: {problems}")
    provenance = capture_provenance()
    run_id = experiment_id(config, provenance, parent_id, attempt)
    if lake.exists(run_id):
        if problems := lake.verify(run_id):
            raise ValueError(f"Existing experiment artifacts failed verification: {problems}")
        return RunOutcome(lake.load_record(run_id), resumed=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "parent_id": parent_id,
        "attempt": attempt,
        "equation": config["equation"],
        "solver": (
            "finite_difference_rk4"
            if config["equation"] == "burgers1d"
            else "darcy_harmonic_sparse"
            if config["equation"] == "darcy2d"
            else "pseudospectral_rk4"
        ),
        "config": config,
        "provenance": provenance,
        "status": "running",
        "metrics": {},
        "error": None,
        "storage_mode": "streamed" if stream and config["equation"] != "darcy2d" else "buffered",
    }
    start = time.perf_counter()
    result, field_store = None, None
    with tempfile.TemporaryDirectory(prefix=".stream-", dir=lake.experiments) as scratch:
        sink = None
        if record["storage_mode"] == "streamed":
            from flowstate.streaming import ZarrFrameSink

            sink = ZarrFrameSink(Path(scratch) / "fields.zarr", config)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = (
                    solve(config, frame_callback=sink, retain_fields=False)
                    if sink
                    else solve(config)
                )
                record["metrics"] = summarize(result, config)
            record["solver_metadata"] = result.metadata
            record["status"] = "completed"
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            record["status"] = "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result = None
        # Storage I/O failures propagate; they are never presented as physical anomalies.
        if result is not None and sink is not None:
            sink.finish(result)
            field_store = sink.path
        record["runtime_seconds"] = time.perf_counter() - start
        try:
            lake.write(run_id, record, result, field_store=field_store)
        except FileExistsError:
            if problems := lake.verify(run_id):
                raise ValueError(f"Concurrent experiment failed verification: {problems}") from None
            return RunOutcome(lake.load_record(run_id), resumed=True)
    return RunOutcome(record, resumed=False)


def expand_sweep(spec: dict) -> list[dict]:
    """Expand a bounded Cartesian grid, validating everything before execution begins."""
    if not isinstance(spec, dict) or set(spec) - {"base", "parameters"}:
        raise ValueError("Sweep must contain only base and parameters objects")
    base, parameters = spec.get("base", {}), spec.get("parameters", {})
    if not isinstance(base, dict) or not isinstance(parameters, dict):
        raise ValueError("Sweep base and parameters must be JSON objects")
    count = 1
    for name, values in parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep parameter {name!r} must be a nonempty list")
        count *= len(values)
        if count > MAX_SWEEP_RUNS:
            raise ValueError(f"Sweep exceeds {MAX_SWEEP_RUNS} experiments")
    keys = sorted(parameters)
    configs = []
    seen = set()
    for values in itertools.product(*(parameters[k] for k in keys)):
        config = normalize_config({**base, **dict(zip(keys, values, strict=True))})
        encoded = json.dumps(config, sort_keys=True)
        if encoded not in seen:
            configs.append(config)
            seen.add(encoded)
    return configs


def _worker(args: tuple) -> RunOutcome:
    config, root, parent, attempt, stream = args
    return run_experiment(config, root, parent_id=parent, attempt=attempt, stream=stream)


def run_sweep(
    spec: dict,
    lake_root: str | Path = "data/lake",
    *,
    workers: int = 1,
    parent_id: str | None = None,
    attempt: int = 0,
    stream: bool = False,
) -> list[RunOutcome]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer between 1 and 32")
    jobs = [(config, str(lake_root), parent_id, attempt, stream) for config in expand_sweep(spec)]
    if workers == 1:
        return [_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, jobs))
