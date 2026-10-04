"""Fixed, committed-source protocol for a larger local data-infrastructure study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from flowstate.engine import capture_provenance
from flowstate.scaling import _aggregate, _isolated_trial, _plan, _runtime_identity, _write_json

THREAD_SETTINGS = {
    key: "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
}
MIN_FREE_BYTES = 4 * 1024**3


def study_plan() -> dict:
    """Freeze workload before execution; count configurations separately from repeats."""
    specs, plans, cases = {}, {}, []
    order_seed = 20261004
    for grid in (64, 128, 256):
        spec = {
            "base": {
                "equation": "navier_stokes2d",
                "grid_size": grid,
                "viscosity": 0.02,
                "amplitude": 0.5,
                "initial_condition": "random",
                "dt": 0.00025,
                "steps": 32,
                "save_every": 16,
            },
            "parameters": {"viscosity": [0.02, 0.05], "seed": list(range(1000, 1036))},
        }
        plan = _plan(spec, [1, 2, 4, 8], ["streamed"], 3, order_seed, large=True)
        specs[str(grid)], plans[str(grid)] = spec, plan
        cases.extend({**case, "grid_size": grid} for case in plan["cases"])
    random.Random(order_seed).shuffle(cases)
    for index, case in enumerate(cases):
        case["id"] = (
            f"trial-{index:03d}-n{case['grid_size']}-w{case['workers']}-r{case['repeat']}"
        )
    estimated = sum(p["estimated_uncompressed_field_bytes"] for p in plans.values())
    return {
        "protocol": "flowstate-local-scaling-v1",
        "order_seed": order_seed,
        "specifications": specs,
        "configs_by_grid": {grid: plan["configs"] for grid, plan in plans.items()},
        "distinct_configurations": sum(len(p["configs"]) for p in plans.values()),
        "initial_condition_families": 36,
        "runs_per_trial": 72,
        "repeats": 3,
        "workers": [1, 2, 4, 8],
        "cases": cases,
        "fresh_runs": sum(p["fresh_runs"] for p in plans.values()),
        "verified_reuse_requests": sum(p["fresh_runs"] for p in plans.values()),
        "integration_steps": sum(p["integration_steps"] for p in plans.values()),
        "estimated_uncompressed_field_bytes": estimated,
        "minimum_free_bytes_before_start": 2 * estimated + MIN_FREE_BYTES,
        "workload_scope": (
            "Short unforced 2D Navier-Stokes trajectories; 72 configurations per grid, "
            "216 total, not 216 per grid. Related seeds across grids/viscosities are "
            "36 families, not independent training examples. Streamed storage only."
        ),
    }


def _summary(trials: list[dict]) -> list[dict]:
    rows = []
    for grid in sorted({t["case"]["grid_size"] for t in trials}):
        rows.extend(
            {"grid_size": grid, **row}
            for row in _aggregate([t for t in trials if t["case"]["grid_size"] == grid])
        )
    return rows


def _existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def run_scaling_study(output: str | Path) -> dict:
    """Run the complete frozen protocol locally; never overwrite or clean up evidence."""
    # Import the renderer before any expensive work so missing extras fail early.
    try:
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Scaling studies require matplotlib; prepare with "
            "uv sync --locked --all-extras --group dev"
        ) from exc

    from flowstate.scaling_chart import render_scaling_chart

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"Use a new study directory: {root}")
    plan = study_plan()
    provenance = capture_provenance()
    if not provenance["git_commit"] or provenance["git_dirty"] is not False:
        raise ValueError("Commit all study code and use a clean Git checkout before measuring")
    disk = shutil.disk_usage(_existing_parent(root.parent))
    if disk.free < plan["minimum_free_bytes_before_start"]:
        raise ValueError("Insufficient free disk for saved fields, scratch, and reserve")
    # Source/package provenance is checked by every trial. Preserve the lockfile as
    # an additional environment identity without recording arbitrary environment vars.
    repo = Path(__file__).resolve().parents[2]
    lock = repo / "uv.lock"
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "plan.json", {**plan, "provenance": provenance})
    env = {**os.environ, **THREAD_SETTINGS}
    report = {
        "schema_version": 1,
        "study": "scaling-large",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "output": str(root),
        "reproduce_command": "uv run --no-sync flowstate scaling-study <NEW_OUTPUT_DIRECTORY>",
        "provenance": provenance,
        "uv_lock_sha256": lock_hash,
        "measurement_dependencies": {
            name: importlib.metadata.version(name) for name in ("psutil", "matplotlib")
        },
        "runtime_thread_settings": THREAD_SETTINGS,
        "host_memory_bytes": psutil.virtual_memory().total,
        "disk_free_bytes_before": disk.free,
        "plan": plan,
        "trials": [],
        "measurement_scope": {
            "wall_time": "run_sweep: worker startup, solve, streamed write, hash, publication",
            "excluded_from_trial_wall": "Trial interpreter startup, audit/fingerprints, reuse",
            "throughput": "Successful completed experiments/hour, not timestep or frame count",
            "attempts": "Completed plus numerical-failed experiments/hour, reported separately",
            "memory": "Sampled sum of trial parent and descendant RSS; shared pages can repeat",
            "memory_limit": "Sampling misses between-sample peaks; not true peak or PSS",
            "reuse": "Verified immediate warm-cache repeat including worker-pool startup",
            "stored_bytes": "Logical lengths of all committed artifacts, not disk allocation",
            "order": "All grid/worker/repeat cases shuffled together with a fixed seed",
            "cache": "OS caches are not flushed; fresh interpreter and lake per trial",
            "failures": "Numerical failures retained and verified; infrastructure faults abort",
            "limits": (
                "One local host and short trajectories; no cloud/distributed/long-time "
                "physics claim. Three repeats show ranges, not statistical significance. "
                "Background load and filesystem effects are not controlled."
            ),
        },
    }

    def event(status, **details):
        with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "time": datetime.now(UTC).isoformat(), "status": status, **details,
            }, allow_nan=False) + "\n")

    references = {}
    started = time.perf_counter()
    try:
        for case in plan["cases"]:
            current_disk = shutil.disk_usage(root)
            if current_disk.free < MIN_FREE_BYTES:
                raise RuntimeError("Disk reserve reached; existing artifacts are retained")
            event("started", case=case)
            request_path = root / f"{case['id']}.json"
            _write_json(request_path, {
                "spec": plan["specifications"][str(case["grid_size"])],
                "case": case,
                "provenance": provenance,
                "allow_numerical_failures": True,
            })
            trial = _isolated_trial(request_path, root / case["id"], env=env)
            report["trials"].append(trial)
            if trial["status"] != "completed":
                raise RuntimeError("A trial did not complete; see retained trial report")
            grid = case["grid_size"]
            values = trial["scientific_results"]
            if grid in references and values != references[grid]:
                raise RuntimeError("Scientific values, identities, or flags differ across trials")
            references[grid] = values
            event("completed", case=case, wall_seconds=trial["wall_seconds"],
                  failure_count=trial["failure_count"])
        after = capture_provenance()
        if _runtime_identity(after) != _runtime_identity(provenance) or after["git_dirty"]:
            raise ValueError("Code or environment changed during the study")
        trials = report["trials"]
        report.update(
            status="completed",
            scientific_values_identical=True,
            summary=_summary(trials),
            totals={
                "trials": len(trials),
                "fresh_runs": sum(t["experiments"] for t in trials),
                "successful_runs": sum(t["completed_experiments"] for t in trials),
                "failure_count": sum(t["failure_count"] for t in trials),
                "verified_reused_runs": sum(t["reused_experiments"] for t in trials),
                "stored_bytes": sum(t["stored_bytes"] for t in trials),
                "fresh_run_wall_seconds": sum(t["wall_seconds"] for t in trials),
                "verified_reuse_seconds": sum(t["verified_reuse_seconds"] for t in trials),
            },
        )
        # Chart generation reads measurements only; its duration is included in
        # study elapsed time but never in trial throughput.
        render_scaling_chart(report, root / "scaling-large.png")
        event("study_completed", trials=len(trials))
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        event("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["study_elapsed_seconds"] = time.perf_counter() - started
        report["disk_free_bytes_after"] = shutil.disk_usage(root).free
        _write_json(root / "scaling-large.json", report)
    return report
