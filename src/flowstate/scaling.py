"""Isolated, bounded comparisons of local sweep throughput and process-tree memory."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
import zarr

from flowstate.engine import capture_provenance, expand_sweep, run_sweep
from flowstate.lake import Lake
from flowstate.numerics import _output_size
from flowstate.resources import ProcessTreeSampler


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _config_key(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _runtime_identity(provenance: dict) -> dict:
    # Creating unignored study artifacts can change git_dirty without changing
    # numerical code. Preserve that annotation; compare the actual run identity.
    return {key: value for key, value in provenance.items() if key != "git_dirty"}


def _plan(spec: dict, workers, modes, repeats: int, seed: int) -> dict:
    configs = expand_sweep(spec)
    if not 1 <= len(configs) <= 32:
        raise ValueError("Scaling studies require 1 to 32 distinct experiments per trial")
    workers, modes = list(workers), list(modes)
    if (
        not workers
        or any(type(n) is not int or not 1 <= n <= min(8, len(configs)) for n in workers)
        or len(set(workers)) != len(workers)
        or 1 not in workers
    ):
        raise ValueError("workers must be unique, include 1, and be <= min(8, experiment count)")
    if not modes or any(m not in ("buffered", "streamed") for m in modes):
        raise ValueError("modes must contain buffered and/or streamed")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must be unique")
    if type(repeats) is not int or not 1 <= repeats <= 7:
        raise ValueError("repeats must be an integer from 1 to 7")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 to 2**32 - 1")
    if any(c["equation"] == "darcy2d" for c in configs):
        raise ValueError("Scaling benchmark supports time-dependent Burgers and Navier-Stokes")
    if any(c["grid_size"] > 512 for c in configs):
        raise ValueError("Scaling benchmark grids must be <= 512 points per dimension")
    cases = [
        {"workers": n, "mode": mode, "repeat": repeat}
        for repeat, n, mode in itertools.product(range(repeats), workers, modes)
    ]
    total_runs = len(cases) * len(configs)
    estimated_bytes = sum(_output_size(c) for c in configs) * len(cases)
    total_steps = sum(c["steps"] for c in configs) * len(cases)
    if len(cases) > 48 or total_runs > 256 or total_steps > 2_000_000:
        raise ValueError("Study exceeds 48 trials, 256 fresh runs, or 2 million integration steps")
    if estimated_bytes > 2 * 1024**3:
        raise ValueError("Study exceeds 2 GiB estimated uncompressed saved fields")
    random.Random(seed).shuffle(cases)
    for index, case in enumerate(cases):
        case["id"] = f"trial-{index:03d}-{case['mode']}-w{case['workers']}-r{case['repeat']}"
    return {
        "configs": configs,
        "cases": cases,
        "seed": seed,
        "repeats": repeats,
        "fresh_runs": total_runs,
        "integration_steps": total_steps,
        "estimated_uncompressed_field_bytes": estimated_bytes,
        "estimate_scope": "Saved fields only; excludes metadata, scratch copies, and caches",
    }


def _fingerprint(path: Path) -> str:
    """Hash scientific values frame by frame, excluding timing/provenance file bytes."""
    group = zarr.open_group(str(path), mode="r")
    arrays = ["time"]
    for collection in ("fields", "coordinates", "diagnostics"):
        arrays.extend(f"{collection}/{name}" for name in sorted(group[collection].array_keys()))
    digest = hashlib.sha256()
    for name in arrays:
        array = group[name]
        header = json.dumps([name, str(array.dtype), list(array.shape)]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        if name.startswith("fields/"):
            for frame in range(array.shape[0]):
                digest.update(array[frame].tobytes())
        else:
            digest.update(array[:].tobytes())
    return digest.hexdigest()


def _execute_case(request: dict, output: Path) -> dict:
    """Run in its own interpreter so prior trials cannot retain parent allocations."""
    output.mkdir(parents=True, exist_ok=False)
    lake_root = output / "lake"
    report = {"schema_version": 1, "case": request["case"], "status": "running"}
    try:
        before = capture_provenance()
        if _runtime_identity(before) != _runtime_identity(request["provenance"]):
            raise ValueError("Code or environment changed between study planning and execution")
        report["provenance"] = before
        case = request["case"]
        common = {"workers": case["workers"], "stream": case["mode"] == "streamed"}
        expected = {_config_key(config) for config in expand_sweep(request["spec"])}
        memory = ProcessTreeSampler()
        try:
            with memory:
                started = time.perf_counter()
                try:
                    outcomes = run_sweep(request["spec"], lake_root, **common)
                finally:
                    report["wall_seconds"] = time.perf_counter() - started
        finally:
            report["memory"] = memory.report()
        elapsed = report["wall_seconds"]
        report.update(
            experiments_per_second=len(outcomes) / elapsed,
            experiments=len(outcomes),
        )
        if len(outcomes) != len(expected) or any(
            o.record["status"] != "completed" or o.resumed for o in outcomes
        ):
            report["outcomes"] = [
                {"id": o.record["id"], "status": o.record["status"], "error": o.record["error"]}
                for o in outcomes
            ]
            raise RuntimeError("Fresh benchmark runs must complete without reuse")
        lake = Lake(lake_root)
        scientific = {}
        for outcome in outcomes:
            record = outcome.record
            if _runtime_identity(record["provenance"]) != _runtime_identity(before):
                raise ValueError("An experiment used a different code/environment snapshot")
            if problems := lake.verify(record["id"]):
                raise ValueError(f"Artifact verification failed: {problems}")
            scientific[_config_key(record["config"])] = {
                "id": record["id"],
                "values_sha256": _fingerprint(lake._path(record["id"]) / "fields.zarr"),
                "needs_review": record["metrics"]["needs_review"],
            }
        report["scientific_results"] = scientific
        if set(scientific) != expected:
            raise RuntimeError("Returned configurations differ from the planned sweep")
        files = [file for file in lake.experiments.rglob("*") if file.is_file()]
        report["stored_bytes"] = sum(file.stat().st_size for file in files)
        report["artifact_files"] = len(files)
        # Reuse still verifies every artifact; it is a separate, warm-cache measurement.
        started = time.perf_counter()
        resumed = run_sweep(request["spec"], lake_root, **common)
        report["verified_reuse_seconds"] = time.perf_counter() - started
        report["reused_experiments"] = sum(out.resumed for out in resumed)
        if report["reused_experiments"] != len(outcomes) or any(
            a.record != b.record for a, b in zip(outcomes, resumed, strict=True)
        ):
            raise RuntimeError("Identical repeat did not reuse every verified experiment")
        if _runtime_identity(capture_provenance()) != _runtime_identity(before):
            raise ValueError("Code or environment changed during a benchmark trial")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        try:
            report["committed_experiment_ids"] = [path.name for path in Lake(lake_root)._run_dirs()]
        except OSError as catalog_error:
            report["partial_catalog_error"] = str(catalog_error)
        raise
    finally:
        _write_json(output / "report.json", report)
    return report


def _isolated_trial(request_path: Path, output: Path) -> dict:
    """Use a fresh interpreter; retain stderr and partial evidence on failure."""
    command = [sys.executable, "-m", "flowstate.scaling", str(request_path), str(output)]
    # Do not pass a shell or credentials. The caller's ordinary numerical runtime
    # environment is inherited, and the small nonsecret thread settings are recorded.
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as job:
        try:
            stdout, stderr = job.communicate()
        except BaseException:
            # Stop only this child and its descendants if the study is interrupted.
            try:
                descendants = psutil.Process(job.pid).children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                descendants = []
            for process in reversed(descendants):
                try:
                    process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if job.poll() is None:
                job.kill()
            job.communicate()
            raise
    if job.returncode:
        raise RuntimeError(f"Trial process exited {job.returncode}: {stderr.strip()[-4000:]}")
    if stdout.strip():
        raise RuntimeError("Unexpected benchmark worker output")
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def _aggregate(trials: list[dict]) -> list[dict]:
    groups = {}
    for trial in trials:
        case = trial["case"]
        groups.setdefault((case["mode"], case["workers"]), []).append(trial)
    summary = []
    for (mode, workers), values in sorted(groups.items()):
        seconds = [v["wall_seconds"] for v in values]
        peaks = [v["memory"]["sampled_peak_aggregate_rss_bytes"] for v in values]
        baseline = statistics.median(v["wall_seconds"] for v in groups[(mode, 1)])
        median = statistics.median(seconds)
        summary.append(
            {
                "mode": mode,
                "workers": workers,
                "trials": len(values),
                "wall_seconds_median": median,
                "wall_seconds_min": min(seconds),
                "wall_seconds_max": max(seconds),
                "speedup_vs_one_worker": baseline / median,
                "experiments_per_second_median": statistics.median(
                    v["experiments_per_second"] for v in values
                ),
                "sampled_peak_aggregate_rss_bytes_max": (
                    max(peaks) if all(p is not None for p in peaks) else None
                ),
                "sampling_errors": sum(v["memory"]["sampling_errors"] for v in values),
                "verified_reuse_seconds_median": statistics.median(
                    v["verified_reuse_seconds"] for v in values
                ),
                "stored_bytes_median": statistics.median(v["stored_bytes"] for v in values),
            }
        )
    return summary


def benchmark_sweep(
    spec: dict,
    output: str | Path,
    *,
    workers=(1, 2, 4),
    modes=("buffered", "streamed"),
    repeats: int = 3,
    seed: int = 0,
) -> dict:
    plan = _plan(spec, workers, modes, repeats, seed)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance()
    _write_json(root / "plan.json", {**plan, "provenance": provenance})

    def event(status: str, **details):
        with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"time": datetime.now(UTC).isoformat(), "status": status, **details},
                    allow_nan=False,
                )
                + "\n"
            )

    trials, reference = [], None
    try:
        for case in plan["cases"]:
            event("started", case=case)
            request_path = root / f"{case['id']}.json"
            _write_json(request_path, {"spec": spec, "case": case, "provenance": provenance})
            trial = _isolated_trial(request_path, root / case["id"])
            if trial["status"] != "completed":
                raise RuntimeError("Trial did not complete")
            values = trial["scientific_results"]
            if reference is None:
                reference = values
            elif values != reference:
                raise RuntimeError("Scientific values, identities, or flags differ across trials")
            trials.append(trial)
            event("completed", case=case, wall_seconds=trial["wall_seconds"])
        if _runtime_identity(capture_provenance()) != _runtime_identity(provenance):
            raise ValueError("Code or environment changed during the study")
        report = {
            "schema_version": 1,
            "status": "completed",
            "provenance": provenance,
            "psutil_version": psutil.__version__,
            "runtime_thread_settings": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "plan": plan,
            "scientific_values_identical": True,
            "summary": _aggregate(trials),
            "trials": trials,
            "measurement_scope": {
                "wall_time": "run_sweep including worker spawn, solve, Zarr, hashing, publication",
                "excluded": "Case-interpreter startup, extra verification/fingerprints, reuse",
                "memory": "Sampled sum of case parent and descendant RSS; not PSS or true peak",
                "isolation": "Fresh interpreter/lake per trial; spawn pool; sampler adds overhead",
                "reuse": "Separate verified repeat with warm local caches; includes pool startup",
                "cache": "OS/filesystem caches not flushed; order shuffled with recorded seed",
                "limitations": "Local study; no distributed/cloud claim or significance test",
            },
        }
        _write_json(root / "report.json", report)
        event("study_completed", trials=len(trials))
        return report
    except BaseException as exc:
        event("failed", error_type=type(exc).__name__, message=str(exc))
        raise


def _main() -> int:
    parser = argparse.ArgumentParser(description="Internal isolated scaling trial")
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _execute_case(json.loads(args.request.read_text(encoding="utf-8")), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
