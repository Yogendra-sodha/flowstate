"""Local, retained evidence from a real process kill and exact solver recovery.

Reproduce from clean committed source with a fresh output directory::

    python -m flowstate.recovery_study --output outputs/recovery-01

Workers are separate interpreters. The parent kills only processes it created;
experiment/checkpoint data and worker logs remain available after every outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psutil
import zarr

from flowstate import numerics
from flowstate.checkpoints import CheckpointStore, validate_checkpoint_options
from flowstate.engine import capture_provenance, run_experiment
from flowstate.lake import Lake
from flowstate.numerics import normalize_config


def _write_json(path: Path, value: dict) -> None:
    """Flush a private file before atomically exposing the complete JSON receipt."""
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_name(f".{path.name}-{uuid.uuid4().hex}.pending")
    with pending.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.rename(pending, path)


def _identity(provenance: dict) -> dict:
    return {key: value for key, value in provenance.items() if key != "git_dirty"}


def _same_runtime(expected: dict, actual: dict) -> None:
    if _identity(expected) != _identity(actual):
        raise RuntimeError("Recovery study source or runtime changed during execution")


def _worker(request: dict) -> dict:
    """Count actual integrator calls and optionally block after a committed checkpoint."""
    before = capture_provenance()
    _same_runtime(request["provenance"], before)
    steps_executed = 0
    original_rk4, original_save = numerics._rk4, CheckpointStore.save
    started = time.perf_counter()

    def count_rk4(*args):
        nonlocal steps_executed
        result = original_rk4(*args)
        steps_executed += 1
        return result

    def save_then_pause(store, checkpoint):
        original_save(store, checkpoint)
        if checkpoint.step == request["checkpoint_every"]:
            _write_json(
                Path(request["ready"]),
                {
                    "pid": os.getpid(),
                    "experiment_id": store.run_id,
                    "checkpoint_step": checkpoint.step,
                    "rk4_calls": steps_executed,
                    "seconds_until_checkpoint": time.perf_counter() - started,
                    "provenance": before,
                    "checkpoint_commit_complete": True,
                },
            )
            while True:
                time.sleep(1)

    numerics._rk4 = count_rk4
    if request["role"] == "interrupted":
        CheckpointStore.save = save_then_pause
    result = {"role": request["role"], "status": "failed", "provenance": before}
    try:
        outcome = run_experiment(
            request["config"],
            request["lake"],
            stream=True,
            checkpoint_every=(
                None if request["role"] == "baseline" else request["checkpoint_every"]
            ),
        )
        result.update(
            experiment_id=outcome.record["id"],
            experiment_status=outcome.record["status"],
            error=outcome.record["error"],
            finalized_result_reused=outcome.resumed,
            resumed_from_step=outcome.checkpoint_step,
        )
        _same_runtime(before, outcome.record["provenance"])
        _same_runtime(before, capture_provenance())
        if outcome.record["status"] != "completed":
            raise RuntimeError(f"Numerical experiment failed: {outcome.record['error']}")
        result["status"] = "completed"
    except BaseException as exc:
        result["worker_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result.update(rk4_calls=steps_executed, engine_seconds=time.perf_counter() - started)
        numerics._rk4, CheckpointStore.save = original_rk4, original_save
        _write_json(Path(request["result"]), result)
    return result


def _kill_owned(process: subprocess.Popen) -> dict:
    """Terminate a known child and its descendants; never scan unrelated processes."""
    if process.poll() is not None:
        raise RuntimeError("Worker exited before the controlled process kill")
    root = psutil.Process(process.pid)
    descendants = root.children(recursive=True)
    targets = [root, *reversed(descendants)]
    killed = []
    for target in targets:
        try:
            target.kill()
            killed.append(target.pid)
        except psutil.NoSuchProcess:
            pass
    # Popen must reap its direct child itself to preserve the real signal/exit
    # status. psutil.wait_procs could otherwise consume that status on POSIX.
    returncode = process.wait(timeout=5)
    _, alive = psutil.wait_procs(descendants, timeout=5)
    if alive:
        raise RuntimeError(f"Owned worker processes did not exit: {[p.pid for p in alive]}")
    return {"terminated_pids": killed, "returncode": returncode}


def _validate_ready(process: subprocess.Popen, ready: dict) -> None:
    """A Windows venv launcher may own a second interpreter with another PID."""
    if process.poll() is not None:
        raise RuntimeError("Worker exited before its checkpoint receipt was checked")
    pid = ready.get("pid")
    if type(pid) is not int or ready.get("checkpoint_commit_complete") is not True:
        raise RuntimeError("Checkpoint receipt does not identify the owned worker")
    root = psutil.Process(process.pid)
    owned = [root, *root.children(recursive=True)]
    if not any(child.pid == pid and child.is_running() for child in owned):
        raise RuntimeError("Checkpoint receipt does not identify the owned worker")


def _invoke_worker(request: dict, directory: Path, timeout: float, *, interrupt=False) -> dict:
    role = request["role"]
    request_path = directory / f"{role}-request.json"
    _write_json(request_path, request)
    started = time.perf_counter()
    with (
        (directory / f"{role}-stdout.log").open("x", encoding="utf-8") as stdout,
        (directory / f"{role}-stderr.log").open("x", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, "-m", "flowstate.recovery_study", "--worker", str(request_path)],
            stdout=stdout,
            stderr=stderr,
        )
        try:
            if interrupt:
                ready_path = Path(request["ready"])
                deadline = time.monotonic() + timeout
                while not ready_path.exists():
                    if process.poll() is not None:
                        raise RuntimeError(f"Interrupted worker exited early: {process.returncode}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Worker did not commit its checkpoint before timeout")
                    time.sleep(0.025)
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                _validate_ready(process, ready)
                ready["launcher_pid"] = process.pid
                ready["termination"] = _kill_owned(process)
                ready["process_seconds"] = time.perf_counter() - started
                return ready
            returncode = process.wait(timeout=timeout)
            if returncode:
                raise RuntimeError(
                    f"{role} worker failed with exit code {returncode}; inspect logs"
                )
            result = json.loads(Path(request["result"]).read_text(encoding="utf-8"))
            result["process_seconds"] = time.perf_counter() - started
            return result
        finally:
            if process.poll() is None:
                _kill_owned(process)


def _scientific_hashes(path: Path) -> dict:
    """Hash every scientific array, retaining dtype/shape and per-array evidence."""
    group = zarr.open_group(str(path), mode="r")
    names = ["time"]
    for collection in ("fields", "coordinates", "diagnostics"):
        names.extend(f"{collection}/{name}" for name in sorted(group[collection].array_keys()))
    evidence = {}
    for name in names:
        array = group[name]
        digest = hashlib.sha256()
        if name.startswith("fields/"):
            for index in range(array.shape[0]):
                digest.update(array[index].tobytes())
        else:
            digest.update(array[:].tobytes())
        evidence[name] = {
            "dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()
        }
    return evidence


def _case(config: dict, directory: Path, provenance: dict, interval: int, timeout: float) -> dict:
    directory.mkdir()
    result = {"config": config, "status": "failed", "checkpoint_every": interval}
    started = time.perf_counter()
    try:
        workers = {}
        for role in ("baseline", "interrupted", "resumed"):
            lake_root = directory / ("baseline-lake" if role == "baseline" else "recovery-lake")
            request = {
                "role": role, "config": config, "provenance": provenance,
                "checkpoint_every": interval, "lake": str(lake_root),
                "ready": str(directory / "checkpoint-ready.json"),
                "result": str(directory / f"{role}-result.json"),
            }
            workers[role] = _invoke_worker(
                request, directory, timeout, interrupt=role == "interrupted"
            )
            result["workers"] = workers
            _same_runtime(provenance, workers[role]["provenance"])
            if role == "interrupted" and Lake(lake_root).exists(workers[role]["experiment_id"]):
                raise RuntimeError("Killed run unexpectedly published a finalized experiment")
        baseline, interrupted, resumed = (workers[name] for name in (
            "baseline", "interrupted", "resumed"
        ))
        if (
            baseline["rk4_calls"] != config["steps"]
            or interrupted["rk4_calls"] != interval
            or interrupted["checkpoint_step"] != interval
            or resumed["rk4_calls"] != config["steps"] - interval
            or resumed["resumed_from_step"] != interval
            or baseline["finalized_result_reused"] or resumed["finalized_result_reused"]
        ):
            raise RuntimeError("Observed RK4 calls or recovery path do not match the protocol")
        if not (
            baseline["experiment_id"] == interrupted["experiment_id"] == resumed["experiment_id"]
        ):
            raise RuntimeError(
                "Experiment identity changed between uninterrupted and recovered runs"
            )
        arrays = {}
        verification = {}
        for role, lake_name in (("baseline", "baseline-lake"), ("resumed", "recovery-lake")):
            lake = Lake(directory / lake_name)
            run_id = workers[role]["experiment_id"]
            verification[role] = lake.verify(run_id)
            if verification[role]:
                raise RuntimeError(f"{role} experiment failed artifact verification")
            arrays[role] = _scientific_hashes(lake._path(run_id) / "fields.zarr")
        result.update(artifact_verification=verification, scientific_arrays=arrays)
        if arrays["baseline"] != arrays["resumed"]:
            raise RuntimeError("Recovered scientific arrays differ from uninterrupted reference")
        result.update(status="completed", all_scientific_arrays_hash_identical=True)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["case_seconds"] = time.perf_counter() - started
        _write_json(directory / "case-report.json", result)
    return result


def run_recovery_study(
    output: str | Path,
    *,
    grid_size: int = 32,
    steps: int = 37,
    save_every: int = 8,
    checkpoint_every: int = 13,
    worker_timeout: float = 120,
    require_clean: bool = True,
) -> dict:
    """Run bounded local kill/recovery checks; failure reports and raw evidence persist."""
    if (
        isinstance(worker_timeout, bool) or not isinstance(worker_timeout, (int, float))
        or not math.isfinite(worker_timeout) or not 1 <= worker_timeout <= 600
    ):
        raise ValueError("worker_timeout must be finite and between 1 and 600 seconds")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    configs = [normalize_config({
        "equation": equation, "grid_size": grid_size, "steps": steps,
        "save_every": save_every, "dt": 0.001, "initial_condition": "random",
        "seed": 51, "viscosity": 0.05, "amplitude": 0.5,
    }) for equation in ("burgers1d", "navier_stokes2d")]
    if steps > 1000 or grid_size > 128 or checkpoint_every >= steps:
        raise ValueError(
            "Study requires grid <= 128, steps <= 1000, and a mid-trajectory checkpoint"
        )
    if checkpoint_every % save_every == 0:
        raise ValueError("Recovery proof checkpoint must fall between scheduled saved frames")
    for config in configs:
        validate_checkpoint_options(config, checkpoint_every)
    before = capture_provenance()
    if require_clean and (before["git_dirty"] is not False or not before["git_commit"]):
        raise ValueError(
            "Commit code and use clean source before running the measured recovery study"
        )
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "provenance": before, "configs": configs, "checkpoint_every": checkpoint_every,
        "worker_timeout_seconds": worker_timeout, "require_clean_source": require_clean,
        "protocol": (
            "uninterrupted reference; commit checkpoint; kill worker; resume; verify and hash"
        ),
    }
    _write_json(root / "plan.json", plan)
    started = time.perf_counter()
    cases = [_case(config, root / config["equation"], before, checkpoint_every, worker_timeout)
             for config in configs]
    failures = [{"equation": case["config"]["equation"], **case["error"]}
                for case in cases if case["status"] != "completed"]
    try:
        _same_runtime(before, capture_provenance())
    except RuntimeError as exc:
        failures.append({"type": type(exc).__name__, "message": str(exc)})
    report = {
        **plan, "status": "failed" if failures else "completed", "cases": cases,
        "failure_count": len(failures), "failures": failures,
        "wall_seconds": time.perf_counter() - started,
        "all_scientific_arrays_hash_identical": not failures,
        "measurement_scope": {
            "kill": "Owned worker process tree killed after a committed checkpoint receipt",
            "work": "Actual completed _rk4 calls counted independently in each interpreter",
            "equality": "All field, time, coordinate and diagnostic dtype/shape/array-byte hashes",
            "timing": "Single observation; engine timing includes checkpoint I/O and publication; "
                      "process timing also includes interpreter startup",
            "limits": (
                "Local process-crash recovery only; not power-loss durability, remote storage, "
                "mid-write kill coverage, hardware-independent equality or performance scaling"
            ),
            "data_retention": "Reference, checkpoint blobs, recovered artifacts and logs retained",
        },
    }
    _write_json(root / "report.json", report)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(args.worker.read_text(encoding="utf-8")))
        return 0
    report = run_recovery_study(args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
