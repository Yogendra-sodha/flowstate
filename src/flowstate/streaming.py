"""Incrementally persist saved fields while retaining only the active solver state."""

import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import zarr


class ZarrFrameSink:
    """Synchronous callback: a frame is written before the solver reuses its arrays.

    Diagnostic histories remain in memory (O(saved frames)); field storage no
    longer grows as O(saved frames * spatial cells). RK/FFT working arrays remain.
    """

    def __init__(self, path: str | Path, config: dict):
        self.path = Path(path)
        self.count = (
            1
            + config["steps"] // config["save_every"]
            + bool(config["steps"] % config["save_every"])
        )
        self.group = zarr.open_group(str(path), mode="w")
        self.fields = self.group.create_group("fields")
        self.times = []

    def __call__(self, time: float, fields: dict) -> None:
        index = len(self.times)
        if index >= self.count:
            raise ValueError("More frames than expected")
        for name, values in fields.items():
            values = np.asarray(values)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite saved field")
            try:
                if index == 0:
                    self.fields.create_array(
                        name,
                        shape=(self.count, *values.shape),
                        dtype=values.dtype,
                        chunks=(1, *(min(n, 64) for n in values.shape)),
                    )
                self.fields[name][index] = values
            except Exception as exc:
                raise OSError(f"Saving streamed field {name} at frame {index} failed") from exc
        self.times.append(time)

    def finish(self, result) -> None:
        if len(self.times) != self.count or not np.array_equal(self.times, result.times):
            raise ValueError("Saved fields and diagnostic times disagree")
        self.group.attrs["simulation_metadata"] = result.metadata
        self.group.create_array("time", data=result.times, chunks=(min(self.count, 1024),))
        for collection in ("coordinates", "diagnostics"):
            target = self.group.create_group(collection)
            for name, values in getattr(result, collection).items():
                target.create_array(name, data=values, chunks=(min(len(values), 1024),))


def benchmark_storage(output: str | Path, *, grid_size: int = 64, steps: int = 32) -> dict:
    """Measure serial local writes and a warm chunk read; Python tracing is not RSS."""
    from flowstate.engine import capture_provenance
    from flowstate.lake import Lake
    from flowstate.numerics import normalize_config, solve

    if tracemalloc.is_tracing():
        raise RuntimeError("Storage benchmark needs exclusive tracemalloc ownership")
    config = normalize_config(
        {
            "equation": "navier_stokes2d",
            "grid_size": grid_size,
            "steps": steps,
            "dt": 0.0005,
            "save_every": 1,
            "initial_condition": "random",
            "seed": 5,
        }
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for mode in ("buffered", "streamed"):
        path = root / f"{mode}.zarr"
        tracemalloc.start()
        start = time.perf_counter()
        try:
            if mode == "buffered":
                result = solve(config)
                Lake._write_arrays(path, result)
            else:
                sink = ZarrFrameSink(path, config)
                result = solve(config, frame_callback=sink, retain_fields=False)
                sink.finish(result)
            elapsed = time.perf_counter() - start
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        del result
        group = zarr.open_group(str(path), mode="r")
        digest = hashlib.sha256()
        # Read one frame at a time; verifying equality must not reassemble a trajectory.
        for name in ("u", "v", "vorticity"):
            for frame in range(steps + 1):
                digest.update(group[f"fields/{name}"][frame].tobytes())
        array = group["fields/u"]
        array[0, : min(grid_size, 32), : min(grid_size, 32)]
        started = time.perf_counter()
        for _ in range(10):
            array[0, : min(grid_size, 32), : min(grid_size, 32)]
        warm_read_seconds = (time.perf_counter() - started) / 10
        files = [path for path in path.rglob("*") if path.is_file()]
        reports.append(
            {
                "mode": mode,
                "solve_and_write_seconds": elapsed,
                "peak_traced_allocation_bytes": peak,
                "stored_bytes": sum(path.stat().st_size for path in files),
                "artifact_files": len(files),
                "field_sha256": digest.hexdigest(),
                "warm_frame_tile_read_seconds": warm_read_seconds,
            }
        )
    report = {
        "schema_version": 1,
        "config": config,
        "cases": reports,
        "field_values_identical": reports[0]["field_sha256"] == reports[1]["field_sha256"],
        "provenance": capture_provenance(),
        "measurement_scope": {
            "memory": "tracemalloc peak, not RSS; excludes untraced native/OS allocations",
            "io": "serial local solve+Zarr writes; tile read is warm and includes decompression",
            "limitations": "No concurrency/cloud latency claim; no performance threshold asserted",
        },
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["field_values_identical"]:
        raise RuntimeError("Buffered and streamed field hashes disagree")
    return report
