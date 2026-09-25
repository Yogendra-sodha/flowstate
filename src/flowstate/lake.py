"""Immutable local experiment artifacts and a reconstructible SQL catalog."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SAFE_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_RENAME_RETRY_DELAYS = (0.01, 0.05, 0.2, 0.5, 1.0)
_RENAME_RETRY_WINERRORS = frozenset({5, 32, 33})
_CONFIG_COLUMNS = ("viscosity", "grid_size", "dt", "steps", "seed")
_METRIC_COLUMNS = (
    "initial_energy",
    "final_energy",
    "max_divergence",
    "mass_drift",
    "reynolds",
    "needs_review",
    "energy_increase_observed",
)


def _json_value(value: Any) -> Any:
    """Convert numerical scalars while rejecting non-portable JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _encode(value: Any) -> str:
    return json.dumps(value, default=_json_value, sort_keys=True, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rename_with_retry(source: Path, destination: Path) -> None:
    """Retry transient Windows rename denials, preserving os.rename semantics.

    Windows access/sharing/lock denials can be temporary, including on synced
    directories. Retry only their explicit Win32 codes, for at most 1.76 seconds
    of delay. Existing-destination and all other errors propagate immediately;
    a persistent denial propagates unchanged after the last attempt.
    """
    for delay in (*_RENAME_RETRY_DELAYS, None):
        try:
            os.rename(source, destination)
        except OSError as error:
            if delay is None or getattr(error, "winerror", None) not in _RENAME_RETRY_WINERRORS:
                raise
            time.sleep(delay)
        else:
            return


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _encode(value)


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: record.get(key)
        for key in ("id", "status", "equation", "solver", "created_at", "parent_id")
    }
    row["error"] = _scalar(record.get("error"))
    config = record.get("config") or {}
    metrics = record.get("metrics") or {}
    row.update({key: _scalar(config.get(key)) for key in _CONFIG_COLUMNS})
    row.update({key: _scalar(metrics.get(key)) for key in _METRIC_COLUMNS})
    for key, value in metrics.items():
        # A metric must never silently replace identity or configuration columns.
        column = key if key not in row or key in _METRIC_COLUMNS else f"metric_{key}"
        row[column] = _scalar(value)
    row["config_json"] = _encode(config)
    row["provenance_json"] = _encode(record.get("provenance") or {})
    row["metrics_json"] = _encode(metrics)
    return row


class Lake:
    """Local, append-only lake; each committed run is its own atomic directory.

    The Parquet catalog is read directly from immutable experiment directories,
    so no separately maintained database can drift out of sync. A write stages
    all files beside the destination and publishes them with one rename.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.experiments = self.root / "experiments"
        self.experiments.mkdir(parents=True, exist_ok=True)

    def _path(self, experiment_id: str) -> Path:
        if not isinstance(experiment_id, str) or not _SAFE_ID.fullmatch(experiment_id):
            raise ValueError("Experiment ID must contain only letters, numbers, '_' or '-'")
        path = self.experiments / experiment_id
        if path.is_symlink():
            raise ValueError("Experiment paths must not be symbolic links")
        return path

    def exists(self, experiment_id: str) -> bool:
        return self._path(experiment_id).is_dir()

    def load_record(self, experiment_id: str) -> dict[str, Any]:
        return json.loads((self._path(experiment_id) / "record.json").read_text("utf-8"))

    def _run_dirs(self) -> list[Path]:
        return sorted(
            path
            for path in self.experiments.iterdir()
            if path.is_dir() and not path.is_symlink() and _SAFE_ID.fullmatch(path.name)
        )

    def records(self) -> list[dict[str, Any]]:
        return [self.load_record(path.name) for path in self._run_dirs()]

    def write(
        self,
        experiment_id: str,
        record: dict,
        result: Any | None,
        *,
        field_store: Path | None = None,
    ) -> Path:
        """Persist one result, refusing to overwrite an existing experiment.

        Failed experiments may have ``result=None``. Invalid input or interrupted
        writes never expose a partially built final experiment directory.
        """
        destination = self._path(experiment_id)
        if destination.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_id}")
        if record.get("id", experiment_id) != experiment_id:
            raise ValueError("Record ID does not match experiment ID")
        document = dict(record, id=experiment_id)
        # Validate and normalize before staging any files.
        document = json.loads(_encode(document))
        staging = Path(tempfile.mkdtemp(prefix=f".staging-{experiment_id}-", dir=self.experiments))
        try:
            (staging / "record.json").write_text(_encode(document) + "\n", encoding="utf-8")
            pq.write_table(pa.Table.from_pylist([_flatten(document)]), staging / "metadata.parquet")
            if field_store is not None:
                # Copy chunk files without loading the full trajectory into memory.
                shutil.copytree(field_store, staging / "fields.zarr")
            elif result is not None:
                self._write_arrays(staging / "fields.zarr", result)
            artifacts = {
                file.relative_to(staging).as_posix(): _sha256(file)
                for file in sorted(staging.rglob("*"))
                if file.is_file()
            }
            (staging / "manifest.json").write_text(
                _encode({"version": 1, "algorithm": "sha256", "artifacts": artifacts}) + "\n",
                encoding="utf-8",
            )
            # os.rename is atomic on this same filesystem. An existing nonempty
            # final directory cannot be replaced, including concurrent writes.
            try:
                _rename_with_retry(staging, destination)
            except OSError as error:
                if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                    raise FileExistsError(
                        errno.EEXIST,
                        f"Experiment already exists: {experiment_id}",
                        str(destination),
                    ) from error
                raise
        except BaseException:
            if staging.resolve().is_relative_to(self.experiments.resolve()):
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    @staticmethod
    def _write_arrays(path: Path, result: Any) -> None:
        times = np.asarray(result.times)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("Result times must be a nonempty one-dimensional array")
        group = zarr.open_group(str(path), mode="w")
        group.attrs["simulation_metadata"] = json.loads(_encode(result.metadata))
        group.create_array("time", data=times, chunks=(min(times.size, 1024),))
        for collection in ("fields", "coordinates", "diagnostics"):
            subgroup = group.create_group(collection)
            for name, values in getattr(result, collection).items():
                if not _SAFE_ARRAY_NAME.fullmatch(name):
                    raise ValueError(f"Unsafe array name: {name!r}")
                array = np.asarray(values)
                if array.ndim == 0 or any(size == 0 for size in array.shape):
                    raise ValueError(f"{collection}/{name} must have nonempty dimensions")
                if array.dtype.kind not in "biufc":
                    raise ValueError(f"{collection}/{name} must be a numeric array")
                if collection != "coordinates" and array.shape[0] != times.size:
                    raise ValueError(f"{collection}/{name} does not match the saved time dimension")
                if collection == "fields":
                    chunks = (1,) + tuple(min(size, 64) for size in array.shape[1:])
                else:
                    chunks = tuple(min(size, 1024) for size in array.shape)
                subgroup.create_array(name, data=array, chunks=chunks)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Evaluate one SELECT against ``experiments`` in an isolated connection.

        Artifacts are ingested first, then filesystem/network access and further
        configuration changes are disabled before user SQL is parsed or run.
        """
        with duckdb.connect(":memory:") as connection:
            paths = [str(path / "metadata.parquet") for path in self._run_dirs()]
            if paths:
                connection.read_parquet(paths, union_by_name=True).create("experiments")
            else:
                # Stable empty schema keeps ordinary filters useful before the
                # first experiment has run.
                connection.execute(
                    "CREATE TABLE experiments (id VARCHAR, status VARCHAR, equation VARCHAR, "
                    "solver VARCHAR, created_at VARCHAR, parent_id VARCHAR, error VARCHAR, "
                    "viscosity DOUBLE, grid_size BIGINT, dt DOUBLE, steps BIGINT, seed BIGINT, "
                    "initial_energy DOUBLE, final_energy DOUBLE, max_divergence DOUBLE, "
                    "mass_drift DOUBLE, reynolds DOUBLE, needs_review BOOLEAN, "
                    "energy_increase_observed BOOLEAN, config_json VARCHAR, "
                    "provenance_json VARCHAR, metrics_json VARCHAR)"
                )
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
            statements = connection.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise ValueError("Only a single read-only SELECT statement is allowed")
            cursor = connection.execute(sql)
            names = [column[0] for column in cursor.description]
            return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        """Return experiment lineage, including explicit absent-parent nodes."""
        records = self.records()
        nodes = {
            record["id"]: {
                "id": record["id"],
                "type": "experiment",
                "status": record.get("status"),
                "equation": record.get("equation"),
                "solver": record.get("solver"),
            }
            for record in records
        }
        edges = []
        for record in records:
            parent = record.get("parent_id")
            if parent:
                nodes.setdefault(parent, {"id": parent, "type": "experiment", "missing": True})
                edges.append({"source": parent, "target": record["id"], "type": "parent_of"})
        return {"nodes": list(nodes.values()), "edges": edges}

    def verify(self, experiment_id: str) -> list[str]:
        """Return integrity problems; an empty list means all hashes match.

        This detects accidental damage, not malicious changes to both artifacts
        and the unsigned manifest. Verification intentionally scans every chunk.
        """
        directory = self._path(experiment_id)
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            artifacts = manifest["artifacts"]
            if (
                manifest.get("version") != 1
                or manifest.get("algorithm") != "sha256"
                or not isinstance(artifacts, dict)
            ):
                return ["Invalid manifest format"]
        except (OSError, ValueError, KeyError, TypeError):
            return ["Missing or invalid manifest.json"]
        problems = []
        for relative, expected in artifacts.items():
            artifact = Path(relative)
            if (
                artifact.is_absolute()
                or ".." in artifact.parts
                or "\\" in relative
                or ":" in relative
            ):
                problems.append(f"Unsafe manifest path: {relative}")
                continue
            path = directory / artifact
            if not path.is_file():
                problems.append(f"Missing artifact: {relative}")
            elif any(
                (directory / Path(*artifact.parts[:index])).is_symlink()
                for index in range(1, len(artifact.parts) + 1)
            ):
                problems.append(f"Symbolic link artifact: {relative}")
            else:
                try:
                    if _sha256(path) != expected:
                        problems.append(f"Hash mismatch: {relative}")
                except OSError:
                    problems.append(f"Unreadable artifact: {relative}")
        actual = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path != manifest_path
        }
        for unexpected in sorted(actual - artifacts.keys()):
            problems.append(f"Unexpected artifact: {unexpected}")
        return problems
