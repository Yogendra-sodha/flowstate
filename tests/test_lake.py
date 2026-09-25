import errno
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest
import zarr

import flowstate.lake as lake_module
from flowstate.lake import Lake


def record(experiment_id="run-1", **overrides):
    value = {
        "id": experiment_id,
        "status": "completed",
        "equation": "burgers_1d",
        "solver": "finite_difference",
        "created_at": "2026-09-22T12:00:00+00:00",
        "parent_id": None,
        "config": {"viscosity": 0.02, "grid_size": 128, "dt": 0.001, "steps": 2, "seed": 7},
        "provenance": {"git_commit": "abc123", "precision": "float64"},
        "metrics": {"initial_energy": 0.5, "final_energy": 0.4, "mass_drift": 1e-14},
        "error": None,
    }
    value.update(overrides)
    return value


def result():
    return SimpleNamespace(
        times=np.array([0.0, 0.001, 0.002]),
        fields={"velocity": np.arange(3 * 128, dtype=np.float64).reshape(3, 128)},
        coordinates={"x": np.linspace(0, 1, 128, endpoint=False)},
        diagnostics={"energy": np.array([0.5, 0.45, 0.4])},
        metadata={"boundary_conditions": "periodic", "precision": "float64"},
    )


def test_zarr_roundtrip_and_chunking(tmp_path):
    lake = Lake(tmp_path)
    simulation = result()
    path = lake.write("run-1", record(), simulation)

    assert lake.exists("run-1")
    assert lake.load_record("run-1") == record()
    saved = zarr.open_group(str(path / "fields.zarr"), mode="r")
    np.testing.assert_array_equal(saved["time"][:], simulation.times)
    np.testing.assert_array_equal(saved["fields/velocity"][:], simulation.fields["velocity"])
    np.testing.assert_array_equal(saved["coordinates/x"][:], simulation.coordinates["x"])
    np.testing.assert_array_equal(saved["diagnostics/energy"][:], simulation.diagnostics["energy"])
    assert saved["fields/velocity"].chunks == (1, 64)
    assert saved.attrs["simulation_metadata"] == simulation.metadata
    assert lake.verify("run-1") == []


def test_query_reconstructs_catalog_and_unions_metric_columns(tmp_path):
    lake = Lake(tmp_path)
    lake.write("run-1", record(), None)
    second = record(
        "run-2", metrics={"initial_energy": 2.0, "final_energy": 1.0, "max_divergence": 0.0}
    )
    second["config"]["viscosity"] = 0.1
    lake.write("run-2", second, None)
    reopened = Lake(tmp_path)
    assert reopened.query("SELECT id, final_energy FROM experiments WHERE viscosity < 0.05") == [
        {"id": "run-1", "final_energy": 0.4}
    ]
    assert reopened.query("SELECT count(*) AS n FROM experiments") == [{"n": 2}]
    assert reopened.query("SELECT mass_drift FROM experiments WHERE id = 'run-2'") == [
        {"mass_drift": None}
    ]


def test_empty_lake_is_queryable(tmp_path):
    assert Lake(tmp_path).query("SELECT id FROM experiments WHERE viscosity > 0") == []


def test_review_flags_are_queryable_in_empty_and_failed_only_lakes(tmp_path):
    lake = Lake(tmp_path)
    query = "SELECT id FROM experiments WHERE needs_review OR energy_increase_observed"
    assert lake.query(query) == []
    lake.write("failure", record("failure", status="failed", metrics={}), None)
    assert lake.query(query) == []
    assert lake.query("SELECT needs_review, energy_increase_observed FROM experiments") == [
        {"needs_review": None, "energy_increase_observed": None}
    ]


def test_failed_null_metrics_union_with_completed_numeric_metrics(tmp_path):
    lake = Lake(tmp_path)
    lake.write("a-failed", record("a-failed", status="failed", metrics={}, error="unstable"), None)
    lake.write("b-completed", record("b-completed"), result())
    assert lake.query("SELECT id, final_energy, error FROM experiments ORDER BY id") == [
        {"id": "a-failed", "final_energy": None, "error": "unstable"},
        {"id": "b-completed", "final_energy": 0.4, "error": None},
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM experiments",
        "CREATE TABLE stolen AS SELECT * FROM experiments",
        "SELECT * FROM experiments; SELECT 1",
        "SET enable_external_access = true",
        "INSTALL httpfs",
        "COPY experiments TO 'export.csv'",
    ],
)
def test_query_rejects_mutation_and_multiple_statements(tmp_path, sql):
    with pytest.raises(ValueError, match="single read-only SELECT"):
        Lake(tmp_path).query(sql)


def test_query_disables_external_file_access(tmp_path):
    lake = Lake(tmp_path)
    path = lake.write("run-1", record(), None)
    external = (path / "record.json").as_posix().replace("'", "''")
    with pytest.raises(duckdb.Error, match="(?i)(permission|disabled|external)"):
        lake.query(f"SELECT * FROM read_json_auto('{external}')")
    assert not (tmp_path / "export.csv").exists()


def test_write_refuses_overwrite_and_cleans_failed_staging(tmp_path):
    lake = Lake(tmp_path)
    lake.write("run-1", record(), None)
    with pytest.raises(FileExistsError):
        lake.write("run-1", record(status="failed"), None)
    assert lake.load_record("run-1")["status"] == "completed"
    invalid = result()
    invalid.fields["velocity"] = np.zeros((1, 128))
    with pytest.raises(ValueError, match="saved time dimension"):
        lake.write("broken", record("broken"), invalid)
    assert not lake.exists("broken")
    assert [path.name for path in lake.experiments.iterdir()] == ["run-1"]


@pytest.mark.parametrize("invalid_id", ["../outside", "..", "", "a/b", "a\\b", "/absolute", "a:b"])
def test_ids_cannot_escape_lake(tmp_path, invalid_id):
    with pytest.raises(ValueError, match="Experiment ID"):
        Lake(tmp_path).write(invalid_id, record(invalid_id), None)


def test_failed_experiment_has_metadata_without_fields(tmp_path):
    lake = Lake(tmp_path)
    path = lake.write(
        "failure", record("failure", status="failed", error="CFL limit exceeded"), None
    )
    assert not (path / "fields.zarr").exists()
    assert lake.query("SELECT id, error FROM experiments WHERE status = 'failed'") == [
        {"id": "failure", "error": "CFL limit exceeded"}
    ]
    assert lake.verify("failure") == []


def test_lineage_and_staging_exclusion(tmp_path):
    lake = Lake(tmp_path)
    lake.write("parent", record("parent"), None)
    lake.write("child", record("child", parent_id="parent"), None)
    (lake.experiments / ".staging-interrupted").mkdir()
    assert len(lake.records()) == 2
    graph = lake.graph()
    assert {node["id"] for node in graph["nodes"]} == {"parent", "child"}
    assert graph["edges"] == [{"source": "parent", "target": "child", "type": "parent_of"}]


def test_tampering_metadata_and_zarr_are_detected(tmp_path):
    lake = Lake(tmp_path)
    path = lake.write("run-1", record(), result())
    (path / "record.json").write_text("{}", encoding="utf-8")
    chunk = next(
        file
        for file in (path / "fields.zarr" / "fields" / "velocity").rglob("*")
        if file.is_file() and file.name != "zarr.json"
    )
    chunk_relative = chunk.relative_to(path).as_posix()
    chunk.unlink()
    (path / "unexpected.txt").write_text("untracked", encoding="utf-8")
    assert set(lake.verify("run-1")) == {
        "Hash mismatch: record.json",
        f"Missing artifact: {chunk_relative}",
        "Unexpected artifact: unexpected.txt",
    }


def test_invalid_manifest_is_reported(tmp_path):
    lake = Lake(tmp_path)
    path = lake.write("run-1", record(), None)
    (path / "manifest.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    assert lake.verify("run-1") == ["Missing or invalid manifest.json"]


def test_concurrent_distinct_writes_are_all_visible(tmp_path):
    lake = Lake(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lake.write, f"run-{i}", record(f"run-{i}"), None) for i in range(6)]
        for future in futures:
            future.result()
    assert len(lake.records()) == 6
    assert lake.query("SELECT count(*) AS n FROM experiments") == [{"n": 6}]
    assert all(lake.verify(f"run-{i}") == [] for i in range(6))


@pytest.mark.parametrize("collision_errno", [errno.EEXIST, errno.ENOTEMPTY])
def test_same_id_publication_race_preserves_winner(tmp_path, monkeypatch, collision_errno):
    lake = Lake(tmp_path)
    original_rename = os.rename
    publishing = Barrier(2)

    def simultaneous_rename(source, destination):
        # Both writers have finished staging and passed the initial exists check.
        publishing.wait(timeout=10)
        try:
            return original_rename(source, destination)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                # Exercise both Windows and POSIX collision codes on either OS.
                raise OSError(collision_errno, "Publication collision") from error
            raise

    monkeypatch.setattr(os, "rename", simultaneous_rename)
    winners = []
    collisions = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(lake.write, "shared", record("shared", provenance={"writer": index}), None)
            for index in range(2)
        ]
        for index, future in enumerate(futures):
            try:
                future.result()
                winners.append(index)
            except FileExistsError:
                collisions.append(index)
    assert len(winners) == len(collisions) == 1
    assert lake.load_record("shared")["provenance"]["writer"] == winners[0]
    assert lake.verify("shared") == []
    assert [path.name for path in lake.experiments.iterdir()] == ["shared"]


def test_noncollision_publication_error_propagates_and_cleans_staging(tmp_path, monkeypatch):
    lake = Lake(tmp_path)

    def denied_rename(source, destination):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(os, "rename", denied_rename)
    with pytest.raises(PermissionError):
        lake.write("denied", record("denied"), None)
    assert list(lake.experiments.iterdir()) == []


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_windows_publication_denial_recovers_without_partial_visibility(
    tmp_path, monkeypatch, winerror
):
    lake = Lake(tmp_path)
    real_rename = os.rename
    attempts, delays = [], []

    def temporarily_denied(source, destination):
        attempts.append((source, destination))
        assert source.is_dir()
        assert not destination.exists()
        assert lake.records() == []
        if len(attempts) <= 2:
            error = PermissionError(errno.EACCES, "Temporary Windows publication denial")
            error.winerror = winerror
            raise error
        return real_rename(source, destination)

    monkeypatch.setattr(os, "rename", temporarily_denied)
    monkeypatch.setattr(lake_module.time, "sleep", delays.append)
    destination = lake.write("recovered", record("recovered"), None)
    assert destination.is_dir()
    assert len(attempts) == 3
    assert delays == [0.01, 0.05]
    assert lake.verify("recovered") == []
    assert [path.name for path in lake.experiments.iterdir()] == ["recovered"]


def test_windows_publication_exhaustion_preserves_error_and_cleans_staging(tmp_path, monkeypatch):
    lake = Lake(tmp_path)
    attempts, delays = [], []
    failure = PermissionError(errno.EACCES, "Persistent Windows publication denial")
    failure.winerror = 5

    def always_denied(source, destination):
        attempts.append((source, destination))
        raise failure

    monkeypatch.setattr(os, "rename", always_denied)
    monkeypatch.setattr(lake_module.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as caught:
        lake.write("denied", record("denied"), None)
    assert caught.value is failure
    assert len(attempts) == 6
    assert delays == [0.01, 0.05, 0.2, 0.5, 1.0]
    assert sum(delays) == pytest.approx(1.76)
    assert list(lake.experiments.iterdir()) == []


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_publication_errors_without_windows_code_are_not_retried(
    tmp_path, monkeypatch, error_number
):
    lake = Lake(tmp_path)
    attempts, delays = [], []
    failure = OSError(error_number, "Non-Windows publication failure")

    def denied(source, destination):
        attempts.append((source, destination))
        raise failure

    monkeypatch.setattr(os, "rename", denied)
    monkeypatch.setattr(lake_module.time, "sleep", delays.append)
    with pytest.raises(OSError) as caught:
        lake.write("denied", record("denied"), None)
    assert caught.value is failure
    assert len(attempts) == 1
    assert delays == []
    assert list(lake.experiments.iterdir()) == []


@pytest.mark.parametrize("collision_errno", [errno.EEXIST, errno.ENOTEMPTY])
def test_publication_collision_is_not_retried(tmp_path, monkeypatch, collision_errno):
    lake = Lake(tmp_path)
    attempts, delays = [], []

    def collision(source, destination):
        attempts.append((source, destination))
        raise OSError(collision_errno, "Publication collision")

    monkeypatch.setattr(os, "rename", collision)
    monkeypatch.setattr(lake_module.time, "sleep", delays.append)
    with pytest.raises(FileExistsError):
        lake.write("collision", record("collision"), None)
    assert len(attempts) == 1
    assert delays == []
    assert list(lake.experiments.iterdir()) == []
