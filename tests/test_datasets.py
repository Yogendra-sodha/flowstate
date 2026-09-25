import hashlib
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import zarr

from flowstate.datasets import export_dataset, import_pdebench, verify_dataset
from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve


def make_lake(path, count=6):
    lake = Lake(path)
    for seed in range(count):
        config = normalize_config(
            {
                "equation": "burgers1d",
                "initial_condition": "random",
                "seed": seed,
                "grid_size": 16,
                "steps": 4,
                "save_every": 1,
            }
        )
        lake.write(
            f"run-{seed}",
            {
                "id": f"run-{seed}",
                "status": "completed",
                "equation": "burgers1d",
                "solver": "finite_difference_rk4",
                "config": config,
                "metrics": {},
            },
            solve(config),
        )
    return lake


def test_export_roundtrip_frozen_splits_and_train_only_normalization(tmp_path):
    lake = make_lake(tmp_path / "lake")
    metadata = export_dataset(lake.root, tmp_path / "dataset", seed=42)
    group = zarr.open_group(str(tmp_path / "dataset"), mode="r")
    assert group["fields/u"].shape == (6, 5, 16)
    assert group["fields/u"].chunks == (1, 5, 16)
    assert group["fields/u"].dtype == np.dtype("float32")
    assert verify_dataset(tmp_path / "dataset") == []
    members = [set(metadata["splits"][name]) for name in ("train", "validation", "test")]
    assert all(members)
    assert set.union(*members) == set(range(6))
    assert all(not (a & b) for index, a in enumerate(members) for b in members[index + 1 :])
    training = np.stack([group["fields/u"][index] for index in metadata["splits"]["train"]])
    assert metadata["normalization"]["mean"] == pytest.approx(training.astype(float).mean())
    assert metadata["normalization"]["std"] == pytest.approx(training.astype(float).std())
    assert metadata["normalization"]["count"] == training.size
    assert metadata["normalization"]["fit_split"] == "train"
    for index, trajectory in enumerate(metadata["trajectories"]):
        source = zarr.open_group(str(lake.experiments / trajectory["id"] / "fields.zarr"))
        np.testing.assert_array_equal(
            group["fields/u"][index], source["fields/velocity"][:].astype("f4")
        )
        np.testing.assert_array_equal(group["x"][:], source["coordinates/x"][:])
        assert len(trajectory["source_manifest_sha256"]) == 64
    repeated = export_dataset(lake.root, tmp_path / "dataset-again", seed=42)
    assert repeated["splits"] == metadata["splits"]
    assert repeated["split_sha256"] == metadata["split_sha256"]
    assert repeated["normalization"] == metadata["normalization"]
    with pytest.raises(FileExistsError):
        export_dataset(lake.root, tmp_path / "dataset", seed=42)


def test_related_initial_conditions_cannot_cross_splits(tmp_path):
    lake = make_lake(tmp_path / "lake")
    config = {**lake.load_record("run-0")["config"], "amplitude": 2.0}
    lake.write(
        "variant",
        {
            "id": "variant",
            "status": "completed",
            "equation": "burgers1d",
            "config": config,
            "metrics": {},
        },
        solve(config),
    )
    metadata = export_dataset(lake.root, tmp_path / "dataset")
    membership = {
        metadata["trajectories"][index]["id"]: split
        for split, indices in metadata["splits"].items()
        for index in indices
    }
    assert membership["run-0"] == membership["variant"]
    families = {
        split: {metadata["trajectories"][i]["family_id"] for i in indices}
        for split, indices in metadata["splits"].items()
    }
    assert families["train"].isdisjoint(families["validation"] | families["test"])
    assert families["validation"].isdisjoint(families["test"])


@pytest.mark.parametrize("changed", [{"grid_size": 32}, {"dt": 0.002}, {"viscosity": 0.1}])
def test_incompatible_physical_contract_is_rejected(tmp_path, changed):
    lake = make_lake(tmp_path / "lake")
    config = {**lake.load_record("run-0")["config"], **changed}
    lake.write(
        "variant",
        {
            "id": "variant",
            "status": "completed",
            "equation": "burgers1d",
            "config": config,
        },
        solve(config),
    )
    with pytest.raises(ValueError, match="same physical time, grid, and viscosity"):
        export_dataset(lake.root, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_sine_seeds_are_not_independent_initial_conditions(tmp_path):
    lake = Lake(tmp_path / "lake")
    for seed in range(3):
        config = normalize_config(
            {"initial_condition": "sine", "seed": seed, "steps": 2, "save_every": 1}
        )
        lake.write(
            f"sine-{seed}",
            {
                "id": f"sine-{seed}",
                "status": "completed",
                "equation": "burgers1d",
                "config": config,
            },
            solve(config),
        )
    with pytest.raises(ValueError, match="3 distinct initial-condition families"):
        export_dataset(lake.root, tmp_path / "dataset")
    assert not list(tmp_path.glob(".staging-*"))


def test_tampered_source_and_duplicate_ids_are_rejected(tmp_path):
    lake = make_lake(tmp_path / "lake")
    with pytest.raises(ValueError, match="Duplicate"):
        export_dataset(lake.root, tmp_path / "dataset", ids=["run-0", "run-0"])
    (lake.experiments / "run-0" / "record.json").write_text("{}")
    with pytest.raises(ValueError, match="verification"):
        export_dataset(lake.root, tmp_path / "dataset", ids=["run-0", "run-1", "run-2"])


def test_flagged_completed_source_is_rejected_and_metrics_are_preserved(tmp_path):
    lake = make_lake(tmp_path / "lake")
    metadata = export_dataset(lake.root, tmp_path / "clean")
    assert all(item["metrics"] == {} for item in metadata["trajectories"])
    config = lake.load_record("run-0")["config"]
    lake.write(
        "flagged",
        {
            "id": "flagged",
            "status": "completed",
            "equation": "burgers1d",
            "config": config,
            "metrics": {"needs_review": True, "mass_drift": 0.1},
        },
        solve(config),
    )
    with pytest.raises(ValueError, match="needs_review"):
        export_dataset(lake.root, tmp_path / "dataset")


def make_hdf5(path, *, extra_time=False, duplicate=False):
    x = np.arange(8) / 8
    time = np.arange(5) * 0.1
    data = np.asarray(
        [[index + np.sin(2 * np.pi * x) * np.exp(-t) for t in time] for index in range(6)],
        dtype=np.float32,
    )
    if duplicate:
        data[1] = data[0]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
        handle.create_dataset("x-coordinate", data=x)
        handle.create_dataset("t-coordinate", data=np.arange(6) * 0.1 if extra_time else time)
    return data


def import_fixture(source, output, **overrides):
    options = {
        "source_url": "https://example.org/fixtures/burgers.hdf5",
        "source_version": "synthetic-test-v1",
        "license_name": "CC0-1.0",
        "viscosity": 0.01 / np.pi,
    }
    return import_pdebench(source, output, **{**options, **overrides})


def test_pdebench_contract_streams_samples_and_records_raw_provenance(tmp_path, monkeypatch):
    source = tmp_path / "burgers.hdf5"
    data = make_hdf5(source, extra_time=True, duplicate=True)
    original_getitem = h5py.Dataset.__getitem__
    reads = []

    def bounded_read(dataset, key, *args, **kwargs):
        if dataset.name == "/tensor":
            assert isinstance(key, (int, tuple)), (
                "Full tensor reads violate bounded-memory contract"
            )
            reads.append(key)
        return original_getitem(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_read)
    metadata = import_fixture(source, tmp_path / "dataset")
    group = zarr.open_group(str(tmp_path / "dataset"), mode="r")
    np.testing.assert_array_equal(group["fields/u"][:], data)
    assert reads and verify_dataset(tmp_path / "dataset") == []
    assert (
        metadata["provenance"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert metadata["provenance"]["discarded_terminal_time_coordinate"] == 0.5
    assert metadata["viscosity"] == pytest.approx(0.01 / np.pi)
    assert metadata["trajectories"][0]["family_id"] == metadata["trajectories"][1]["family_id"]
    assert metadata["provenance"]["license_name"] == "CC0-1.0"


@pytest.mark.parametrize(
    "options",
    [
        {"source_url": ""},
        {"source_version": ""},
        {"license_name": ""},
        {"viscosity": -1},
    ],
)
def test_import_requires_provenance_and_physical_viscosity(tmp_path, options):
    source = tmp_path / "burgers.hdf5"
    make_hdf5(source)
    with pytest.raises(ValueError):
        import_fixture(source, tmp_path / "dataset", **options)


def test_import_rejects_nonfinite_payload_without_publishing(tmp_path):
    source = tmp_path / "burgers.hdf5"
    make_hdf5(source)
    with h5py.File(source, "a") as handle:
        handle["tensor"][3, 2, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        import_fixture(source, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".staging-*"))


def test_manifest_tampering_is_detected(tmp_path):
    source = tmp_path / "burgers.hdf5"
    make_hdf5(source)
    import_fixture(source, tmp_path / "dataset")
    metadata_file = tmp_path / "dataset" / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["normalization"]["mean"] = 12345
    metadata_file.write_text(json.dumps(metadata))
    assert verify_dataset(tmp_path / "dataset") == ["Hash mismatch: metadata.json"]


def test_shortened_last_time_step_rejected(tmp_path):
    lake = make_lake(tmp_path / "lake")
    config = lake.load_record("run-0")["config"]
    solved = solve(config)
    invalid = SimpleNamespace(
        **{**vars(solved), "times": np.array([0.0, 0.001, 0.002, 0.003, 0.0035])}
    )
    lake.write(
        "nonuniform",
        {
            "id": "nonuniform",
            "status": "completed",
            "equation": "burgers1d",
            "config": config,
        },
        invalid,
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        export_dataset(lake.root, tmp_path / "dataset")
