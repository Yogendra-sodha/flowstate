import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import zarr

import flowstate.public_data as public_data
from flowstate.datasets import import_pdebench, verify_dataset


@pytest.fixture
def remote_fixture(tmp_path, monkeypatch):
    """Exercise the real HDF5 acquisition without depending on a public server."""
    path = tmp_path / "remote.hdf5"
    x = (np.arange(8, dtype=np.float32) + 0.5) / 8
    time = np.arange(6, dtype=np.float32) / 10
    values = np.asarray(
        [
            [index + (index + 1) * np.sin(2 * np.pi * x) * np.exp(-t) for t in time[:5]]
            for index in range(8)
        ],
        dtype="f4",
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=values)
        handle.create_dataset("x-coordinate", data=x)
        handle.create_dataset("t-coordinate", data=time)
    source = {
        **public_data.SOURCE,
        "source_url": "https://example.test/pdebench-fixture.hdf5",
        "source_version": "Synthetic test fixture v1",
        "tensor_shape": [8, 5, 8],
        "size_bytes": path.stat().st_size,
        "publisher_checksum": {
            "algorithm": "MD5",
            "value": hashlib.md5(path.read_bytes()).hexdigest(),
        },
    }
    calls, handles = [], []
    transfer = {
        "source_url": source["source_url"],
        "source_size_bytes": source["size_bytes"],
        "strong_etag": '"fixture-etag"',
        "requests": 1,
        "bytes_received": source["size_bytes"],
        "full_remote_checksum_verified": False,
    }

    @contextmanager
    def local_reader(url, expected_size, *, max_bytes):
        calls.append((url, expected_size, max_bytes))
        assert url == source["source_url"]
        assert expected_size == source["size_bytes"]
        with path.open("rb") as handle:
            handle.receipt = lambda: transfer.copy()
            handles.append(handle)
            yield handle

    monkeypatch.setattr(public_data, "SOURCE", source)
    monkeypatch.setattr(public_data, "HTTPRangeReader", local_reader)
    monkeypatch.setattr(public_data, "capture_provenance", lambda: {"git_commit": "fixture"})
    return SimpleNamespace(
        path=path,
        values=values,
        x=x,
        time=time,
        source=source,
        calls=calls,
        handles=handles,
        transfer=transfer,
    )


def acquire(output, **options):
    return public_data.acquire_pdebench(
        output, **{"samples": 4, "sample_seed": 12, "time_stop": 5, **options}
    )


def assert_unpublished(output):
    assert not output.exists()
    assert not list(output.parent.glob(f".staging-{output.name}-*"))


def test_acquisition_preserves_selected_values_coordinates_and_provenance(
    tmp_path, remote_fixture, monkeypatch
):
    output = tmp_path / "acquired"
    expected = sorted(np.random.default_rng(12).choice(8, size=4, replace=False).tolist())
    original_getitem = h5py.Dataset.__getitem__
    reads = []

    def selected_read(dataset, key, *args, **kwargs):
        if dataset.name == "/tensor":
            assert key[0] in expected
            assert key[1:] == (slice(None, 5), slice(None))
            reads.append(key[0])
        return original_getitem(dataset, key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset, "__getitem__", selected_read)
        metadata = acquire(output)
    assert reads == expected
    assert public_data.verify_acquisition(output) == []
    assert all(handle.closed for handle in remote_fixture.handles)
    with h5py.File(output / "source.hdf5", "r") as handle:
        np.testing.assert_array_equal(handle["tensor"][:], remote_fixture.values[expected])
        np.testing.assert_array_equal(handle["x-coordinate"][:], remote_fixture.x)
        np.testing.assert_array_equal(handle["t-coordinate"][:], remote_fixture.time[:5])
    assert metadata["selection"] == {
        "sample_indices": expected,
        "time_start": 0,
        "time_stop": 5,
        "spatial_stride": 1,
    }
    assert metadata["shape"] == [4, 5, 8]
    assert metadata["time_range"] == [0.0, float(remote_fixture.time[4])]
    assert metadata["discarded_terminal_time_coordinate"] == float(remote_fixture.time[-1])
    assert metadata["source"] == remote_fixture.source
    assert metadata["transfer"] == remote_fixture.transfer
    assert metadata["transfer"]["full_remote_checksum_verified"] is False
    assert metadata["provenance"] == {"git_commit": "fixture"}
    assert (
        metadata["source_hdf5_sha256"]
        == hashlib.sha256((output / "source.hdf5").read_bytes()).hexdigest()
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["artifacts"]) == {"metadata.json", "source.hdf5"}
    for name, digest in manifest["artifacts"].items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()
    assert acquire(tmp_path / "repeated")["selection"] == metadata["selection"]
    calls = len(remote_fixture.calls)
    with pytest.raises(FileExistsError):
        acquire(output)
    assert len(remote_fixture.calls) == calls


def test_import_acquisition_binds_remote_indices_and_local_hashes(tmp_path, remote_fixture):
    acquired = tmp_path / "acquired"
    receipt = acquire(acquired)
    output = tmp_path / "canonical"
    metadata = public_data.import_acquired_pdebench(acquired, output, seed=17)
    assert verify_dataset(output) == []
    group = zarr.open_group(str(output), mode="r")
    selected = receipt["selection"]["sample_indices"]
    np.testing.assert_array_equal(group["fields/u"][:], remote_fixture.values[selected])
    np.testing.assert_array_equal(group["x"][:], remote_fixture.x)
    np.testing.assert_array_equal(group["time"][:], remote_fixture.time[:5])
    assert [item["source_index"] for item in metadata["trajectories"]] == list(range(4))
    assert [item["remote_source_index"] for item in metadata["trajectories"]] == selected
    assert metadata["provenance"]["acquisition"] == receipt
    assert (
        metadata["provenance"]["acquisition_manifest_sha256"]
        == hashlib.sha256((acquired / "manifest.json").read_bytes()).hexdigest()
    )
    assert metadata["provenance"]["source_sha256"] == receipt["source_hdf5_sha256"]
    assert (
        metadata["provenance"]["source_sha256_scope"] == "Complete derived local subset HDF5 file"
    )
    assert metadata["viscosity"] == pytest.approx(0.01 / np.pi)
    training = remote_fixture.values[[selected[index] for index in metadata["splits"]["train"]]]
    assert metadata["normalization"]["mean"] == pytest.approx(training.astype(float).mean())
    assert metadata["normalization"]["count"] == training.size


@pytest.mark.parametrize("artifact", ["source.hdf5", "metadata.json", "manifest.json"])
def test_acquisition_tampering_blocks_import(tmp_path, remote_fixture, artifact):
    acquired = tmp_path / "acquired"
    acquire(acquired)
    path = acquired / artifact
    if artifact == "source.hdf5":
        with h5py.File(path, "a") as handle:
            handle["tensor"][0, 0, 0] += 1
    else:
        path.write_text("{}")
    assert public_data.verify_acquisition(acquired)
    output = tmp_path / "canonical"
    with pytest.raises(ValueError, match="Acquisition verification failed"):
        public_data.import_acquired_pdebench(acquired, output)
    assert_unpublished(output)


def test_unexpected_acquisition_file_is_rejected(tmp_path, remote_fixture):
    acquired = tmp_path / "acquired"
    acquire(acquired)
    (acquired / "unrecorded.txt").write_text("unrecorded")
    assert public_data.verify_acquisition(acquired) == ["Unexpected acquisition artifacts"]


@pytest.mark.parametrize(
    "options, match",
    [
        ({"samples": 2}, "samples"),
        ({"samples": True}, "samples"),
        ({"sample_seed": -1}, "sample_seed"),
        ({"time_stop": 6}, "time_stop"),
        ({"max_bytes": 7}, "max_bytes"),
        ({"max_bytes": True}, "max_bytes"),
        ({"max_bytes": 256 * 1024**2 + 1}, "max_bytes"),
    ],
)
def test_invalid_selection_and_budget_fail_before_remote_access(
    tmp_path, remote_fixture, options, match
):
    output = tmp_path / "acquired"
    with pytest.raises(ValueError, match=match):
        acquire(output, **options)
    assert remote_fixture.calls == []
    assert_unpublished(output)


def test_transfer_budget_failure_cleans_staging(tmp_path, remote_fixture, monkeypatch):
    @contextmanager
    def exhausted_reader(*args, **kwargs):
        raise ValueError("HTTP range transfer budget exhausted")
        yield  # pragma: no cover

    monkeypatch.setattr(public_data, "HTTPRangeReader", exhausted_reader)
    output = tmp_path / "acquired"
    with pytest.raises(ValueError, match="transfer budget exhausted"):
        acquire(output, max_bytes=8)
    assert_unpublished(output)


@pytest.mark.parametrize("failure", ["remote", "nonfinite"])
def test_failed_trajectory_read_removes_partial_acquisition(
    tmp_path, remote_fixture, monkeypatch, failure
):
    original_getitem = h5py.Dataset.__getitem__
    read_count = 0

    def failing_read(dataset, key, *args, **kwargs):
        nonlocal read_count
        if dataset.name == "/tensor":
            read_count += 1
            if read_count == 2:
                assert list(tmp_path.glob(".staging-acquired-*/source.hdf5"))
                if failure == "remote":
                    raise OSError("Remote connection failed")
                values = original_getitem(dataset, key, *args, **kwargs)
                values[0, 0] = np.nan
                return values
        return original_getitem(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", failing_read)
    output = tmp_path / "acquired"
    with pytest.raises((ValueError, OSError), match="Remote connection|Non-finite"):
        acquire(output)
    assert read_count == 2
    assert all(handle.closed for handle in remote_fixture.handles)
    assert_unpublished(output)


@pytest.mark.parametrize("mismatch", ["file", "version", "viscosity"])
def test_import_cannot_attach_receipt_to_other_source(tmp_path, remote_fixture, mismatch):
    acquired = tmp_path / "acquired"
    acquire(acquired)
    source = acquired / "source.hdf5"
    options = {
        "source_url": remote_fixture.source["source_url"],
        "source_version": remote_fixture.source["source_version"],
        "license_name": remote_fixture.source["license_name"],
        "viscosity": remote_fixture.source["effective_viscosity"],
        "acquisition": acquired,
    }
    if mismatch == "file":
        copied = tmp_path / "copied.hdf5"
        copied.write_bytes(source.read_bytes())
        source = copied
    elif mismatch == "version":
        options["source_version"] = "Different version"
    else:
        options["viscosity"] = 0.01
    output = tmp_path / "canonical"
    with pytest.raises(ValueError, match="does not belong|provenance differs"):
        import_pdebench(source, output, **options)
    assert_unpublished(output)
