"""GCS transfer contracts against an in-memory SDK fake; no network or credentials."""

import hashlib
import io
import json

import pytest
from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed
from google.auth.exceptions import DefaultCredentialsError

from flowstate import gcs_store
from flowstate.cli import main
from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve


class FakeBlob:
    def __init__(self, client, bucket, name):
        self.client = client
        self.key = (bucket, name)
        self.name = name
        self.metadata = None
        self.generation = None

    def _read(self):
        try:
            return self.client.objects[self.key]
        except KeyError as error:
            raise NotFound(self.name) from error

    def download_as_bytes(self, **kwargs):
        return self._read()

    def reload(self, **kwargs):
        # A deterministic fake generation identifies the exact current payload.
        self.generation = int.from_bytes(hashlib.sha256(self._read()).digest()[:8], "big")

    def open(self, mode, **kwargs):
        assert mode == "rb"
        current_generation = int.from_bytes(hashlib.sha256(self._read()).digest()[:8], "big")
        assert self.generation is not None
        assert kwargs["if_generation_match"] == self.generation == current_generation
        return io.BytesIO(self._read())

    def upload_from_file(self, stream, *, size=None, if_generation_match=None, **kwargs):
        # Every publication must be create-only, including the final manifest.
        assert if_generation_match == 0
        if self.key in self.client.objects:
            raise PreconditionFailed(self.name)
        if self.client.fail_after is not None and len(self.client.writes) >= self.client.fail_after:
            raise RuntimeError("simulated connection loss")
        data = stream.read()
        if size is not None:
            assert len(data) == size
        self.client.objects[self.key] = data
        self.client.writes.append(self.key)


class FakeBucket:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def blob(self, name, **kwargs):
        return FakeBlob(self.client, self.name, name)


class FakeClient:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.fail_after = None
        self.projects = []

    def bucket(self, name):
        return FakeBucket(self, name)

    def names(self):
        return [name for bucket, name in self.objects if bucket == "flowstate-test"]


@pytest.fixture
def gcs(monkeypatch):
    client = FakeClient()

    def fake_client(project):
        client.projects.append(project)
        return client

    monkeypatch.setattr(gcs_store, "_client", fake_client)
    return client


def source_lake(path, *, status="completed"):
    lake = Lake(path)
    config = normalize_config({"grid_size": 8, "steps": 2, "save_every": 1})
    lake.write(
        "example",
        {
            "id": "example",
            "status": status,
            "equation": "burgers1d",
            "config": config,
            "metrics": {},
            "error": None if status == "completed" else "test failure",
        },
        solve(config) if status == "completed" else None,
    )
    return lake


def test_gcs_roundtrip_and_idempotent_retries(tmp_path, gcs):
    source = source_lake(tmp_path / "source")
    options = {"prefix": "study/one", "project": "flowstate-project"}
    first = gcs_store.upload_experiment(source.root, "example", "flowstate-test", **options)
    assert first["committed"] and first["uploaded_objects"] > 0
    again = gcs_store.upload_experiment(source.root, "example", "flowstate-test", **options)
    assert again["resumed"] and again["uploaded_objects"] == 0
    assert again["reused_objects"] == first["uploaded_objects"]
    destination = tmp_path / "restored"
    assert not gcs_store.download_experiment(
        destination, "example", "flowstate-test", **options
    )["resumed"]
    restored = Lake(destination)
    assert restored.verify("example") == []
    assert restored.load_record("example") == source.load_record("example")
    originals = source.experiments / "example"
    copies = restored.experiments / "example"
    for path in originals.rglob("*"):
        if path.is_file():
            assert (copies / path.relative_to(originals)).read_bytes() == path.read_bytes()
    assert gcs_store.download_experiment(destination, "example", "flowstate-test", **options)[
        "resumed"
    ]
    assert gcs.projects and set(gcs.projects) == {"flowstate-project"}


def test_interrupted_upload_has_no_commit_and_retry_reuses_objects(tmp_path, gcs):
    source = source_lake(tmp_path / "source")
    gcs.fail_after = 1
    with pytest.raises(RuntimeError, match="connection loss"):
        gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert len(gcs.objects) == 1
    assert "experiments/example/manifest.json" not in gcs.names()
    with pytest.raises(FileNotFoundError, match="No committed"):
        gcs_store.download_experiment(tmp_path / "restored", "example", "flowstate-test")
    gcs.fail_after = None
    resumed = gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert resumed["reused_objects"] == 1
    assert resumed["committed"]


def test_manifest_is_last_and_writes_are_conditional(tmp_path, gcs):
    source = source_lake(tmp_path / "source", status="failed")
    result = gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    seen = [name for _, name in gcs.writes]
    assert seen[-1] == result["manifest_key"] == "experiments/example/manifest.json"
    assert all(key.startswith("artifacts/example/") for key in seen[:-1])
    expected = hashlib.sha256(
        (source.experiments / "example" / "manifest.json").read_bytes()
    ).hexdigest()
    assert result["manifest_sha256"] == expected
    assert f"artifacts/example/{expected}/record.json" in seen


def test_conflicting_partial_upload_cannot_be_committed(tmp_path, gcs):
    source = source_lake(tmp_path / "source")
    gcs.fail_after = 1
    with pytest.raises(RuntimeError, match="connection loss"):
        gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    key = next(iter(gcs.objects))
    gcs.objects[key] = b"unexpected existing bytes"
    gcs.fail_after = None
    with pytest.raises(ValueError, match="conflicting bytes"):
        gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert "experiments/example/manifest.json" not in gcs.names()
    assert gcs.objects[key] == b"unexpected existing bytes"


def test_conflicting_remote_commit_and_local_run_are_not_overwritten(tmp_path, gcs):
    first = source_lake(tmp_path / "first")
    second = source_lake(tmp_path / "second", status="failed")
    gcs_store.upload_experiment(first.root, "example", "flowstate-test")
    before = dict(gcs.objects)
    with pytest.raises(FileExistsError, match="different committed manifest"):
        gcs_store.upload_experiment(second.root, "example", "flowstate-test")
    assert gcs.objects == before
    with pytest.raises(FileExistsError, match="different manifest"):
        gcs_store.download_experiment(second.root, "example", "flowstate-test")
    assert second.load_record("example")["status"] == "failed"
    assert second.verify("example") == []


def test_remote_corruption_is_detected_on_download_and_upload_retry(tmp_path, gcs):
    source = source_lake(tmp_path / "source")
    gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    key = next(key for key in gcs.objects if key[1].endswith("/record.json"))
    gcs.objects[key] = b"corrupted"
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="hash mismatch"):
        gcs_store.download_experiment(destination, "example", "flowstate-test")
    assert list(Lake(destination).experiments.iterdir()) == []
    with pytest.raises(ValueError, match="conflicting bytes"):
        gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert gcs.objects[key] == b"corrupted"


def test_missing_remote_chunk_never_publishes_locally(tmp_path, gcs):
    source = source_lake(tmp_path / "source")
    gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    key = next(key for key in gcs.objects if key[1].endswith("/record.json"))
    del gcs.objects[key]
    destination = tmp_path / "restored"
    with pytest.raises(NotFound):
        gcs_store.download_experiment(destination, "example", "flowstate-test")
    assert list(Lake(destination).experiments.iterdir()) == []


@pytest.mark.parametrize(
    "unsafe", ["../escape", "/absolute", "a/../escape", "a\\escape", "C:escape"]
)
def test_untrusted_manifest_cannot_escape_destination(tmp_path, gcs, unsafe):
    gcs.objects[("flowstate-test", "experiments/example/manifest.json")] = json.dumps(
        {
            "version": 1,
            "algorithm": "sha256",
            "artifacts": {
                "record.json": "0" * 64,
                "metadata.parquet": "0" * 64,
                unsafe: "0" * 64,
            },
        }
    ).encode()
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe path"):
        gcs_store.download_experiment(destination, "example", "flowstate-test")
    assert list(Lake(destination).experiments.iterdir()) == []
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("experiment_id", ["../outside", "a/b", "a\\b", "", "a:b"])
@pytest.mark.parametrize("operation", ["upload_experiment", "download_experiment"])
def test_invalid_experiment_ids_fail_before_cloud_access(tmp_path, gcs, experiment_id, operation):
    with pytest.raises(ValueError, match="Experiment ID"):
        getattr(gcs_store, operation)(tmp_path / "lake", experiment_id, "flowstate-test")
    assert gcs.projects == []
    assert gcs.objects == {}


@pytest.mark.parametrize("prefix", ["../outside", "a/../b", "a//b", "a\\b"])
@pytest.mark.parametrize("operation", ["upload_experiment", "download_experiment"])
def test_invalid_prefixes_fail_before_cloud_access(tmp_path, gcs, prefix, operation):
    source = source_lake(tmp_path / "source")
    with pytest.raises(ValueError):
        getattr(gcs_store, operation)(source.root, "example", "flowstate-test", prefix=prefix)
    assert gcs.projects == []
    assert gcs.objects == {}


def test_local_corruption_prevents_upload(tmp_path, gcs):
    source = source_lake(tmp_path / "source")
    (source.experiments / "example" / "record.json").write_text("{}")
    with pytest.raises(ValueError, match="Source experiment failed verification"):
        gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert gcs.objects == {}


def test_source_mutation_after_initial_verification_cannot_poison_remote(
    tmp_path, gcs, monkeypatch
):
    source = source_lake(tmp_path / "source")
    record = source.experiments / "example" / "record.json"
    original = record.read_bytes()

    def mutate_source_after_verification(project):
        record.write_bytes(b"changed source record")
        return gcs

    monkeypatch.setattr(gcs_store, "_client", mutate_source_after_verification)
    with pytest.raises(ValueError, match="changed during upload"):
        gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert not any(key.endswith("/record.json") for key in gcs.names())
    assert "experiments/example/manifest.json" not in gcs.names()
    partial_objects = len(gcs.objects)
    assert partial_objects > 0

    record.write_bytes(original)
    monkeypatch.setattr(gcs_store, "_client", lambda project: gcs)
    result = gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert result["committed"]
    assert result["reused_objects"] == partial_objects
    destination = tmp_path / "restored"
    gcs_store.download_experiment(destination, "example", "flowstate-test")
    assert Lake(destination).verify("example") == []
    assert (Lake(destination).experiments / "example" / "record.json").read_bytes() == original


def test_upload_uses_verified_snapshot_when_original_changes_during_transfer(
    tmp_path, gcs, monkeypatch
):
    source = source_lake(tmp_path / "source")
    record = source.experiments / "example" / "record.json"
    original = record.read_bytes()
    original_upload = FakeBlob.upload_from_file

    def mutate_original_before_read(self, stream, **kwargs):
        if self.name.endswith("/record.json"):
            record.write_bytes(b"source modified during upload")
        return original_upload(self, stream, **kwargs)

    monkeypatch.setattr(FakeBlob, "upload_from_file", mutate_original_before_read)
    result = gcs_store.upload_experiment(source.root, "example", "flowstate-test")
    assert result["committed"]
    assert record.read_bytes() != original
    remote_record = next(key for key in gcs.objects if key[1].endswith("/record.json"))
    assert gcs.objects[remote_record] == original
    destination = tmp_path / "restored"
    gcs_store.download_experiment(destination, "example", "flowstate-test")
    assert Lake(destination).verify("example") == []
    assert (Lake(destination).experiments / "example" / "record.json").read_bytes() == original


def test_cli_upload_and_download_roundtrip_with_project_and_prefix(tmp_path, gcs, capsys):
    source = source_lake(tmp_path / "source")
    options = ["--prefix", "flowstate/demo", "--project", "flowstate-project"]
    assert main(
        ["--lake", str(source.root), "gcs", "upload", "example", "flowstate-test", *options]
    ) == 0
    uploaded = json.loads(capsys.readouterr().out)
    assert uploaded["committed"]
    assert uploaded["manifest_key"] == "flowstate/demo/experiments/example/manifest.json"
    assert all(key.startswith("flowstate/demo/") for key in gcs.names())
    destination = tmp_path / "restored"
    assert main(
        ["--lake", str(destination), "gcs", "download", "example", "flowstate-test", *options]
    ) == 0
    downloaded = json.loads(capsys.readouterr().out)
    assert downloaded["id"] == "example" and not downloaded["resumed"]
    restored = Lake(destination)
    assert restored.load_record("example") == source.load_record("example")
    assert restored.verify("example") == []
    assert gcs.projects == ["flowstate-project", "flowstate-project"]


@pytest.mark.parametrize("operation", ["upload", "download"])
@pytest.mark.parametrize("error_type", [Forbidden, DefaultCredentialsError])
def test_cli_cloud_errors_return_actionable_failure(
    tmp_path, gcs, monkeypatch, capsys, operation, error_type
):
    source = source_lake(tmp_path / "source")

    def fail_client(project):
        raise error_type("test credentials or permission failure")

    monkeypatch.setattr(gcs_store, "_client", fail_client)
    assert main(
        ["--lake", str(source.root), "gcs", operation, "example", "flowstate-test"]
    ) == 2
    output = capsys.readouterr()
    assert "flowstate: GCS request failed:" in output.err
    assert "test credentials or permission failure" in output.err
    assert output.out == ""
    assert gcs.objects == {}


def test_cli_missing_remote_experiment_reports_failure(tmp_path, gcs, capsys):
    destination = tmp_path / "restored"
    assert main(
        ["--lake", str(destination), "gcs", "download", "missing", "flowstate-test"]
    ) == 2
    output = capsys.readouterr()
    assert "No committed remote experiment" in output.err
    assert output.out == ""
    assert list(Lake(destination).experiments.iterdir()) == []
