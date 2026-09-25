import hashlib
import json

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from flowstate import object_store
from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve
from flowstate.object_store import download_experiment, upload_experiment


@pytest.fixture
def s3():
    # Moto intercepts SDK requests and supplies dummy credentials: no live cloud use.
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="flowstate-test")
        yield client


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


def keys(client):
    return [
        item["Key"] for item in client.list_objects_v2(Bucket="flowstate-test").get("Contents", [])
    ]


def test_s3_roundtrip_and_idempotent_retries(tmp_path, s3):
    source = source_lake(tmp_path / "source")
    first = upload_experiment(source.root, "example", "flowstate-test", prefix="study/one")
    assert first["committed"] and first["uploaded_objects"] > 0
    again = upload_experiment(source.root, "example", "flowstate-test", prefix="study/one")
    assert again["resumed"] and again["uploaded_objects"] == 0
    assert again["reused_objects"] == first["uploaded_objects"]
    destination = tmp_path / "restored"
    assert not download_experiment(destination, "example", "flowstate-test", prefix="study/one")[
        "resumed"
    ]
    restored = Lake(destination)
    assert restored.verify("example") == []
    assert restored.load_record("example") == source.load_record("example")
    originals = source.experiments / "example"
    copies = restored.experiments / "example"
    for path in originals.rglob("*"):
        if path.is_file():
            assert (copies / path.relative_to(originals)).read_bytes() == path.read_bytes()
    assert download_experiment(destination, "example", "flowstate-test", prefix="study/one")[
        "resumed"
    ]


def test_interrupted_upload_has_no_commit_and_retry_reuses_objects(tmp_path, s3, monkeypatch):
    source = source_lake(tmp_path / "source")
    original_put = s3.put_object
    uploaded = []

    def fail_second_put(**kwargs):
        if len(uploaded) == 1:
            raise RuntimeError("simulated connection loss")
        uploaded.append(kwargs["Key"])
        return original_put(**kwargs)

    monkeypatch.setattr(object_store, "_client", lambda endpoint_url: s3)
    monkeypatch.setattr(s3, "put_object", fail_second_put)
    with pytest.raises(RuntimeError, match="connection loss"):
        upload_experiment(source.root, "example", "flowstate-test")
    assert keys(s3) == uploaded
    assert "experiments/example/manifest.json" not in keys(s3)
    with pytest.raises(FileNotFoundError, match="No committed"):
        download_experiment(tmp_path / "restored", "example", "flowstate-test")
    monkeypatch.setattr(s3, "put_object", original_put)
    resumed = upload_experiment(source.root, "example", "flowstate-test")
    assert resumed["reused_objects"] == 1
    assert resumed["committed"]


def test_manifest_is_last_and_all_puts_are_conditional(tmp_path, s3, monkeypatch):
    source = source_lake(tmp_path / "source", status="failed")
    original_put = s3.put_object
    seen = []

    def inspect_put(**kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        seen.append(kwargs["Key"])
        return original_put(**kwargs)

    monkeypatch.setattr(object_store, "_client", lambda endpoint_url: s3)
    monkeypatch.setattr(s3, "put_object", inspect_put)
    result = upload_experiment(source.root, "example", "flowstate-test")
    assert seen[-1] == result["manifest_key"] == "experiments/example/manifest.json"
    assert all(key.startswith("artifacts/example/") for key in seen[:-1])


def test_conflicting_remote_commit_and_existing_local_run_are_not_overwritten(tmp_path, s3):
    first = source_lake(tmp_path / "first")
    second = source_lake(tmp_path / "second", status="failed")
    upload_experiment(first.root, "example", "flowstate-test")
    with pytest.raises(FileExistsError, match="different committed manifest"):
        upload_experiment(second.root, "example", "flowstate-test")
    with pytest.raises(FileExistsError, match="different manifest"):
        download_experiment(second.root, "example", "flowstate-test")
    assert second.load_record("example")["status"] == "failed"
    assert second.verify("example") == []


def test_remote_corruption_is_detected_on_download_and_upload_retry(tmp_path, s3):
    source = source_lake(tmp_path / "source")
    upload_experiment(source.root, "example", "flowstate-test")
    key = next(key for key in keys(s3) if key.endswith("/record.json"))
    s3.put_object(Bucket="flowstate-test", Key=key, Body=b"corrupted")
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="hash mismatch"):
        download_experiment(destination, "example", "flowstate-test")
    assert list(Lake(destination).experiments.iterdir()) == []
    with pytest.raises(ValueError, match="conflicting bytes"):
        upload_experiment(source.root, "example", "flowstate-test")


def test_missing_remote_chunk_never_publishes_locally(tmp_path, s3):
    source = source_lake(tmp_path / "source")
    upload_experiment(source.root, "example", "flowstate-test")
    key = next(key for key in keys(s3) if key.endswith("/record.json"))
    s3.delete_object(Bucket="flowstate-test", Key=key)
    with pytest.raises(ClientError):
        download_experiment(tmp_path / "restored", "example", "flowstate-test")
    assert list(Lake(tmp_path / "restored").experiments.iterdir()) == []


def test_untrusted_manifest_cannot_escape_destination(tmp_path, s3):
    payload = json.dumps(
        {
            "version": 1,
            "algorithm": "sha256",
            "artifacts": {
                "record.json": "0" * 64,
                "metadata.parquet": "0" * 64,
                "../escape": "0" * 64,
            },
        }
    ).encode()
    s3.put_object(Bucket="flowstate-test", Key="experiments/example/manifest.json", Body=payload)
    with pytest.raises(ValueError, match="Unsafe path"):
        download_experiment(tmp_path / "restored", "example", "flowstate-test")
    assert not (tmp_path / "escape").exists()


def test_local_corruption_prevents_upload(tmp_path, s3):
    source = source_lake(tmp_path / "source")
    (source.experiments / "example" / "record.json").write_text("{}")
    with pytest.raises(ValueError, match="Source experiment failed verification"):
        upload_experiment(source.root, "example", "flowstate-test")
    assert keys(s3) == []


def test_manifest_hash_drives_content_address(tmp_path, s3):
    source = source_lake(tmp_path / "source")
    metadata = upload_experiment(source.root, "example", "flowstate-test")
    expected = hashlib.sha256(
        (source.experiments / "example" / "manifest.json").read_bytes()
    ).hexdigest()
    assert metadata["manifest_sha256"] == expected
    assert f"artifacts/example/{expected}/record.json" in keys(s3)
