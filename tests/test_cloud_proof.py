"""Complete proof contract using the existing in-memory GCS SDK fake, no credentials."""

import copy
import json
from pathlib import Path

import pytest
from test_gcs_store import FakeClient

from flowstate import cloud_proof, engine, gcs_store
from flowstate.lake import Lake


@pytest.fixture
def proof(tmp_path, monkeypatch):
    provenance = engine.capture_provenance()
    monkeypatch.setattr(engine, "capture_provenance", lambda: copy.deepcopy(provenance))
    monkeypatch.setattr(cloud_proof, "capture_provenance", lambda: copy.deepcopy(provenance))
    client = FakeClient()

    def factory(project):
        client.projects.append(project)
        return client

    monkeypatch.setattr(gcs_store, "_client", factory)
    root = tmp_path / "proof"
    prepared = cloud_proof.prepare_proof(root, require_clean=False)
    return root, prepared, client


def execute(root, prepared):
    return cloud_proof.execute_proof(
        root, plan_sha256=prepared["plan_sha256"], allow_cloud=True,
        allow_local_delete=True, require_clean=False,
    )


def test_preparation_has_no_cloud_access_and_exact_bound(proof):
    root, prepared, client = proof
    assert not client.projects and not client.objects
    assert not prepared["cloud_accessed"]
    assert prepared["file_count"] <= cloud_proof.MAX_FILES
    assert prepared["local_bytes"] <= cloud_proof.MAX_BYTES
    assert prepared["gcs"]["prefix"].startswith("flowstate/proofs/")
    assert cloud_proof._hash(root / "plan.json") == prepared["plan_sha256"]
    assert (root / "viewer.html").is_file()
    plan = json.loads((root / "plan.json").read_text())
    assert not plan["approved"]
    assert plan["local_removal_target"] == prepared["local_removal_target"]


def test_fake_cloud_proof_removes_original_before_actual_download(proof, monkeypatch):
    root, prepared, client = proof
    original = Path(prepared["local_removal_target"])
    download = gcs_store.download_experiment
    checked = []

    def observe(lake, run_id, **options):
        assert not original.exists()
        assert not Path(lake).exists()
        checked.append(True)
        return download(lake, run_id, **options)

    monkeypatch.setattr(gcs_store, "download_experiment", observe)
    report = execute(root, prepared)
    assert checked and report["status"] == "completed", report.get("error")
    assert report["local_removal_started"] and report["local_removal_completed"]
    assert report["stages"]["remote_verification"]["uploaded_objects"] == 0
    assert report["stages"]["verification"]["all_files_byte_identical"]
    assert report["stages"]["verification"]["file_count"] == prepared["file_count"]
    assert not report["remote_objects_deleted"]
    assert len(client.objects) == prepared["file_count"]
    assert set(client.projects) == {"flowstate-510320"}
    assert not original.exists()
    assert not Lake(root / "restored").verify(prepared["experiment_id"])
    assert (root / "restored-viewer.html").is_file()
    assert json.loads((root / "receipt.json").read_text()) == report


@pytest.mark.parametrize("flags", [{}, {"allow_cloud": True}, {"allow_local_delete": True},
                                   {"allow_cloud": 1, "allow_local_delete": True}])
def test_missing_approval_never_uses_credentials_or_deletes(proof, flags):
    root, prepared, client = proof
    with pytest.raises(PermissionError):
        cloud_proof.execute_proof(root, plan_sha256=prepared["plan_sha256"], **flags)
    assert not client.projects
    assert Path(prepared["local_removal_target"]).exists()
    assert not (root / "execution-started.json").exists()


def test_tampered_plan_is_rejected_before_cloud(proof):
    root, prepared, client = proof
    (root / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        execute(root, prepared)
    assert not client.projects
    assert Path(prepared["local_removal_target"]).exists()


def test_unexpected_local_file_aborts_before_cloud(proof):
    root, prepared, client = proof
    original = Path(prepared["local_removal_target"])
    (original / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="local files changed"):
        execute(root, prepared)
    assert not client.projects
    assert (original / "keep.txt").read_text() == "user data"


@pytest.mark.parametrize("failure", ["upload", "remote_corrupt", "local_changed"])
def test_pre_removal_failures_preserve_source_and_failure_receipt(proof, monkeypatch, failure):
    root, prepared, client = proof
    original = Path(prepared["local_removal_target"])
    upload = gcs_store.upload_experiment
    calls = 0

    def fault(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = upload(*args, **kwargs)
        if calls == 1 and failure == "remote_corrupt":
            client.objects[next(iter(client.objects))] = b"corrupt cloud bytes"
        if calls == 2 and failure == "local_changed":
            (original / "keep.txt").write_text("preserve", encoding="utf-8")
        return result

    monkeypatch.setattr(gcs_store, "upload_experiment", fault)
    if failure == "upload":
        client.fail_after = 1
    report = execute(root, prepared)
    assert report["status"] == "failed"
    assert not report["local_removal_started"]
    assert original.exists()
    assert json.loads((root / "receipt.json").read_text())["error"]
    assert not (root / "restored").exists()


def test_missing_remote_after_removal_keeps_restore_failure_evidence(proof, monkeypatch):
    root, prepared, client = proof
    download = gcs_store.download_experiment

    def fault(*args, **kwargs):
        client.objects.clear()
        return download(*args, **kwargs)

    monkeypatch.setattr(gcs_store, "download_experiment", fault)
    report = execute(root, prepared)
    assert report["status"] == "failed"
    assert report["local_removal_completed"]
    assert report["error"]["stage"] == "download"
    assert not Path(prepared["local_removal_target"]).exists()
    assert (root / "after-removal.json").exists()
    assert json.loads((root / "receipt.json").read_text())["plan"]["inventory"]
    assert not Lake(root / "restored").records()


def test_execution_marker_blocks_duplicate_attempt_before_cloud(proof):
    root, prepared, client = proof
    (root / "execution-started.json").write_text("previous attempt", encoding="utf-8")
    with pytest.raises(FileExistsError):
        execute(root, prepared)
    assert not client.projects
    assert Path(prepared["local_removal_target"]).exists()


@pytest.mark.parametrize("name", ["receipt.json", "before-removal.json", "after-removal.json",
                                  "restored-viewer.html"])
def test_existing_stage_outputs_stop_before_cloud_or_removal(proof, name):
    root, prepared, client = proof
    (root / name).write_text("preserve existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        execute(root, prepared)
    assert not client.projects
    assert Path(prepared["local_removal_target"]).exists()
    assert (root / name).read_text() == "preserve existing evidence"


def test_partial_removal_is_reported_with_exact_removed_file_list(proof, monkeypatch):
    root, prepared, _ = proof
    directory = Path(prepared["local_removal_target"])
    unlink = Path.unlink
    removed = []

    def fail_second(path, *args, **kwargs):
        if path.is_relative_to(directory):
            if removed:
                raise OSError("Injected second removal failure")
            removed.append(path.relative_to(directory).as_posix())
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_second)
    report = execute(root, prepared)
    assert report["status"] == "failed"
    assert report["local_removal_started"] and not report["local_removal_completed"]
    assert report["stages"]["local_removal"]["removed_files"] == removed
    assert len(removed) == 1 and directory.exists()
    assert not (directory / removed[0]).exists()
    assert not (root / "restored").exists()


def test_oversize_extra_file_is_rejected_before_hashing(proof, monkeypatch):
    root, prepared, client = proof
    extra = Path(prepared["local_removal_target"]) / "oversize.bin"
    with extra.open("wb") as stream:
        stream.truncate(cloud_proof.MAX_BYTES + 1)
    digest = cloud_proof._hash

    def reject_hash(path):
        assert path != extra, "Size bound must be checked before reading oversized data"
        return digest(path)

    monkeypatch.setattr(cloud_proof, "_hash", reject_hash)
    with pytest.raises(ValueError, match="transfer bound"):
        execute(root, prepared)
    assert not client.projects and extra.exists()


@pytest.mark.parametrize("target", ["source", "restored"])
def test_junction_check_prevents_local_escape(proof, monkeypatch, target):
    root, prepared, client = proof
    directory = Path(prepared["local_removal_target"])
    junction = directory / "fields.zarr" if target == "source" else root / "restored"
    actual = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda path: path == junction or actual(path))
    with pytest.raises(ValueError, match="links or junctions"):
        execute(root, prepared)
    assert not client.projects and directory.exists()


def test_real_symlink_cannot_remove_outside_file(proof, tmp_path):
    root, prepared, client = proof
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    link = Path(prepared["local_removal_target"]) / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Windows account lacks permission to create symbolic links")
    with pytest.raises(ValueError, match="links or junctions"):
        execute(root, prepared)
    assert outside.read_text() == "keep" and not client.projects


def test_changed_runtime_cannot_execute_proof(proof, monkeypatch):
    root, prepared, client = proof
    provenance = cloud_proof.capture_provenance()
    monkeypatch.setattr(cloud_proof, "capture_provenance", lambda: {
        **provenance, "source_sha256": "different source",
    })
    with pytest.raises(ValueError, match="runtime differs"):
        execute(root, prepared)
    assert not client.projects


def test_documentation_only_commit_can_advance_without_rewriting_run_provenance(proof, monkeypatch):
    root, prepared, _ = proof
    old = cloud_proof.capture_provenance()
    monkeypatch.setattr(cloud_proof, "capture_provenance", lambda: {
        **old, "git_commit": "b" * 40,
    })
    report = execute(root, prepared)
    assert report["status"] == "completed"
    assert report["execution_provenance"]["git_commit"] == "b" * 40
    assert Lake(root / "restored").load_record(prepared["experiment_id"])["provenance"] == old


@pytest.mark.parametrize("option", [{"project": "bad"}, {"bucket": "gs://bad"},
                                    {"prefix": "../escape"}, {"prefix": ""}])
def test_bad_plan_options_have_no_filesystem_effects(tmp_path, option):
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        cloud_proof.prepare_proof(root, require_clean=False, **option)
    assert not root.exists()


def test_dirty_preparation_is_refused_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_proof, "capture_provenance", lambda: {
        "git_commit": "a" * 40, "git_dirty": True,
    })
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="clean checkout"):
        cloud_proof.prepare_proof(root)
    assert not root.exists()
