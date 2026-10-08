"""Checkpoint requests survive CLI parsing, queue storage, and leased execution."""

import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from flowstate import checkpoints, cli, engine
from flowstate import queue as queue_module
from flowstate.lake import Lake


@pytest.fixture
def checkpoint_queue(tmp_path, monkeypatch):
    # These tests exercise request handling rather than edits in unrelated files.
    provenance = engine.capture_provenance()
    monkeypatch.setattr(engine, "capture_provenance", lambda: copy.deepcopy(provenance))
    monkeypatch.setattr(queue_module, "capture_provenance", lambda: copy.deepcopy(provenance))
    monkeypatch.setattr(
        queue_module, "_IMPORTED_RUNTIME_IDENTITY", queue_module._runtime_identity(provenance)
    )
    return {
        "spec": {"base": {"grid_size": 8, "steps": 4, "save_every": 2}},
        "queue_path": tmp_path / "queue-parent" / "jobs.sqlite",
        "lake_root": tmp_path / "lake",
    }


@pytest.mark.parametrize("command", ["run", "sweep"])
def test_cli_forwards_checkpoint_interval_and_reports_actual_resume(
    tmp_path, monkeypatch, capsys, command,
):
    config = tmp_path / "request.json"
    config.write_text("{}", encoding="utf-8")
    captured = {}
    outcome = SimpleNamespace(
        record={"id": "example", "status": "completed", "metrics": {}, "error": None},
        resumed=False, checkpoint_step=3,
    )

    def execute(value, lake, **kwargs):
        captured.update(kwargs)
        return outcome if command == "run" else [outcome]

    monkeypatch.setattr(cli, "run_experiment" if command == "run" else "run_sweep", execute)
    assert cli.main([
        "--lake", str(tmp_path / "lake"), command, str(config), "--checkpoint-every", "3",
    ]) == 0
    assert captured["checkpoint_every"] == 3
    result = json.loads(capsys.readouterr().out)[0]
    assert result["checkpoint_resumed_from"] == 3
    assert result["resumed"] is False  # Finalized reuse remains a distinct event.


@pytest.mark.parametrize("command", ["run", "sweep", "queue"])
def test_invalid_checkpoint_cli_request_creates_no_lake_or_queue(tmp_path, capsys, command):
    config = tmp_path / "request.json"
    config.write_text("{}", encoding="utf-8")
    lake = tmp_path / "lake"
    database = tmp_path / "jobs.sqlite"
    argv = ["--lake", str(lake), command]
    argv += ["submit", str(config), str(database)] if command == "queue" else [str(config)]
    assert cli.main([*argv, "--checkpoint-every", "0"]) == 2
    assert "checkpoint" in capsys.readouterr().err.lower()
    assert not lake.exists()
    assert not database.exists()


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "2"])
def test_invalid_checkpoint_queue_interval_has_no_filesystem_effects(checkpoint_queue, interval):
    with pytest.raises(ValueError, match="checkpoint"):
        queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=interval)
    assert not checkpoint_queue["queue_path"].parent.exists()
    assert not checkpoint_queue["lake_root"].exists()


def test_darcy_checkpoint_queue_rejected_before_database_creation(checkpoint_queue):
    checkpoint_queue["spec"] = {"base": {"equation": "darcy2d", "grid_size": 9}}
    with pytest.raises(ValueError):
        queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=1)
    assert not checkpoint_queue["queue_path"].parent.exists()
    assert not checkpoint_queue["lake_root"].exists()


def test_queue_validates_every_checkpoint_config_before_creating_files(
    checkpoint_queue, monkeypatch,
):
    checkpoint_queue["spec"]["parameters"] = {"seed": [1, 2]}
    checked = []

    def reject_second(config, interval):
        checked.append((config["seed"], interval))
        if config["seed"] == 2:
            raise ValueError("checkpoint budget exceeded")

    monkeypatch.setattr(queue_module, "validate_checkpoint_options", reject_second)
    with pytest.raises(ValueError, match="checkpoint budget"):
        queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=2)
    assert checked == [(1, 2), (2, 2)]
    assert not checkpoint_queue["queue_path"].parent.exists()
    assert not checkpoint_queue["lake_root"].exists()


def test_omitted_checkpoint_option_preserves_old_request_identity(checkpoint_queue):
    submission = queue_module.submit_sweep(**checkpoint_queue)
    with sqlite3.connect(checkpoint_queue["queue_path"]) as connection:
        request = json.loads(connection.execute("SELECT request_json FROM jobs").fetchone()[0])
    assert "checkpoint_every" not in request
    assert submission["job_ids"] == [queue_module._identity(request)]
    assert queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=None)["existing"] == 1
    enabled = queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=2)
    assert enabled["submitted"] == 1 and enabled["job_ids"] != submission["job_ids"]


def test_queue_persists_and_reports_resumed_checkpoint_step(checkpoint_queue, monkeypatch):
    queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=2)
    job = queue_module.queue_status(checkpoint_queue["queue_path"])["jobs"][0]
    assert job["checkpoint_every"] == 2
    assert job["checkpoint_resumed_from"] is None

    def resumed(config, lake, **kwargs):
        assert kwargs["checkpoint_every"] == 2
        return SimpleNamespace(
            record={"id": job["expected_result_id"], "status": "completed", "error": None},
            resumed=False, checkpoint_step=2,
        )

    monkeypatch.setattr(queue_module, "run_experiment", resumed)
    result = queue_module.work_queue(checkpoint_queue["queue_path"])
    assert result["completed"] == 1
    assert result["jobs"][0]["checkpoint_resumed_from"] == 2
    assert result["jobs"][0]["resumed"] is False
    status = queue_module.queue_status(checkpoint_queue["queue_path"])
    assert status["jobs"][0]["checkpoint_resumed_from"] == 2
    assert status["events"][-1]["detail"]["checkpoint_resumed_from"] == 2


def test_checkpoint_cli_queue_submit_and_real_drain(checkpoint_queue, tmp_path, capsys):
    config = tmp_path / "sweep.json"
    config.write_text(json.dumps(checkpoint_queue["spec"]), encoding="utf-8")
    assert cli.main([
        "--lake", str(checkpoint_queue["lake_root"]), "queue", "submit", str(config),
        str(checkpoint_queue["queue_path"]), "--checkpoint-every", "2",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["submitted"] == 1
    report = queue_module.work_queue(checkpoint_queue["queue_path"])
    assert report["completed"] == 1
    assert report["jobs"][0]["checkpoint_resumed_from"] is None
    assert report["jobs"][0]["resumed"] is False
    run_id = report["jobs"][0]["result_id"]
    assert not Lake(checkpoint_queue["lake_root"]).verify(run_id)

    # A different recovery frequency changes the job request, not the scientific result.
    queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=1)
    reused = queue_module.work_queue(checkpoint_queue["queue_path"])
    assert reused["completed"] == 1
    assert reused["jobs"][0]["result_id"] == run_id
    assert reused["jobs"][0]["resumed"] is True
    assert reused["jobs"][0]["checkpoint_resumed_from"] is None


def test_queue_recovers_solver_checkpoint_after_worker_interruption(checkpoint_queue, monkeypatch):
    queue_module.submit_sweep(**checkpoint_queue, checkpoint_every=2)
    save = checkpoints.CheckpointStore.save

    def interrupt_after_commit(store, checkpoint):
        save(store, checkpoint)
        if checkpoint.step == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(checkpoints.CheckpointStore, "save", interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        queue_module.work_queue(checkpoint_queue["queue_path"])
    status = queue_module.queue_status(checkpoint_queue["queue_path"])
    assert status["counts"]["running"] == 1
    assert status["jobs"][0]["claim_count"] == 1
    assert Lake(checkpoint_queue["lake_root"]).records() == []

    # Make this test job's lease expire without sleeping; production uses wall time.
    with sqlite3.connect(checkpoint_queue["queue_path"]) as connection:
        connection.execute("UPDATE jobs SET lease_until=0 WHERE status='running'")
    monkeypatch.setattr(checkpoints.CheckpointStore, "save", save)
    recovered = queue_module.work_queue(checkpoint_queue["queue_path"])
    assert recovered["completed"] == 1
    assert recovered["jobs"][0]["checkpoint_resumed_from"] == 2
    assert recovered["jobs"][0]["resumed"] is False
    status = queue_module.queue_status(checkpoint_queue["queue_path"])
    assert status["jobs"][0]["claim_count"] == 2
    assert status["jobs"][0]["checkpoint_resumed_from"] == 2
    assert [event["kind"] for event in status["events"]] == [
        "submitted", "claimed", "lease_expired", "claimed", "completed",
    ]
    assert not Lake(checkpoint_queue["lake_root"]).verify(recovered["jobs"][0]["result_id"])
