import copy
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from flowstate import engine
from flowstate import queue as queue_module
from flowstate.lake import Lake


@pytest.fixture
def setup(tmp_path, monkeypatch):
    provenance = engine.capture_provenance()
    monkeypatch.setattr(
        queue_module, "_IMPORTED_RUNTIME_IDENTITY", queue_module._runtime_identity(provenance)
    )
    monkeypatch.setattr(engine, "capture_provenance", lambda: copy.deepcopy(provenance))
    monkeypatch.setattr(queue_module, "capture_provenance", lambda: copy.deepcopy(provenance))
    return {
        "queue_path": tmp_path / "queue.sqlite", "lake_root": tmp_path / "lake",
        "spec": {"base": {"grid_size": 16, "steps": 4, "save_every": 2, "dt": 0.001}},
    }


def test_submission_is_idempotent_and_keeps_terminal_state(setup):
    first = queue_module.submit_sweep(**setup)
    second = queue_module.submit_sweep(**setup)
    assert (first["submitted"], second["existing"]) == (1, 1)
    assert first["job_ids"] == second["job_ids"]
    report = queue_module.work_queue(setup["queue_path"])
    assert report["processed"] == report["completed"] == 1
    assert report["jobs"][0]["resumed"] is False
    assert queue_module.submit_sweep(**setup)["submitted"] == 0
    assert queue_module.work_queue(setup["queue_path"])["processed"] == 0
    status = queue_module.queue_status(setup["queue_path"])
    assert status["counts"] == {"queued": 0, "running": 0, "completed": 1, "failed": 0}
    assert [event["kind"] for event in status["events"]] == ["submitted", "claimed", "completed"]
    assert not Lake(setup["lake_root"]).verify(report["jobs"][0]["result_id"])


def test_submission_validates_all_configs_before_creating_database(setup):
    setup["spec"]["parameters"] = {"viscosity": [0.01, -1]}
    with pytest.raises(ValueError):
        queue_module.submit_sweep(**setup)
    assert not setup["queue_path"].exists()


def test_identity_tracks_request_and_actual_runtime_but_not_dirty_annotation(setup, monkeypatch):
    baseline = queue_module.submit_sweep(**setup)["job_ids"]
    provenance = queue_module.capture_provenance()
    monkeypatch.setattr(queue_module, "capture_provenance", lambda: {
        **provenance, "git_dirty": not provenance["git_dirty"],
    })
    assert queue_module.submit_sweep(**setup)["job_ids"] == baseline
    assert queue_module.submit_sweep(**setup, attempt=1)["job_ids"] != baseline
    assert queue_module.submit_sweep(**setup, stream=False)["job_ids"] != baseline
    assert queue_module.submit_sweep(**{
        **setup, "lake_root": setup["lake_root"] / "other",
    })["job_ids"] != baseline
    monkeypatch.setattr(queue_module, "capture_provenance", lambda: {
        **provenance, "source_sha256": "other-source",
    })
    assert queue_module.submit_sweep(**setup)["job_ids"] != baseline


def test_competing_claims_only_one_owner(setup):
    queue_module.submit_sweep(**setup)
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        return queue_module._claim(setup["queue_path"], 60)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _: claim(), range(2)))
    assert sum(claim is not None for claim in claims) == 1
    assert queue_module.queue_status(setup["queue_path"])["jobs"][0]["claim_count"] == 1


def test_competing_submissions_insert_once(setup):
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait(timeout=5)
        return queue_module.submit_sweep(**setup)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(lambda _: submit(), range(2)))
    assert sum(report["submitted"] for report in reports) == 1
    assert sum(report["existing"] for report in reports) == 1
    assert queue_module.queue_status(setup["queue_path"])["counts"]["queued"] == 1


def test_expiry_fences_stale_renewal_and_acknowledgement(setup):
    queue_module.submit_sweep(**setup)
    path = setup["queue_path"]
    first = queue_module._claim(path, 10, now=100)
    assert queue_module._renew(path, first, 10, now=105)
    assert queue_module._claim(path, 10, now=114) is None
    second = queue_module._claim(path, 10, now=115)
    assert first["id"] == second["id"] and first["token"] != second["token"]
    outcome = {"status": "completed", "result_id": "result"}
    assert not queue_module._renew(path, first, 10, now=116)
    assert not queue_module._finish(path, first, outcome, now=116)
    assert queue_module._finish(path, second, outcome, now=116)
    status = queue_module.queue_status(path)
    assert status["jobs"][0]["claim_count"] == 2
    assert [e["kind"] for e in status["events"]].count("lease_expired") == 1


def test_expired_owner_cannot_revive_without_reclaim(setup):
    queue_module.submit_sweep(**setup)
    path = setup["queue_path"]
    claim = queue_module._claim(path, 1, now=100)
    assert not queue_module._renew(path, claim, 60, now=101)
    assert not queue_module._finish(path, claim, {"status": "failed"}, now=101)


def test_expiry_is_checked_after_waiting_for_write_lock(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    path = setup["queue_path"]
    claim = queue_module._claim(path, 1, now=100)
    clock = [100.5]
    original = queue_module._transaction

    @contextmanager
    def lock_after_expiry(db):
        with original(db):
            clock[0] = 101.5
            yield

    monkeypatch.setattr(queue_module, "_transaction", lock_after_expiry)
    monkeypatch.setattr(queue_module.time, "time", lambda: clock[0])
    assert not queue_module._finish(path, claim, {"status": "completed"})
    clock[0] = 100.5
    assert not queue_module._renew(path, claim, 60)


def test_status_is_read_only_and_does_not_recover_expired_jobs(setup):
    queue_module.submit_sweep(**setup)
    queue_module._claim(setup["queue_path"], 1, now=100)
    first = queue_module.queue_status(setup["queue_path"])
    second = queue_module.queue_status(setup["queue_path"])
    assert first == second
    assert first["expired_leases"] == 1 and first["counts"]["running"] == 1
    assert all(e["kind"] != "lease_expired" for e in first["events"])


def test_crash_after_publication_recovers_and_verifies_existing_result(setup):
    queue_module.submit_sweep(**setup)
    old_claim = queue_module._claim(setup["queue_path"], 1, now=time.time() - 10)
    request = old_claim["request"]
    published = engine.run_experiment(request["config"], request["lake_root"], stream=True)
    # Simulated process death after lake publication but before queue acknowledgement.
    report = queue_module.work_queue(setup["queue_path"])
    assert report["completed"] == 1 and report["jobs"][0]["resumed"] is True
    assert report["jobs"][0]["result_id"] == published.record["id"]
    assert not queue_module._finish(setup["queue_path"], old_claim, {"status": "failed"})
    assert len(Lake(setup["lake_root"]).records()) == 1


def test_recovery_does_not_accept_corrupt_published_result(setup):
    queue_module.submit_sweep(**setup)
    claim = queue_module._claim(setup["queue_path"], 1, now=time.time() - 10)
    request = claim["request"]
    result = engine.run_experiment(request["config"], request["lake_root"])
    (setup["lake_root"] / "experiments" / result.record["id"] / "record.json").write_text("{}")
    report = queue_module.work_queue(setup["queue_path"])
    assert report["failed"] == 1
    assert "verification" in report["jobs"][0]["error"]["message"]


def test_changed_runtime_records_terminal_failure_before_solver(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    changed = {**queue_module.capture_provenance(), "source_sha256": "changed"}
    monkeypatch.setattr(queue_module, "capture_provenance", lambda: changed)

    def forbidden(*args, **kwargs):
        raise AssertionError("Solver must not run under mismatched provenance")

    monkeypatch.setattr(queue_module, "run_experiment", forbidden)
    report = queue_module.work_queue(setup["queue_path"])
    assert report["failed"] == 1
    assert "changed since submission" in report["jobs"][0]["error"]["message"]
    assert queue_module.queue_status(setup["queue_path"])["counts"]["running"] == 0
    assert not setup["lake_root"].exists()


def test_runtime_change_between_precheck_and_execution_not_acknowledged(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    original = queue_module.run_experiment

    def changed_run(*args, **kwargs):
        result = original(*args, **kwargs)
        result.record["id"] = "unexpected-result"
        return result

    monkeypatch.setattr(queue_module, "run_experiment", changed_run)
    report = queue_module.work_queue(setup["queue_path"])
    assert report["failed"] == 1
    assert "identity changed" in report["jobs"][0]["error"]["message"]


def test_cached_worker_cannot_relabel_imported_code_for_new_submission(setup, monkeypatch):
    changed = {**queue_module.capture_provenance(), "source_sha256": "new-disk-code"}
    monkeypatch.setattr(queue_module, "capture_provenance", lambda: changed)
    queue_module.submit_sweep(**setup)

    def forbidden(*args, **kwargs):
        pytest.fail("Cached old code must not run a matching new-disk-code submission")

    monkeypatch.setattr(queue_module, "run_experiment", forbidden)
    report = queue_module.work_queue(setup["queue_path"])
    assert report["failed"] == 1
    assert "changed after import" in report["jobs"][0]["error"]["message"]
    assert not setup["lake_root"].exists()


def test_code_edit_during_solver_cannot_be_acknowledged_as_success(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    original = queue_module.run_experiment
    changed = {**queue_module.capture_provenance(), "source_sha256": "changed-during-solve"}

    def editing_run(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(queue_module, "capture_provenance", lambda: changed)
        return result

    monkeypatch.setattr(queue_module, "run_experiment", editing_run)
    report = queue_module.work_queue(setup["queue_path"])
    assert report["failed"] == 1
    assert "changed during execution" in report["jobs"][0]["error"]["message"]
    assert len(Lake(setup["lake_root"]).records()) == 1


def test_numerical_failures_are_terminal_and_other_jobs_continue(setup):
    setup["spec"]["parameters"] = {"dt": [0.001, 20]}
    queue_module.submit_sweep(**setup)
    report = queue_module.work_queue(setup["queue_path"], max_jobs=2)
    assert report["processed"] == 2 and report["completed"] == report["failed"] == 1
    failed = next(job for job in report["jobs"] if job["status"] == "failed")
    assert failed["result_id"] and "stability" in failed["error"]["message"].lower()
    assert queue_module.submit_sweep(**setup)["existing"] == 2
    assert queue_module.work_queue(setup["queue_path"], max_jobs=2)["processed"] == 0


def test_io_exception_persisted_without_stopping_other_jobs(setup, monkeypatch):
    setup["spec"]["parameters"] = {"seed": [1, 2]}
    queue_module.submit_sweep(**setup)
    original = queue_module.run_experiment
    calls = []

    def sometimes_fails(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_module, "run_experiment", sometimes_fails)
    report = queue_module.work_queue(setup["queue_path"], max_jobs=2)
    assert report["completed"] == report["failed"] == 1
    status = queue_module.queue_status(setup["queue_path"])
    failed = next(j for j in status["jobs"] if j["status"] == "failed")
    assert failed["error"] == {"type": "OSError", "message": "disk full"}
    assert failed["result_id"] is None


def test_heartbeat_renews_and_joins_after_exit(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    claim = queue_module._claim(setup["queue_path"], 60)
    renewed = threading.Event()
    original = queue_module._renew

    def renew(*args, **kwargs):
        result = original(*args, **kwargs)
        renewed.set()
        return result

    monkeypatch.setattr(queue_module, "_renew", renew)
    heartbeat = queue_module._Heartbeat(setup["queue_path"], claim, 0.03)
    with heartbeat:
        assert renewed.wait(5)
    assert not heartbeat.thread.is_alive()
    assert not heartbeat.lost.is_set()


def test_interrupt_stops_heartbeat_and_leaves_claim_recoverable(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    observed = []
    original = queue_module._Heartbeat

    class ObservedHeartbeat(original):
        def __init__(self, *args):
            super().__init__(*args)
            observed.append(self)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(queue_module, "_Heartbeat", ObservedHeartbeat)
    monkeypatch.setattr(queue_module, "run_experiment", interrupted)
    with pytest.raises(KeyboardInterrupt):
        queue_module.work_queue(setup["queue_path"])
    assert len(observed) == 1 and not observed[0].thread.is_alive()
    status = queue_module.queue_status(setup["queue_path"])
    assert status["counts"]["running"] == 1
    assert queue_module._claim(
        setup["queue_path"], 60, now=status["jobs"][0]["lease_until"] + 1,
    ) is not None


def test_heartbeat_error_is_visible_and_worker_does_not_acknowledge(setup, monkeypatch):
    queue_module.submit_sweep(**setup)
    failed = threading.Event()
    original = queue_module._Heartbeat

    class ObservedHeartbeat(original):
        def _run(self):
            try:
                super()._run()
            finally:
                failed.set()

    def cannot_renew(*args):
        raise OSError("database unavailable")

    def wait_for_failure(*args, **kwargs):
        assert failed.wait(5)
        raise OSError("solver also failed")

    monkeypatch.setattr(queue_module, "_Heartbeat", ObservedHeartbeat)
    monkeypatch.setattr(queue_module, "_renew", cannot_renew)
    monkeypatch.setattr(queue_module, "run_experiment", wait_for_failure)
    report = queue_module.work_queue(setup["queue_path"], lease_seconds=1)
    assert report["lost_claim"] == 1
    assert report["jobs"][0]["heartbeat_error"]["message"] == "database unavailable"
    assert queue_module.queue_status(setup["queue_path"])["counts"]["running"] == 1


def test_max_jobs_bounds_work_and_missing_database_is_not_created(setup):
    with pytest.raises(FileNotFoundError):
        queue_module.queue_status(setup["queue_path"])
    with pytest.raises(FileNotFoundError):
        queue_module.work_queue(setup["queue_path"])
    assert not setup["queue_path"].exists()
    setup["spec"]["parameters"] = {"seed": [1, 2]}
    queue_module.submit_sweep(**setup)
    assert queue_module.work_queue(setup["queue_path"], max_jobs=1)["processed"] == 1
    assert queue_module.queue_status(setup["queue_path"])["counts"]["queued"] == 1


@pytest.mark.parametrize("kwargs", [
    {"max_jobs": 0}, {"max_jobs": True}, {"max_jobs": 10001},
    {"lease_seconds": 0}, {"lease_seconds": 0.5}, {"lease_seconds": float("nan")},
    {"lease_seconds": float("inf")}, {"lease_seconds": True}, {"lease_seconds": 86401},
])
def test_worker_rejects_invalid_limits(setup, kwargs):
    with pytest.raises(ValueError):
        queue_module.work_queue(setup["queue_path"], **kwargs)


def test_foreign_sqlite_database_is_rejected_without_modification(setup):
    with sqlite3.connect(setup["queue_path"]) as db:
        db.execute("CREATE TABLE other(data TEXT)")
    for operation in (
        lambda: queue_module.submit_sweep(**setup),
        lambda: queue_module.queue_status(setup["queue_path"]),
        lambda: queue_module.work_queue(setup["queue_path"]),
    ):
        with pytest.raises(ValueError, match="supported Flowstate queue"):
            operation()
    with sqlite3.connect(setup["queue_path"]) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("other",),
        ]
