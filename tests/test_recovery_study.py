import json
from types import SimpleNamespace

import psutil
import pytest

from flowstate import recovery_study


def test_actual_worker_kill_resumes_both_equations_without_repeating_completed_steps(tmp_path):
    root = tmp_path / "study"
    report = recovery_study.run_recovery_study(
        root, grid_size=8, steps=11, save_every=4, checkpoint_every=3,
        require_clean=False,
    )
    assert report == json.loads((root / "report.json").read_text())
    assert report["status"] == "completed", report["failures"]
    assert report["failure_count"] == 0
    assert report["all_scientific_arrays_hash_identical"]
    assert {case["config"]["equation"] for case in report["cases"]} == {
        "burgers1d", "navier_stokes2d",
    }
    for case in report["cases"]:
        assert case["status"] == "completed"
        assert case["scientific_arrays"]["baseline"] == case["scientific_arrays"]["resumed"]
        assert all(not failures for failures in case["artifact_verification"].values())
        baseline, interrupted, resumed = (
            case["workers"][role] for role in ("baseline", "interrupted", "resumed")
        )
        assert baseline["rk4_calls"] == 11
        assert interrupted["checkpoint_step"] == interrupted["rk4_calls"] == 3
        assert resumed["resumed_from_step"] == 3 and resumed["rk4_calls"] == 8
        assert not resumed["finalized_result_reused"]
        assert interrupted["termination"]["returncode"] != 0
        assert interrupted["pid"] in interrupted["termination"]["terminated_pids"]
        assert all(not psutil.pid_exists(pid)
                   for pid in interrupted["termination"]["terminated_pids"])
        assert any(name.startswith("fields/") for name in case["scientific_arrays"]["resumed"])
        evidence = root / case["config"]["equation"]
        assert (evidence / "checkpoint-ready.json").is_file()
        assert (evidence / "interrupted-stderr.log").is_file()
        assert list((evidence / "recovery-lake" / "checkpoints").glob("*/step-3.json"))
        assert not (evidence / "interrupted-result.json").exists()


@pytest.mark.parametrize("options", [
    {"worker_timeout": float("inf")},
    {"worker_timeout": True},
    {"checkpoint_every": True},
    {"checkpoint_every": 0},
    {"checkpoint_every": 37},
    {"checkpoint_every": 8},
    {"steps": 1001},
    {"grid_size": 256},
    {"steps": 1000, "checkpoint_every": 1},
])
def test_invalid_study_plan_creates_no_output(tmp_path, options):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        recovery_study.run_recovery_study(root, require_clean=False, **options)
    assert not root.exists()


def test_measured_study_requires_committed_clean_source(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery_study, "capture_provenance", lambda: {
        "git_commit": "a" * 40, "git_dirty": True,
    })
    root = tmp_path / "dirty"
    with pytest.raises(ValueError, match="Commit code and use clean source"):
        recovery_study.run_recovery_study(root)
    assert not root.exists()


def test_existing_evidence_is_never_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("retained", encoding="utf-8")
    with pytest.raises(FileExistsError):
        recovery_study.run_recovery_study(tmp_path, require_clean=False)
    assert marker.read_text() == "retained"


def test_worker_failure_keeps_failure_report_and_raw_evidence(tmp_path, monkeypatch):
    def fail(request, directory, timeout, *, interrupt=False):
        (directory / "injected-stderr.log").write_text("worker fault", encoding="utf-8")
        raise OSError("Injected worker failure")

    monkeypatch.setattr(recovery_study, "_invoke_worker", fail)
    root = tmp_path / "failed"
    report = recovery_study.run_recovery_study(root, require_clean=False)
    assert report == json.loads((root / "report.json").read_text())
    assert report["status"] == "failed" and report["failure_count"] == 2
    assert not report["all_scientific_arrays_hash_identical"]
    assert all(failure["message"] == "Injected worker failure" for failure in report["failures"])
    for case in report["cases"]:
        evidence = root / case["config"]["equation"]
        assert (evidence / "injected-stderr.log").read_text() == "worker fault"
        assert json.loads((evidence / "case-report.json").read_text())["status"] == "failed"
    assert (root / "plan.json").is_file()


def test_ready_receipt_accepts_owned_interpreter_but_rejects_unrelated_pid(monkeypatch):
    interpreter = SimpleNamespace(pid=12, is_running=lambda: True)
    launcher = SimpleNamespace(
        pid=11, is_running=lambda: True, children=lambda recursive: [interpreter],
    )
    monkeypatch.setattr(recovery_study.psutil, "Process", lambda pid: launcher)
    process = SimpleNamespace(pid=11, poll=lambda: None)
    recovery_study._validate_ready(process, {"pid": 12, "checkpoint_commit_complete": True})
    with pytest.raises(RuntimeError, match="does not identify the owned worker"):
        recovery_study._validate_ready(process, {"pid": 99, "checkpoint_commit_complete": True})
    with pytest.raises(RuntimeError, match="does not identify the owned worker"):
        recovery_study._validate_ready(process, {"pid": 12, "checkpoint_commit_complete": 1})
