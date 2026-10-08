import json

import pytest

from flowstate import scaling_study
from flowstate.engine import capture_provenance
from flowstate.scaling import _config_key

pytest.importorskip("matplotlib")


def test_frozen_protocol_counts_configurations_families_and_executions_separately():
    plan = scaling_study.study_plan()
    assert plan == scaling_study.study_plan()
    assert plan["distinct_configurations"] == 216
    assert plan["initial_condition_families"] == 36
    assert plan["fresh_runs"] == plan["verified_reuse_requests"] == 2592
    assert plan["integration_steps"] == 2592 * 32
    assert plan["estimated_uncompressed_field_bytes"] == 5_357_449_728
    assert len(plan["cases"]) == len({c["id"] for c in plan["cases"]}) == 36
    all_configs = [c for configs in plan["configs_by_grid"].values() for c in configs]
    assert len({_config_key(c) for c in all_configs}) == 216
    for grid in (64, 128, 256):
        assert len(plan["configs_by_grid"][str(grid)]) == 72
        cases = [c for c in plan["cases"] if c["grid_size"] == grid]
        assert {(c["workers"], c["repeat"]) for c in cases} == {
            (workers, repeat) for workers in (1, 2, 4, 8) for repeat in range(3)
        }


@pytest.fixture
def clean_source(monkeypatch):
    provenance = {**capture_provenance(), "git_dirty": False}
    monkeypatch.setattr(scaling_study, "capture_provenance", lambda: provenance)
    return provenance


def test_dirty_source_rejected_before_creating_study(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling_study, "capture_provenance", lambda: {
        "git_commit": "a" * 40, "git_dirty": True,
    })
    root = tmp_path / "study"
    with pytest.raises(ValueError, match="Commit all study code"):
        scaling_study.run_scaling_study(root)
    assert not root.exists()


def test_existing_directory_is_never_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("retained", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new study directory"):
        scaling_study.run_scaling_study(tmp_path)
    assert marker.read_text() == "retained"


def test_insufficient_disk_rejected_before_creating_study(tmp_path, clean_source, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(scaling_study.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    root = tmp_path / "study"
    with pytest.raises(ValueError, match="Insufficient free disk"):
        scaling_study.run_scaling_study(root)
    assert not root.exists()


def _tiny_plan():
    return {
        "cases": [
            {"id": "one", "grid_size": 8, "workers": 1, "mode": "streamed", "repeat": 0},
            {"id": "two", "grid_size": 8, "workers": 2, "mode": "streamed", "repeat": 0},
        ],
        "specifications": {"8": {
            "base": {"grid_size": 8, "steps": 2, "save_every": 1},
            "parameters": {"seed": [1, 2]},
        }},
        "minimum_free_bytes_before_start": 1,
        "distinct_configurations": 2,
        "runs_per_trial": 2,
        "repeats": 1,
    }


def test_tiny_study_verifies_serial_parallel_results_and_retains_report(
    tmp_path, clean_source, monkeypatch,
):
    monkeypatch.setattr(scaling_study, "study_plan", _tiny_plan)
    root = tmp_path / "study"
    report = scaling_study.run_scaling_study(root)
    assert report == json.loads((root / "scaling-large.json").read_text())
    assert report["status"] == "completed"
    assert report["scientific_values_identical"]
    assert report["totals"]["fresh_runs"] == report["totals"]["verified_reused_runs"] == 4
    assert report["totals"]["failure_count"] == 0
    assert {row["workers"] for row in report["summary"]} == {1, 2}
    assert (root / "scaling-large.png").read_bytes().startswith(b"\x89PNG")
    assert all(t["failure_count"] == 0 for t in report["trials"])
    assert all(t["runtime_thread_settings"] == scaling_study.THREAD_SETTINGS
               for t in report["trials"])


def test_summary_keeps_separate_serial_baselines_per_grid():
    def trial(grid, workers, seconds):
        return {
            "case": {"grid_size": grid, "workers": workers, "mode": "streamed"},
            "wall_seconds": seconds,
            "experiments_per_second": 72 / seconds,
            "memory": {"sampled_peak_aggregate_rss_bytes": 100, "sampling_errors": 0},
            "verified_reuse_seconds": 2,
            "stored_bytes": 1000,
        }

    summary = scaling_study._summary([
        trial(64, 1, 12), trial(64, 2, 6), trial(256, 1, 20), trial(256, 2, 5),
    ])
    assert [(row["grid_size"], row["workers"], row["speedup_vs_one_worker"])
            for row in summary] == [(64, 1, 1), (64, 2, 2), (256, 1, 1), (256, 2, 4)]


def test_infrastructure_failure_retains_failed_study_without_success_summary(
    tmp_path, clean_source, monkeypatch,
):
    monkeypatch.setattr(scaling_study, "study_plan", _tiny_plan)

    def fail(request, output, *, env):
        assert all(env[key] == "1" for key in scaling_study.THREAD_SETTINGS)
        raise RuntimeError("Injected child failure")

    monkeypatch.setattr(scaling_study, "_isolated_trial", fail)
    root = tmp_path / "study"
    with pytest.raises(RuntimeError, match="Injected child failure"):
        scaling_study.run_scaling_study(root)
    report = json.loads((root / "scaling-large.json").read_text())
    assert report["status"] == "failed"
    assert "summary" not in report and "scientific_values_identical" not in report
    assert report["study_elapsed_seconds"] >= 0
    assert report["error"]["message"] == "Injected child failure"
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["started", "failed"]


def test_cli_dispatches_without_creating_an_unrelated_lake(tmp_path, monkeypatch, capsys):
    from flowstate.cli import main

    monkeypatch.setattr(scaling_study, "run_scaling_study", lambda output: {
        "status": "completed", "output": output, "totals": {}, "summary": [],
    })
    unused = tmp_path / "unused-lake"
    assert main(["--lake", str(unused), "scaling-study", str(tmp_path / "study")]) == 0
    assert not unused.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
