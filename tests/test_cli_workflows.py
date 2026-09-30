"""User-facing dispatch and exit codes for durable jobs and constrained learning."""

import json

from flowstate.cli import main


def test_dashboard_cli_exports_snapshot_and_rejects_overwrite(tmp_path, capsys):
    from flowstate.engine import run_experiment

    lake, output = tmp_path / "lake", tmp_path / "snapshot.html"
    run_experiment({"grid_size": 8, "steps": 2}, lake)
    args = ["--lake", str(lake), "dashboard", str(output), "--max-experiments", "1"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["displayed"] == 1 and result["output_html"] == str(output.resolve())
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err


def test_dashboard_cli_does_not_create_missing_lake(tmp_path, capsys):
    lake = tmp_path / "missing"
    assert main(["--lake", str(lake), "dashboard", str(tmp_path / "snapshot.html")]) == 2
    assert not lake.exists()
    assert "flowstate:" in capsys.readouterr().err


def test_queue_cli_submits_executes_and_reports_the_same_jobs(tmp_path, capsys):
    spec = tmp_path / "sweep.json"
    spec.write_text(
        json.dumps(
            {"base": {"grid_size": 8, "steps": 2, "save_every": 1}, "parameters": {"seed": [0, 1]}}
        )
    )
    queue, lake = str(tmp_path / "jobs.sqlite"), str(tmp_path / "lake")
    prefix = ["--lake", lake, "queue"]
    assert main([*prefix, "submit", str(spec), queue]) == 0
    submitted = json.loads(capsys.readouterr().out)
    assert submitted["submitted"] == 2
    assert main([*prefix, "work", queue, "--max-jobs", "2"]) == 0
    worked = json.loads(capsys.readouterr().out)
    assert worked["completed"] == 2 and worked["failed"] == 0
    assert main([*prefix, "status", queue]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["counts"] == {"queued": 0, "running": 0, "completed": 2, "failed": 0}
    assert {job["id"] for job in status["jobs"]} == set(submitted["job_ids"])
    assert main([*prefix, "submit", str(spec), queue]) == 0
    assert json.loads(capsys.readouterr().out)["existing"] == 2


def test_queue_cli_returns_failure_for_numerical_failures(tmp_path, capsys):
    spec = tmp_path / "unstable.json"
    spec.write_text(
        json.dumps({"base": {"grid_size": 8, "steps": 2, "dt": 100.0}, "parameters": {"seed": [0]}})
    )
    queue = str(tmp_path / "jobs.sqlite")
    prefix = ["--lake", str(tmp_path / "lake"), "queue"]
    assert main([*prefix, "submit", str(spec), queue]) == 0
    capsys.readouterr()
    assert main([*prefix, "work", queue]) == 1
    assert json.loads(capsys.readouterr().out)["failed"] == 1


def test_invalid_queue_cli_returns_intelligible_error(tmp_path, capsys):
    invalid = tmp_path / "invalid.sqlite"
    invalid.write_text("not a database")
    assert main(["--lake", str(tmp_path / "lake"), "queue", "status", str(invalid)]) == 2
    assert "flowstate:" in capsys.readouterr().err


def test_train_cli_delivers_conservation_flag(tmp_path, monkeypatch, capsys):
    import pytest

    ml = pytest.importorskip("flowstate.ml")
    calls = []

    def train(dataset, output, **options):
        calls.append((dataset, output, options))
        return {"conserve_mean": options["conserve_mean"]}

    monkeypatch.setattr(ml, "train_fno", train)
    assert (
        main(
            [
                "--lake",
                str(tmp_path / "lake"),
                "train-fno",
                "dataset",
                "model",
                "--conserve-mean",
                "--epochs",
                "2",
            ]
        )
        == 0
    )
    assert calls[0][2]["conserve_mean"] is True and calls[0][2]["epochs"] == 2
    assert json.loads(capsys.readouterr().out)["conserve_mean"] is True
