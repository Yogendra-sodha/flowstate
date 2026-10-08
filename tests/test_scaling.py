import json

import pytest

from flowstate import scaling
from flowstate.engine import capture_provenance


@pytest.fixture
def spec():
    return {"base": {"grid_size": 8, "steps": 2, "save_every": 1}, "parameters": {"seed": [1, 2]}}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": [2]},
        {"workers": [1, 3]},
        {"workers": [1, 1]},
        {"workers": [True]},
        {"modes": []},
        {"modes": ["invalid"]},
        {"modes": ["streamed", "streamed"]},
        {"repeats": True},
        {"repeats": 8},
        {"seed": -1},
    ],
)
def test_invalid_plan_has_no_filesystem_side_effects(tmp_path, spec, kwargs):
    options = {"workers": [1, 2], **kwargs}
    with pytest.raises(ValueError):
        scaling.benchmark_sweep(spec, tmp_path / "study", **options)
    assert not (tmp_path / "study").exists()


def test_plan_bounds_all_trials_before_running(spec):
    too_many = {"base": spec["base"], "parameters": {"seed": list(range(32))}}
    with pytest.raises(ValueError, match="256 fresh runs"):
        scaling._plan(too_many, [1, 2, 4], ["buffered", "streamed"], 3, 0)
    large = {
        "base": {"equation": "navier_stokes2d", "grid_size": 256, "steps": 32, "save_every": 1},
        "parameters": {"seed": [1, 2, 3, 4]},
    }
    with pytest.raises(ValueError, match="2 GiB"):
        scaling._plan(large, [1, 2, 4], ["buffered", "streamed"], 3, 0)


def test_large_plan_accepts_explicitly_bounded_larger_workload(spec):
    larger = {"base": spec["base"], "parameters": {"seed": list(range(256))}}
    plan = scaling._plan(larger, [1, 2, 4, 8], ["streamed"], 3, 0, large=True)
    assert len(plan["configs"]) == 256
    assert plan["fresh_runs"] == 3072
    with pytest.raises(ValueError, match="1 to 32"):
        scaling._plan(larger, [1, 2, 4, 8], ["streamed"], 3, 0)
    with pytest.raises(ValueError, match="4096 fresh runs"):
        scaling._plan(larger, [1, 2, 4, 8], ["streamed"], 5, 0, large=True)
    larger["parameters"]["seed"].append(256)
    with pytest.raises(ValueError, match="1 to 256"):
        scaling._plan(larger, [1], ["streamed"], 1, 0, large=True)


def test_large_plan_retains_step_trial_and_storage_bounds(spec):
    with pytest.raises(ValueError, match="48 trials"):
        scaling._plan(
            {"base": spec["base"], "parameters": {"seed": list(range(8))}},
            [1, 2, 4, 8],
            ["buffered", "streamed"],
            7,
            0,
            large=True,
        )
    with pytest.raises(ValueError, match="2 million integration steps"):
        scaling._plan(
            {"base": {**spec["base"], "steps": 2000}, "parameters": {"seed": list(range(256))}},
            [1, 2, 4, 8],
            ["streamed"],
            1,
            0,
            large=True,
        )
    with pytest.raises(ValueError, match="16 GiB"):
        scaling._plan(
            {
                "base": {
                    "equation": "navier_stokes2d",
                    "grid_size": 256,
                    "steps": 32,
                    "save_every": 1,
                },
                "parameters": {"seed": list(range(64))},
            },
            [1, 2, 4, 8],
            ["streamed"],
            2,
            0,
            large=True,
        )


def test_order_is_seeded_and_contains_each_combination(spec):
    first = scaling._plan(spec, [1, 2], ["buffered", "streamed"], 2, 7)
    assert first == scaling._plan(spec, [1, 2], ["buffered", "streamed"], 2, 7)
    assert first["cases"] != scaling._plan(spec, [1, 2], ["buffered", "streamed"], 2, 8)["cases"]
    assert len({(c["workers"], c["mode"], c["repeat"]) for c in first["cases"]}) == 8
    assert first["fresh_runs"] == 16


def test_isolated_serial_parallel_streamed_outputs_and_reuse_agree(tmp_path, spec, capsys):
    from flowstate.cli import main

    specification = tmp_path / "sweep.json"
    specification.write_text(json.dumps(spec), encoding="utf-8")
    assert (
        main(
            [
                "--lake",
                str(tmp_path / "unused"),
                "sweep-benchmark",
                str(specification),
                str(tmp_path / "study"),
                "--workers",
                "1",
                "2",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "completed" and report["scientific_values_identical"]
    assert len(report["trials"]) == 4
    assert all(t["reused_experiments"] == 2 for t in report["trials"])
    assert all(t["memory"]["sample_count"] >= 2 for t in report["trials"])
    assert all(t["memory"]["sampled_peak_aggregate_rss_bytes"] > 0 for t in report["trials"])
    assert all(t["stored_bytes"] > 0 for t in report["trials"])
    assert all(t["failure_count"] == 0 for t in report["trials"])
    assert all(t["completed_experiments"] == 2 for t in report["trials"])
    assert all(t["runs_per_hour"] == t["attempts_per_hour"] for t in report["trials"])
    assert (tmp_path / "study" / "report.json").exists()
    with pytest.raises(FileExistsError):
        scaling.benchmark_sweep(spec, tmp_path / "study", workers=[1, 2], repeats=1)


def test_numerical_failure_is_retained_and_invalidates_trial(tmp_path):
    request = {
        "spec": {"base": {"grid_size": 8, "dt": 100}},
        "case": {"workers": 1, "mode": "buffered", "repeat": 0},
        "provenance": capture_provenance(),
    }
    with pytest.raises(RuntimeError, match="must complete"):
        scaling._execute_case(request, tmp_path / "trial")
    report = json.loads((tmp_path / "trial" / "report.json").read_text())
    assert report["status"] == "failed" and report["outcomes"][0]["status"] == "failed"
    assert report["failure_count"] == 1 and report["completed_experiments"] == 0
    assert report["runs_per_hour"] == 0 and report["attempts_per_hour"] > 0
    assert len(list((tmp_path / "trial" / "lake" / "experiments").iterdir())) == 1


def test_allowed_numerical_failure_is_verified_and_reused(tmp_path, monkeypatch):
    verified = []
    original_verify = scaling.Lake.verify

    def record_verification(lake, experiment_id):
        verified.append(experiment_id)
        return original_verify(lake, experiment_id)

    monkeypatch.setattr(scaling.Lake, "verify", record_verification)
    request = {
        "spec": {"base": {"grid_size": 8, "steps": 2}, "parameters": {"dt": [0.001, 100]}},
        "case": {"workers": 1, "mode": "streamed", "repeat": 0},
        "provenance": capture_provenance(),
        "allow_numerical_failures": True,
    }
    report = scaling._execute_case(request, tmp_path / "trial")
    assert report["status"] == "completed"
    assert report["completed_experiments"] == report["failure_count"] == 1
    assert report["reused_experiments"] == 2
    assert report["attempts_per_hour"] == 2 * report["runs_per_hour"]
    results = list(report["scientific_results"].values())
    failure = next(result for result in results if result.get("status") == "failed")
    assert failure["values_sha256"] is None
    assert failure["error"]["type"] == "ValueError"
    assert failure["error"]["message"]
    assert verified.count(failure["id"]) >= 2
    assert not (
        tmp_path / "trial" / "lake" / "experiments" / failure["id"] / "fields.zarr"
    ).exists()


def test_allowing_numerical_failure_never_bypasses_artifact_verification(tmp_path, monkeypatch):
    original_verify = scaling.Lake.verify

    def reject_failed_artifact(lake, experiment_id):
        if lake.load_record(experiment_id)["status"] == "failed":
            return ["Injected failed-record checksum mismatch"]
        return original_verify(lake, experiment_id)

    monkeypatch.setattr(scaling.Lake, "verify", reject_failed_artifact)
    request = {
        "spec": {"base": {"grid_size": 8, "dt": 100}},
        "case": {"workers": 1, "mode": "streamed"},
        "provenance": capture_provenance(),
        "allow_numerical_failures": True,
    }
    with pytest.raises(ValueError, match="failed-record checksum mismatch"):
        scaling._execute_case(request, tmp_path / "trial")
    report = json.loads((tmp_path / "trial" / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_count"] == 1
    assert "reused_experiments" not in report


@pytest.mark.parametrize("alteration", ["reused", "missing", "unfinished"])
def test_allowing_failures_rejects_invalid_fresh_outcomes(tmp_path, monkeypatch, alteration):
    from flowstate.engine import RunOutcome, run_experiment

    def invalid_sweep(spec, lake_root, **kwargs):
        outcome = run_experiment(spec["base"], lake_root)
        if alteration == "missing":
            return []
        if alteration == "reused":
            return [RunOutcome(outcome.record, resumed=True)]
        return [RunOutcome({**outcome.record, "status": "running"}, resumed=False)]

    monkeypatch.setattr(scaling, "run_sweep", invalid_sweep)
    request = {
        "spec": {"base": {"grid_size": 8, "steps": 2}},
        "case": {"workers": 1, "mode": "buffered"},
        "provenance": capture_provenance(),
        "allow_numerical_failures": True,
    }
    with pytest.raises(RuntimeError, match="must complete without reuse"):
        scaling._execute_case(request, tmp_path / "trial")
    report = json.loads((tmp_path / "trial" / "report.json").read_text())
    assert report["status"] == "failed"
    assert len(report["committed_experiment_ids"]) == 1


@pytest.mark.parametrize("returncode", [0, 7])
def test_isolated_trial_passes_environment_and_retains_stderr(tmp_path, monkeypatch, returncode):
    output = tmp_path / "trial"
    output.mkdir()
    (output / "report.json").write_text('{"status": "completed"}', encoding="utf-8")
    environment = {"OMP_NUM_THREADS": "1"}
    seen = {}

    class Job:
        def __init__(self, command, **kwargs):
            seen.update(kwargs)
            self.returncode = returncode

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self):
            return "", "retained worker diagnostic\n"

    monkeypatch.setattr(scaling.subprocess, "Popen", Job)
    if returncode:
        with pytest.raises(RuntimeError, match="Trial process exited 7"):
            scaling._isolated_trial(tmp_path / "request.json", output, env=environment)
    else:
        assert scaling._isolated_trial(tmp_path / "request.json", output, env=environment) == {
            "status": "completed"
        }
    assert seen["env"] == environment
    assert (output / "stderr.txt").read_text(encoding="utf-8") == ("retained worker diagnostic\n")
    with pytest.raises(FileExistsError):
        scaling._isolated_trial(tmp_path / "request.json", output, env=environment)


def test_git_dirty_annotation_does_not_change_runtime_identity():
    provenance = capture_provenance()
    assert scaling._runtime_identity({**provenance, "git_dirty": True}) == (
        scaling._runtime_identity({**provenance, "git_dirty": False})
    )
    assert scaling._runtime_identity({**provenance, "source_sha256": "changed"}) != (
        scaling._runtime_identity(provenance)
    )


def test_io_failure_retains_elapsed_memory_and_partial_publication(tmp_path, spec, monkeypatch):
    from flowstate.engine import run_experiment

    def partial_failure(spec, lake_root, **kwargs):
        run_experiment(spec["base"], lake_root)
        raise OSError("Injected storage failure")

    monkeypatch.setattr(scaling, "run_sweep", partial_failure)
    request = {
        "spec": spec,
        "case": {"workers": 1, "mode": "buffered"},
        "provenance": capture_provenance(),
    }
    with pytest.raises(OSError, match="Injected storage"):
        scaling._execute_case(request, tmp_path / "trial")
    report = json.loads((tmp_path / "trial" / "report.json").read_text())
    assert report["wall_seconds"] > 0
    assert report["memory"]["sample_count"] >= 2
    assert report["memory"]["finished"]
    assert len(report["committed_experiment_ids"]) == 1
    assert report["completed_experiments"] == 1
    assert report["failure_count"] == 0
    assert report["runs_per_hour"] > 0
    assert report["attempts_per_hour"] is None
    assert report["error"]["type"] == "OSError"


def test_cross_trial_value_mismatch_prevents_success_summary(tmp_path, spec, monkeypatch):
    counter = 0

    def altered_trial(request, output):
        nonlocal counter
        counter += 1
        return {
            "status": "completed",
            "scientific_results": {"config": str(counter)},
            "wall_seconds": 1,
        }

    monkeypatch.setattr(scaling, "_isolated_trial", altered_trial)
    with pytest.raises(RuntimeError, match="Scientific values"):
        scaling.benchmark_sweep(spec, tmp_path / "study", workers=[1], repeats=1)
    assert not (tmp_path / "study" / "report.json").exists()
    events = [
        json.loads(line) for line in (tmp_path / "study" / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["status"] == "failed"


def test_aggregation_uses_medians_and_matching_mode_baselines():
    def trial(mode, workers, seconds, peak):
        return {
            "case": {"mode": mode, "workers": workers},
            "wall_seconds": seconds,
            "memory": {"sampled_peak_aggregate_rss_bytes": peak, "sampling_errors": 0},
            "experiments_per_second": 8 / seconds,
            "verified_reuse_seconds": 1,
            "stored_bytes": 100,
        }

    summary = scaling._aggregate(
        [
            trial("buffered", 1, 9, 100),
            trial("buffered", 1, 11, 120),
            trial("buffered", 2, 4, 200),
            trial("buffered", 2, 6, 220),
            trial("streamed", 1, 20, 80),
            trial("streamed", 2, 4, None),
        ]
    )
    assert summary[1]["speedup_vs_one_worker"] == 2
    assert summary[1]["sampled_peak_aggregate_rss_bytes_max"] == 220
    assert summary[1]["completed_experiments"] == 16
    assert summary[1]["failure_count"] == 0
    assert summary[1]["runs_per_hour_median"] == pytest.approx((8 / 4 + 8 / 6) * 1800)
    assert summary[1]["attempts_per_hour_median"] == pytest.approx(
        summary[1]["runs_per_hour_median"]
    )
    assert summary[1]["verified_reuse_seconds_min"] == 1
    assert summary[1]["verified_reuse_seconds_max"] == 1
    assert summary[1]["stored_bytes_min"] == 100
    assert summary[1]["stored_bytes_max"] == 100
    assert summary[3]["speedup_vs_one_worker"] == 5
    assert summary[3]["sampled_peak_aggregate_rss_bytes_max"] is None


def test_aggregation_does_not_count_failures_as_successful_throughput():
    summary = scaling._aggregate(
        [
            {
                "case": {"mode": "streamed", "workers": 1},
                "wall_seconds": 36,
                "memory": {"sampled_peak_aggregate_rss_bytes": 100, "sampling_errors": 0},
                "experiments_per_second": 2 / 36,
                "completed_experiments": 1,
                "failure_count": 1,
                "runs_per_hour": 100,
                "attempts_per_hour": 200,
                "verified_reuse_seconds": 1,
                "stored_bytes": 100,
            }
        ]
    )[0]
    assert summary["failure_count"] == summary["completed_experiments"] == 1
    assert summary["runs_per_hour_median"] == 100
    assert summary["attempts_per_hour_median"] == 200
