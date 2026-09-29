"""Offline orchestration checks for the fixed public-data comparison plan."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import flowstate.public_study as public_study


@pytest.fixture
def study_pipeline(monkeypatch):
    """Mock external stages while retaining real plan/events/report file writing."""
    state = SimpleNamespace(
        calls=[],
        plan_bytes=None,
        failure_stage=None,
        failure=RuntimeError("stage failed"),
        dataset_problems=[],
        model_problem=None,
    )
    model_module = ModuleType("flowstate.ml")
    dataset = {
        "shape": [24, 201, 1024],
        "viscosity": 0.01 / 3.141592653589793,
        "domain_length": 1.0,
        "saved_dt": 0.01,
        "splits": {
            "train": list(range(16)),
            "validation": list(range(16, 20)),
            "test": list(range(20, 24)),
        },
        "split_sha256": "frozen-split-hash",
        "normalization": {"fit_split": "train", "mean": 0.0, "std": 0.5},
        "trajectories": [{"id": f"trajectory-{index}"} for index in range(24)],
    }

    def observe(stage, output):
        root = Path(output).parent
        plan_bytes = (root / "plan.json").read_bytes()
        if state.plan_bytes is None:
            state.plan_bytes = plan_bytes
        assert plan_bytes == state.plan_bytes, "Plan changed after execution began"
        state.calls.append(stage)
        if state.failure_stage == stage:
            raise state.failure
        return json.loads(plan_bytes)

    def acquire(output, **options):
        plan = observe("acquisition", output)
        assert options == {name: plan[name] for name in ("samples", "sample_seed", "time_stop")}
        output.mkdir()
        (output / "manifest.json").write_text('{"kind":"fixture-acquisition"}', "utf-8")
        return {"transfer": {"bytes_received": 12345, "full_remote_checksum_verified": False}}

    def import_dataset(acquired, output, *, seed):
        plan = observe("dataset", output)
        assert acquired == output.parent / "acquisition"
        assert seed == plan["split_seed"]
        output.mkdir()
        (output / "manifest.json").write_text('{"kind":"fixture-dataset"}', "utf-8")
        return dataset

    def train_fno(dataset_path, output, *, seed, **options):
        plan = observe(f"fno_seed_{seed}", output)
        assert dataset_path == output.parent / "dataset"
        assert options == plan["fno"]
        assert seed in plan["fno_seeds"]
        output.mkdir()
        trained = {
            "best_epoch": seed + 1,
            "training_seconds": seed + 0.25,
            "checkpoint_sha256": f"checkpoint-{seed}",
        }
        (output / "report.json").write_text(json.dumps(trained), "utf-8")
        return trained

    def evaluate_fno(dataset_path, model_path, *, output):
        seed = int(model_path.name.removeprefix("fno_seed_"))
        observe(f"evaluation_{seed}", output)
        assert dataset_path == output.parent / "dataset"
        output.mkdir()
        # The middle seed wins this fake metric. Every seed must still appear.
        report = {
            "split": "test",
            "rollout": {"rmse": [0.9, 0.1, 0.5][seed]},
            "persistence_rollout": {"rmse": 0.7},
        }
        (output / "report.json").write_text(json.dumps(report), "utf-8")
        return report

    def train_pinn(dataset_path, output, **options):
        plan = observe("pinn", output)
        assert dataset_path == output.parent / "dataset"
        assert options == plan["pinn"]
        assert "trajectory_index" not in options  # Trainer's frozen-test-first default.
        output.mkdir()
        report = {
            "trajectory_id": "trajectory-20",
            "training_seconds": 2.5,
            "trajectory_error": {"rmse": 0.2},
            "persistence_trajectory_error": {"rmse": 0.7},
            "physical_residual_rms": 0.03,
            "checkpoint_sha256": "pinn-checkpoint",
            "supervision": "initial frame only",
            "comparison_scope": "per-instance physics",
        }
        (output / "report.json").write_text(json.dumps(report), "utf-8")
        return report

    def verify_model(path):
        state.calls.append(f"verify_{path.name}")
        return ["corrupt model"] if path.name == state.model_problem else []

    model_module.train_fno = train_fno
    model_module.evaluate_fno = evaluate_fno
    model_module.train_pinn = train_pinn
    model_module.verify_model = verify_model
    monkeypatch.setitem(sys.modules, "flowstate.ml", model_module)
    monkeypatch.setattr(public_study, "capture_provenance", lambda: {"git_commit": "fixture"})
    monkeypatch.setattr(public_study, "acquire_pdebench", acquire)
    monkeypatch.setattr(public_study, "import_acquired_pdebench", import_dataset)
    monkeypatch.setattr(public_study, "verify_dataset", lambda path: state.dataset_problems)
    return state


def read_events(output):
    return [json.loads(line) for line in (output / "events.jsonl").read_text("utf-8").splitlines()]


def test_success_retains_all_three_seeds_and_hashes_the_unchanged_plan(tmp_path, study_pipeline):
    output = tmp_path / "study"
    report = public_study.run_public_study(output, epochs=3, pinn_epochs=2)
    plan_bytes = (output / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    assert plan_bytes == study_pipeline.plan_bytes
    assert plan["fno_seeds"] == [0, 1, 2]
    assert plan["samples"] == 24 and plan["time_stop"] == 201
    assert plan["sample_seed"] == 20260926 and plan["split_seed"] == 17
    assert plan["fno"]["epochs"] == 3 and plan["pinn"]["epochs"] == 2
    assert report["status"] == "completed"
    assert report["plan"] == plan
    assert report["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert [run["seed"] for run in report["fno_runs"]] == [0, 1, 2]
    assert [run["evaluation"]["rollout"]["rmse"] for run in report["fno_runs"]] == [0.9, 0.1, 0.5]
    assert [run["checkpoint_sha256"] for run in report["fno_runs"]] == [
        "checkpoint-0",
        "checkpoint-1",
        "checkpoint-2",
    ]
    assert "best_seed" not in report and "winner" not in report
    assert report["pinn"]["trajectory_id"] == "trajectory-20"
    for name in ("dataset", "acquisition"):
        assert (
            report[f"{name}_manifest_sha256"]
            == hashlib.sha256((output / name / "manifest.json").read_bytes()).hexdigest()
        )
    assert json.loads((output / "report.json").read_text("utf-8")) == report
    assert not (output / "failure.json").exists()
    assert study_pipeline.calls == [
        "acquisition",
        "dataset",
        "fno_seed_0",
        "verify_fno_seed_0",
        "evaluation_0",
        "fno_seed_1",
        "verify_fno_seed_1",
        "evaluation_1",
        "fno_seed_2",
        "verify_fno_seed_2",
        "evaluation_2",
        "pinn",
        "verify_pinn",
    ]
    events = read_events(output)
    assert events[-1]["stage"] == "study" and events[-1]["status"] == "completed"
    for stage in ("acquisition", "dataset", "fno_seed_0", "fno_seed_1", "fno_seed_2", "pinn"):
        assert [event["status"] for event in events if event["stage"] == stage] == [
            "started",
            "completed",
        ]


@pytest.mark.parametrize("failed_stage", ["acquisition", "dataset", "fno_seed_1", "pinn"])
def test_failure_preserves_original_plan_failure_record_events_and_prior_evidence(
    tmp_path, study_pipeline, failed_stage
):
    output = tmp_path / "failed-study"
    study_pipeline.failure_stage = failed_stage
    with pytest.raises(RuntimeError) as caught:
        public_study.run_public_study(output, epochs=3, pinn_epochs=2)
    assert caught.value is study_pipeline.failure
    assert (output / "plan.json").read_bytes() == study_pipeline.plan_bytes
    assert not (output / "report.json").exists()
    failure = json.loads((output / "failure.json").read_text("utf-8"))
    assert failure == {
        "stage": failed_stage,
        "error_type": "RuntimeError",
        "message": "stage failed",
        "plan_sha256": hashlib.sha256(study_pipeline.plan_bytes).hexdigest(),
    }
    events = read_events(output)
    assert events[-1]["stage"] == failed_stage and events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "RuntimeError"
    assert not any(event["stage"] == "study" and event["status"] == "completed" for event in events)
    if failed_stage in ("fno_seed_1", "pinn"):
        assert (output / "fno_seed_0" / "report.json").is_file()
        assert (output / "fno_seed_0_evaluation" / "report.json").is_file()
        assert (output / "dataset" / "manifest.json").is_file()
    previous_calls = list(study_pipeline.calls)
    with pytest.raises(FileExistsError):
        public_study.run_public_study(output, epochs=3, pinn_epochs=2)
    assert study_pipeline.calls == previous_calls
    assert (output / "plan.json").read_bytes() == study_pipeline.plan_bytes


def test_keyboard_interrupt_is_recorded_and_propagated(tmp_path, study_pipeline):
    study_pipeline.failure_stage = "acquisition"
    study_pipeline.failure = KeyboardInterrupt("cancelled")
    output = tmp_path / "cancelled-study"
    with pytest.raises(KeyboardInterrupt) as caught:
        public_study.run_public_study(output)
    assert caught.value is study_pipeline.failure
    assert (
        json.loads((output / "failure.json").read_text("utf-8"))["error_type"]
        == "KeyboardInterrupt"
    )
    assert read_events(output)[-1]["status"] == "failed"


@pytest.mark.parametrize("integrity_failure", ["dataset", "fno_seed_1", "pinn"])
def test_verification_failure_stops_and_is_recorded(tmp_path, study_pipeline, integrity_failure):
    if integrity_failure == "dataset":
        study_pipeline.dataset_problems = ["corrupt dataset"]
    else:
        study_pipeline.model_problem = integrity_failure
    output = tmp_path / "integrity-failure"
    with pytest.raises(ValueError, match="verification failed"):
        public_study.run_public_study(output, epochs=3, pinn_epochs=2)
    failure = json.loads((output / "failure.json").read_text("utf-8"))
    assert failure["stage"] == integrity_failure
    assert failure["error_type"] == "ValueError"
    assert not (output / "report.json").exists()
    if integrity_failure == "fno_seed_1":
        assert "evaluation_1" not in study_pipeline.calls
        assert "fno_seed_2" not in study_pipeline.calls


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 51},
        {"epochs": 1.5},
        {"pinn_epochs": 0},
        {"pinn_epochs": True},
        {"pinn_epochs": 2001},
    ],
)
def test_invalid_training_budget_does_not_create_study(tmp_path, study_pipeline, options):
    output = tmp_path / "invalid-study"
    with pytest.raises(ValueError):
        public_study.run_public_study(output, **options)
    assert not output.exists()
    assert study_pipeline.calls == []
