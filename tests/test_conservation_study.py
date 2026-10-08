"""Offline checks for a frozen, paired study with an excluded earlier cohort."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import flowstate.conservation_study as study


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "manifest.json").write_text('{"kind":"previous-fixture"}', "utf-8")
    old_indices = sorted(np.random.default_rng(20260926).choice(10000, 24, replace=False).tolist())
    new_indices = sorted(np.random.default_rng(20260929).choice(10000, 24, replace=False).tolist())
    old_metadata = {
        "provenance": {"acquisition": {"source": dict(study.SOURCE)}},
        "trajectories": [
            {"remote_source_index": index, "initial_field_sha256": f"old-field-{index}"}
            for index in old_indices
        ],
    }
    dataset = {
        "shape": [24, 201, 1024],
        "viscosity": 0.01 / np.pi,
        "domain_length": 2.0,
        "saved_dt": 0.01,
        "splits": {
            "train": list(range(16)),
            "validation": list(range(16, 20)),
            "test": list(range(20, 24)),
        },
        "split_sha256": "new-frozen-split",
        "normalization": {"fit_split": "train", "mean": 0.1, "std": 0.5},
        "trajectories": [
            {
                "id": f"new-{index}",
                "remote_source_index": index,
                "initial_field_sha256": f"new-field-{index}",
            }
            for index in new_indices
        ],
    }
    state = SimpleNamespace(
        previous=previous,
        old_metadata=old_metadata,
        dataset=dataset,
        new_indices=new_indices,
        acquired_indices=list(new_indices),
        calls=[],
        plan_bytes=None,
        previous_problems=[],
        dataset_problems=[],
        model_problem=None,
        failure_stage=None,
        failure=RuntimeError("stage failed"),
    )

    def save_previous():
        (previous / "metadata.json").write_text(json.dumps(old_metadata), "utf-8")

    state.save_previous = save_previous
    save_previous()

    def observe(stage, output):
        root = Path(output).parent
        plan_bytes = (root / "plan.json").read_bytes()
        if state.plan_bytes is None:
            state.plan_bytes = plan_bytes
        assert plan_bytes == state.plan_bytes, "The study plan changed after execution began"
        state.calls.append(stage)
        if state.failure_stage == stage:
            raise state.failure
        return json.loads(plan_bytes)

    def acquire(output, **options):
        plan = observe("acquisition", output)
        assert options == {key: plan[key] for key in ("samples", "sample_seed", "time_stop")}
        assert plan["source_indices"] == new_indices
        assert plan["previous_cohort"]["source_indices"] == old_indices
        output.mkdir()
        (output / "manifest.json").write_text('{"kind":"acquisition-fixture"}', "utf-8")
        return {
            "selection": {"sample_indices": state.acquired_indices},
            "transfer": {"bytes_received": 12345, "full_remote_checksum_verified": False},
        }

    def import_dataset(acquired, output, *, seed):
        plan = observe("dataset", output)
        assert acquired == output.parent / "acquisition"
        assert seed == plan["split_seed"]
        output.mkdir()
        (output / "manifest.json").write_text('{"kind":"dataset-fixture"}', "utf-8")
        return dataset

    def train(dataset_path, output, *, seed, conserve_mean, **options):
        plan = observe(output.name, output)
        assert dataset_path == output.parent / "dataset"
        assert options == plan["training"]
        assert {"seed": seed, "conserve_mean": conserve_mean} in plan["runs"]
        variant = "mean_preserving" if conserve_mean else "baseline"
        assert output.name == f"{variant}_seed_{seed}"
        output.mkdir()
        report = {
            "best_epoch": seed + 1,
            "training_seconds": 0.25 + seed,
            "checkpoint_sha256": f"{variant}-{seed}-checkpoint",
        }
        (output / "report.json").write_text(json.dumps(report), "utf-8")
        return report

    def evaluate(dataset_path, model_path, *, output):
        observe(f"evaluate_{model_path.name}", output)
        assert dataset_path == output.parent / "dataset"
        assert output.name == f"{model_path.name}_evaluation"
        seed = int(model_path.name[-1])
        projected = model_path.name.startswith("mean_preserving")
        # Projection loses field accuracy in one pair; the study must retain it.
        rmse = [0.3, 0.6, 0.4][seed] if projected else [0.5, 0.2, 0.7][seed]
        report = {
            "split": "test",
            "conserve_mean": projected,
            "rollout": {"rmse": rmse},
            "conservation": {"rollout_mean_drift": {"rms": 1e-6 if projected else 0.1}},
            "persistence_rollout": {"rmse": 0.8},
        }
        output.mkdir()
        (output / "report.json").write_text(json.dumps(report), "utf-8")
        return report

    def verify_dataset(path):
        return state.previous_problems if path == previous else state.dataset_problems

    def verify_model(path):
        state.calls.append(f"verify_{path.name}")
        return ["corrupt model"] if path.name == state.model_problem else []

    ml = ModuleType("flowstate.ml")
    ml.train_fno, ml.evaluate_fno, ml.verify_model = train, evaluate, verify_model
    monkeypatch.setitem(sys.modules, "flowstate.ml", ml)
    monkeypatch.setattr(study, "capture_provenance", lambda: {"git_commit": "fixture"})
    monkeypatch.setattr(study, "acquire_pdebench", acquire)
    monkeypatch.setattr(study, "import_acquired_pdebench", import_dataset)
    monkeypatch.setattr(study, "verify_dataset", verify_dataset)
    return state


def events(output):
    return [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]


def test_success_freezes_plan_and_retains_all_paired_runs(tmp_path, pipeline):
    output = tmp_path / "study"
    report = study.run_conservation_study(pipeline.previous, output, epochs=3)
    plan_bytes = (output / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    expected_pairs = [(0, False), (0, True), (1, True), (1, False), (2, False), (2, True)]
    expected_names = [
        f"{'mean_preserving' if flag else 'baseline'}_seed_{seed}" for seed, flag in expected_pairs
    ]
    assert plan_bytes == pipeline.plan_bytes
    assert plan["samples"] == 24 and plan["time_stop"] == 201
    assert plan["sample_seed"] == 20260929 and plan["split_seed"] == 17
    assert plan["training"]["epochs"] == 3
    assert [(item["seed"], item["conserve_mean"]) for item in plan["runs"]] == expected_pairs
    assert [(item["seed"], item["conserve_mean"]) for item in report["runs"]] == expected_pairs
    assert [item["evaluation"]["rollout"]["rmse"] for item in report["runs"]] == [
        0.5,
        0.3,
        0.6,
        0.2,
        0.7,
        0.4,
    ]
    assert len({item["checkpoint_sha256"] for item in report["runs"]}) == 6
    assert report["cohort_checks"] == {
        "source_rows_disjoint": True,
        "initial_field_hashes_disjoint": True,
    }
    assert report["status"] == "completed" and report["plan"] == plan
    assert report["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert "best_seed" not in report and "winner" not in report
    assert (
        plan["previous_cohort"]["dataset_manifest_sha256"]
        == hashlib.sha256((pipeline.previous / "manifest.json").read_bytes()).hexdigest()
    )
    for name in ("acquisition", "dataset"):
        assert (
            report[f"{name}_manifest_sha256"]
            == hashlib.sha256((output / name / "manifest.json").read_bytes()).hexdigest()
        )
    assert json.loads((output / "report.json").read_text()) == report
    assert not (output / "failure.json").exists()
    expected_calls = ["acquisition", "dataset"]
    for name in expected_names:
        expected_calls.extend([name, f"verify_{name}", f"evaluate_{name}"])
        assert [event["status"] for event in events(output) if event["stage"] == name] == [
            "started",
            "completed",
        ]
    assert pipeline.calls == expected_calls
    assert events(output)[-1]["stage"] == "study"
    assert events(output)[-1]["status"] == "completed"


def test_source_index_overlap_stops_before_publication_or_remote_work(tmp_path, pipeline):
    pipeline.old_metadata["trajectories"][0]["remote_source_index"] = pipeline.new_indices[0]
    pipeline.save_previous()
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="overlaps the previous source rows"):
        study.run_conservation_study(pipeline.previous, output)
    assert pipeline.calls == []
    assert not output.exists()


def test_initial_field_overlap_stops_before_training(tmp_path, pipeline):
    pipeline.dataset["trajectories"][0]["initial_field_sha256"] = pipeline.old_metadata[
        "trajectories"
    ][0]["initial_field_sha256"]
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="repeats an earlier initial field"):
        study.run_conservation_study(pipeline.previous, output)
    assert pipeline.calls == ["acquisition", "dataset"]
    assert json.loads((output / "failure.json").read_text())["stage"] == "dataset"
    assert (output / "dataset" / "manifest.json").is_file()
    assert not (output / "report.json").exists()


def test_acquired_rows_must_match_frozen_selection(tmp_path, pipeline):
    pipeline.acquired_indices = pipeline.new_indices[::-1]
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="differ from the frozen plan"):
        study.run_conservation_study(pipeline.previous, output)
    assert pipeline.calls == ["acquisition"]
    assert json.loads((output / "failure.json").read_text())["stage"] == "acquisition"
    assert not (output / "dataset").exists()


@pytest.mark.parametrize("corruption", ["previous", "dataset", "mean_preserving_seed_0"])
def test_integrity_failure_stops_before_dependent_work(tmp_path, pipeline, corruption):
    if corruption == "previous":
        pipeline.previous_problems = ["corrupt previous dataset"]
    elif corruption == "dataset":
        pipeline.dataset_problems = ["corrupt dataset"]
    else:
        pipeline.model_problem = corruption
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="verification failed"):
        study.run_conservation_study(pipeline.previous, output)
    if corruption == "previous":
        assert not output.exists() and pipeline.calls == []
    else:
        assert json.loads((output / "failure.json").read_text())["stage"] == corruption
        assert not (output / "report.json").exists()
        if corruption == "dataset":
            assert pipeline.calls == ["acquisition", "dataset"]
        else:
            assert f"evaluate_{corruption}" not in pipeline.calls
            assert "mean_preserving_seed_1" not in pipeline.calls


@pytest.mark.parametrize(
    "failed_stage", ["acquisition", "dataset", "mean_preserving_seed_1", "evaluate_baseline_seed_1"]
)
def test_failure_retains_original_plan_completed_stages_and_error_record(
    tmp_path, pipeline, failed_stage
):
    pipeline.failure_stage = failed_stage
    output = tmp_path / "study"
    with pytest.raises(RuntimeError) as caught:
        study.run_conservation_study(pipeline.previous, output)
    assert caught.value is pipeline.failure
    recorded_stage = failed_stage.removeprefix("evaluate_")
    assert (output / "plan.json").read_bytes() == pipeline.plan_bytes
    assert json.loads((output / "failure.json").read_text()) == {
        "stage": recorded_stage,
        "error_type": "RuntimeError",
        "message": "stage failed",
        "plan_sha256": hashlib.sha256(pipeline.plan_bytes).hexdigest(),
    }
    assert not (output / "report.json").exists()
    assert events(output)[-1]["stage"] == recorded_stage
    assert events(output)[-1]["status"] == "failed"
    if "seed_1" in failed_stage:
        assert (output / "baseline_seed_0" / "report.json").is_file()
        assert (output / "mean_preserving_seed_0_evaluation" / "report.json").is_file()
        assert "baseline_seed_2" not in pipeline.calls
    prior_calls = list(pipeline.calls)
    with pytest.raises(FileExistsError):
        study.run_conservation_study(pipeline.previous, output)
    assert pipeline.calls == prior_calls
    assert (output / "plan.json").read_bytes() == pipeline.plan_bytes


def test_keyboard_interrupt_is_recorded_and_propagated(tmp_path, pipeline):
    pipeline.failure_stage = "acquisition"
    pipeline.failure = KeyboardInterrupt("cancelled")
    output = tmp_path / "study"
    with pytest.raises(KeyboardInterrupt) as caught:
        study.run_conservation_study(pipeline.previous, output)
    assert caught.value is pipeline.failure
    assert json.loads((output / "failure.json").read_text())["error_type"] == "KeyboardInterrupt"
    assert events(output)[-1]["status"] == "failed"


@pytest.mark.parametrize("epochs", [0, True, 51, 1.5])
def test_invalid_budget_stops_without_publication(tmp_path, pipeline, epochs):
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="epochs"):
        study.run_conservation_study(pipeline.previous, output, epochs=epochs)
    assert pipeline.calls == [] and not output.exists()


def test_output_cannot_modify_previous_dataset(pipeline):
    output = pipeline.previous / "study"
    with pytest.raises(ValueError, match="outside the previous immutable dataset"):
        study.run_conservation_study(pipeline.previous, output)
    assert pipeline.calls == [] and not output.exists()


def test_previous_dataset_must_come_from_pinned_source(tmp_path, pipeline):
    pipeline.old_metadata["provenance"]["acquisition"]["source"]["source_url"] = (
        "https://example.test/unrelated.hdf5"
    )
    pipeline.save_previous()
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="pinned public source"):
        study.run_conservation_study(pipeline.previous, output)
    assert pipeline.calls == [] and not output.exists()
