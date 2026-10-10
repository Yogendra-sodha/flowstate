"""Family-level comparisons, calibration boundaries, conservation and study guards."""

import numpy as np
import pytest

from flowstate.model_study import (
    calibrate,
    check_families,
    paired_summary,
    physical_summary,
    run_model_study,
    study_plan,
    uncertainty_report,
)


def test_frozen_plan_has_enough_training_families_and_seeds():
    plan = study_plan()
    assert len(set(plan["family_seeds"])) == 180
    assert 180 - 2 * (180 // 5) >= plan["minimum_training_families"] == 100
    assert plan["training_seeds"] == [0, 1, 2, 3, 4]
    assert plan["solver"]["steps"] * plan["solver"]["dt"] == 2
    assert plan["training"]["conserve_mean"]


def metadata():
    return {
        "trajectories": [{"family_id": str(i), "initial_field_sha256": str(i)} for i in range(180)],
        "splits": {
            "train": list(range(108)),
            "validation": list(range(108, 144)),
            "test": list(range(144, 180)),
        },
    }


def test_counts_are_families_not_windows_or_parameter_variants():
    data = metadata()
    assert check_families(data) == {"train": 108, "validation": 36, "test": 36}
    for i in range(108):
        data["trajectories"][i]["family_id"] = str(i // 2)
    with pytest.raises(ValueError, match="100 independent"):
        check_families(data)


def test_overlap_and_exact_duplicate_initial_fields_rejected():
    data = metadata()
    data["trajectories"][108]["family_id"] = "0"
    with pytest.raises(ValueError, match="overlap"):
        check_families(data)
    data = metadata()
    data["trajectories"][108]["initial_field_sha256"] = "0"
    with pytest.raises(ValueError, match="Repeated initial"):
        check_families(data)


def test_uncertainty_uses_family_maxima_and_frozen_validation_calibration():
    # Two ensemble members centered at zero, unit sample std.
    predictions = np.stack(
        [np.full((10, 2, 3), -1 / np.sqrt(2)), np.full((10, 2, 3), 1 / np.sqrt(2))]
    )
    truth = np.zeros((10, 2, 3))
    truth[:, 1, 2] = np.arange(10)
    fitted = calibrate(predictions, truth)
    assert fitted["multiplier"] == pytest.approx(9)
    assert fitted["calibration_families"] == 10
    frozen = dict(fitted)
    tested = uncertainty_report(predictions, np.full_like(truth, 10), fitted)
    assert tested["simultaneous_trajectory_coverage"] == 0
    assert tested["uncovered_family_positions"] == list(range(10))
    assert fitted == frozen and fitted["fit_split"] == "validation"


def test_shared_bias_is_not_hidden_by_zero_ensemble_spread():
    predictions = np.ones((5, 4, 2, 8))
    fit = calibrate(predictions, np.zeros((4, 2, 8)))
    assert fit["multiplier"] == 1e6
    test = uncertainty_report(predictions, np.full((4, 2, 8), 3), fit)
    assert test["cell_coverage"] == 0
    assert test["spread_error_pearson"] is None


def test_paired_bootstrap_retains_losses_and_uses_families():
    truth = np.zeros((3, 2, 8))
    baseline = np.ones_like(truth)
    predictions = np.repeat(np.array([0.5, 2, 1])[:, None, None], 2, axis=1)
    predictions = np.repeat(predictions, 8, axis=2)
    result = paired_summary(predictions, truth, baseline)
    assert result["per_family_paired_rmse_difference"] == [-0.5, 1, 0]
    assert result["loss_family_positions"] == [1] and result["ties"] == 1
    assert result == paired_summary(predictions, truth, baseline)
    assert result["bootstrap_95_percent_mean_difference"][0] <= 1 / 6
    assert result["bootstrap_95_percent_mean_difference"][1] >= 1 / 6


def test_conservation_mass_and_energy_include_initial_transition():
    initial = np.ones((2, 8))
    prediction = np.full((2, 3, 8), 2.0)
    result = physical_summary(prediction, prediction, initial, 4)
    assert result["mean_drift"]["rms"] == 1
    assert result["mass_drift"]["rms"] == 4
    assert result["energy_increase_transitions"] == 2
    assert result["energy_increase_family_positions"] == [0, 1]


@pytest.mark.parametrize("bad", [np.full((5, 4, 2, 8), np.nan), np.ones((1, 4, 2, 8))])
def test_invalid_ensemble_fails_instead_of_claiming_coverage(bad):
    with pytest.raises(ValueError, match="Ensemble"):
        calibrate(bad, np.zeros((4, 2, 8)))


def test_dirty_source_and_existing_output_refused_before_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "flowstate.model_study.capture_provenance",
        lambda: {"git_commit": "fixture", "git_dirty": True},
    )
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="clean checkout"):
        run_model_study(output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        run_model_study(output)


def test_failed_generation_retains_plan_and_receipt(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(
        "flowstate.model_study.capture_provenance",
        lambda: {"git_commit": "fixture", "git_dirty": False},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("storage failed")

    monkeypatch.setattr("flowstate.model_study.run_experiment", fail)
    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="storage failed"):
        run_model_study(output)
    assert json.loads((output / "plan.json").read_text())["training_seeds"] == [0, 1, 2, 3, 4]
    assert json.loads((output / "failure.json").read_text())["stage"] == "generation"
    assert not (output / "report.json").exists()


def test_real_small_pipeline_freezes_calibration_before_any_test_inference(tmp_path, monkeypatch):
    import json

    import flowstate.model_study as study

    plan = study_plan()
    plan["family_seeds"] = list(range(6))
    plan["minimum_training_families"] = 3
    plan["reference_audit_seeds"] = [0]
    plan["solver"].update(grid_size=16, steps=8, save_every=2)
    plan["reference_audit"].update(grid_size=32, steps=16, save_every=4)
    plan["training"].update(epochs=1, width=4, modes=3, depth=1)
    provenance = study.capture_provenance()
    provenance["git_dirty"] = False
    monkeypatch.setattr(study, "capture_provenance", lambda: provenance)
    monkeypatch.setattr(study, "study_plan", lambda: plan)
    output = tmp_path / "study"
    report = run_model_study(output)
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    stages = [event["stage"] for event in events]
    assert stages.index("calibration") < stages.index("test")
    assert report["family_counts"] == {"train": 4, "validation": 1, "test": 1}
    assert len(report["training"]) == len(report["evaluations"]) == 5
    assert report["query"] == [{"status": "completed", "runs": 6}]
    assert all(
        json.loads((output / f"seed-{seed}" / "report.json").read_text())["evaluation"] is None
        for seed in range(5)
    )
    assert report["summary"]["rollout"]["uncertainty"]["calibration"]["fit_split"] == "validation"
