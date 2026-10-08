import json
import subprocess
import sys

import pytest

from flowstate import engine
from flowstate.lake import Lake


@pytest.fixture
def stable_provenance(monkeypatch):
    provenance = engine.capture_provenance()
    monkeypatch.setattr(engine, "capture_provenance", lambda: provenance)
    return provenance


@pytest.fixture
def config():
    return {"grid_size": 16, "steps": 6, "save_every": 2, "dt": 0.001}


def test_run_reuses_verified_artifacts(tmp_path, config, stable_provenance):
    first = engine.run_experiment(config, tmp_path)
    second = engine.run_experiment(config, tmp_path)
    assert first.record["status"] == "completed"
    assert not first.resumed and second.resumed
    assert first.record == second.record
    assert not Lake(tmp_path).verify(first.record["id"])
    assert first.record["metrics"]["final_time"] == pytest.approx(0.006)
    assert first.record["metrics"]["final_energy"] < first.record["metrics"]["initial_energy"]


def test_code_and_parent_and_attempt_are_part_of_identity(config, stable_provenance):
    get_id = engine.experiment_id
    original = get_id(config, stable_provenance, None, 0)
    changed = dict(stable_provenance, source_sha256="different-code")
    assert get_id(config, changed, None, 0) != original
    assert get_id(config, stable_provenance, "parent", 0) != original
    assert get_id(config, stable_provenance, None, 1) != original


def test_child_lineage_and_distinct_attempt(tmp_path, config, stable_provenance):
    parent = engine.run_experiment(config, tmp_path).record["id"]
    child = engine.run_experiment({**config, "viscosity": 0.1}, tmp_path, parent_id=parent)
    assert child.record["parent_id"] == parent
    assert {"source": parent, "target": child.record["id"], "type": "parent_of"} in (
        Lake(tmp_path).graph()["edges"]
    )
    repeat = engine.run_experiment(config, tmp_path, attempt=1)
    assert repeat.record["id"] != parent and not repeat.resumed


def test_failed_solver_is_preserved_and_new_attempt_is_separate(
    tmp_path, config, stable_provenance
):
    unstable = {**config, "dt": 20}
    first = engine.run_experiment(unstable, tmp_path)
    assert first.record["status"] == "failed"
    assert "stability" in first.record["error"]["message"].lower()
    assert not (tmp_path / "experiments" / first.record["id"] / "fields.zarr").exists()
    assert engine.run_experiment(unstable, tmp_path).resumed
    retry = engine.run_experiment(unstable, tmp_path, attempt=1, parent_id=first.record["id"])
    assert retry.record["status"] == "failed" and not retry.resumed
    assert len(Lake(tmp_path).records()) == 2


def test_corrupt_artifacts_are_not_silently_reused(tmp_path, config, stable_provenance):
    outcome = engine.run_experiment(config, tmp_path)
    path = tmp_path / "experiments" / outcome.record["id"] / "record.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="verification"):
        engine.run_experiment(config, tmp_path)


def test_nonexistent_parent_and_invalid_attempt_rejected(tmp_path, config):
    with pytest.raises(ValueError, match="Parent"):
        engine.run_experiment(config, tmp_path, parent_id="missing")
    with pytest.raises(ValueError, match="attempt"):
        engine.run_experiment(config, tmp_path, attempt=True)


def test_zero_viscosity_is_not_a_divide_by_zero_failure(tmp_path, config, stable_provenance):
    result = engine.run_experiment({**config, "viscosity": 0}, tmp_path)
    assert result.record["status"] == "completed"
    assert result.record["metrics"]["reynolds"] is None


def test_sweep_validates_all_configs_and_deduplicates(config):
    spec = {"base": config, "parameters": {"viscosity": [0.05, 0.05, 0.1], "seed": [0, 1]}}
    assert len(engine.expand_sweep(spec)) == 4
    with pytest.raises(ValueError):
        engine.expand_sweep({"base": config, "parameters": {"viscosity": [0.1, -0.1]}})
    with pytest.raises(ValueError, match="nonempty"):
        engine.expand_sweep({"parameters": {"dt": []}})
    with pytest.raises(ValueError, match="exceeds"):
        engine.expand_sweep({"parameters": {"seed": list(range(10_001))}})


def test_sweep_continues_after_numerical_failure(tmp_path, config, stable_provenance):
    outcomes = engine.run_sweep({"base": config, "parameters": {"dt": [0.001, 20]}}, tmp_path)
    assert [out.record["status"] for out in outcomes] == ["completed", "failed"]


def test_cli_parallel_sweep_and_resume(tmp_path, config):
    spec = tmp_path / "sweep.json"
    spec.write_text(json.dumps({"base": config, "parameters": {"viscosity": [0.02, 0.05]}}))
    command = [sys.executable, "-m", "flowstate", "--lake", str(tmp_path / "lake"),
               "sweep", str(spec), "--workers", "2"]
    first = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    first_outcomes = json.loads(first.stdout)
    assert len(first_outcomes) == 2
    assert all(out["status"] == "completed" and not out["resumed"] for out in first_outcomes)
    second = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    assert all(out["resumed"] for out in json.loads(second.stdout))


def test_cli_bad_input_is_actionable(tmp_path):
    from flowstate.cli import main

    assert main(["--lake", str(tmp_path), "run", str(tmp_path / "missing.json")]) == 2
