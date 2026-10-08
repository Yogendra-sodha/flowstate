"""Checkpoint integrity, publication races, and integration with the immutable lake."""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from flowstate import checkpoints, engine
from flowstate.checkpoints import CheckpointError, CheckpointStore, validate_checkpoint_options
from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve
from flowstate.scaling import _fingerprint


@pytest.fixture
def stable_code(monkeypatch):
    provenance = engine.capture_provenance()
    monkeypatch.setattr(engine, "capture_provenance", lambda: copy.deepcopy(provenance))
    return provenance


def config_for(equation="burgers1d"):
    return normalize_config({
        "equation": equation, "grid_size": 8, "steps": 11,
        "save_every": 4, "initial_condition": "random", "seed": 123,
    })


def interrupt_at(monkeypatch, step):
    save = CheckpointStore.save

    def interrupt(store, checkpoint):
        save(store, checkpoint)
        if checkpoint.step == step:
            raise KeyboardInterrupt("test interruption after committed state")

    monkeypatch.setattr(CheckpointStore, "save", interrupt)
    return save


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
@pytest.mark.parametrize("step", [3, 11])
def test_engine_resume_preserves_every_scientific_array(
    tmp_path, monkeypatch, stable_code, equation, step,
):
    config = config_for(equation)
    baseline = engine.run_experiment(config, tmp_path / "baseline")
    lake_root = tmp_path / "recovery"
    save = interrupt_at(monkeypatch, step)
    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(config, lake_root, checkpoint_every=step)
    assert Lake(lake_root).records() == []
    monkeypatch.setattr(CheckpointStore, "save", save)
    actual = engine.run_experiment(config, lake_root, checkpoint_every=step)
    assert actual.checkpoint_step == step and not actual.resumed
    assert actual.record["recovery"]["steps_executed"] == config["steps"] - step
    assert actual.record["status"] == "completed"
    assert actual.record["storage_mode"] == "streamed"
    assert actual.record["metrics"] == baseline.record["metrics"]
    run_id = actual.record["id"]
    assert run_id == baseline.record["id"]
    assert not Lake(lake_root).verify(run_id)
    assert _fingerprint(lake_root / "experiments" / run_id / "fields.zarr") == _fingerprint(
        tmp_path / "baseline" / "experiments" / run_id / "fields.zarr"
    )
    reused = engine.run_experiment(config, lake_root, checkpoint_every=2)
    assert reused.resumed and reused.checkpoint_step is None


@pytest.mark.parametrize("kind", ["state", "frame", "marker", "missing"])
def test_corrupt_latest_checkpoint_fails_closed_without_failure_record(
    tmp_path, monkeypatch, stable_code, kind,
):
    save = interrupt_at(monkeypatch, 6)
    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    monkeypatch.setattr(CheckpointStore, "save", save)
    root = next((tmp_path / "checkpoints").iterdir())
    marker = root / "step-6.json"
    document = json.loads(marker.read_bytes())
    if kind == "marker":
        marker.write_text("incomplete", encoding="utf-8")
    elif kind == "missing":
        document["payload"]["state"] = "f" * 64
        document["sha256"] = hashlib.sha256(
            checkpoints._encoded(document["payload"])
        ).hexdigest()
        marker.write_bytes(checkpoints._encoded(document))
    else:
        payload = document["payload"]
        digest = payload["state"] if kind == "state" else payload["frames"][-1]
        (root / "blobs" / f"{digest}.npz").write_bytes(b"corrupt")
    with pytest.raises(CheckpointError):
        engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    assert Lake(tmp_path).records() == []
    assert (root / "step-3.json").exists()  # No silent fallback to this older state.


def test_pending_artifacts_are_ignored_and_preserved(tmp_path, monkeypatch, stable_code):
    save = interrupt_at(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    monkeypatch.setattr(CheckpointStore, "save", save)
    root = next((tmp_path / "checkpoints").iterdir())
    pending = root / ".pending-test"
    pending.write_bytes(b"half written marker")
    orphan = root / "blobs" / ("e" * 64 + ".npz")
    orphan.write_bytes(b"unreferenced interrupted blob")
    outcome = engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    assert outcome.checkpoint_step == 3
    assert pending.read_bytes() == b"half written marker"
    assert orphan.read_bytes() == b"unreferenced interrupted blob"


def test_concurrent_checkpoint_writers_publish_identical_artifacts(tmp_path, stable_code):
    config = config_for()
    run_id = engine.experiment_id(config, stable_code, None, 0)

    def write():
        store = CheckpointStore(tmp_path, run_id, config, stable_code)
        solve(config, frame_callback=store.append_frame, retain_fields=False,
              checkpoint_every=3, checkpoint_callback=store.save)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: write(), range(2)))
    store = CheckpointStore(tmp_path, run_id, config, stable_code)
    state = store.load_latest()
    assert state.step == config["steps"]
    assert len(list(store.root.glob("step-*.json"))) == 4
    assert not list(store.root.rglob(".pending-*"))
    state.state[0] += 0.5
    with pytest.raises(CheckpointError, match="Conflicting immutable"):
        store.save(state)


def test_manifest_identity_and_frame_schedule_are_verified(tmp_path, stable_code):
    config = config_for()
    run_id = engine.experiment_id(config, stable_code, None, 0)
    store = CheckpointStore(tmp_path, run_id, config, stable_code)
    solve(config, frame_callback=store.append_frame, retain_fields=False,
          checkpoint_every=3, checkpoint_callback=store.save)
    incompatible = CheckpointStore(tmp_path, run_id, config, {**stable_code, "python": "different"})
    with pytest.raises(CheckpointError, match="runtime mismatch"):
        incompatible.load_latest()
    marker = store.root / "step-11.json"
    document = json.loads(marker.read_bytes())
    document["payload"]["frames"].reverse()
    document["sha256"] = hashlib.sha256(checkpoints._encoded(document["payload"])).hexdigest()
    marker.write_bytes(checkpoints._encoded(document))
    with pytest.raises(CheckpointError, match="saved schedule"):
        store.load_latest()


def test_storage_encoding_error_is_not_a_numerical_failure(tmp_path, monkeypatch, stable_code):
    def fail(*args, **kwargs):
        raise ValueError("encoder failed")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(CheckpointError, match="encode checkpoint"):
        engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    assert Lake(tmp_path).records() == []


def test_numerical_failure_after_recovery_is_an_immutable_failure(
    tmp_path, monkeypatch, stable_code,
):
    save = interrupt_at(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    monkeypatch.setattr(CheckpointStore, "save", save)

    def unstable(*args, **kwargs):
        assert kwargs["checkpoint"].step == 3
        raise ValueError("unstable later step")

    monkeypatch.setattr(engine, "solve", unstable)
    failed = engine.run_experiment(config_for(), tmp_path, checkpoint_every=3)
    assert failed.record["status"] == "failed"
    assert failed.checkpoint_step == 3
    assert failed.record["recovery"]["steps_executed"] is None
    assert not (tmp_path / "experiments" / failed.record["id"] / "fields.zarr").exists()
    assert not Lake(tmp_path).verify(failed.record["id"])
    assert engine.run_experiment(config_for(), tmp_path, checkpoint_every=3).resumed


@pytest.mark.parametrize("config,interval", [
    ({"equation": "darcy2d"}, 1), ({}, True), ({}, 0),
    ({"steps": 257}, 1), ({"steps": 10000, "save_every": 1}, 100),
    ({"equation": "navier_stokes2d", "grid_size": 512,
      "steps": 10000, "save_every": 50}, 100),
])
def test_invalid_checkpoint_options_do_not_create_a_lake(tmp_path, config, interval):
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        engine.run_experiment(config, root, checkpoint_every=interval)
    assert not root.exists()


def test_sweep_validates_all_checkpoint_options_before_execution(tmp_path, monkeypatch):
    checked = []

    def reject_second(config, interval):
        checked.append(config["seed"])
        if config["seed"] == 2:
            raise ValueError("checkpoint budget")

    monkeypatch.setattr(engine, "validate_checkpoint_options", reject_second)
    with pytest.raises(ValueError, match="checkpoint budget"):
        engine.run_sweep({"parameters": {"seed": [1, 2]}}, tmp_path / "absent",
                         checkpoint_every=1)
    assert checked == [1, 2]
    assert not (tmp_path / "absent").exists()


def test_default_execution_remains_checkpoint_free(tmp_path, stable_code):
    validate_checkpoint_options(config_for(), None)
    engine.run_experiment(config_for(), tmp_path)
    assert not (tmp_path / "checkpoints").exists()
