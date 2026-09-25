import numpy as np
import pytest
import zarr

from flowstate.engine import run_experiment
from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve
from flowstate.streaming import ZarrFrameSink, benchmark_storage


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
def test_streamed_fields_equal_buffered_fields_without_retention(tmp_path, equation):
    config = normalize_config({"equation": equation, "steps": 7, "save_every": 3})
    expected = solve(config)
    sink = ZarrFrameSink(tmp_path / "stream.zarr", config)
    actual = solve(config, frame_callback=sink, retain_fields=False)
    sink.finish(actual)
    assert actual.fields == {}
    group = zarr.open_group(str(sink.path), mode="r")
    for name, values in expected.fields.items():
        np.testing.assert_array_equal(group[f"fields/{name}"][:], values)
    np.testing.assert_array_equal(group["time"][:], expected.times)


def test_streamed_run_is_published_and_failure_has_no_partial_trajectory(tmp_path):
    config = {"steps": 5, "save_every": 2}
    good = run_experiment(config, tmp_path, stream=True)
    assert good.record["status"] == "completed"
    assert not Lake(tmp_path).verify(good.record["id"])
    bad = run_experiment({**config, "dt": 100}, tmp_path, stream=True)
    assert bad.record["status"] == "failed"
    assert not (tmp_path / "experiments" / bad.record["id"] / "fields.zarr").exists()
    assert all(
        not directory.name.startswith(".") for directory in (tmp_path / "experiments").iterdir()
    )


def test_darcy_runs_through_common_engine_and_catalog(tmp_path):
    result = run_experiment({"equation": "darcy2d", "grid_size": 17}, tmp_path)
    assert result.record["status"] == "completed"
    assert result.record["metrics"]["pressure_l2_error"] < 0.01
    assert result.record["metrics"]["residual_l2"] < 1e-9
    assert Lake(tmp_path).query("SELECT equation FROM experiments") == [{"equation": "darcy2d"}]


def test_storage_benchmark_preserves_fields_and_reports_scoped_resources(tmp_path):
    report = benchmark_storage(tmp_path / "benchmark", grid_size=8, steps=2)
    assert report["field_values_identical"]
    assert {case["mode"] for case in report["cases"]} == {"buffered", "streamed"}
    assert all(case["stored_bytes"] > 0 for case in report["cases"])
    assert "not RSS" in report["measurement_scope"]["memory"]


def test_streaming_storage_error_is_not_recorded_as_a_numerical_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("codec storage failure")

    monkeypatch.setattr(zarr.Group, "create_array", fail)
    with pytest.raises(OSError, match="Saving streamed field"):
        run_experiment({"steps": 2}, tmp_path, stream=True)
    assert Lake(tmp_path).records() == []
