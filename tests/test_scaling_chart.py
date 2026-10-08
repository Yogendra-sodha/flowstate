import copy
import math

import pytest

from flowstate.scaling_chart import _chart_rows, render_scaling_chart


@pytest.fixture
def report():
    return {
        "status": "completed",
        "provenance": {
            "git_commit": "a" * 40,
            "hardware": {"platform": "Windows-test", "logical_cpu_count": 8},
        },
        "plan": {"distinct_configurations": 216, "repeats": 3, "runs_per_trial": 72},
        "summary": [
            {
                "grid_size": grid,
                "workers": workers,
                "wall_seconds_median": 12 / workers,
                "wall_seconds_min": 10 / workers,
                "wall_seconds_max": 15 / workers,
                "runs_per_hour_median": 72 * workers * 300,
                "sampled_peak_aggregate_rss_bytes_max": workers * 1024**2,
                "verified_reuse_seconds_median": 2,
                "verified_reuse_seconds_min": 1,
                "verified_reuse_seconds_max": 3,
                "stored_bytes_median": grid * 1024**2,
                "failure_count": 0,
                "trials": 3,
            }
            for grid in (64, 128, 256)
            for workers in (1, 2, 4, 8)
        ],
    }


def test_chart_writes_png_without_mutating_report(tmp_path, report):
    pytest.importorskip("matplotlib")
    original = copy.deepcopy(report)
    output = tmp_path / "chart.png"
    assert render_scaling_chart(report, output) == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report == original


def test_chart_refuses_to_overwrite_existing_file(tmp_path, report):
    output = tmp_path / "chart.png"
    output.write_bytes(b"previous chart")
    with pytest.raises(FileExistsError):
        render_scaling_chart(report, output)
    assert output.read_bytes() == b"previous chart"


def test_missing_memory_is_a_gap_not_zero(tmp_path, report):
    pytest.importorskip("matplotlib")
    for row in report["summary"]:
        row["sampled_peak_aggregate_rss_bytes_max"] = None
    output = render_scaling_chart(report, tmp_path / "missing-memory.png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all(
        math.isnan(row["sampled_peak_aggregate_rss_bytes_max"]) for row in _chart_rows(report)
    )
    assert all(row["sampled_peak_aggregate_rss_bytes_max"] is None for row in report["summary"])


@pytest.mark.parametrize("status", ["running", "failed"])
def test_chart_rejects_incomplete_reports(tmp_path, report, status):
    report["status"] = status
    with pytest.raises(ValueError, match="completed"):
        render_scaling_chart(report, tmp_path / "chart.png")
    assert not (tmp_path / "chart.png").exists()


def test_chart_rejects_invalid_metric_before_writing(tmp_path, report):
    report["summary"][0]["wall_seconds_max"] = 1
    with pytest.raises(ValueError, match="range"):
        render_scaling_chart(report, tmp_path / "chart.png")
    assert not (tmp_path / "chart.png").exists()
