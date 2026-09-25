"""The validation report must contain measured artifacts and honest scope."""

import json
import tracemalloc

import numpy as np
import pytest

from flowstate.validation import run_validation


def test_validation_writes_real_refinement_evidence_and_artifacts(tmp_path):
    output = tmp_path / "validation.json"
    report = run_validation(output)
    assert report["passed"], report["studies"]
    assert len(report["studies"]) == 6
    assert len(report["runs"]) == 19
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["studies"]["taylor_green_spatial"]["observed_orders"] is None
    assert "NOT process RSS" in report["measurement_scope"]["memory"]
    total_bytes = 0
    for run in report["runs"]:
        artifact = tmp_path / run["artifact"]
        assert artifact.is_file()
        assert artifact.stat().st_size == run["stored_output_bytes"]
        assert run["solver_wall_seconds"] > 0
        assert run["peak_traced_allocation_bytes"] > 0
        total_bytes += artifact.stat().st_size
        with np.load(artifact, allow_pickle=False) as arrays:
            assert arrays["times"].ndim == 1
            assert sum(array.nbytes for array in arrays.values()) == run["raw_array_bytes"]
    assert total_bytes == report["stored_output_bytes"]
    assert not tracemalloc.is_tracing()
    with pytest.raises(FileExistsError):
        run_validation(output)


def test_validation_refuses_non_json_output_and_preserves_existing_files(tmp_path):
    with pytest.raises(ValueError, match=".json"):
        run_validation(tmp_path / "report")
    output = tmp_path / "existing.json"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_validation(output)
    assert output.read_text(encoding="utf-8") == "keep me"
