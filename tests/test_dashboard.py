import hashlib
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from flowstate import dashboard
from flowstate.engine import summarize
from flowstate.lake import Lake
from flowstate.numerics import normalize_config, solve


class SnapshotHTML(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.script_count = 0
        self.tags = []
        self.in_snapshot = False
        self.payload = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self.script_count += 1
            self.in_snapshot = dict(attrs).get("id") == "snapshot"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_snapshot = False

    def handle_data(self, value):
        if self.in_snapshot:
            self.payload.append(value)


def read_snapshot(path):
    source = Path(path).read_text(encoding="utf-8")
    parsed = SnapshotHTML(source)
    return json.loads("".join(parsed.payload)), parsed


def saved_record(run_id, equation="burgers1d", **updates):
    record = {
        "id": run_id,
        "equation": equation,
        "status": "completed",
        "solver": "fixture",
        "created_at": "2026-09-29T12:00:00Z",
        "parent_id": None,
        "config": {},
        "metrics": {},
        "error": None,
        "provenance": {"git_commit": "fixture-commit", "precision": "float64"},
    }
    return {**record, **updates}


def save_solver_result(lake, run_id, equation="burgers1d", **parameters):
    config = normalize_config(
        {
            "equation": equation,
            **(
                {"grid_size": 9}
                if equation == "darcy2d"
                else {
                    "grid_size": 16,
                    "steps": 4,
                    "save_every": 2,
                    "dt": 0.001,
                }
            ),
            **parameters,
        }
    )
    result = solve(config)
    record = saved_record(run_id, equation, config=config, metrics=summarize(result, config))
    directory = lake.write(run_id, record, result)
    return directory, record, result


def hashes_under(path):
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def make_symlink(link, destination, *, directory=False):
    try:
        link.symlink_to(destination, target_is_directory=directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows account lacks permission to create symbolic links")
        raise


@pytest.mark.parametrize(
    "equation,field",
    [
        ("burgers1d", "velocity"),
        ("navier_stokes2d", "vorticity"),
        ("darcy2d", "pressure"),
    ],
)
def test_export_preserves_verified_record_and_actual_solver_arrays(tmp_path, equation, field):
    lake = Lake(tmp_path / "lake")
    directory, record, result = save_solver_result(lake, "run-1", equation)
    before = hashes_under(lake.root)
    output = tmp_path / "reports" / "snapshot.html"
    report = dashboard.export_dashboard(lake.root, output)
    payload, parsed = read_snapshot(output)
    assert parsed.script_count == 2
    assert report["displayed"] == report["total"] == 1 and not report["truncated"]
    assert report["html_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert hashes_under(lake.root) == before
    assert lake.verify("run-1") == []
    exported = payload["experiments"][0]
    assert exported["record"] == record
    for artifact in ("manifest", "record"):
        assert (
            exported[f"{artifact}_sha256"]
            == hashlib.sha256((directory / f"{artifact}.json").read_bytes()).hexdigest()
        )
    preview = exported["preview"]
    assert preview["field"] == field
    assert preview["steady"] is (equation == "darcy2d")
    assert preview["time"] == result.times[-1]
    assert preview["source_frame_index"] == len(result.times) - 1
    np.testing.assert_array_equal(preview["values"], result.fields[field][-1])
    for axis, values in result.coordinates.items():
        np.testing.assert_array_equal(preview["coordinates"][axis], values)
    if equation == "darcy2d":
        assert exported["energy"] is None
    else:
        np.testing.assert_array_equal(exported["energy"]["time"], result.times)
        np.testing.assert_array_equal(exported["energy"]["values"], result.diagnostics["energy"])


def test_large_history_decodes_only_bounded_final_frame_and_diagnostic_samples(
    tmp_path,
    monkeypatch,
):
    lake = Lake(tmp_path / "lake")
    # Non-round history length exposes omitted endpoints. These are stored fixture
    # values, not a claimed physical solution. Keep the field small for fast I/O.
    count, width = 205, 16
    result = SimpleNamespace(
        times=np.linspace(0, 1, count),
        fields={"velocity": np.arange(count * width, dtype=float).reshape(count, width)},
        coordinates={"x": np.arange(width, dtype=float) / 4},
        diagnostics={"energy": np.linspace(5, 1, count)},
        metadata={"fixture": "known values for bounded extraction"},
    )
    lake.write("large", saved_record("large"), result)
    original = zarr.Array.__getitem__
    field_selections = []

    def bounded_read(array, selection):
        if array.path.startswith("fields/"):
            field_selections.append(selection)
            assert isinstance(selection, tuple) and selection[0] == count - 1
            assert all(isinstance(s, slice) and s.step >= 1 for s in selection[1:])
        return original(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", bounded_read)
    output = tmp_path / "snapshot.html"
    dashboard.export_dashboard(lake.root, output)
    exported = read_snapshot(output)[0]["experiments"][0]
    assert len(field_selections) == 1
    preview, energy = exported["preview"], exported["energy"]
    assert preview["source_shape"] == [width]
    assert preview["display_shape"][0] <= 256
    assert preview["strides"] == [1]
    np.testing.assert_array_equal(preview["values"], result.fields["velocity"][-1])
    np.testing.assert_array_equal(preview["coordinates"]["x"], result.coordinates["x"])
    assert energy["source_samples"] == count and energy["display_samples"] == 201
    indices = energy["sample_indices"]
    assert len(indices) == len(set(indices)) == 201
    assert indices == sorted(indices) and indices[0] == 0 and indices[-1] == count - 1
    np.testing.assert_array_equal(energy["time"], result.times[indices])
    np.testing.assert_array_equal(energy["values"], result.diagnostics["energy"][indices])


def test_1d_preview_rounds_stride_up_to_respect_spatial_bound(tmp_path):
    lake = Lake(tmp_path / "lake")
    count, width = 3, 513
    result = SimpleNamespace(
        times=np.linspace(0, 1, count),
        fields={"velocity": np.arange(count * width, dtype=float).reshape(count, width)},
        coordinates={"x": np.arange(width, dtype=float) / 4},
        diagnostics={"energy": np.linspace(5, 1, count)},
        metadata={"fixture": "known values for bounded spatial extraction"},
    )
    lake.write("wide", saved_record("wide"), result)
    output = tmp_path / "snapshot.html"
    dashboard.export_dashboard(lake.root, output)
    preview = read_snapshot(output)[0]["experiments"][0]["preview"]
    assert preview["source_shape"] == [width]
    assert preview["display_shape"] == [171] and preview["strides"] == [3]
    np.testing.assert_array_equal(preview["values"], result.fields["velocity"][-1, ::3])
    np.testing.assert_array_equal(preview["coordinates"]["x"], result.coordinates["x"][::3])


def test_2d_preview_uses_64_point_bound_and_correct_spatial_coordinates(tmp_path):
    lake = Lake(tmp_path / "lake")
    _, _, result = save_solver_result(
        lake,
        "vorticity",
        "navier_stokes2d",
        grid_size=128,
        steps=2,
        save_every=1,
    )
    output = tmp_path / "snapshot.html"
    dashboard.export_dashboard(lake.root, output)
    preview = read_snapshot(output)[0]["experiments"][0]["preview"]
    assert preview["source_shape"] == [128, 128]
    assert preview["display_shape"] == [64, 64] and preview["strides"] == [2, 2]
    np.testing.assert_array_equal(preview["values"], result.fields["vorticity"][-1, ::2, ::2])
    for axis in ("x", "y"):
        np.testing.assert_array_equal(preview["coordinates"][axis], result.coordinates[axis][::2])


def test_stored_nonfinite_preview_and_energy_values_become_json_null(tmp_path):
    lake = Lake(tmp_path / "lake")
    special = np.array([1.25, np.nan, np.inf, -np.inf])
    result = SimpleNamespace(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        fields={"velocity": np.tile(special, (4, 1))},
        coordinates={"x": np.arange(4, dtype=float)},
        diagnostics={"energy": special.copy()},
        metadata={"fixture": "nonfinite stored samples are missing in previews"},
    )
    lake.write("nonfinite", saved_record("nonfinite"), result)
    assert lake.verify("nonfinite") == []
    before = hashes_under(lake.root)
    output = tmp_path / "snapshot.html"
    dashboard.export_dashboard(lake.root, output)
    _, parsed = read_snapshot(output)

    def reject_nonstandard_number(value):
        pytest.fail(f"Embedded snapshot contains a nonstandard JSON number: {value}")

    payload = json.loads("".join(parsed.payload), parse_constant=reject_nonstandard_number)
    exported = payload["experiments"][0]
    assert exported["preview"]["values"] == [1.25, None, None, None]
    assert exported["energy"]["values"] == [1.25, None, None, None]
    assert exported["energy"]["time"] == [0.0, 0.1, 0.2, 0.3]
    assert exported["preview"]["coordinates"]["x"] == [0.0, 1.0, 2.0, 3.0]
    assert hashes_under(lake.root) == before


@pytest.mark.parametrize(
    "malformed,reason",
    [
        ("energy", "Energy diagnostic must align"),
        ("coordinates", "coordinates do not match"),
        ("field", "real scalar field aligned"),
    ],
)
def test_verified_but_incompatible_array_dimensions_refuse_export(tmp_path, malformed, reason):
    lake = Lake(tmp_path / "lake")
    result = SimpleNamespace(
        times=np.array([0.0, 0.1]),
        fields={"velocity": np.zeros((2, 4))},
        coordinates={"x": np.arange(4, dtype=float)},
        diagnostics={"energy": np.array([1.0, 0.5])},
        metadata={"fixture": "array semantics must be checked after checksum verification"},
    )
    if malformed == "energy":
        # Lake permits multidimensional diagnostics, but the energy chart requires
        # one scalar per saved time; its first dimension alone is insufficient.
        result.diagnostics["energy"] = np.zeros((2, 2))
    elif malformed == "coordinates":
        result.coordinates["x"] = np.arange(3, dtype=float)
    else:
        result.fields["velocity"] = np.zeros((2, 2, 2, 2))
    lake.write("incompatible", saved_record("incompatible"), result)
    assert lake.verify("incompatible") == []
    before = hashes_under(lake.root)
    output = tmp_path / "reports" / "snapshot.html"
    with pytest.raises(ValueError, match=reason):
        dashboard.export_dashboard(lake.root, output)
    assert not output.parent.exists()
    assert hashes_under(lake.root) == before


@pytest.mark.parametrize("artifact", ["record.json", "field_chunk"])
def test_corrupted_selected_artifact_refuses_export_without_creating_output(tmp_path, artifact):
    lake = Lake(tmp_path / "lake")
    directory, _, _ = save_solver_result(lake, "corrupt")
    if artifact == "field_chunk":
        target = next(
            p
            for p in (directory / "fields.zarr" / "fields" / "velocity").rglob("*")
            if p.is_file() and p.name != "zarr.json"
        )
    else:
        target = directory / artifact
    target.write_bytes(target.read_bytes() + b"damage")
    before = hashes_under(lake.root)
    output = tmp_path / "new-directory" / "snapshot.html"
    with pytest.raises(ValueError, match="verification failed"):
        dashboard.export_dashboard(lake.root, output)
    assert hashes_under(lake.root) == before
    assert not output.parent.exists()


def test_record_text_cannot_escape_embedded_json_into_executable_html(tmp_path):
    attack = '</script><script>alert("injected")</script><img src=x onerror=alert(1)>&\u2028\u2029'
    lake = Lake(tmp_path / "lake")
    record = saved_record("hostile-text", status="failed", error={"message": attack})
    lake.write("hostile-text", record, None)
    output = tmp_path / "snapshot.html"
    dashboard.export_dashboard(lake.root, output)
    payload, parsed = read_snapshot(output)
    assert payload["experiments"][0]["record"]["error"]["message"] == attack
    assert parsed.script_count == 2
    assert not any(tag == "img" for tag, _ in parsed.tags)
    assert not any(any(name.startswith("on") for name in attrs) for _, attrs in parsed.tags)
    assert "\\u003c/script\\u003e" in "".join(parsed.payload)
    assert "\\u2028" in "".join(parsed.payload) and "\\u2029" in "".join(parsed.payload)
    source = output.read_text(encoding="utf-8")
    assert attack not in source
    assert "connect-src 'none'" in source
    assert payload["experiments"][0]["preview"] is None
    assert payload["experiments"][0]["energy"] is None


def test_existing_output_is_never_overwritten(tmp_path):
    lake = Lake(tmp_path / "lake")
    output = tmp_path / "snapshot.html"
    output.write_text("keep original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        dashboard.export_dashboard(lake.root, output)
    assert output.read_text(encoding="utf-8") == "keep original"


def test_racing_output_creation_preserves_winner_and_cleans_staging_file(tmp_path, monkeypatch):
    lake = Lake(tmp_path / "lake")
    output = tmp_path / "snapshot.html"
    original = dashboard.os.link

    def competing_link(source, destination):
        Path(destination).write_text("other exporter won", encoding="utf-8")
        original(source, destination)

    monkeypatch.setattr(dashboard.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        dashboard.export_dashboard(lake.root, output)
    assert output.read_text(encoding="utf-8") == "other exporter won"
    assert not list(tmp_path.glob(".snapshot.html-*.tmp"))


@pytest.mark.parametrize("dangling", [False, True])
def test_output_leaf_symlink_is_not_followed(tmp_path, dangling):
    lake = Lake(tmp_path / "lake")
    destination = tmp_path / "original.html"
    if not dangling:
        destination.write_text("keep original", encoding="utf-8")
    output = tmp_path / "snapshot.html"
    make_symlink(output, destination)
    with pytest.raises(FileExistsError):
        dashboard.export_dashboard(lake.root, output)
    assert output.is_symlink()
    assert not destination.exists() if dangling else destination.read_text() == "keep original"


@pytest.mark.parametrize("inside", ["snapshot.html", "experiments/new/snapshot.html"])
def test_output_anywhere_inside_lake_is_rejected_without_changes(tmp_path, inside):
    lake = Lake(tmp_path / "lake")
    before = sorted(str(p) for p in lake.root.rglob("*"))
    with pytest.raises(ValueError, match="outside the entire immutable lake"):
        dashboard.export_dashboard(lake.root, lake.root / inside)
    assert sorted(str(p) for p in lake.root.rglob("*")) == before


def test_symlinked_output_parent_cannot_bypass_lake_boundary(tmp_path):
    lake = Lake(tmp_path / "lake")
    alias = tmp_path / "lake-alias"
    make_symlink(alias, lake.root, directory=True)
    with pytest.raises(ValueError, match="outside the entire immutable lake"):
        dashboard.export_dashboard(lake.root, alias / "snapshot.html")
    assert not (lake.root / "snapshot.html").exists()


def test_symlinked_experiments_directory_is_rejected(tmp_path):
    source = tmp_path / "lake"
    source.mkdir()
    actual = tmp_path / "actual-experiments"
    actual.mkdir()
    make_symlink(source / "experiments", actual, directory=True)
    with pytest.raises(ValueError, match="real experiments directory"):
        dashboard.export_dashboard(source, tmp_path / "snapshot.html")


@pytest.mark.parametrize("limit", [0, -1, 201, True, 1.5, "2", None])
def test_invalid_export_limit_is_rejected_before_any_files_are_created(tmp_path, limit):
    with pytest.raises(ValueError, match="max_experiments"):
        dashboard.export_dashboard(
            tmp_path / "missing-lake",
            tmp_path / "report" / "snapshot.html",
            max_experiments=limit,
        )
    assert list(tmp_path.iterdir()) == []


def test_limit_selects_sorted_ids_and_reports_unverified_truncation(tmp_path):
    lake = Lake(tmp_path / "lake")
    for run_id in ("c-run", "a-run", "b-run"):
        lake.write(
            run_id,
            saved_record(run_id, status="failed", error={"message": "fixture"}),
            None,
        )
    (lake.experiments / ".staging-not-committed").mkdir()
    # The third run is intentionally damaged. A cap does not promise to verify
    # records it did not export; this limitation must remain visible to the reader.
    (lake.experiments / "c-run" / "record.json").write_text("damaged", encoding="utf-8")
    output = tmp_path / "snapshot.html"
    report = dashboard.export_dashboard(lake.root, output, max_experiments=2)
    payload, _ = read_snapshot(output)
    assert report["displayed"] == 2 and report["total"] == 3 and report["truncated"]
    assert [item["record"]["id"] for item in payload["experiments"]] == ["a-run", "b-run"]
    assert "undisplayed records are not verified" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        dashboard.export_dashboard(lake.root, tmp_path / "all.html", max_experiments=3)


def test_empty_lake_exports_usable_zero_record_snapshot(tmp_path):
    lake = Lake(tmp_path / "lake")
    output = tmp_path / "empty.html"
    report = dashboard.export_dashboard(lake.root, output)
    payload, parsed = read_snapshot(output)
    assert report["displayed"] == report["total"] == 0 and report["truncated"] is False
    assert payload["experiments"] == [] and parsed.script_count == 2
    assert "No experiments match these filters" in output.read_text(encoding="utf-8")


def test_missing_lake_does_not_create_an_empty_source(tmp_path):
    source = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        dashboard.export_dashboard(source, tmp_path / "snapshot.html")
    assert not source.exists() and not (tmp_path / "snapshot.html").exists()


def test_nonfinite_plot_values_become_json_null(tmp_path):
    lake = Lake(tmp_path / "lake")
    result = SimpleNamespace(
        times=np.array([0.0, 1.0]),
        fields={"velocity": np.array([[0.0, 1.0, 2.0, 3.0], [np.nan, np.inf, -np.inf, 4.0]])},
        coordinates={"x": np.arange(4, dtype=float)},
        diagnostics={"energy": np.array([1.0, np.nan])},
        metadata={"fixture": "nonfinite values are gaps, not zeros"},
    )
    lake.write("nonfinite", saved_record("nonfinite"), result)
    output = tmp_path / "snapshot.html"
    dashboard.export_dashboard(lake.root, output)
    exported = read_snapshot(output)[0]["experiments"][0]
    assert exported["preview"]["values"] == [None, None, None, 4.0]
    assert exported["energy"]["values"] == [1.0, None]


@pytest.mark.parametrize("invalid", ["coordinates", "diagnostics"])
def test_inconsistent_plot_dimensions_refuse_export(tmp_path, invalid):
    lake = Lake(tmp_path / "lake")
    result = SimpleNamespace(
        times=np.array([0.0, 1.0]),
        fields={"velocity": np.zeros((2, 4))},
        coordinates={"x": np.arange(3 if invalid == "coordinates" else 4, dtype=float)},
        diagnostics={"energy": np.zeros((2, 1)) if invalid == "diagnostics" else np.zeros(2)},
        metadata={"fixture": "checksum-valid arrays with incompatible dimensions"},
    )
    lake.write("mismatched", saved_record("mismatched"), result)
    output = tmp_path / "snapshot.html"
    with pytest.raises(ValueError, match="coordinates|diagnostic"):
        dashboard.export_dashboard(lake.root, output)
    assert not output.exists()
