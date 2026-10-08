"""Render a retained local scaling report without changing its measurements."""

from __future__ import annotations

import io
import math
from numbers import Real
from pathlib import Path


def _number(row: dict, name: str, *, optional: bool = False) -> float:
    value = row.get(name)
    if optional and value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Chart metric {name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Chart metric {name} must be a finite nonnegative number")
    return value


def _chart_rows(report: dict) -> list[dict]:
    if report.get("status") != "completed":
        raise ValueError("A scaling chart requires a completed report")
    summary = report.get("summary")
    if not isinstance(summary, list) or not summary:
        raise ValueError("A scaling chart requires nonempty summary rows")
    rows = []
    identities = set()
    for original in summary:
        if not isinstance(original, dict):
            raise ValueError("Scaling summary rows must be objects")
        row = dict(original)
        for name in ("grid_size", "workers"):
            if type(row.get(name)) is not int or row[name] < 1:
                raise ValueError(f"Chart {name} must be a positive integer")
        identity = (row["grid_size"], row["workers"])
        if identity in identities:
            raise ValueError("Scaling summary contains duplicate grid/worker rows")
        identities.add(identity)
        for prefix in ("wall_seconds", "verified_reuse_seconds"):
            for suffix in ("median", "min", "max"):
                key = f"{prefix}_{suffix}"
                row[key] = _number(row, key)
            if not row[f"{prefix}_min"] <= row[f"{prefix}_median"] <= row[f"{prefix}_max"]:
                raise ValueError(f"Chart {prefix} range must contain its median")
        for name in ("runs_per_hour_median", "stored_bytes_median", "failure_count"):
            row[name] = _number(row, name)
        row["sampled_peak_aggregate_rss_bytes_max"] = _number(
            row, "sampled_peak_aggregate_rss_bytes_max", optional=True
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (row["grid_size"], row["workers"]))


def render_scaling_chart(report: dict, output: str | Path) -> Path:
    """Write a new PNG with six measurement panels; existing files are refused.

    Memory gaps stay gaps. Timing error bars show observed minimum/maximum across
    trials, not confidence intervals. This function does not mutate the report.
    """
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Scaling chart output must have a .png suffix")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows = _chart_rows(report)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError(
            "Scaling charts require matplotlib; install with "
            "uv sync --extra benchmark --group dev (or uv sync --all-extras --group dev)"
        ) from exc

    figure = Figure(figsize=(13, 10), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.15, hspace=0.56, wspace=0.25)
    panels = [
        ("Fresh sweep wall time", "Seconds / trial", "wall_seconds_median", 1, "wall_seconds"),
        (
            "Successful experiment throughput", "Successful runs / hour",
            "runs_per_hour_median", 1, None,
        ),
        (
            "Sampled peak process-tree RSS",
            "MiB (maximum across trials)",
            "sampled_peak_aggregate_rss_bytes_max",
            1024**2,
            None,
        ),
        (
            "Verified reuse of the same sweep",
            "Seconds / trial (warm cache)",
            "verified_reuse_seconds_median",
            1,
            "verified_reuse_seconds",
        ),
        ("Stored artifacts", "MiB / trial (median)", "stored_bytes_median", 1024**2, None),
        ("Recorded numerical failures", "Failures (sum across trials)", "failure_count", 1, None),
    ]
    colors = ("#2166ac", "#b35806", "#1b7837", "#762a83")
    markers = ("o", "s", "^")
    grids = sorted({row["grid_size"] for row in rows})
    workers = sorted({row["workers"] for row in rows})
    legend_handles = []
    for index, grid in enumerate(grids):
        selected = [row for row in rows if row["grid_size"] == grid]
        x = [row["workers"] for row in selected]
        style = {
            "color": colors[index % len(colors)],
            "marker": markers[index % len(markers)],
            "linewidth": 1.7,
            "markersize": 5,
            "label": f"Grid {grid} × {grid}",
        }
        for axis, (_, _, metric, divisor, interval) in zip(axes.flat, panels, strict=True):
            y = [row[metric] / divisor for row in selected]
            if interval is not None:
                error = [
                    [(row[metric] - row[f"{interval}_min"]) / divisor for row in selected],
                    [(row[f"{interval}_max"] - row[metric]) / divisor for row in selected],
                ]
                plotted = axis.errorbar(x, y, yerr=error, capsize=4, **style)
                if metric == "wall_seconds_median":
                    legend_handles.append(plotted)
            else:
                axis.plot(x, y, **style)

    for axis, (title, ylabel, metric, _, _) in zip(axes.flat, panels, strict=True):
        axis.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_xlabel("Local workers", fontsize=9)
        axis.set_xscale("log", base=2)
        axis.set_xticks(workers, labels=[str(worker) for worker in workers])
        axis.minorticks_off()
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        if metric == "failure_count":
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            if all(row[metric] == 0 for row in rows):
                axis.set_ylim(0, 1)
        if metric == "sampled_peak_aggregate_rss_bytes_max" and any(
            math.isnan(row[metric]) for row in rows
        ):
            axis.text(
                0.03, 0.95, "Missing measurements omitted", transform=axis.transAxes,
                va="top", fontsize=8, color="#555555",
            )

    plan = report.get("plan", {})
    provenance = report.get("provenance", {})
    hardware = provenance.get("hardware", {})
    cpu_count = hardware.get("logical_cpu_count", "unrecorded")
    platform = str(hardware.get("platform", "machine unrecorded")).split("-")[0]
    commit = str(provenance.get("git_commit") or "unrecorded")[:12]
    subtitle = (
        f"{plan.get('distinct_configurations', '?')} distinct configurations · "
        f"{plan.get('runs_per_trial', '?')} runs / trial · "
        f"{plan.get('repeats', '?')} repeats / setting · "
        "timing bars: observed min–max"
    )
    figure.suptitle(
        "Flowstate | Local experiment engine scaling", fontsize=17, x=0.09, ha="left", y=0.965
    )
    figure.text(0.09, 0.925, subtitle, fontsize=9, color="#444444")
    figure.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.083, 0.908),
                  ncol=len(grids), frameon=False, fontsize=10)
    figure.text(
        0.09, 0.065,
        "RSS is sampled, can double-count shared pages, and is not a true peak. "
        "Reuse uses warm filesystem caches.\n"
        "Short trajectories measure this local workload; they do not establish "
        "long-horizon accuracy or distributed scaling.\n"
        "Worker count is not thread count: Zarr may use additional threads "
        "even when BLAS threads are limited.",
        fontsize=8, color="#444444", linespacing=1.6,
    )
    figure.text(0.09, 0.027, f"Source commit {commit} · {platform} · {cpu_count} logical CPUs",
                fontsize=8, color="#555555")
    image = io.BytesIO()
    figure.savefig(image, format="png", metadata={"Title": "Flowstate local scaling measurements"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(image.getvalue())
    return output
