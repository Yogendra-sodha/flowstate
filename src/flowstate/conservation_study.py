"""Paired mean-conservation study on a fresh cohort, with a frozen comparison plan."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from flowstate.datasets import verify_dataset
from flowstate.engine import capture_provenance
from flowstate.lake import _sha256
from flowstate.public_data import SOURCE, acquire_pdebench, import_acquired_pdebench


def _previous_cohort(path: Path) -> dict:
    if problems := verify_dataset(path):
        raise ValueError(f"Previous dataset verification failed: {problems}")
    metadata = json.loads((path / "metadata.json").read_text("utf-8"))
    acquisition = metadata.get("provenance", {}).get("acquisition", {})
    if acquisition.get("source", {}).get("source_url") != SOURCE["source_url"]:
        raise ValueError("Previous dataset must be an acquired subset of the pinned public source")
    return {
        "dataset_manifest_sha256": _sha256(path / "manifest.json"),
        "source_indices": sorted(item["remote_source_index"] for item in metadata["trajectories"]),
        "initial_hashes": sorted(
            {item["initial_field_sha256"] for item in metadata["trajectories"]}
        ),
    }


def run_conservation_study(
    previous_dataset: str | Path, output: str | Path, *, epochs: int = 10
) -> dict:
    """Compare all paired seeds on a non-overlapping public sample at one viscosity.

    This is a fresh small cohort, not independent-source validation or benchmark
    reproduction. No observed test result selects the projection or a winning seed.
    """
    from flowstate.ml import evaluate_fno, train_fno, verify_model

    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
        raise ValueError("epochs must be an integer in [1, 50]")
    previous_path, root = Path(previous_dataset).resolve(), Path(output).resolve()
    if root.is_relative_to(previous_path):
        raise ValueError("Study output must be outside the previous immutable dataset")
    previous = _previous_cohort(previous_path)
    sample_seed, samples = 20260929, 24
    proposed_indices = sorted(
        np.random.default_rng(sample_seed)
        .choice(SOURCE["tensor_shape"][0], size=samples, replace=False)
        .tolist()
    )
    if set(proposed_indices) & set(previous["source_indices"]):
        raise ValueError("Frozen new cohort overlaps the previous source rows; study refused")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 1,
        "question": (
            "Does a zero-mean learned update improve Burgers mean conservation and rollout error?"
        ),
        "source": SOURCE,
        "previous_cohort": previous,
        "samples": samples,
        "sample_seed": sample_seed,
        "source_indices": proposed_indices,
        "time_stop": 201,
        "split_seed": 17,
        "runs": [
            {"seed": seed, "conserve_mean": flag}
            for seed in (0, 1, 2)
            for flag in ((False, True) if seed % 2 == 0 else (True, False))
        ],
        "training": {
            "epochs": epochs,
            "width": 16,
            "modes": 8,
            "depth": 3,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "selection": "Minimum validation MSE within each run; report all test results",
        "primary_measure": "RMS rollout spatial-mean drift from each trajectory's initial mean",
        "secondary_measures": [
            "Field RMSE",
            "Energy error",
            "Mean discrepancy against numerical reference",
        ],
        "provenance": capture_provenance(),
    }

    def write(name, value):
        (root / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", "utf-8")

    def event(stage, status, **details):
        with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": datetime.now(UTC).isoformat(),
                        "stage": stage,
                        "status": status,
                        **details,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    write("plan.json", plan)
    stage = "acquisition"
    try:
        event(stage, "started")
        acquisition = acquire_pdebench(
            root / "acquisition",
            samples=samples,
            sample_seed=sample_seed,
            time_stop=plan["time_stop"],
        )
        if acquisition["selection"]["sample_indices"] != proposed_indices:
            raise ValueError("Acquired source rows differ from the frozen plan")
        event(stage, "completed", bytes_received=acquisition["transfer"]["bytes_received"])
        stage = "dataset"
        event(stage, "started")
        dataset = import_acquired_pdebench(root / "acquisition", root / "dataset", seed=17)
        if problems := verify_dataset(root / "dataset"):
            raise ValueError(f"New dataset verification failed: {problems}")
        current_hashes = {item["initial_field_sha256"] for item in dataset["trajectories"]}
        if current_hashes & set(previous["initial_hashes"]):
            raise ValueError("New cohort repeats an earlier initial field; learning stopped")
        event(stage, "completed", shape=dataset["shape"], splits=dataset["splits"])
        results = []
        for run in plan["runs"]:
            variant = "mean_preserving" if run["conserve_mean"] else "baseline"
            stage = f"{variant}_seed_{run['seed']}"
            event(stage, "started")
            trained = train_fno(root / "dataset", root / stage, **run, **plan["training"])
            if problems := verify_model(root / stage):
                raise ValueError(f"Model verification failed: {problems}")
            evaluation = evaluate_fno(
                root / "dataset", root / stage, output=root / f"{stage}_evaluation"
            )
            results.append(
                {
                    **run,
                    "variant": variant,
                    "best_epoch": trained["best_epoch"],
                    "checkpoint_sha256": trained["checkpoint_sha256"],
                    "training_seconds": trained["training_seconds"],
                    "evaluation": evaluation,
                }
            )
            event(stage, "completed", best_epoch=trained["best_epoch"])
        report = {
            "schema_version": 1,
            "status": "completed",
            "plan": plan,
            "plan_sha256": _sha256(root / "plan.json"),
            "dataset_manifest_sha256": _sha256(root / "dataset" / "manifest.json"),
            "acquisition_manifest_sha256": _sha256(root / "acquisition" / "manifest.json"),
            "acquisition": acquisition,
            "dataset": {
                key: dataset[key]
                for key in (
                    "shape",
                    "viscosity",
                    "domain_length",
                    "saved_dt",
                    "splits",
                    "split_sha256",
                    "normalization",
                    "trajectories",
                )
            },
            "cohort_checks": {"source_rows_disjoint": True, "initial_field_hashes_disjoint": True},
            "runs": results,
            "limitations": [
                "Fresh cohort from the same public generator, not independent-source validation",
                "24 trajectories, one viscosity, one split; seeds measure training randomness only",
                "Mean projection does not guarantee energy dissipation, stability, or accuracy",
                "Numerical references can drift in mean; reference error is not invariant drift",
                "Only received ranges and the complete local subset were checksum-verified",
            ],
        }
        write("report.json", report)
        event("study", "completed")
        return report
    except BaseException as exc:
        event(stage, "failed", error_type=type(exc).__name__, message=str(exc))
        write(
            "failure.json",
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "plan_sha256": _sha256(root / "plan.json"),
            },
        )
        raise
