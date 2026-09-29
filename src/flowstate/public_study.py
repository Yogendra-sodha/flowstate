"""Predeclared small-sample comparison using actual public PDEBench trajectories."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from flowstate.datasets import verify_dataset
from flowstate.engine import capture_provenance
from flowstate.lake import _sha256
from flowstate.public_data import SOURCE, acquire_pdebench, import_acquired_pdebench


def run_public_study(output: str | Path, *, epochs: int = 10, pinn_epochs: int = 200) -> dict:
    """Keep all three fixed FNO seeds; never choose a winner using test results."""
    from flowstate.ml import evaluate_fno, train_fno, train_pinn, verify_model

    for name, value, maximum in (("epochs", epochs, 50), ("pinn_epochs", pinn_epochs, 2000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 1,
        "question": (
            "How do fixed-budget FNO seeds and one PINN compare with persistence "
            "on a small public subset?"
        ),
        "source": SOURCE,
        "samples": 24,
        "sample_seed": 20260926,
        "time_stop": 201,
        "split_seed": 17,
        "fno_seeds": [0, 1, 2],
        "fno": {
            "epochs": epochs,
            "width": 16,
            "modes": 8,
            "depth": 3,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "pinn": {
            "epochs": pinn_epochs,
            "seed": 0,
            "width": 32,
            "depth": 3,
            "collocation_points": 128,
            "learning_rate": 0.001,
        },
        "pinn_selection": "First trajectory in the frozen test split, chosen before fitting",
        "checkpoint_selection": "FNO minimum validation MSE; PINN last epoch",
        "reporting": "Report all seeds and failures; no test-based tuning or seed selection",
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
            samples=plan["samples"],
            sample_seed=plan["sample_seed"],
            time_stop=plan["time_stop"],
        )
        event(stage, "completed", bytes_received=acquisition["transfer"]["bytes_received"])
        stage = "dataset"
        event(stage, "started")
        dataset = import_acquired_pdebench(
            root / "acquisition", root / "dataset", seed=plan["split_seed"]
        )
        if problems := verify_dataset(root / "dataset"):
            raise ValueError(f"Dataset verification failed: {problems}")
        event(stage, "completed", shape=dataset["shape"], splits=dataset["splits"])
        results = []
        for seed in plan["fno_seeds"]:
            stage = f"fno_seed_{seed}"
            event(stage, "started")
            trained = train_fno(root / "dataset", root / stage, seed=seed, **plan["fno"])
            if problems := verify_model(root / stage):
                raise ValueError(f"Model verification failed: {problems}")
            evaluation = evaluate_fno(
                root / "dataset", root / stage, output=root / f"{stage}_evaluation"
            )
            results.append(
                {
                    "seed": seed,
                    "best_epoch": trained["best_epoch"],
                    "training_seconds": trained["training_seconds"],
                    "checkpoint_sha256": trained["checkpoint_sha256"],
                    "evaluation": evaluation,
                }
            )
            event(stage, "completed", best_epoch=trained["best_epoch"])
        stage = "pinn"
        event(stage, "started")
        pinn = train_pinn(root / "dataset", root / "pinn", **plan["pinn"])
        if problems := verify_model(root / "pinn"):
            raise ValueError(f"Model verification failed: {problems}")
        event(stage, "completed")
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
            "fno_runs": results,
            "pinn": {
                key: pinn[key]
                for key in (
                    "trajectory_id",
                    "training_seconds",
                    "trajectory_error",
                    "persistence_trajectory_error",
                    "physical_residual_rms",
                    "checkpoint_sha256",
                    "supervision",
                    "comparison_scope",
                )
            },
            "limitations": [
                "24 trajectories at one viscosity; not a full PDEBench benchmark reproduction",
                "Three FNO seeds measure training randomness on one fixed split",
                "PINN fits one initial-value problem; FNO is trained across other trajectories",
                "Errors compare numerical references, not exact PDE solutions",
                "Full remote file checksum was not verified; see range and local subset hashes",
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
