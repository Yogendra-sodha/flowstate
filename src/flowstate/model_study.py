"""Frozen five-seed Burgers evaluation; test labels never choose study settings."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

from flowstate.datasets import export_dataset
from flowstate.engine import capture_provenance, run_experiment
from flowstate.lake import Lake, _sha256


def study_plan() -> dict:
    return {
        "protocol": "flowstate-trustworthy-model-v1",
        "family_seeds": list(range(20000, 20180)),
        "split_seed": 20261010,
        "minimum_training_families": 100,
        "training_seeds": [0, 1, 2, 3, 4],
        "solver": {
            "equation": "burgers1d",
            "initial_condition": "random",
            "grid_size": 64,
            "viscosity": 0.1,
            "amplitude": 0.5,
            "dt": 0.0025,
            "steps": 800,
            "save_every": 20,
        },
        "training": {
            "epochs": 30,
            "width": 16,
            "modes": 8,
            "depth": 3,
            "batch_size": 64,
            "learning_rate": 0.001,
            "conserve_mean": True,
        },
        "reference_audit_seeds": list(range(20000, 20012)),
        "reference_audit": {"grid_size": 128, "dt": 0.00125, "steps": 1600, "save_every": 40},
        "uncertainty": {
            "nominal_coverage": 0.9,
            "std_floor": 1e-6,
            "score": "trajectory maximum absolute error / floored ensemble std",
            "calibration": "higher empirical validation quantile; no test fitting",
        },
        "bootstrap": {"resamples": 2000, "seed": 20261010},
        "selection": "Each seed uses minimum validation MSE, including epoch zero",
        "test_policy": "All seeds and calibration frozen before test inference; no tuning",
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def check_families(metadata: dict, minimum: int = 100) -> dict:
    families = {
        split: {metadata["trajectories"][i]["family_id"] for i in indices}
        for split, indices in metadata["splits"].items()
    }
    if any(
        families[a] & families[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Trajectory families overlap across splits")
    hashes = [t["initial_field_sha256"] for t in metadata["trajectories"]]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Repeated initial fields are not independent study families")
    if len(families["train"]) < minimum:
        raise ValueError("Study requires at least 100 independent training families")
    return {split: len(values) for split, values in families.items()}


def _ensemble(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 4 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Ensemble requires finite [seed,family,time,x] predictions")
    return values.mean(axis=0), np.maximum(values.std(axis=0, ddof=1), 1e-6)


def calibrate(predictions: np.ndarray, truth: np.ndarray, coverage: float = 0.9) -> dict:
    """Use whole trajectories as calibration units, rather than correlated cells."""
    mean, std = _ensemble(predictions)
    if truth.shape != mean.shape or not np.isfinite(truth).all() or not 0 < coverage < 1:
        raise ValueError("Invalid calibration truth or coverage")
    scores = np.max(np.abs(mean - truth) / std, axis=(1, 2))
    multiplier = float(np.quantile(scores, coverage, method="higher"))
    return {
        "fit_split": "validation",
        "multiplier": multiplier,
        "nominal_coverage": coverage,
        "std_floor": 1e-6,
        "trajectory_scores": scores.tolist(),
        "calibration_families": len(scores),
        "guarantee": "Empirical only: validation also selects checkpoints; no conformal guarantee",
    }


def uncertainty_report(predictions: np.ndarray, truth: np.ndarray, calibration: dict) -> dict:
    mean, std = _ensemble(predictions)
    radius = calibration["multiplier"] * std
    covered = np.abs(mean - truth) <= radius
    error = np.sqrt(np.mean((mean - truth) ** 2, axis=(1, 2)))
    spread = np.sqrt(np.mean(std**2, axis=(1, 2)))
    correlation = None
    if np.std(error) > 0 and np.std(spread) > 0:
        correlation = float(np.corrcoef(error, spread)[0, 1])
    return {
        "calibration": calibration,
        "cell_coverage": float(covered.mean()),
        "simultaneous_trajectory_coverage": float(covered.all(axis=(1, 2)).mean()),
        "mean_interval_width": float((2 * radius).mean()),
        "per_saved_time_cell_coverage": covered.mean(axis=(0, 2)).tolist(),
        "per_family_rmse": error.tolist(),
        "per_family_rms_std": spread.tolist(),
        "spread_error_pearson": correlation,
        "uncovered_family_positions": np.flatnonzero(~covered.all(axis=(1, 2))).tolist(),
    }


def paired_summary(predictions: np.ndarray, truth: np.ndarray, baseline: np.ndarray) -> dict:
    """Retain all family losses and a paired bootstrap over families, not cells."""
    model = np.sqrt(np.mean((predictions.astype(np.float64) - truth) ** 2, axis=(1, 2)))
    persistence = np.sqrt(np.mean((baseline.astype(np.float64) - truth) ** 2, axis=(1, 2)))
    delta = model - persistence
    draws = np.random.default_rng(20261010).integers(0, len(delta), (2000, len(delta)))
    return {
        "per_family_model_rmse": model.tolist(),
        "per_family_persistence_rmse": persistence.tolist(),
        "per_family_paired_rmse_difference": delta.tolist(),
        "mean_paired_rmse_difference": float(delta.mean()),
        "bootstrap_95_percent_mean_difference": np.quantile(
            delta[draws].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "loss_family_positions": np.flatnonzero(delta > 0).tolist(),
        "ties": int(np.count_nonzero(delta == 0)),
        "bootstrap_scope": "Families conditional on this split; not independent seed replicates",
    }


def physical_summary(
    prediction: np.ndarray, truth: np.ndarray, initial: np.ndarray, length: float
) -> dict:
    from flowstate.ml import _drift_metrics, _metrics

    drift = prediction.astype(np.float64).mean(axis=-1) - initial.mean(axis=-1)[:, None]
    energy = 0.5 * np.mean(prediction.astype(np.float64) ** 2, axis=-1)
    initial_energy = 0.5 * np.mean(initial.astype(np.float64) ** 2, axis=-1)
    increases = np.diff(np.column_stack((initial_energy, energy)), axis=1) > 1e-8
    return {
        "field": _metrics(prediction, truth),
        "mean_drift": _drift_metrics(drift),
        "mass_drift": _drift_metrics(length * drift),
        "energy_increase_transitions": int(increases.sum()),
        "energy_increase_family_positions": np.flatnonzero(increases.any(axis=1)).tolist(),
        "energy_increase_tolerance": 1e-8,
    }


def _predictions(root: Path, split: str, kind: str, seeds: list[int]) -> np.ndarray:
    return np.stack(
        [
            np.load(root / f"seed-{seed}-{split}" / f"{kind}.npy", allow_pickle=False)
            for seed in seeds
        ]
    )


def run_model_study(output: str | Path) -> dict:
    from flowstate.ml import _load_data, evaluate_fno, train_fno, verify_model

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("Choose a fresh study output directory")
    provenance = capture_provenance()
    if not provenance["git_commit"] or provenance["git_dirty"] is not False:
        raise ValueError("Commit study code and use a clean checkout before measurement")
    plan = study_plan()
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "plan.json", {**plan, "provenance": provenance})
    started = time.perf_counter()

    def event(stage: str, **details):
        with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"at": datetime.now(UTC).isoformat(), "stage": stage, **details},
                    allow_nan=False,
                )
                + "\n"
            )
        print(stage, flush=True)

    stage = "generation"
    try:
        ids, by_seed = [], {}
        lake = Lake(root / "lake")
        for seed in plan["family_seeds"]:
            result = run_experiment({**plan["solver"], "seed": seed}, root / "lake", stream=True)
            record = result.record
            if record["status"] != "completed" or record["metrics"]["needs_review"]:
                raise ValueError(f"Numerical reference failed quality checks: {record['id']}")
            ids.append(record["id"])
            by_seed[seed] = record["id"]
            if len(ids) % 20 == 0:
                event(stage, completed=len(ids))
        stage = "dataset"
        dataset = export_dataset(root / "lake", root / "dataset", ids=ids, seed=plan["split_seed"])
        counts = check_families(dataset, plan["minimum_training_families"])
        data = _load_data(root / "dataset")
        event(stage, family_counts=counts)
        stage = "reference-audit"
        audits = []
        for seed in plan["reference_audit_seeds"]:
            fine = run_experiment(
                {**plan["solver"], **plan["reference_audit"], "seed": seed},
                root / "reference-lake",
                stream=True,
            )
            if fine.record["status"] != "completed" or fine.record["metrics"]["needs_review"]:
                raise ValueError("Finer reference failed quality checks")
            group = zarr.open_group(
                str(root / "reference-lake" / "experiments" / fine.record["id"] / "fields.zarr"),
                mode="r",
            )
            coarse = zarr.open_group(
                str(lake.experiments / by_seed[seed] / "fields.zarr"), mode="r"
            )
            if not np.allclose(coarse["time"][:], group["time"][:], rtol=0, atol=1e-12):
                raise ValueError("Reference audit saved times differ")
            error = coarse["fields/velocity"][:] - group["fields/velocity"][:, ::2]
            audits.append(
                {
                    "seed": seed,
                    "coarse_id": by_seed[seed],
                    "fine_id": fine.record["id"],
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "max_abs": float(np.abs(error).max()),
                }
            )
        event(stage, completed=len(audits))
        training = []
        for seed in plan["training_seeds"]:
            stage = f"training-seed-{seed}"
            event(stage, status="started")
            trained = train_fno(
                root / "dataset",
                root / f"seed-{seed}",
                seed=seed,
                evaluate_test=False,
                **plan["training"],
            )
            if problems := verify_model(root / f"seed-{seed}"):
                raise ValueError(f"Model verification failed: {problems}")
            training.append(
                {
                    key: trained[key]
                    for key in (
                        "config",
                        "best_epoch",
                        "history",
                        "training_seconds",
                        "checkpoint_sha256",
                    )
                }
            )
            evaluate_fno(
                root / "dataset",
                root / f"seed-{seed}",
                split="validation",
                output=root / f"seed-{seed}-validation",
            )
            event(stage, status="completed", best_epoch=trained["best_epoch"])
        stage = "calibration"
        validation = data["u"][data["splits"]["validation"], 1:].astype(np.float64)
        calibration = {
            kind: calibrate(
                _predictions(root, "validation", kind, plan["training_seeds"]),
                validation,
                plan["uncertainty"]["nominal_coverage"],
            )
            for kind in ("one_step", "rollout")
        }
        _write(root / "calibration.json", calibration)
        calibration_hash = _sha256(root / "calibration.json")
        event(stage, sha256=calibration_hash)
        stage = "test"
        evaluations = []
        for seed in plan["training_seeds"]:
            evaluations.append(
                evaluate_fno(
                    root / "dataset", root / f"seed-{seed}", output=root / f"seed-{seed}-test"
                )
            )
            event(stage, seed=seed)
        truth = data["u"][data["splits"]["test"]].astype(np.float64)
        summary = {}
        for kind in ("one_step", "rollout"):
            predictions = _predictions(root, "test", kind, plan["training_seeds"])
            mean, _ = _ensemble(predictions)
            baseline = (
                truth[:, :-1]
                if kind == "one_step"
                else np.repeat(truth[:, :1], truth.shape[1] - 1, axis=1)
            )
            rmse = [evaluation[kind]["rmse"] for evaluation in evaluations]
            paired = [paired_summary(p, truth[:, 1:], baseline) for p in predictions]
            summary[kind] = {
                "seed_rmse_mean": float(np.mean(rmse)),
                "seed_rmse_sample_std": float(np.std(rmse, ddof=1)),
                "seed_rmse_range": [min(rmse), max(rmse)],
                "per_seed_paired": paired,
                "per_seed_physical": [
                    physical_summary(
                        p, truth[:, 1:], truth[:, 0], data["contract"]["domain_length"]
                    )
                    for p in predictions
                ],
                "ensemble_paired": paired_summary(mean, truth[:, 1:], baseline),
                "uncertainty": uncertainty_report(predictions, truth[:, 1:], calibration[kind]),
                "ensemble_physical": physical_summary(
                    mean, truth[:, 1:], truth[:, 0], data["contract"]["domain_length"]
                ),
                "persistence_physical": physical_summary(
                    baseline, truth[:, 1:], truth[:, 0], data["contract"]["domain_length"]
                ),
                "reference_physical": physical_summary(
                    truth[:, 1:], truth[:, 1:], truth[:, 0], data["contract"]["domain_length"]
                ),
            }
        stage = "verification"
        for source_lake in (lake, Lake(root / "reference-lake")):
            for record in source_lake.records():
                if problems := source_lake.verify(record["id"]):
                    raise ValueError(f"Reference manifest failed: {problems}")
        query = lake.query("SELECT status, count(*) AS runs FROM experiments GROUP BY status")
        if _sha256(root / "calibration.json") != calibration_hash:
            raise ValueError("Calibration changed during test evaluation")
        current = capture_provenance()
        if current != provenance:
            raise ValueError("Source or runtime changed during the study")
        report = {
            "schema_version": 1,
            "status": "completed",
            "plan": plan,
            "provenance": provenance,
            "uv_lock_sha256": _sha256(Path(__file__).parents[2] / "uv.lock"),
            "plan_sha256": _sha256(root / "plan.json"),
            "dataset_manifest_sha256": data["dataset_sha256"],
            "calibration_sha256": calibration_hash,
            "family_counts": counts,
            "dataset": dataset,
            "reference_audit": audits,
            "training": training,
            "evaluations": evaluations,
            "summary": summary,
            "query": query,
            "elapsed_seconds": time.perf_counter() - started,
            "evidence_hashes": {
                p.relative_to(root).as_posix(): _sha256(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
                and (
                    p.name in ("manifest.json", "report.json", "checkpoint.pt")
                    or p.suffix == ".npy"
                )
            },
            "limitations": [
                "One smooth low-amplitude random Fourier generator, viscosity, grid and split",
                "108 training families are independent initial draws, not independent PDE sources",
                "Five seeds measure optimization variation; they are not five independent datasets",
                "Finite-difference references are not exact truth; twelve refinement audits only",
                "Validation selects checkpoints and calibrates intervals: empirical coverage only",
                "Ensemble disagreement can miss shared bias; intervals are not an OOD detector",
                "Mean projection does not ensure energy dissipation or rollout accuracy",
                "No test tuning; no public benchmark, shock, long-time or other-PDE claim",
                "One-step receives preceding truth; rollout feeds its own predictions for 40 steps",
            ],
        }
        _write(root / "report.json", report)
        event("completed")
        return report
    except BaseException as exc:
        _write(
            root / "failure.json",
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "provenance": provenance,
            },
        )
        event(stage, status="failed", message=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_model_study(args.output)


if __name__ == "__main__":
    main()
