"""A small reproducible study exercising the implemented research workflow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from flowstate.datasets import export_dataset
from flowstate.engine import capture_provenance, run_experiment, run_sweep
from flowstate.lake import Lake
from flowstate.research import ResearchGraph, register_model_artifact, research_cycle
from flowstate.streaming import benchmark_storage
from flowstate.validation import run_validation


def run_demo(output: str | Path, *, epochs: int = 20, pinn_epochs: int = 200) -> dict:
    """Create a new study directory; interrupted stages remain available for inspection."""
    from flowstate.ml import evaluate_fno, train_fno, train_pinn

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    lake_root = root / "lake"

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

    stage = "validation"
    try:
        event(stage, "started")
        validation = run_validation(root / "validation.json")
        if not validation["passed"]:
            raise RuntimeError("Numerical validation failed; learning stages stopped")
        event(stage, "completed")
        stage = "storage_benchmark"
        event(stage, "started")
        storage = benchmark_storage(root / "storage_benchmark", grid_size=32, steps=16)
        event(stage, "completed")
        stage = "trajectories"
        event(stage, "started")
        spec = {
            "base": {
                "equation": "burgers1d",
                "grid_size": 32,
                "viscosity": 0.1,
                "amplitude": 0.5,
                "dt": 0.002,
                "steps": 100,
                "save_every": 10,
                "initial_condition": "random",
            },
            "parameters": {"seed": list(range(12))},
        }
        (root / "training_sweep.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        outcomes = run_sweep(spec, lake_root, workers=1, stream=True)
        if any(out.record["status"] != "completed" for out in outcomes):
            raise RuntimeError("A training trajectory failed; dataset creation stopped")
        repeated = run_sweep(spec, lake_root, workers=1, stream=True)
        if not all(out.resumed for out in repeated):
            raise RuntimeError("Trajectory resumption failed")
        event(stage, "completed", runs=len(outcomes), resumed=len(repeated))
        stage = "dataset_etl"
        event(stage, "started")
        dataset = export_dataset(
            lake_root, root / "dataset", ids=[o.record["id"] for o in outcomes]
        )
        event(stage, "completed")
        stage = "fno"
        event(stage, "started")
        fno = train_fno(root / "dataset", root / "fno", epochs=epochs)
        evaluation = evaluate_fno(root / "dataset", root / "fno", output=root / "fno_evaluation")
        fno_entities = register_model_artifact(lake_root, root / "fno")
        event(stage, "completed", best_epoch=fno["best_epoch"])
        stage = "pinn"
        event(stage, "started")
        pinn = train_pinn(root / "dataset", root / "pinn", epochs=pinn_epochs)
        pinn_entities = register_model_artifact(lake_root, root / "pinn")
        event(stage, "completed")
        stage = "research_cycle"
        event(stage, "started")
        darcy = run_experiment({"equation": "darcy2d", "grid_size": 17}, lake_root)
        if darcy.record["status"] != "completed":
            raise RuntimeError("The Darcy demonstration failed")
        # Intentionally violate the timestep screen, then give the policy evidence to refine.
        unstable = run_experiment(
            {
                "grid_size": 32,
                "viscosity": 0.1,
                "amplitude": 0.5,
                "dt": 0.08,
                "steps": 4,
                "save_every": 1,
            },
            lake_root,
        )
        if unstable.record["status"] != "failed":
            raise RuntimeError("Expected diagnostic failure fixture did not fail")
        cycle = research_cycle(lake_root, max_runs=2, max_total_steps=500, execute=True)
        graph = ResearchGraph(lake_root).build()
        (root / "research_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
        (root / "research_cycle.json").write_text(json.dumps(cycle, indent=2), encoding="utf-8")
        event(stage, "completed", executed=len(cycle["executed"]))
        report = {
            "schema_version": 1,
            "status": "completed",
            "provenance": capture_provenance(),
            "validation_report": "validation.json",
            "storage_benchmark": storage,
            "validation_summary": {
                "passed": validation["passed"],
                "studies": len(validation["studies"]),
                "runs": len(validation["runs"]),
                "stored_output_bytes": validation["stored_output_bytes"],
            },
            "training_trajectories": len(outcomes),
            "resumed_trajectories": len(repeated),
            "dataset": dataset,
            "fno_evaluation": evaluation,
            "pinn_report": "pinn/report.json",
            "pinn_evaluation": pinn["trajectory_error"],
            "fno_entities": fno_entities,
            "pinn_entities": pinn_entities,
            "darcy_id": darcy.record["id"],
            "expected_failure_id": unstable.record["id"],
            "research_executed": cycle["executed"],
            "catalog": Lake(lake_root).query(
                "SELECT equation, status, count(*) AS runs FROM experiments "
                "GROUP BY equation, status ORDER BY equation, status"
            ),
            "graph_nodes": len(graph["nodes"]),
            "graph_edges": len(graph["edges"]),
            "limitations": [
                "Small CPU demonstration, not a reproduction of a published benchmark",
                "FNO learns across training trajectories; PINN fits one held-out initial condition",
                "Finite-difference trajectories are numerical references, not exact truth",
                "S3 cloud deployment and public PDEBench downloads are separate operations",
            ],
        }
        (root / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        event("study", "completed")
        return report
    except BaseException as exc:
        event(stage, "failed", error_type=type(exc).__name__, message=str(exc))
        raise
