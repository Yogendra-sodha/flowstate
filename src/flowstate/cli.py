"""Small, scriptable CLI: JSON in; structured results out."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import duckdb

from flowstate.engine import run_experiment, run_sweep
from flowstate.lake import Lake


def _read_json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flowstate", description="Reproducible numerical PDE experiments"
    )
    parser.add_argument("--lake", default="data/lake", help="Local experiment lake directory")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "Run one experiment or resume an identical finalized result"),
        ("sweep", "Run a parameter grid, preserving every successful or failed result"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", help="JSON configuration file")
        command.add_argument("--parent", help="Existing parent experiment ID in the same lake")
        command.add_argument(
            "--stream", action="store_true", help="Write saved fields incrementally"
        )
        command.add_argument(
            "--attempt", type=int, default=0, help="New attempt identity (default 0)"
        )
        if name == "sweep":
            command.add_argument("--workers", type=int, default=1)
    query = commands.add_parser("query", help="Read-only SQL over the experiments table")
    query.add_argument("sql")
    commands.add_parser("list", help="List experiment records")
    dashboard = commands.add_parser("dashboard", help="Export a verified offline research snapshot")
    dashboard.add_argument("output", help="New HTML file outside the lake")
    dashboard.add_argument("--max-experiments", type=int, default=50)
    graph = commands.add_parser("graph", help="Export lineage or the typed research ontology")
    graph.add_argument("--ontology", action="store_true")
    show = commands.add_parser("show", help="Show full configuration, provenance, and metrics")
    show.add_argument("id")
    verify = commands.add_parser("verify", help="Verify artifact checksums")
    verify.add_argument("id")
    validation = commands.add_parser("validate", help="Run numerical/resource validation studies")
    validation.add_argument("output", help="New JSON report path")
    benchmark = commands.add_parser(
        "storage-benchmark", help="Measure local buffered/streamed writes"
    )
    benchmark.add_argument("output", help="New output directory")
    benchmark.add_argument("--grid-size", type=int, default=64)
    benchmark.add_argument("--steps", type=int, default=32)
    scaling = commands.add_parser(
        "sweep-benchmark", help="Compare isolated local worker/storage configurations"
    )
    scaling.add_argument("config", help="JSON sweep specification")
    scaling.add_argument("output", help="New study directory")
    scaling.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    scaling.add_argument(
        "--modes", nargs="+", choices=["buffered", "streamed"], default=["buffered", "streamed"]
    )
    scaling.add_argument("--repeats", type=int, default=3)
    scaling.add_argument("--seed", type=int, default=0)
    large_scaling = commands.add_parser(
        "scaling-study", help="Run the fixed larger local scaling protocol and generate a chart"
    )
    large_scaling.add_argument("output", help="New study directory; requires clean committed code")
    queue = commands.add_parser("queue", help="Durable local jobs with lease recovery")
    queue_commands = queue.add_subparsers(dest="queue_command", required=True)
    submit = queue_commands.add_parser("submit")
    submit.add_argument("config", help="JSON sweep specification")
    submit.add_argument("path", help="Local SQLite queue file; do not share over network storage")
    submit.add_argument("--parent")
    submit.add_argument("--attempt", type=int, default=0)
    submit.add_argument("--buffered", action="store_true", help="Keep fields in memory")
    work = queue_commands.add_parser("work")
    work.add_argument("path")
    work.add_argument("--max-jobs", type=int, default=1)
    work.add_argument("--lease-seconds", type=float, default=60)
    status = queue_commands.add_parser("status")
    status.add_argument("path")
    dataset = commands.add_parser("dataset", help="Curate or verify trajectory datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    export = dataset_commands.add_parser("export")
    export.add_argument("output")
    export.add_argument("--ids", nargs="+")
    export.add_argument("--seed", type=int, default=0)
    ingest = dataset_commands.add_parser("import-pdebench")
    ingest.add_argument("source")
    ingest.add_argument("output")
    ingest.add_argument("--source-url", required=True)
    ingest.add_argument("--source-version", required=True)
    ingest.add_argument("--license", dest="license_name", required=True)
    ingest.add_argument(
        "--viscosity",
        type=float,
        required=True,
        help="Effective PDE coefficient, not an inferred filename label",
    )
    ingest.add_argument("--seed", type=int, default=0)
    ingest.add_argument("--sample-indices", nargs="+", type=int)
    ingest.add_argument("--time-stop", type=int, help="Exclusive time prefix; preserve full x grid")
    ingest.add_argument("--max-values", type=int, default=16_000_000)
    acquire = dataset_commands.add_parser("acquire-pdebench", help="Fetch a bounded public subset")
    acquire.add_argument("output")
    acquire.add_argument("--samples", type=int, default=24)
    acquire.add_argument("--sample-seed", type=int, default=20260926)
    acquire.add_argument("--time-stop", type=int, default=201)
    acquire.add_argument("--max-bytes", type=int, default=64 * 1024**2)
    acquired = dataset_commands.add_parser("import-acquired")
    acquired.add_argument("source")
    acquired.add_argument("output")
    acquired.add_argument("--seed", type=int, default=17)
    acquisition_verify = dataset_commands.add_parser("verify-acquisition")
    acquisition_verify.add_argument("path")
    dataset_verify = dataset_commands.add_parser("verify")
    dataset_verify.add_argument("path")
    for name, default_epochs in (("train-fno", 20), ("train-pinn", 200)):
        train = commands.add_parser(name, help="Train a bounded CPU baseline (requires ml extra)")
        train.add_argument("dataset")
        train.add_argument("output")
        train.add_argument("--epochs", type=int, default=default_epochs)
        train.add_argument("--seed", type=int, default=0)
        train.add_argument("--width", type=int, default=16 if name == "train-fno" else 32)
        train.add_argument("--depth", type=int, default=3)
        train.add_argument("--learning-rate", type=float, default=0.001)
        train.add_argument("--resume", help="Previous training directory or checkpoint")
        if name == "train-fno":
            train.add_argument("--modes", type=int, default=8)
            train.add_argument("--batch-size", type=int, default=16)
            train.add_argument("--conserve-mean", action="store_true")
        else:
            train.add_argument("--trajectory-index", type=int)
            train.add_argument("--collocation-points", type=int, default=128)
    evaluate = commands.add_parser("evaluate-fno", help="Evaluate one-step and rollout predictions")
    evaluate.add_argument("dataset")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    evaluate.add_argument("--output", help="New immutable evaluation artifact directory")
    model = commands.add_parser("model", help="Verify or register a trained model bundle")
    model.add_argument("model_command", choices=["verify", "register"])
    model.add_argument("path")
    research = commands.add_parser(
        "research", help="Propose or execute a budgeted refinement cycle"
    )
    research.add_argument("--execute", action="store_true")
    research.add_argument("--max-runs", type=int, default=3)
    research.add_argument("--max-total-steps", type=int, default=2000)
    hypothesis = commands.add_parser("hypothesis", help="Record an explicit scientific hypothesis")
    hypothesis.add_argument("statement")
    hypothesis.add_argument("--criterion", required=True)
    storage = commands.add_parser(
        "s3", help="Mirror verified experiment artifacts (requires s3 extra)"
    )
    storage.add_argument("storage_command", choices=["upload", "download"])
    storage.add_argument("id")
    storage.add_argument("bucket")
    storage.add_argument("--prefix", default="")
    storage.add_argument("--endpoint-url")
    gcs = commands.add_parser("gcs", help="Mirror verified artifacts to Google Cloud Storage")
    gcs.add_argument("storage_command", choices=["upload", "download"])
    gcs.add_argument("id")
    gcs.add_argument("bucket", help="Bucket name without gs://")
    gcs.add_argument("--prefix", default="")
    gcs.add_argument("--project", help="Google Cloud project ID (or use the ADC default)")
    demo = commands.add_parser("demo", help="Run a small reproducible end-to-end research study")
    demo.add_argument("output")
    demo.add_argument("--epochs", type=int, default=20)
    demo.add_argument("--pinn-epochs", type=int, default=200)
    public = commands.add_parser("public-study", help="Run the fixed small public Burgers study")
    public.add_argument("output")
    public.add_argument("--epochs", type=int, default=10)
    public.add_argument("--pinn-epochs", type=int, default=200)
    conservation = commands.add_parser(
        "conservation-study", help="Paired FNO study on a fresh public-data cohort"
    )
    conservation.add_argument("previous_dataset", help="Previous acquired dataset to exclude")
    conservation.add_argument("output")
    conservation.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "sweep"}:
            common = {"parent_id": args.parent, "attempt": args.attempt, "stream": args.stream}
            if args.command == "run":
                outcomes = [run_experiment(_read_json(args.config), args.lake, **common)]
            else:
                outcomes = run_sweep(
                    _read_json(args.config), args.lake, workers=args.workers, **common
                )
            _print(
                [
                    {
                        "id": out.record["id"],
                        "status": out.record["status"],
                        "resumed": out.resumed,
                        "metrics": out.record["metrics"],
                        "error": out.record["error"],
                    }
                    for out in outcomes
                ]
            )
            return int(any(out.record["status"] == "failed" for out in outcomes))
        if args.command in {"dashboard", "scaling-study"}:
            return _extended_command(args)
        lake = Lake(args.lake)
        if args.command == "query":
            _print(lake.query(args.sql))
        elif args.command == "list":
            _print(
                [
                    {key: record.get(key) for key in ("id", "equation", "status", "parent_id")}
                    for record in lake.records()
                ]
            )
        elif args.command == "show":
            _print(lake.load_record(args.id))
        elif args.command == "graph":
            if args.ontology:
                from flowstate.research import ResearchGraph

                _print(ResearchGraph(args.lake).build())
            else:
                _print(lake.graph())
        elif args.command == "verify":
            problems = lake.verify(args.id)
            _print({"id": args.id, "valid": not problems, "problems": problems})
            return int(bool(problems))
        else:
            return _extended_command(args)
    except (ValueError, OSError, RuntimeError, ImportError, duckdb.Error, sqlite3.Error) as exc:
        print(f"flowstate: {exc}", file=sys.stderr)
        return 2
    return 0


def _extended_command(args) -> int:
    if args.command == "dashboard":
        from flowstate.dashboard import export_dashboard

        _print(export_dashboard(args.lake, args.output, max_experiments=args.max_experiments))
    elif args.command == "scaling-study":
        from flowstate.scaling_study import run_scaling_study

        report = run_scaling_study(args.output)
        _print({key: report[key] for key in ("status", "output", "totals", "summary")})
    elif args.command == "queue":
        from flowstate.queue import queue_status, submit_sweep, work_queue

        if args.queue_command == "submit":
            _print(
                submit_sweep(
                    _read_json(args.config),
                    args.path,
                    args.lake,
                    parent_id=args.parent,
                    attempt=args.attempt,
                    stream=not args.buffered,
                )
            )
        elif args.queue_command == "work":
            result = work_queue(args.path, max_jobs=args.max_jobs, lease_seconds=args.lease_seconds)
            _print(result)
            return int(bool(result["failed"] or result["lost_claim"]))
        else:
            _print(queue_status(args.path))
    elif args.command == "sweep-benchmark":
        from flowstate.scaling import benchmark_sweep

        _print(
            benchmark_sweep(
                _read_json(args.config),
                args.output,
                workers=args.workers,
                modes=args.modes,
                repeats=args.repeats,
                seed=args.seed,
            )
        )
    elif args.command == "storage-benchmark":
        from flowstate.streaming import benchmark_storage

        _print(benchmark_storage(args.output, grid_size=args.grid_size, steps=args.steps))
    elif args.command == "validate":
        from flowstate.validation import run_validation

        _print(run_validation(args.output))
    elif args.command == "dataset":
        from flowstate.datasets import export_dataset, import_pdebench, verify_dataset

        if args.dataset_command == "export":
            _print(export_dataset(args.lake, args.output, ids=args.ids, seed=args.seed))
        elif args.dataset_command == "import-pdebench":
            _print(
                import_pdebench(
                    args.source,
                    args.output,
                    source_url=args.source_url,
                    source_version=args.source_version,
                    license_name=args.license_name,
                    viscosity=args.viscosity,
                    seed=args.seed,
                    sample_indices=args.sample_indices,
                    time_stop=args.time_stop,
                    max_values=args.max_values,
                )
            )
        elif args.dataset_command in {"acquire-pdebench", "import-acquired", "verify-acquisition"}:
            from flowstate.public_data import (
                acquire_pdebench,
                import_acquired_pdebench,
                verify_acquisition,
            )

            if args.dataset_command == "acquire-pdebench":
                _print(
                    acquire_pdebench(
                        args.output,
                        samples=args.samples,
                        sample_seed=args.sample_seed,
                        time_stop=args.time_stop,
                        max_bytes=args.max_bytes,
                    )
                )
            elif args.dataset_command == "import-acquired":
                _print(import_acquired_pdebench(args.source, args.output, seed=args.seed))
            else:
                problems = verify_acquisition(args.path)
                _print({"valid": not problems, "problems": problems})
                return int(bool(problems))
        else:
            problems = verify_dataset(args.path)
            _print({"path": args.path, "valid": not problems, "problems": problems})
            return int(bool(problems))
    elif args.command in {"train-fno", "train-pinn", "evaluate-fno", "model"}:
        from flowstate.ml import evaluate_fno, train_fno, train_pinn, verify_model

        if args.command in {"train-fno", "train-pinn"}:
            common = {
                "epochs": args.epochs,
                "seed": args.seed,
                "width": args.width,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "resume": args.resume,
            }
            if args.command == "train-fno":
                result = train_fno(
                    args.dataset,
                    args.output,
                    modes=args.modes,
                    batch_size=args.batch_size,
                    conserve_mean=args.conserve_mean,
                    **common,
                )
            else:
                result = train_pinn(
                    args.dataset,
                    args.output,
                    trajectory_index=args.trajectory_index,
                    collocation_points=args.collocation_points,
                    **common,
                )
            _print(result)
        elif args.command == "evaluate-fno":
            _print(
                evaluate_fno(args.dataset, args.checkpoint, split=args.split, output=args.output)
            )
        elif args.model_command == "verify":
            problems = verify_model(args.path)
            _print({"valid": not problems, "problems": problems})
            return int(bool(problems))
        else:
            from flowstate.research import register_model_artifact

            _print(register_model_artifact(args.lake, args.path))
    elif args.command in {"research", "hypothesis"}:
        from flowstate.research import ResearchGraph, research_cycle

        if args.command == "hypothesis":
            _print(
                ResearchGraph(args.lake).register(
                    "Hypothesis",
                    {
                        "statement": args.statement,
                        "criterion": args.criterion,
                        "status": "untested",
                    },
                )
            )
        else:
            _print(
                research_cycle(
                    args.lake,
                    max_runs=args.max_runs,
                    max_total_steps=args.max_total_steps,
                    execute=args.execute,
                )
            )
    elif args.command == "s3":
        from botocore.exceptions import BotoCoreError, ClientError

        from flowstate.object_store import download_experiment, upload_experiment

        operation = upload_experiment if args.storage_command == "upload" else download_experiment
        try:
            _print(
                operation(
                    args.lake,
                    args.id,
                    args.bucket,
                    prefix=args.prefix,
                    endpoint_url=args.endpoint_url,
                )
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"S3 request failed: {exc}") from exc
    elif args.command == "gcs":
        try:
            from google.api_core.exceptions import GoogleAPICallError
            from google.auth.exceptions import GoogleAuthError
            from google.cloud.storage.exceptions import DataCorruption
        except ImportError as exc:
            raise RuntimeError("Install GCS support with: uv sync --extra gcs") from exc

        from flowstate.gcs_store import download_experiment, upload_experiment

        operation = upload_experiment if args.storage_command == "upload" else download_experiment
        try:
            _print(
                operation(
                    args.lake, args.id, args.bucket, prefix=args.prefix, project=args.project
                )
            )
        except (GoogleAPICallError, GoogleAuthError, DataCorruption) as exc:
            raise RuntimeError(f"GCS request failed: {exc}") from exc
    elif args.command == "demo":
        from flowstate.demo import run_demo

        _print(run_demo(args.output, epochs=args.epochs, pinn_epochs=args.pinn_epochs))
    elif args.command == "public-study":
        from flowstate.public_study import run_public_study

        _print(run_public_study(args.output, epochs=args.epochs, pinn_epochs=args.pinn_epochs))
    elif args.command == "conservation-study":
        from flowstate.conservation_study import run_conservation_study

        _print(run_conservation_study(args.previous_dataset, args.output, epochs=args.epochs))
    return 0
