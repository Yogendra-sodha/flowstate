"""Small, scriptable CLI: JSON in; structured results out."""

from __future__ import annotations

import argparse
import json
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
            "--attempt", type=int, default=0, help="New attempt identity (default 0)"
        )
        if name == "sweep":
            command.add_argument("--workers", type=int, default=1)
    query = commands.add_parser("query", help="Read-only SQL over the experiments table")
    query.add_argument("sql")
    commands.add_parser("list", help="List experiment records")
    commands.add_parser("graph", help="Export experiment lineage as JSON nodes and edges")
    show = commands.add_parser("show", help="Show full configuration, provenance, and metrics")
    show.add_argument("id")
    verify = commands.add_parser("verify", help="Verify artifact checksums")
    verify.add_argument("id")
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "sweep"}:
            common = {"parent_id": args.parent, "attempt": args.attempt}
            if args.command == "run":
                outcomes = [run_experiment(_read_json(args.config), args.lake, **common)]
            else:
                outcomes = run_sweep(
                    _read_json(args.config), args.lake, workers=args.workers, **common
                )
            _print([
                {
                    "id": out.record["id"],
                    "status": out.record["status"],
                    "resumed": out.resumed,
                    "metrics": out.record["metrics"],
                    "error": out.record["error"],
                }
                for out in outcomes
            ])
            return int(any(out.record["status"] == "failed" for out in outcomes))
        lake = Lake(args.lake)
        if args.command == "query":
            _print(lake.query(args.sql))
        elif args.command == "list":
            _print([
                {key: record.get(key) for key in ("id", "equation", "status", "parent_id")}
                for record in lake.records()
            ])
        elif args.command == "show":
            _print(lake.load_record(args.id))
        elif args.command == "graph":
            _print(lake.graph())
        elif args.command == "verify":
            problems = lake.verify(args.id)
            _print({"id": args.id, "valid": not problems, "problems": problems})
            return int(bool(problems))
    except (ValueError, OSError, duckdb.Error) as exc:
        print(f"flowstate: {exc}", file=sys.stderr)
        return 2
    return 0
