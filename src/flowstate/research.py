"""Typed scientific evidence and a deterministic, budgeted experiment policy."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from flowstate.engine import run_experiment
from flowstate.lake import Lake
from flowstate.numerics import normalize_config

ENTITY_TYPES = {
    "Equation",
    "Solver",
    "Experiment",
    "InitialCondition",
    "BoundaryCondition",
    "Dataset",
    "Model",
    "Checkpoint",
    "Metric",
    "Anomaly",
    "Hypothesis",
    "Finding",
    "Proposal",
}
LINK_TYPES = {
    "solves",
    "uses",
    "starts_from",
    "has_boundary",
    "produces",
    "trained_on",
    "belongs_to",
    "measures",
    "observed_in",
    "cites",
    "supports",
    "contradicts",
    "parent_of",
    "tests",
}


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(kind: str, properties: dict) -> str:
    return f"{kind.lower()}-{hashlib.sha256(_encoded(properties)).hexdigest()[:24]}"


def _verified_records(lake: Lake) -> list[dict]:
    records = []
    for directory in lake._run_dirs():
        if problems := lake.verify(directory.name):
            raise ValueError(f"Evidence failed verification: {directory.name}: {problems}")
        record = lake.load_record(directory.name)
        if record.get("id") != directory.name:
            raise ValueError("Experiment record identity disagrees with its directory")
        records.append(record)
    return records


class ResearchGraph:
    """Rebuildable ontology view plus immutable, explicit research assertions."""

    def __init__(self, lake_root: str | Path):
        self.lake = Lake(lake_root)
        self.entities = self.lake.root / "research" / "entities"

    def register(self, kind: str, properties: dict, links: list[dict] | None = None) -> dict:
        if kind not in ENTITY_TYPES or not isinstance(properties, dict):
            raise ValueError("Unknown entity type or invalid properties")
        links = links or []
        for link in links:
            if (
                set(link) != {"type", "target"}
                or link["type"] not in LINK_TYPES
                or not isinstance(link["target"], str)
                or not link["target"]
            ):
                raise ValueError("Links require a supported type and nonempty target ID")
        if kind in {"Hypothesis", "Finding"} and not properties.get("statement"):
            raise ValueError("A hypothesis or finding needs a statement")
        if kind == "Finding" and not any(link["type"] == "cites" for link in links):
            raise ValueError("A finding must cite explicit evidence")
        identity = _identity(kind, {"properties": properties, "links": links})
        directory = self.entities / identity
        if directory.exists():
            return self._read(directory)
        document = {
            "id": identity,
            "type": kind,
            "properties": properties,
            "links": links,
            "created_at": datetime.now(UTC).isoformat(),
            "schema_version": 1,
        }
        self.entities.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.entities))
        try:
            encoded = _encoded(document)
            (staging / "record.json").write_bytes(encoded)
            (staging / "sha256").write_text(hashlib.sha256(encoded).hexdigest(), encoding="ascii")
            try:
                os.rename(staging, directory)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                return self._read(directory)
        finally:
            if staging.exists() and staging.resolve().is_relative_to(self.entities.resolve()):
                shutil.rmtree(staging)
        return document

    @staticmethod
    def _read(directory: Path) -> dict:
        encoded = (directory / "record.json").read_bytes()
        expected = (directory / "sha256").read_text(encoding="ascii")
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError(f"Research entity checksum mismatch: {directory.name}")
        return json.loads(encoded)

    def build(self) -> dict:
        nodes, edges = {}, []

        def node(kind, properties, identity=None):
            identity = identity or _identity(kind, properties)
            nodes[identity] = {"id": identity, "type": kind, "properties": properties}
            return identity

        def edge(source, relation, target):
            edges.append({"source": source, "type": relation, "target": target})

        for record in _verified_records(self.lake):
            run_id = record["id"]
            config = record["config"]
            node(
                "Experiment",
                {key: record.get(key) for key in ("status", "config", "provenance", "created_at")},
                run_id,
            )
            edge(run_id, "solves", node("Equation", {"name": record["equation"]}))
            edge(run_id, "uses", node("Solver", {"name": record["solver"]}))
            metadata = record.get("solver_metadata", {})
            edge(
                run_id,
                "has_boundary",
                node("BoundaryCondition", {"name": metadata.get("boundary_conditions", "unknown")}),
            )
            if "initial_condition" in config:
                edge(
                    run_id,
                    "starts_from",
                    node(
                        "InitialCondition",
                        {
                            **{
                                key: config[key]
                                for key in (
                                    "equation",
                                    "grid_size",
                                    "initial_condition",
                                    "seed",
                                    "amplitude",
                                    "domain_length",
                                )
                            },
                            "generator_source_sha256": record["provenance"].get("source_sha256"),
                        },
                    ),
                )
            if record.get("parent_id"):
                edge(record["parent_id"], "parent_of", run_id)
            if record["status"] == "completed":
                edge(
                    run_id,
                    "produces",
                    node(
                        "Dataset",
                        {
                            "experiment_id": run_id,
                            "format": "zarr",
                            "path": f"experiments/{run_id}/fields.zarr",
                        },
                    ),
                )
            for name, value in record.get("metrics", {}).items():
                metric = node("Metric", {"experiment_id": run_id, "name": name, "value": value})
                edge(metric, "measures", run_id)
            if record["status"] == "failed" or record.get("metrics", {}).get("needs_review"):
                anomaly = node(
                    "Anomaly",
                    {
                        "experiment_id": run_id,
                        "kind": "computational_failure"
                        if record["status"] == "failed"
                        else "sampled_diagnostic_flag",
                        "error": record.get("error"),
                        "metrics": record.get("metrics", {}),
                        "interpretation": "Numerical evidence requiring review",
                    },
                )
                edge(anomaly, "observed_in", run_id)
        if self.entities.exists():
            for directory in sorted(self.entities.iterdir()):
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                entity = self._read(directory)
                node(entity["type"], entity["properties"], entity["id"])
                for link in entity["links"]:
                    edge(entity["id"], link["type"], link["target"])
        for item in edges:
            for identity in (item["source"], item["target"]):
                if identity not in nodes:
                    nodes[identity] = {"id": identity, "type": "Unresolved", "properties": {}}
        return {"schema_version": 1, "nodes": list(nodes.values()), "edges": edges}


def propose(lake_root: str | Path, *, max_runs: int = 3, max_total_steps: int = 2000) -> dict:
    """Propose temporal refinement or Darcy spatial refinement using cited records.

    Unsteady proposals preserve physical final time: half dt, twice as many steps.
    This is an auditable rule-based policy, not an LLM or a proof of improvement.
    """
    if (
        isinstance(max_runs, bool)
        or not isinstance(max_runs, int)
        or not 1 <= max_runs <= 32
        or isinstance(max_total_steps, bool)
        or not isinstance(max_total_steps, int)
        or not 1 <= max_total_steps <= 100_000
    ):
        raise ValueError("Budgets require 1..32 runs and 1..100000 total integration steps")
    graph = ResearchGraph(lake_root)
    records = _verified_records(graph.lake)
    # Failure first, then diagnostic flags, then stable candidates; ties use stable IDs.
    records.sort(
        key=lambda r: (
            r["status"] != "failed",
            not r.get("metrics", {}).get("needs_review", False),
            r["id"],
        )
    )
    proposals, skipped = [], []
    steps = 0
    for record in records:
        if len(proposals) == max_runs:
            break
        original = record["config"]
        candidate = dict(original)
        if record["equation"] == "darcy2d":
            candidate["grid_size"] = 2 * (original["grid_size"] - 1) + 1
            cost = 1
            rationale = "Refine Darcy spacing to check manufactured-solution error reduction"
        else:
            candidate["dt"] = original["dt"] / 2
            candidate["steps"] = original["steps"] * 2
            candidate["save_every"] = original["save_every"] * 2
            cost = candidate["steps"]
            rationale = "Halve timestep at the same physical duration and saved times"
        try:
            candidate = normalize_config(candidate)
        except ValueError as exc:
            skipped.append({"parent_id": record["id"], "reason": str(exc)})
            continue
        if any(r.get("parent_id") == record["id"] and r["config"] == candidate for r in records):
            continue
        if steps + cost > max_total_steps:
            skipped.append({"parent_id": record["id"], "reason": "total step budget"})
            continue
        hypothesis = graph.register(
            "Hypothesis",
            {
                "statement": rationale,
                "criterion": "Compare matched diagnostics and reference error",
                "status": "untested",
                "policy": "refinement-v1",
            },
            [{"type": "cites", "target": record["id"]}],
        )
        properties = {
            "parent_id": record["id"],
            "config": candidate,
            "rationale": rationale,
            "changed_parameters": {
                k: {"before": original[k], "after": v}
                for k, v in candidate.items()
                if original[k] != v
            },
            "estimated_steps": cost,
            "hypothesis_id": hypothesis["id"],
            "policy": "refinement-v1",
        }
        entity = graph.register(
            "Proposal",
            properties,
            [
                {"type": "cites", "target": record["id"]},
                {"type": "tests", "target": hypothesis["id"]},
            ],
        )
        proposals.append({"id": entity["id"], **properties})
        steps += cost
    return {
        "policy": "refinement-v1",
        "max_runs": max_runs,
        "max_total_steps": max_total_steps,
        "planned_steps": steps,
        "proposals": proposals,
        "skipped": skipped,
    }


def research_cycle(
    lake_root: str | Path,
    *,
    max_runs: int = 3,
    max_total_steps: int = 2000,
    execute: bool = False,
) -> dict:
    plan = propose(lake_root, max_runs=max_runs, max_total_steps=max_total_steps)
    plan["executed"] = []
    if execute:
        graph = ResearchGraph(lake_root)
        for proposal in plan["proposals"]:
            outcome = run_experiment(proposal["config"], lake_root, parent_id=proposal["parent_id"])
            record = outcome.record
            finding = graph.register(
                "Finding",
                {
                    "statement": f"Refinement {record['id']} ended as {record['status']}",
                    "interpretation": "Observation only; no automatic scientific judgment",
                    "metrics": record["metrics"],
                    "error": record["error"],
                    "proposal_id": proposal["id"],
                },
                [
                    {"type": "cites", "target": record["id"]},
                    {"type": "tests", "target": proposal["hypothesis_id"]},
                ],
            )
            plan["executed"].append(
                {
                    "id": record["id"],
                    "status": record["status"],
                    "resumed": outcome.resumed,
                    "finding_id": finding["id"],
                }
            )
    return plan


def register_model_artifact(lake_root: str | Path, artifact: str | Path) -> dict:
    """Register a locally verified training bundle; preserve model/dataset links."""
    from flowstate.ml import verify_model

    artifact = Path(artifact)
    if problems := verify_model(artifact):
        raise ValueError(f"Model artifact failed verification: {problems}")
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    graph = ResearchGraph(lake_root)
    dataset = graph.register(
        "Dataset",
        {
            "artifact_kind": "training_dataset",
            "identity": report.get("dataset_sha256"),
            "path": report.get("dataset_path"),
        },
    )
    model = graph.register(
        "Model",
        {"artifact_path": str(artifact.resolve()), "report": report},
        [{"type": "trained_on", "target": dataset["id"]}],
    )
    checkpoint_hash = hashlib.sha256((artifact / "checkpoint.pt").read_bytes()).hexdigest()
    checkpoint = graph.register(
        "Checkpoint", {"sha256": checkpoint_hash}, [{"type": "belongs_to", "target": model["id"]}]
    )
    return {"model_id": model["id"], "checkpoint_id": checkpoint["id"], "dataset_id": dataset["id"]}
