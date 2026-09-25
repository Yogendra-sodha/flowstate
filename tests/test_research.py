import pytest

from flowstate.engine import run_experiment
from flowstate.research import ResearchGraph, propose, research_cycle


def test_graph_has_equation_solver_fields_and_metrics(tmp_path):
    run = run_experiment({"steps": 4, "save_every": 2}, tmp_path).record
    graph = ResearchGraph(tmp_path).build()
    assert {
        "Experiment",
        "Equation",
        "Solver",
        "Dataset",
        "Metric",
        "InitialCondition",
        "BoundaryCondition",
    } <= {node["type"] for node in graph["nodes"]}
    assert any(
        edge["source"] == run["id"] and edge["type"] == "produces" for edge in graph["edges"]
    )


def test_refinement_preserves_physical_time_and_respects_budget(tmp_path):
    run_experiment({"steps": 4, "save_every": 2, "dt": 0.001}, tmp_path)
    plan = propose(tmp_path, max_runs=1, max_total_steps=8)
    proposal = plan["proposals"][0]
    assert proposal["config"]["dt"] == 0.0005
    assert proposal["config"]["steps"] == 8
    assert proposal["config"]["save_every"] == 4
    assert propose(tmp_path, max_runs=1, max_total_steps=7)["proposals"] == []
    assert propose(tmp_path, max_runs=1, max_total_steps=8)["proposals"][0]["id"] == proposal["id"]


def test_research_cycle_executes_bounded_proposals_and_records_observation(tmp_path):
    run_experiment({"steps": 4}, tmp_path)
    report = research_cycle(tmp_path, max_runs=1, max_total_steps=8, execute=True)
    assert len(report["executed"]) == 1 and report["executed"][0]["status"] == "completed"
    graph = ResearchGraph(tmp_path).build()
    assert {"Hypothesis", "Proposal", "Finding"} <= {node["type"] for node in graph["nodes"]}
    assert not any(edge["type"] in {"supports", "contradicts"} for edge in graph["edges"])


def test_darcy_refinement_and_corrupt_evidence(tmp_path):
    run = run_experiment({"equation": "darcy2d", "grid_size": 9}, tmp_path).record
    assert propose(tmp_path)["proposals"][0]["config"]["grid_size"] == 17
    (tmp_path / "experiments" / run["id"] / "record.json").write_text("{}")
    with pytest.raises(ValueError, match="verification"):
        ResearchGraph(tmp_path).build()
    with pytest.raises(ValueError, match="verification"):
        propose(tmp_path)


def test_research_assertions_need_evidence_and_are_immutable(tmp_path):
    graph = ResearchGraph(tmp_path)
    with pytest.raises(ValueError, match="cite"):
        graph.register("Finding", {"statement": "a claim"})
    first = graph.register("Hypothesis", {"statement": "refinement improves accuracy"})
    assert graph.register("Hypothesis", first["properties"]) == first
    path = graph.entities / first["id"] / "record.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        graph.build()


def test_initial_condition_nodes_distinguish_equations_and_grids(tmp_path):
    base = {"steps": 1, "initial_condition": "random", "seed": 3}
    for equation, grid in (("burgers1d", 16), ("burgers1d", 32), ("navier_stokes2d", 16)):
        run_experiment({**base, "equation": equation, "grid_size": grid}, tmp_path)
    nodes = ResearchGraph(tmp_path).build()["nodes"]
    conditions = [node for node in nodes if node["type"] == "InitialCondition"]
    assert len(conditions) == 3
    assert all(node["properties"]["generator_source_sha256"] for node in conditions)
