import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kindpress.self_correction import build_correction_lattice, run_self_correction


def test_lattice_builds_combinations_across_levels():
    levels = {
        "individual": ["i1", "i2"],
        "family": ["i1", "i2", "i3"],
        "neighborhood": ["i1", "i3"],
        "community": ["i1"],
    }
    lattice = build_correction_lattice(levels)

    # non-empty combinations for 2 + 3 + 2 + 1 members = 3 + 7 + 3 + 1 = 14 nodes
    assert len(lattice.nodes) == 14
    assert len(lattice.edges) > 0


def test_self_correction_applies_consensus_shift():
    levels = {
        "individual": ["a", "b"],
        "family": ["a", "b"],
        "neighbourhood": ["a", "b"],
        "community": ["a", "b"],
    }

    observed = {
        "individual:a": 1.0,
        "individual:b": 0.0,
        "individual:a+b": 0.5,
        "family:a+b": 0.1,
    }

    result = run_self_correction(levels, observed, damping=0.5)

    node_a = result["nodes"]["individual:a"]
    assert node_a["corrected_error"] != node_a["observed_error"]


def test_self_correction_outputs_counts_and_edges():
    levels = {
        "individual": ["x"],
        "family": ["x"],
        "neighbourhood": ["x"],
        "community": ["x"],
    }
    result = run_self_correction(levels, {}, damping=0.3)
    assert result["counts"]["node_count"] == 4
    assert result["counts"]["edge_count"] >= 6
