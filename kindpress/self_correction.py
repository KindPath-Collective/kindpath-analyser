"""
KindPress :: self_correction.py

Multiscale self-correction lattice for individual/family/neighbourhood/community.

The mechanism is intentionally uniform across scales:
- Build all non-empty combinations within each level.
- Link nodes via the same correction rule (intra + inter level edges).
- Compute corrected error score by local neighbourhood consensus.

This keeps the correction process structurally identical whether we are
correcting one individual signal or a whole community communication fabric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from statistics import median
from typing import Dict, List, Tuple


LEVEL_ORDER = ["individual", "family", "neighbourhood", "community"]


@dataclass
class CorrectionNode:
    node_id: str
    level: str
    members: Tuple[str, ...]
    observed_error: float = 0.0
    corrected_error: float = 0.0


@dataclass
class CorrectionEdge:
    src: str
    dst: str
    relation: str  # intra | inter


@dataclass
class CorrectionLattice:
    nodes: Dict[str, CorrectionNode] = field(default_factory=dict)
    edges: List[CorrectionEdge] = field(default_factory=list)


def _normalise_level(level: str) -> str:
    level_l = level.strip().lower()
    if level_l == "neighborhood":
        return "neighbourhood"
    return level_l


def generate_level_combinations(level: str, members: List[str]) -> List[Tuple[str, ...]]:
    """Return all non-empty combinations for one level."""
    uniq = sorted({m for m in members if m})
    out: List[Tuple[str, ...]] = []
    for r in range(1, len(uniq) + 1):
        out.extend(combinations(uniq, r))
    return out


def node_id(level: str, members: Tuple[str, ...]) -> str:
    return f"{level}:{'+'.join(members)}"


def build_correction_lattice(level_members: Dict[str, List[str]]) -> CorrectionLattice:
    """
    Build a lattice across all levels using one correction mechanism.

    Intra-level links: nodes that differ by one member (adjacent combinations).
    Inter-level links: nodes sharing at least one member token across adjacent levels.
    """
    lattice = CorrectionLattice()

    normalized = {_normalise_level(k): v for k, v in level_members.items()}
    for level in LEVEL_ORDER:
        members = normalized.get(level, [])
        for combo in generate_level_combinations(level, members):
            nid = node_id(level, combo)
            lattice.nodes[nid] = CorrectionNode(node_id=nid, level=level, members=combo)

    # Intra edges
    by_level: Dict[str, List[CorrectionNode]] = {lvl: [] for lvl in LEVEL_ORDER}
    for n in lattice.nodes.values():
        by_level.setdefault(n.level, []).append(n)

    for level, nodes in by_level.items():
        for i, a in enumerate(nodes):
            set_a = set(a.members)
            for b in nodes[i + 1:]:
                set_b = set(b.members)
                # adjacent combinations at same level
                if abs(len(set_a) - len(set_b)) == 1 and len(set_a.symmetric_difference(set_b)) == 1:
                    lattice.edges.append(CorrectionEdge(src=a.node_id, dst=b.node_id, relation="intra"))
                    lattice.edges.append(CorrectionEdge(src=b.node_id, dst=a.node_id, relation="intra"))

    # Inter edges (adjacent levels only)
    for i, level in enumerate(LEVEL_ORDER[:-1]):
        nxt = LEVEL_ORDER[i + 1]
        for a in by_level.get(level, []):
            set_a = set(a.members)
            for b in by_level.get(nxt, []):
                if set_a.intersection(set(b.members)):
                    lattice.edges.append(CorrectionEdge(src=a.node_id, dst=b.node_id, relation="inter"))
                    lattice.edges.append(CorrectionEdge(src=b.node_id, dst=a.node_id, relation="inter"))

    return lattice


def run_self_correction(
    level_members: Dict[str, List[str]],
    observed_errors: Dict[str, float],
    damping: float = 0.5,
) -> Dict[str, dict]:
    """
    Apply one-step consensus correction across all levels.

    corrected_error = (1-damping)*self + damping*median(neighbourhood)
    where neighbourhood includes self and all adjacent nodes (intra/inter).
    """
    lattice = build_correction_lattice(level_members)

    neighbours: Dict[str, List[str]] = {nid: [] for nid in lattice.nodes}
    for edge in lattice.edges:
        neighbours.setdefault(edge.src, []).append(edge.dst)

    for nid, node in lattice.nodes.items():
        node.observed_error = float(observed_errors.get(nid, 0.0))

    for nid, node in lattice.nodes.items():
        samples = [node.observed_error]
        for other in neighbours.get(nid, []):
            samples.append(lattice.nodes[other].observed_error)
        consensus = median(samples)
        node.corrected_error = (1.0 - damping) * node.observed_error + damping * consensus

    return {
        "nodes": {
            nid: {
                "level": n.level,
                "members": list(n.members),
                "observed_error": n.observed_error,
                "corrected_error": round(n.corrected_error, 6),
            }
            for nid, n in lattice.nodes.items()
        },
        "edges": [
            {"src": e.src, "dst": e.dst, "relation": e.relation}
            for e in lattice.edges
        ],
        "counts": {
            "node_count": len(lattice.nodes),
            "edge_count": len(lattice.edges),
        },
    }
