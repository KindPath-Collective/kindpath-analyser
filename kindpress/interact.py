"""
interact.py — KindPress Agent Interaction Signal

KindPress for agent interactions uses the same k/Δ model as audio analysis,
applied to a completely different signal domain: the reasoning, decisions,
and uncertainties produced by AI agents as they work.

The audio k/Δ model compresses a signal record (waveform features + history)
into a portable, storable profile. The interaction k/Δ model does the
same for an agent session: it compresses reasoning traces, uncertainty deposits,
and cross-agent insights into a signal that can be stored in KCE, diffed
over time, and eventually used as training data.

Why this belongs in KindPress:
  The KindPress principle is that knowledge has a constant layer (k) and
  a delta layer (Δ). The k layer describes the stable context: which agent,
  which doctrine version, which domain, what the standard was. The Δ layer
  describes what actually happened that session: the decisions taken, the
  uncertainties surfaced, the insights offered to other domains.

  k can be updated (doctrine evolves, agent role broadens) without losing the Δ.
  The Δ is permanent. Add, never edit.

  For agent interactions:
    k = (agent_domain, doctrine_version, primary_repo, session_date_utc)
    Δ = compressed vector of (thinking_traces, uncertainty_deposits, cross_agent_insights)
        from a given session window

The output InteractionPacket is designed to be deposited to the KCE seedbank
or stored as a JSON record for corpus-level training analysis.

This module does NOT touch press.py, reason.py, or validate.py — those are
built for audio fingerprint data and must stay pure. interact.py is the
sibling module, applying the same conceptual architecture to a different domain.

See kindpath-analyser/docs/KINDPRESS_SPEC.md for the k/Δ model reference.
See kindpath-canon/AGENT_COMMUNITY_CHARTER.md §6+§7 for the doctrine.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional
import statistics


# ── k layer: the stable context that frames the session ──────────────────────


@dataclass
class InteractionContext:
    """
    The k layer for an agent interaction session.

    This is the stable frame: who, in what role, under what doctrine,
    working on what. Two sessions by the same agent on the same repo
    share k if nothing structural has changed.

    k can be updated when doctrine evolves or agent roles change.
    The Δ records deposited under k are permanent.
    """

    agent_domain: str
    # e.g. 'devops-lead', 'analyser', 'kindai-dev', 'oversight', 'research'

    doctrine_version: str
    # ISO date of the AGENT_COMMUNITY_CHARTER.md version in effect
    # e.g. '2026-03-13'. Update this when Charter is revised.

    primary_repo: str
    # The repo this agent was primarily working in this session

    session_date_utc: str
    # ISO8601 UTC date (date only, not time — sessions are dated, not timestamped)

    role_tags: list[str] = field(default_factory=list)
    # Stable tags describing the agent's function during this context period
    # e.g. ['infrastructure', 'cross-repo', 'architecture']

    def fingerprint(self) -> str:
        """
        A stable identifier for this k layer.
        Same agent + doctrine + repo + date = same fingerprint.
        Used to group Δ records from the same session window.
        """
        raw = f"{self.agent_domain}:{self.doctrine_version}:{self.primary_repo}:{self.session_date_utc}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Δ layer: what actually happened in the session ────────────────────────────


@dataclass
class ThinkingSignal:
    """
    A single thinking trace compressed for the Δ layer.
    Fields that carry training signal (not raw verbatim content).
    """

    task_domain: str
    # The type of task: 'schema', 'api', 'doc', 'test', 'config', 'debug', etc.

    alternatives_considered: int
    # How many alternatives were genuinely considered (0 = no visible alternatives)

    had_cross_domain_signal: bool
    # Whether this trace produced a signal for another agent domain

    confidence: float
    # Agent's stated confidence in the decision taken (0-1)

    decision_category: str
    # Broad category: 'build', 'defer', 'route', 'escalate', 'research', 'fix'

    @classmethod
    def from_kce_record(cls, record: dict) -> "ThinkingSignal":
        """Build from a KCE /agent/thinking record."""
        decision = record.get("decision_taken", "")
        # Infer decision category from the first verb of the decision
        category = "build"
        for keyword, cat in [
            ("defer", "defer"), ("route", "route"), ("escalate", "escalate"),
            ("research", "research"), ("fix", "fix"), ("cannot", "defer"),
            ("skip", "defer"), ("document", "build"),
        ]:
            if keyword in decision.lower():
                category = cat
                break

        alts = record.get("alternatives_considered") or []
        return cls(
            task_domain=_infer_task_domain(record.get("task_context", "")),
            alternatives_considered=len(alts),
            had_cross_domain_signal=bool(record.get("cross_domain_signal")),
            confidence=float(record.get("confidence", 0.7)),
            decision_category=category,
        )


@dataclass
class UncertaintySignal:
    """
    A single uncertainty deposit compressed for the Δ layer.
    """

    uncertainty_type: str
    # One of: technical, doctrine, cross_domain, data, architectural, precedent

    confidence: float
    # Agent's confidence in the action taken despite the uncertainty

    was_routed: bool
    # Whether this uncertainty was auto-routed to another agent

    resolution_status: str
    # 'open' | 'resolved' | 'escalated' | 'doctrine_gap' | 'wont_fix'

    @classmethod
    def from_kce_record(cls, record: dict) -> "UncertaintySignal":
        """Build from a KCE /agent/uncertainties record."""
        return cls(
            uncertainty_type=record.get("type", "technical"),
            confidence=float(record.get("confidence", 0.5)),
            was_routed=bool(record.get("routed_to")),
            resolution_status=record.get("status", "open"),
        )


@dataclass
class InsightSignal:
    """
    A cross-agent insight compressed for the Δ layer.
    """

    target_domain: str
    # Which agent domain received the insight

    was_responded_to: bool
    # Whether the target agent responded

    implied_action_present: bool
    # Whether the insight included an implied action (i.e. was actionable)

    @classmethod
    def from_kce_record(cls, record: dict) -> "InsightSignal":
        """Build from a KCE /agent/insights record."""
        return cls(
            target_domain=record.get("target_domain", "unknown"),
            was_responded_to=record.get("status") == "responded",
            implied_action_present=bool(record.get("implied_action")),
        )


@dataclass
class InteractionDelta:
    """
    The Δ layer: everything that happened in a session window.

    This is permanent. It records characteristics of the session's reasoning
    output — not the raw content (which is in KCE) but the compressed signal
    that has training value.

    Aggregate metrics derived from the thinking traces, uncertainty deposits,
    and cross-agent insights of the session.
    """

    # Raw counts
    thinking_trace_count: int
    uncertainty_deposit_count: int
    cross_agent_insight_count: int

    # Thinking quality signals
    mean_alternatives_considered: float
    # Proxy for deliberateness — agents that consider alternatives make better decisions
    mean_confidence: float
    # Average stated confidence — calibration target
    cross_domain_signal_rate: float
    # Fraction of traces that produced a cross-domain signal (0-1)
    decision_category_distribution: dict[str, int]
    # How the session's decisions were categorised

    # Uncertainty profile
    uncertainty_type_distribution: dict[str, int]
    # How many of each type were deposited this session
    uncertainty_routing_rate: float
    # Fraction that were successfully routed (vs orphaned)
    open_uncertainty_count: int
    # Uncertainties still unresolved at session close

    # Insight activity
    insight_response_rate: float
    # Fraction of deposited insights that received a response
    actionable_insight_rate: float
    # Fraction of insights that included an implied action

    # Aggregate health scores (0-1, higher = healthier)
    decisiveness_score: float
    # 1 - (open_uncertainties / total_decisions) — how much was resolved vs deferred
    collaboration_score: float
    # cross_domain_signal_rate * insight_response_rate — how active cross-domain exchange was
    calibration_score: float
    # How close mean_confidence is to the empirical resolve rate
    # (if mean_confidence=0.8 but 40% uncertainties remain, calibration is poor)

    # Raw signal lists for downstream analysis
    thinking_signals: list[ThinkingSignal] = field(default_factory=list)
    uncertainty_signals: list[UncertaintySignal] = field(default_factory=list)
    insight_signals: list[InsightSignal] = field(default_factory=list)


@dataclass
class InteractionPacket:
    """
    The complete KindPress packet for an agent interaction session.

    k + Δ together. Portable, storable, diffable.

    InteractionPacket is designed to:
    1. Be deposited to the KCE residue corpus as evidence of an agent session
    2. Be stored as JSON records in a local corpus for training analysis
    3. Be diffed over time (how has this agent's decisiveness/calibration changed?)
    4. Be aggregated across agents (what is the community's uncertainty profile?)

    The packet is immutable. If the context changes, create a new packet.
    Stack Δ records under the same k fingerprint to build session history.
    """

    context: InteractionContext
    delta: InteractionDelta
    k_fingerprint: str
    packet_id: str
    created_at: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ── Public API ────────────────────────────────────────────────────────────────


def compress(
    context: InteractionContext,
    thinking_records: list[dict],
    uncertainty_records: list[dict],
    insight_records: list[dict],
) -> InteractionPacket:
    """
    Compress a set of KCE agent intelligence records into an InteractionPacket.

    Input records are the raw JSON objects returned by:
      GET /agent/thinking?agent_domain=...
      GET /agent/uncertainties?agent_domain=...
      GET /agent/insights?source_domain=...

    Returns a complete InteractionPacket (k + Δ).

    The packet is the training-safe compressed form. Use it for:
    - Depositing to KCE residue corpus
    - Storing in a local corpus file for analysis
    - Diffing against previous sessions
    """
    import uuid

    # Build signal lists
    thinking_signals = [ThinkingSignal.from_kce_record(r) for r in thinking_records]
    uncertainty_signals = [UncertaintySignal.from_kce_record(r) for r in uncertainty_records]
    insight_signals = [InsightSignal.from_kce_record(r) for r in insight_records]

    delta = _compute_delta(thinking_signals, uncertainty_signals, insight_signals)

    k_fp = context.fingerprint()
    packet_id = f"ip-{k_fp[:8]}-{uuid.uuid4().hex[:6]}"

    return InteractionPacket(
        context=context,
        delta=delta,
        k_fingerprint=k_fp,
        packet_id=packet_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def diff(packet_a: InteractionPacket, packet_b: InteractionPacket) -> dict:
    """
    Compute the delta between two InteractionPackets for the same agent domain.

    Returns a dict of metric → (a_value, b_value, change) for all numeric metrics.
    Positive change = improvement (more decisive, better calibrated, more collaborative).

    Typical use: compare this session's packet against last session's packet
    to see if the agent is improving, drifting, or holding steady.
    """
    def _metric_change(a_val: float, b_val: float) -> tuple:
        change = b_val - a_val
        return (round(a_val, 3), round(b_val, 3), round(change, 3))

    da = packet_a.delta
    db = packet_b.delta

    return {
        "thinking_trace_count": _metric_change(da.thinking_trace_count, db.thinking_trace_count),
        "uncertainty_deposit_count": _metric_change(da.uncertainty_deposit_count, db.uncertainty_deposit_count),
        "mean_alternatives_considered": _metric_change(da.mean_alternatives_considered, db.mean_alternatives_considered),
        "mean_confidence": _metric_change(da.mean_confidence, db.mean_confidence),
        "cross_domain_signal_rate": _metric_change(da.cross_domain_signal_rate, db.cross_domain_signal_rate),
        "uncertainty_routing_rate": _metric_change(da.uncertainty_routing_rate, db.uncertainty_routing_rate),
        "open_uncertainty_count": _metric_change(da.open_uncertainty_count, db.open_uncertainty_count),
        "insight_response_rate": _metric_change(da.insight_response_rate, db.insight_response_rate),
        "decisiveness_score": _metric_change(da.decisiveness_score, db.decisiveness_score),
        "collaboration_score": _metric_change(da.collaboration_score, db.collaboration_score),
        "calibration_score": _metric_change(da.calibration_score, db.calibration_score),
        "_agent_domain_a": packet_a.context.agent_domain,
        "_agent_domain_b": packet_b.context.agent_domain,
        "_same_agent": packet_a.context.agent_domain == packet_b.context.agent_domain,
    }


def aggregate(packets: list[InteractionPacket]) -> dict:
    """
    Aggregate a list of InteractionPackets (typically the full community corpus
    for a given time window) into a community-level health profile.

    Returns summary statistics across agents and time.
    Useful for the periodic synthesis script and governance reporting.
    """
    if not packets:
        return {"error": "no packets to aggregate", "count": 0}

    deltas = [p.delta for p in packets]
    domains = [p.context.agent_domain for p in packets]

    def _mean(values: list[float]) -> float:
        return round(statistics.mean(values), 3) if values else 0.0

    # Collect uncertainty type distributions across all packets
    all_uncertainty_types: dict[str, int] = {}
    for d in deltas:
        for t, count in d.uncertainty_type_distribution.items():
            all_uncertainty_types[t] = all_uncertainty_types.get(t, 0) + count

    return {
        "packet_count": len(packets),
        "agent_domains": list(set(domains)),
        "total_thinking_traces": sum(d.thinking_trace_count for d in deltas),
        "total_uncertainty_deposits": sum(d.uncertainty_deposit_count for d in deltas),
        "total_insights": sum(d.cross_agent_insight_count for d in deltas),
        "community_decisiveness": _mean([d.decisiveness_score for d in deltas]),
        "community_collaboration": _mean([d.collaboration_score for d in deltas]),
        "community_calibration": _mean([d.calibration_score for d in deltas]),
        "mean_open_uncertainties_per_session": _mean([float(d.open_uncertainty_count) for d in deltas]),
        "community_uncertainty_type_distribution": all_uncertainty_types,
        "highest_uncertainty_type": max(all_uncertainty_types, key=all_uncertainty_types.get) if all_uncertainty_types else None,
        # Which type of uncertainty is most common → doctrine gap candidate
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _compute_delta(
    thinking: list[ThinkingSignal],
    uncertainties: list[UncertaintySignal],
    insights: list[InsightSignal],
) -> InteractionDelta:
    """Compute all aggregate metrics from the signal lists."""

    n_thinking = len(thinking)
    n_uncertainty = len(uncertainties)
    n_insights = len(insights)

    # ── Think signals ──
    mean_alts = (
        statistics.mean(t.alternatives_considered for t in thinking)
        if thinking else 0.0
    )
    mean_conf = (
        statistics.mean(t.confidence for t in thinking)
        if thinking else 0.7
    )
    xd_signal_rate = (
        sum(1 for t in thinking if t.had_cross_domain_signal) / n_thinking
        if n_thinking else 0.0
    )
    decision_dist: dict[str, int] = {}
    for t in thinking:
        decision_dist[t.decision_category] = decision_dist.get(t.decision_category, 0) + 1

    # ── Uncertainty signals ──
    uncertainty_type_dist: dict[str, int] = {}
    for u in uncertainties:
        uncertainty_type_dist[u.uncertainty_type] = uncertainty_type_dist.get(u.uncertainty_type, 0) + 1
    routing_rate = (
        sum(1 for u in uncertainties if u.was_routed) / n_uncertainty
        if n_uncertainty else 1.0
    )
    open_count = sum(1 for u in uncertainties if u.resolution_status == "open")

    # ── Insight signals ──
    response_rate = (
        sum(1 for i in insights if i.was_responded_to) / n_insights
        if n_insights else 0.0
    )
    actionable_rate = (
        sum(1 for i in insights if i.implied_action_present) / n_insights
        if n_insights else 0.0
    )

    # ── Health scores ──
    total_decisions = n_thinking + n_uncertainty  # every trace + every uncertainty = a decision point
    decisiveness = (
        1.0 - (open_count / total_decisions)
        if total_decisions > 0 else 1.0
    )
    collaboration = xd_signal_rate * max(response_rate, 0.1)
    # Calibration: how well does stated confidence predict resolution rate?
    resolve_rate = 1.0 - (open_count / n_uncertainty) if n_uncertainty else 1.0
    calibration = max(0.0, 1.0 - abs(mean_conf - resolve_rate))

    return InteractionDelta(
        thinking_trace_count=n_thinking,
        uncertainty_deposit_count=n_uncertainty,
        cross_agent_insight_count=n_insights,
        mean_alternatives_considered=round(mean_alts, 3),
        mean_confidence=round(mean_conf, 3),
        cross_domain_signal_rate=round(xd_signal_rate, 3),
        decision_category_distribution=decision_dist,
        uncertainty_type_distribution=uncertainty_type_dist,
        uncertainty_routing_rate=round(routing_rate, 3),
        open_uncertainty_count=open_count,
        insight_response_rate=round(response_rate, 3),
        actionable_insight_rate=round(actionable_rate, 3),
        decisiveness_score=round(decisiveness, 3),
        collaboration_score=round(collaboration, 3),
        calibration_score=round(calibration, 3),
        thinking_signals=thinking,
        uncertainty_signals=uncertainties,
        insight_signals=insights,
    )


def _infer_task_domain(task_context: str) -> str:
    """
    Infer a broad task domain tag from a task context string.
    Used for compressing thinking traces without storing raw content.
    """
    import re
    text = task_context.lower()
    # Use word-boundary checks to avoid substring false matches (e.g. 'docker' ≠ 'doc')
    def has_word(word: str) -> bool:
        return bool(re.search(r'\b' + re.escape(word) + r'\b', text))
    if any(has_word(w) for w in ["schema", "table", "migration", "database"]):
        return "schema"
    if text and " db " in f" {text} ":
        return "schema"
    if any(has_word(w) for w in ["api", "route", "endpoint", "server", "http"]):
        return "api"
    if any(has_word(w) for w in ["test", "pytest", "spec", "assert"]):
        return "test"
    if any(has_word(w) for w in ["readme", "comment", "docstring"]) or has_word("doc"):
        return "doc"
    if any(has_word(w) for w in ["config", "yaml", "env", "settings"]):
        return "config"
    if any(has_word(w) for w in ["bug", "fix", "error", "fail", "broken"]):
        return "debug"
    if any(has_word(w) for w in ["deploy", "cloud", "docker"]):
        return "deploy"
    if any(has_word(w) for w in ["build", "cmake", "compile", "webpack", "npm"]):
        return "build"
    if any(has_word(w) for w in ["architecture", "design", "structure", "refactor"]):
        return "architecture"
    return "general"
