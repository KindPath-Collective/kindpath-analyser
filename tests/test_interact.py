"""
tests/test_interact.py

Tests for kindpress/interact.py — the k/Δ agent interaction compression module.

These tests verify the module in isolation: no KCE connection required,
no audio processing. Everything is tested with synthetic dict inputs
matching the KCE API response shapes.

Tests must not break when run alongside the existing 57 tests in the suite.
This file is purely additive.
"""

import pytest
import json
from kindpress.interact import (
    InteractionContext,
    InteractionDelta,
    InteractionPacket,
    ThinkingSignal,
    UncertaintySignal,
    InsightSignal,
    compress,
    diff,
    aggregate,
    _infer_task_domain,
    _compute_delta,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CONTEXT = InteractionContext(
    agent_domain="devops-lead",
    doctrine_version="2026-03-13",
    primary_repo="kindpath-kce",
    session_date_utc="2026-03-20",
    role_tags=["infrastructure", "cross-repo"],
)

SAMPLE_THINKING_RECORDS = [
    {
        "id": "abc-001",
        "agent_domain": "devops-lead",
        "task_context": "Designing the schema for agent thinking traces",
        "decision_taken": "build the schema with FK references for cross-agent insights",
        "alternatives_considered": ["flat JSON blob", "no FK constraints"],
        "cross_domain_signal": "testing agent needs to validate schema",
        "target_domains": ["testing"],
        "confidence": 0.85,
        "session_id": "sess-001",
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:00:00Z",
    },
    {
        "id": "abc-002",
        "agent_domain": "devops-lead",
        "task_context": "Adding API route for uncertainty deposits",
        "decision_taken": "build route with auto-routing logic embedded",
        "alternatives_considered": [],
        "cross_domain_signal": None,
        "target_domains": [],
        "confidence": 0.9,
        "session_id": "sess-001",
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:05:00Z",
    },
    {
        "id": "abc-003",
        "agent_domain": "devops-lead",
        "task_context": "Fix TypeScript error in server.ts",
        "decision_taken": "fix EVENT_TYPES reference to use ANALYSIS_COMPLETE",
        "alternatives_considered": ["add MODULE_UPDATED to EVENT_TYPES"],
        "cross_domain_signal": None,
        "target_domains": [],
        "confidence": 0.95,
        "session_id": "sess-001",
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:10:00Z",
    },
]

SAMPLE_UNCERTAINTY_RECORDS = [
    {
        "id": "unc-001",
        "type": "technical",
        "agent_domain": "devops-lead",
        "question": "Which drizzle-kit command syntax applies to version 0.20.18?",
        "context": "npm run db:push fails with 'unknown command push'",
        "taken": "Tried push:pg variant",
        "resolve_via": None,
        "confidence": 0.6,
        "routed_to": "testing",
        "status": "resolved",
        "resolution_text": "Use push:pg --driver pg --schema ... --connectionString ...",
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:07:00Z",
        "resolved_at": "2026-03-20T10:12:00Z",
    },
    {
        "id": "unc-002",
        "type": "doctrine",
        "agent_domain": "devops-lead",
        "question": "Should cross-agent insights auto-create on every thinking trace?",
        "context": "Current implementation auto-creates only when cross_domain_signal set",
        "taken": "Only auto-create when cross_domain_signal is present",
        "resolve_via": None,
        "confidence": 0.5,
        "routed_to": "oversight",
        "status": "open",
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:08:00Z",
        "resolved_at": None,
    },
]

SAMPLE_INSIGHT_RECORDS = [
    {
        "id": "ins-001",
        "source_domain": "devops-lead",
        "target_domain": "testing",
        "insight": "Three new KCE schema tables added — tests needed",
        "connection": "Every new table needs validation before production use",
        "implied_action": "Write tests for agent_thinking_traces, uncertainty_deposits, cross_agent_insights",
        "status": "open",
        "response_text": None,
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:00:00Z",
        "responded_at": None,
    },
    {
        "id": "ins-002",
        "source_domain": "devops-lead",
        "target_domain": "oversight",
        "insight": "Auto-cascading insight creation may be too noisy",
        "connection": "Charter §7 governs what warrants cross-domain signal",
        "implied_action": None,
        "status": "responded",
        "response_text": "Agreed — keep auto-cascade behind explicit flag only",
        "source_repo": "kindpath-kce",
        "created_at": "2026-03-20T10:01:00Z",
        "responded_at": "2026-03-20T11:00:00Z",
    },
]


# ── InteractionContext tests ──────────────────────────────────────────────────

class TestInteractionContext:
    def test_fingerprint_is_16_chars(self):
        fp = SAMPLE_CONTEXT.fingerprint()
        assert len(fp) == 16

    def test_fingerprint_is_deterministic(self):
        fp1 = SAMPLE_CONTEXT.fingerprint()
        fp2 = SAMPLE_CONTEXT.fingerprint()
        assert fp1 == fp2

    def test_different_domain_different_fingerprint(self):
        other = InteractionContext(
            agent_domain="testing",
            doctrine_version=SAMPLE_CONTEXT.doctrine_version,
            primary_repo=SAMPLE_CONTEXT.primary_repo,
            session_date_utc=SAMPLE_CONTEXT.session_date_utc,
        )
        assert SAMPLE_CONTEXT.fingerprint() != other.fingerprint()

    def test_different_date_different_fingerprint(self):
        other = InteractionContext(
            agent_domain=SAMPLE_CONTEXT.agent_domain,
            doctrine_version=SAMPLE_CONTEXT.doctrine_version,
            primary_repo=SAMPLE_CONTEXT.primary_repo,
            session_date_utc="2026-03-21",
        )
        assert SAMPLE_CONTEXT.fingerprint() != other.fingerprint()

    def test_role_tags_are_optional(self):
        ctx = InteractionContext(
            agent_domain="research",
            doctrine_version="2026-03-13",
            primary_repo="kindpath-canon",
            session_date_utc="2026-03-20",
        )
        assert ctx.role_tags == []
        assert ctx.fingerprint()  # Should not raise


# ── ThinkingSignal tests ──────────────────────────────────────────────────────

class TestThinkingSignal:
    def test_from_kce_record_basic(self):
        sig = ThinkingSignal.from_kce_record(SAMPLE_THINKING_RECORDS[0])
        assert sig.task_domain == "schema"
        assert sig.alternatives_considered == 2
        assert sig.had_cross_domain_signal is True
        assert sig.confidence == 0.85
        assert sig.decision_category == "build"

    def test_from_kce_record_no_alternatives(self):
        sig = ThinkingSignal.from_kce_record(SAMPLE_THINKING_RECORDS[1])
        assert sig.alternatives_considered == 0
        assert sig.had_cross_domain_signal is False

    def test_decision_category_fix(self):
        record = {"decision_taken": "fix the broken import", "alternatives_considered": [], "confidence": 0.9}
        sig = ThinkingSignal.from_kce_record(record)
        assert sig.decision_category == "fix"

    def test_decision_category_defer(self):
        record = {"decision_taken": "defer this until oversight responds", "alternatives_considered": [], "confidence": 0.5}
        sig = ThinkingSignal.from_kce_record(record)
        assert sig.decision_category == "defer"

    def test_decision_category_default_is_build(self):
        record = {"decision_taken": "created new endpoint with validation", "alternatives_considered": [], "confidence": 0.8}
        sig = ThinkingSignal.from_kce_record(record)
        assert sig.decision_category == "build"


# ── UncertaintySignal tests ───────────────────────────────────────────────────

class TestUncertaintySignal:
    def test_from_kce_resolved(self):
        sig = UncertaintySignal.from_kce_record(SAMPLE_UNCERTAINTY_RECORDS[0])
        assert sig.uncertainty_type == "technical"
        assert sig.confidence == 0.6
        assert sig.was_routed is True
        assert sig.resolution_status == "resolved"

    def test_from_kce_open(self):
        sig = UncertaintySignal.from_kce_record(SAMPLE_UNCERTAINTY_RECORDS[1])
        assert sig.resolution_status == "open"
        assert sig.was_routed is True  # routed_to = oversight

    def test_unrouted_uncertainty(self):
        record = {"type": "data", "confidence": 0.4, "routed_to": None, "status": "open"}
        sig = UncertaintySignal.from_kce_record(record)
        assert sig.was_routed is False


# ── InsightSignal tests ───────────────────────────────────────────────────────

class TestInsightSignal:
    def test_from_kce_open_with_action(self):
        sig = InsightSignal.from_kce_record(SAMPLE_INSIGHT_RECORDS[0])
        assert sig.target_domain == "testing"
        assert sig.was_responded_to is False
        assert sig.implied_action_present is True

    def test_from_kce_responded_no_action(self):
        sig = InsightSignal.from_kce_record(SAMPLE_INSIGHT_RECORDS[1])
        assert sig.was_responded_to is True
        assert sig.implied_action_present is False


# ── compress() tests ──────────────────────────────────────────────────────────

class TestCompress:
    def test_returns_interaction_packet(self):
        packet = compress(
            SAMPLE_CONTEXT,
            SAMPLE_THINKING_RECORDS,
            SAMPLE_UNCERTAINTY_RECORDS,
            SAMPLE_INSIGHT_RECORDS,
        )
        assert isinstance(packet, InteractionPacket)

    def test_packet_has_k_fingerprint(self):
        packet = compress(SAMPLE_CONTEXT, [], [], [])
        assert len(packet.k_fingerprint) == 16

    def test_packet_id_has_prefix(self):
        packet = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, [], [])
        assert packet.packet_id.startswith("ip-")

    def test_packet_ids_are_unique(self):
        p1 = compress(SAMPLE_CONTEXT, [], [], [])
        p2 = compress(SAMPLE_CONTEXT, [], [], [])
        assert p1.packet_id != p2.packet_id

    def test_delta_counts_are_correct(self):
        packet = compress(
            SAMPLE_CONTEXT,
            SAMPLE_THINKING_RECORDS,
            SAMPLE_UNCERTAINTY_RECORDS,
            SAMPLE_INSIGHT_RECORDS,
        )
        assert packet.delta.thinking_trace_count == 3
        assert packet.delta.uncertainty_deposit_count == 2
        assert packet.delta.cross_agent_insight_count == 2

    def test_empty_inputs_do_not_raise(self):
        packet = compress(SAMPLE_CONTEXT, [], [], [])
        assert packet.delta.thinking_trace_count == 0
        assert packet.delta.uncertainty_deposit_count == 0
        assert packet.delta.cross_agent_insight_count == 0
        assert packet.delta.decisiveness_score == 1.0  # no open uncertainties

    def test_to_json_is_valid(self):
        packet = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, SAMPLE_UNCERTAINTY_RECORDS, SAMPLE_INSIGHT_RECORDS)
        json_str = packet.to_json()
        parsed = json.loads(json_str)
        assert "context" in parsed
        assert "delta" in parsed
        assert "k_fingerprint" in parsed

    def test_to_dict_is_serialisable(self):
        packet = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, SAMPLE_UNCERTAINTY_RECORDS, SAMPLE_INSIGHT_RECORDS)
        d = packet.to_dict()
        # All values must be JSON-serialisable
        json.dumps(d)  # Should not raise


# ── InteractionDelta derived metrics tests ────────────────────────────────────

class TestInteractionDeltaMetrics:
    def setup_method(self):
        self.packet = compress(
            SAMPLE_CONTEXT,
            SAMPLE_THINKING_RECORDS,
            SAMPLE_UNCERTAINTY_RECORDS,
            SAMPLE_INSIGHT_RECORDS,
        )

    def test_mean_confidence_is_in_range(self):
        d = self.packet.delta
        assert 0.0 <= d.mean_confidence <= 1.0

    def test_cross_domain_signal_rate(self):
        d = self.packet.delta
        # 1 of 3 thinking traces had cross_domain_signal
        assert abs(d.cross_domain_signal_rate - (1 / 3)) < 0.01

    def test_open_uncertainty_count(self):
        d = self.packet.delta
        # 1 of 2 uncertainties is open
        assert d.open_uncertainty_count == 1

    def test_insight_response_rate(self):
        d = self.packet.delta
        # 1 of 2 insights was responded to
        assert abs(d.insight_response_rate - 0.5) < 0.01

    def test_decisiveness_score_is_float_in_range(self):
        d = self.packet.delta
        assert 0.0 <= d.decisiveness_score <= 1.0

    def test_collaboration_score_is_float_in_range(self):
        d = self.packet.delta
        assert 0.0 <= d.collaboration_score <= 1.0

    def test_calibration_score_is_float_in_range(self):
        d = self.packet.delta
        assert 0.0 <= d.calibration_score <= 1.0

    def test_decision_category_distribution_present(self):
        d = self.packet.delta
        assert isinstance(d.decision_category_distribution, dict)
        assert sum(d.decision_category_distribution.values()) == 3

    def test_uncertainty_type_distribution_present(self):
        d = self.packet.delta
        assert "technical" in d.uncertainty_type_distribution
        assert "doctrine" in d.uncertainty_type_distribution

    def test_actionable_insight_rate(self):
        d = self.packet.delta
        # 1 of 2 insights had implied_action
        assert abs(d.actionable_insight_rate - 0.5) < 0.01


# ── diff() tests ──────────────────────────────────────────────────────────────

class TestDiff:
    def test_diff_returns_expected_keys(self):
        p1 = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS[:1], SAMPLE_UNCERTAINTY_RECORDS[:1], [])
        p2 = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, SAMPLE_UNCERTAINTY_RECORDS, SAMPLE_INSIGHT_RECORDS)
        result = diff(p1, p2)
        assert "decisiveness_score" in result
        assert "collaboration_score" in result
        assert "calibration_score" in result
        assert "_same_agent" in result

    def test_diff_same_agent_flag(self):
        p1 = compress(SAMPLE_CONTEXT, [], [], [])
        p2 = compress(SAMPLE_CONTEXT, [], [], [])
        result = diff(p1, p2)
        assert result["_same_agent"] is True

    def test_diff_different_agent_flag(self):
        other_ctx = InteractionContext(
            agent_domain="testing",
            doctrine_version="2026-03-13",
            primary_repo="kindpath-analyser",
            session_date_utc="2026-03-20",
        )
        p1 = compress(SAMPLE_CONTEXT, [], [], [])
        p2 = compress(other_ctx, [], [], [])
        result = diff(p1, p2)
        assert result["_same_agent"] is False

    def test_diff_change_values_are_tuples(self):
        p1 = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS[:1], [], [])
        p2 = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, [], [])
        result = diff(p1, p2)
        # Each value should be (a, b, change) tuple
        a, b, change = result["thinking_trace_count"]
        assert a == 1
        assert b == 3
        assert change == 2


# ── aggregate() tests ─────────────────────────────────────────────────────────

class TestAggregate:
    def test_empty_list_returns_error(self):
        result = aggregate([])
        assert "error" in result
        assert result["count"] == 0

    def test_aggregate_basic_counts(self):
        p1 = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, SAMPLE_UNCERTAINTY_RECORDS, SAMPLE_INSIGHT_RECORDS)
        other_ctx = InteractionContext(
            agent_domain="testing",
            doctrine_version="2026-03-13",
            primary_repo="kindpath-analyser",
            session_date_utc="2026-03-20",
        )
        p2 = compress(other_ctx, SAMPLE_THINKING_RECORDS[:1], [], [])
        result = aggregate([p1, p2])
        assert result["packet_count"] == 2
        assert result["total_thinking_traces"] == 4  # 3 + 1
        assert "community_decisiveness" in result
        assert "community_collaboration" in result

    def test_aggregate_domain_list(self):
        p1 = compress(SAMPLE_CONTEXT, [], [], [])
        other_ctx = InteractionContext(
            agent_domain="oversight",
            doctrine_version="2026-03-13",
            primary_repo="kindpath-analyser",
            session_date_utc="2026-03-20",
        )
        p2 = compress(other_ctx, [], [], [])
        result = aggregate([p1, p2])
        assert set(result["agent_domains"]) == {"devops-lead", "oversight"}

    def test_aggregate_single_packet(self):
        packet = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS, SAMPLE_UNCERTAINTY_RECORDS, SAMPLE_INSIGHT_RECORDS)
        result = aggregate([packet])
        assert result["packet_count"] == 1
        assert result["total_thinking_traces"] == 3


# ── _infer_task_domain() tests ────────────────────────────────────────────────

class TestInferTaskDomain:
    @pytest.mark.parametrize("text,expected", [
        ("Designing the schema for postgres", "schema"),
        ("Adding API route for endpoint", "api"),
        ("Writing pytest test assertions", "test"),
        ("Updating the README docstring", "doc"),
        ("Editing config.yaml settings", "config"),
        ("Fix broken error in module", "debug"),
        ("Running npm build webpack", "build"),
        ("Deploy to Cloud Run docker", "deploy"),
        ("Architecture design refactor", "architecture"),
        ("Something entirely unrelated", "general"),
    ])
    def test_domain_inference(self, text, expected):
        assert _infer_task_domain(text) == expected

    def test_empty_string_returns_general(self):
        assert _infer_task_domain("") == "general"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_compress_with_none_alternatives(self):
        """KCE records might have null alternatives_considered."""
        records = [{
            "task_context": "test",
            "decision_taken": "build it",
            "alternatives_considered": None,  # null from DB
            "cross_domain_signal": None,
            "confidence": 0.8,
        }]
        # Should not raise — None alternatives treated as 0
        packet = compress(SAMPLE_CONTEXT, records, [], [])
        assert packet.delta.thinking_trace_count == 1

    def test_compress_with_missing_confidence(self):
        """Records without confidence field use default 0.7."""
        records = [{
            "task_context": "build",
            "decision_taken": "done",
            "alternatives_considered": [],
            "cross_domain_signal": None,
            # no 'confidence' key
        }]
        packet = compress(SAMPLE_CONTEXT, records, [], [])
        assert abs(packet.delta.mean_confidence - 0.7) < 0.01

    def test_all_uncertainties_open_gives_low_decisiveness(self):
        """Session with all uncertainties open should score low decisiveness."""
        open_records = [
            {"type": "technical", "confidence": 0.5, "routed_to": "testing", "status": "open"},
            {"type": "doctrine", "confidence": 0.4, "routed_to": "oversight", "status": "open"},
            {"type": "data", "confidence": 0.3, "routed_to": "research", "status": "open"},
        ]
        packet = compress(SAMPLE_CONTEXT, [], open_records, [])
        # 3 open uncertainties / 3 total decisions = decisiveness = 1 - 1.0 = 0.0
        assert packet.delta.decisiveness_score == 0.0

    def test_all_uncertainties_resolved_gives_high_decisiveness(self):
        """All resolved uncertainties → decisiveness = 1.0."""
        resolved = [{
            "type": "technical", "confidence": 0.9, "routed_to": "testing", "status": "resolved"
        }]
        # With 1 thinking trace and 1 resolved uncertainty: open_count = 0
        packet = compress(SAMPLE_CONTEXT, SAMPLE_THINKING_RECORDS[:1], resolved, [])
        assert packet.delta.decisiveness_score == 1.0
