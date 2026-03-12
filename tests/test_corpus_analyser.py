"""
Tests for core/corpus_analyser.py

All tests use real numpy computations on synthetic profile dictionaries.
No audio files needed — the corpus analyser works entirely on already-extracted
analysis dicts. Tests are fast (no audio processing).

Test coverage:
- CorpusTemporalProfile dataclass structure
- Period bucketing (period_size, boundary computation)
- Trend extraction from profiles
- Composite index computation (creative freedom, cultural pressure)
- SOS signature detection
- Inflection point detection
- Prediction via linear extrapolation
- Narrative summary
- Edge cases: 0 profiles, 1 profile, missing fields, single period
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.corpus_analyser import (
    CorpusTemporalProfile,
    analyse_corpus,
    _mean_field,
    _deep_get,
    _compute_creative_freedom_index,
    _compute_cultural_pressure_index,
    _detect_sos_signature_at_period,
    _is_declining,
    _is_rising,
    _find_inflection_points,
    _predict_next_period,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_profile(
    year: int,
    lsii: float = 0.2,
    authentic: float = 0.6,
    manufacturing: float = 0.3,
    residue: float = 0.4,
    dynamic_range: float = 12.0,
    valence: float = 0.3,
    arousal: float = 0.5,
    tension: float = 0.4,
    complexity: float = 0.5,
    groove: float = 15.0,
    crest: float = 12.0,
    harmonic: float = 0.5,
) -> dict:
    """Make a minimal profile dict for corpus analysis testing."""
    return {
        "year": year,
        "lsii_score": lsii,
        "authentic_emission_score": authentic,
        "manufacturing_score": manufacturing,
        "creative_residue": residue,
        "dynamic_range_db": dynamic_range,
        "valence": valence,
        "arousal": arousal,
        "tension": tension,
        "complexity": complexity,
        "groove_deviation_ms": groove,
        "crest_factor_db": crest,
        "harmonic_complexity": harmonic,
    }


def make_corpus_declining(
    start_year: int = 1970,
    n_periods: int = 5,
    period_size: int = 5,
) -> list:
    """
    Build a corpus showing a clear declining creative freedom pattern:
    - Dynamic range narrows over time
    - Groove deviation decreases (more quantised)
    - Creative residue drops
    - LSII rises
    - Manufacturing rises
    Each 5-year period has 3 records.
    """
    profiles = []
    for p in range(n_periods):
        year_start = start_year + p * period_size
        decay = p / (n_periods - 1)  # 0 → 1 over the range
        for _ in range(3):
            profiles.append(make_profile(
                year=year_start + 1,
                lsii=0.1 + 0.6 * decay,
                authentic=0.8 - 0.5 * decay,
                manufacturing=0.2 + 0.5 * decay,
                residue=0.7 - 0.5 * decay,
                dynamic_range=18.0 - 12.0 * decay,
                groove=25.0 - 20.0 * decay,
                crest=15.0 - 10.0 * decay,
                harmonic=0.7 - 0.4 * decay,
            ))
    return profiles


def make_corpus_recovery(start_year: int = 2000, n_periods: int = 4) -> list:
    """A corpus where creative freedom improves over time."""
    profiles = []
    for p in range(n_periods):
        year = start_year + p * 5
        recovery = p / max(n_periods - 1, 1)
        for _ in range(2):
            profiles.append(make_profile(
                year=year + 1,
                lsii=0.6 - 0.4 * recovery,
                authentic=0.3 + 0.4 * recovery,
                manufacturing=0.7 - 0.4 * recovery,
                residue=0.2 + 0.4 * recovery,
                dynamic_range=6.0 + 10.0 * recovery,
                groove=5.0 + 20.0 * recovery,
            ))
    return profiles


# ── Test: CorpusTemporalProfile dataclass ─────────────────────────────────────

class TestCorpusTemporalProfile:
    def test_default_fields_exist(self):
        p = CorpusTemporalProfile()
        for field in [
            "periods", "period_boundaries", "record_counts",
            "lsii_trend", "authentic_emission_trend", "manufacturing_trend",
            "creative_residue_trend", "dynamic_range_trend",
            "valence_trend", "arousal_trend", "tension_trend", "complexity_trend",
            "groove_deviation_trend", "compression_trend", "harmonic_complexity_trend",
            "creative_freedom_index", "cultural_pressure_index",
            "sos_periods", "sos_confidence",
            "next_period_prediction", "prediction_confidence", "prediction_reasoning",
            "trend_summary", "inflection_points",
        ]:
            assert hasattr(p, field), f"Missing field: {field}"

    def test_default_lists_are_empty(self):
        p = CorpusTemporalProfile()
        assert p.periods == []
        assert p.lsii_trend == []
        assert p.sos_periods == []


# ── Test: Empty / minimal corpus ──────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_corpus_returns_profile(self):
        result = analyse_corpus([])
        assert isinstance(result, CorpusTemporalProfile)
        assert "insufficient" in result.trend_summary.lower()

    def test_single_profile_returns_profile(self):
        result = analyse_corpus([make_profile(year=2010)])
        assert isinstance(result, CorpusTemporalProfile)
        assert result.trend_summary != ""

    def test_profiles_without_year_field_skipped(self):
        profiles = [{"lsii_score": 0.3, "authentic_emission_score": 0.5}]
        result = analyse_corpus(profiles)
        assert isinstance(result, CorpusTemporalProfile)

    def test_invalid_year_values_skipped(self):
        profiles = [
            {"year": "not-a-year", "lsii_score": 0.3},
            make_profile(year=2000),
            make_profile(year=2005),
        ]
        result = analyse_corpus(profiles, period_size=5)
        assert isinstance(result, CorpusTemporalProfile)

    def test_single_period_no_trend(self):
        profiles = [make_profile(year=2000), make_profile(year=2003)]
        result = analyse_corpus(profiles, period_size=5)
        assert isinstance(result, CorpusTemporalProfile)

    def test_missing_fields_default_to_zero(self):
        """Profiles with missing fields should default gracefully."""
        sparse = [{"year": 2000}, {"year": 2005}]
        result = analyse_corpus(sparse, period_size=5)
        assert isinstance(result, CorpusTemporalProfile)


# ── Test: Period bucketing ─────────────────────────────────────────────────────

class TestPeriodBucketing:
    def test_correct_period_count(self):
        """5 periods × 3 records = should produce 5 time buckets."""
        profiles = make_corpus_declining(start_year=1970, n_periods=5, period_size=5)
        result = analyse_corpus(profiles, period_size=5)
        assert len(result.periods) == 5

    def test_period_labels_format(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=3)
        result = analyse_corpus(profiles, period_size=5)
        for label in result.periods:
            parts = label.split("-")
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()

    def test_record_counts_sum_correct(self):
        """Total record count should match input profiles that have valid years."""
        profiles = make_corpus_declining(start_year=1970, n_periods=4)
        result = analyse_corpus(profiles, period_size=5)
        assert sum(result.record_counts) == len(profiles)

    def test_all_trends_same_length(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        n = len(result.periods)
        for trend in [
            result.lsii_trend, result.authentic_emission_trend,
            result.manufacturing_trend, result.creative_residue_trend,
            result.dynamic_range_trend,
        ]:
            assert len(trend) == n, f"Expected {n} periods, got {len(trend)}"

    def test_decade_period_size(self):
        """Period size of 10 years."""
        profiles = []
        for year in range(1960, 2010, 2):
            profiles.append(make_profile(year=year))
        result = analyse_corpus(profiles, period_size=10)
        assert len(result.periods) >= 3


# ── Test: Trend extraction ─────────────────────────────────────────────────────

class TestTrendExtraction:
    def test_lsii_trend_reflects_data(self):
        """LSII trend should match the input data direction."""
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        # Declining corpus: LSII should increase over time
        assert result.lsii_trend[-1] > result.lsii_trend[0], \
            f"LSII should rise in declining corpus: {result.lsii_trend}"

    def test_authentic_emission_trend_reflects_data(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        # Authentic emission should decrease
        assert result.authentic_emission_trend[-1] < result.authentic_emission_trend[0], \
            f"Authentic emission should fall in declining corpus"

    def test_dynamic_range_trend_reflects_data(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        assert result.dynamic_range_trend[-1] < result.dynamic_range_trend[0]

    def test_recovery_corpus_shows_improvement(self):
        profiles = make_corpus_recovery(start_year=2000, n_periods=4)
        result = analyse_corpus(profiles, period_size=5)
        if len(result.authentic_emission_trend) >= 2:
            assert result.authentic_emission_trend[-1] > result.authentic_emission_trend[0]


# ── Test: Mean field extraction ────────────────────────────────────────────────

class TestMeanField:
    def test_simple_field(self):
        profiles = [{"lsii_score": 0.3}, {"lsii_score": 0.5}]
        assert abs(_mean_field(profiles, "lsii_score") - 0.4) < 0.001

    def test_nested_field(self):
        profiles = [{"psychosomatic": {"valence": 0.3}}, {"psychosomatic": {"valence": 0.7}}]
        result = _mean_field(profiles, "psychosomatic.valence")
        assert abs(result - 0.5) < 0.001

    def test_missing_field_returns_default(self):
        profiles = [{"other_field": 1.0}]
        result = _mean_field(profiles, "nonexistent_field", default=0.5)
        assert result == 0.5

    def test_empty_profiles_returns_default(self):
        result = _mean_field([], "lsii_score", default=0.0)
        assert result == 0.0


# ── Test: Deep get ────────────────────────────────────────────────────────────

class TestDeepGet:
    def test_top_level_key(self):
        d = {"a": 1}
        assert _deep_get(d, "a") == 1

    def test_nested_key(self):
        d = {"a": {"b": 2}}
        assert _deep_get(d, "a.b") == 2

    def test_missing_key_returns_none(self):
        assert _deep_get({}, "missing") is None

    def test_missing_nested_key(self):
        d = {"a": {"b": 2}}
        assert _deep_get(d, "a.c") is None


# ── Test: Composite indices ────────────────────────────────────────────────────

class TestCompositeIndices:
    def test_creative_freedom_range(self):
        ae = [0.5, 0.4, 0.3]
        dr = [12.0, 10.0, 8.0]
        gd = [15.0, 10.0, 5.0]
        result = _compute_creative_freedom_index(ae, dr, gd)
        assert len(result) == 3
        for v in result:
            assert 0.0 <= v <= 1.0

    def test_cultural_pressure_range(self):
        mfg = [0.3, 0.5, 0.7]
        cmp = [15.0, 10.0, 5.0]
        result = _compute_cultural_pressure_index(mfg, cmp)
        assert len(result) == 3
        for v in result:
            assert 0.0 <= v <= 1.0

    def test_high_dynamic_range_increases_freedom(self):
        ae = [0.5, 0.5]
        dr_high = [20.0, 20.0]
        dr_low = [2.0, 2.0]
        gd = [15.0, 15.0]
        freedom_high = _compute_creative_freedom_index(ae, dr_high, gd)
        freedom_low = _compute_creative_freedom_index(ae, dr_low, gd)
        assert freedom_high[0] > freedom_low[0]

    def test_high_compression_increases_pressure(self):
        mfg = [0.5, 0.5]
        cmp_low = [3.0, 3.0]   # near-hypercompressed
        cmp_high = [18.0, 18.0] # natural dynamics
        pressure_low = _compute_cultural_pressure_index(mfg, cmp_low)
        pressure_high = _compute_cultural_pressure_index(mfg, cmp_high)
        assert pressure_low[0] > pressure_high[0], (
            f"Low crest factor should give higher pressure: low={pressure_low[0]:.2f}, high={pressure_high[0]:.2f}"
        )

    def test_empty_inputs_return_empty(self):
        assert _compute_creative_freedom_index([], [], []) == []
        assert _compute_cultural_pressure_index([], []) == []


# ── Test: SOS detection ───────────────────────────────────────────────────────

class TestSOSDetection:
    def test_declining_corpus_triggers_sos(self):
        """A corpus with clear multi-indicator decline should trigger SOS."""
        profiles = make_corpus_declining(start_year=1970, n_periods=6)
        result = analyse_corpus(profiles, period_size=5)
        assert len(result.sos_periods) > 0, \
            f"Expected SOS detection in declining corpus. Periods: {result.periods}, " \
            f"LSII: {result.lsii_trend}, DR: {result.dynamic_range_trend}"

    def test_sos_confidence_in_range(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=6)
        result = analyse_corpus(profiles, period_size=5)
        for conf in result.sos_confidence:
            assert 0.0 <= conf <= 1.0

    def test_recovery_corpus_fewer_sos(self):
        """A recovering corpus should have fewer SOS flags than a declining one."""
        declining = make_corpus_declining(start_year=1970, n_periods=5)
        recovering = make_corpus_recovery(start_year=1970, n_periods=5)
        result_d = analyse_corpus(declining, period_size=5)
        result_r = analyse_corpus(recovering, period_size=5)
        # Recovery should have fewer or equal SOS periods
        assert len(result_r.sos_periods) <= len(result_d.sos_periods)

    def test_sos_single_indicator_not_enough(self):
        """Only one indicator changing should not trigger SOS (threshold is 4/6)."""
        # Only LSII rises, everything else constant
        profiles = []
        for i, year in enumerate([1970, 1975, 1980]):
            lsii = 0.2 + i * 0.2
            for _ in range(2):
                profiles.append(make_profile(year=year + 1, lsii=lsii,
                                              authentic=0.5, manufacturing=0.4,
                                              residue=0.5, dynamic_range=12.0,
                                              groove=15.0, harmonic=0.5))
        result = analyse_corpus(profiles, period_size=5)
        # With only 1 indicator, no SOS
        assert len(result.sos_periods) == 0

    def test_is_declining_helper(self):
        values = [10.0, 8.0, 6.0]
        assert _is_declining(values, 1)   # 8 < 10
        assert _is_declining(values, 2)   # 6 < 8
        assert not _is_declining(values, 0)  # No prior period

    def test_is_rising_helper(self):
        values = [5.0, 7.0, 9.0]
        assert _is_rising(values, 1)
        assert _is_rising(values, 2)
        assert not _is_rising(values, 0)


# ── Test: Inflection points ───────────────────────────────────────────────────

class TestInflectionPoints:
    def test_sharp_decline_produces_inflection(self):
        periods = ["1970-1974", "1975-1979", "1980-1984", "1985-1989"]
        cf = [0.8, 0.6, 0.3, 0.25]  # Sharp decline in period 3
        cp = [0.2, 0.3, 0.4, 0.5]
        inflections = _find_inflection_points(periods, cf, cp)
        assert len(inflections) > 0

    def test_stable_corpus_no_inflection(self):
        periods = ["1970-1974", "1975-1979", "1980-1984"]
        cf = [0.5, 0.5, 0.5]
        cp = [0.4, 0.4, 0.4]
        inflections = _find_inflection_points(periods, cf, cp)
        assert len(inflections) == 0

    def test_inflection_has_required_fields(self):
        periods = ["1970-1974", "1975-1979", "1980-1984", "1985-1989"]
        cf = [0.8, 0.5, 0.3, 0.3]
        cp = [0.2, 0.4, 0.6, 0.6]
        inflections = _find_inflection_points(periods, cf, cp)
        for inf in inflections:
            assert "period" in inf
            assert "type" in inf
            assert "description" in inf

    def test_fewer_than_3_periods_no_inflection(self):
        inflections = _find_inflection_points(["A", "B"], [0.5, 0.3], [0.4, 0.6])
        assert inflections == []


# ── Test: Prediction ──────────────────────────────────────────────────────────

class TestPrediction:
    def test_prediction_contains_required_keys(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        assert "period" in result.next_period_prediction
        assert "lsii" in result.next_period_prediction
        assert "creative_freedom" in result.next_period_prediction

    def test_prediction_confidence_in_range(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        assert 0.0 <= result.prediction_confidence <= 1.0

    def test_prediction_reasoning_non_empty(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        assert isinstance(result.prediction_reasoning, str)
        assert len(result.prediction_reasoning) > 5

    def test_declining_trend_predicts_lower_freedom(self):
        """For a declining corpus, predicted creative freedom should be ≤ current."""
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        if result.creative_freedom_index and result.next_period_prediction.get("creative_freedom"):
            predicted = result.next_period_prediction["creative_freedom"]
            current = result.creative_freedom_index[-1]
            # Allow small tolerance
            assert predicted <= current + 0.1


# ── Test: Trend summary ───────────────────────────────────────────────────────

class TestTrendSummary:
    def test_summary_is_non_empty_string(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        assert isinstance(result.trend_summary, str)
        assert len(result.trend_summary) > 10

    def test_summary_mentions_time_span(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=5)
        result = analyse_corpus(profiles, period_size=5)
        assert "1970" in result.trend_summary or "1994" in result.trend_summary

    def test_summary_mentions_sos_when_detected(self):
        profiles = make_corpus_declining(start_year=1970, n_periods=6)
        result = analyse_corpus(profiles, period_size=5)
        if result.sos_periods:
            assert "SOS" in result.trend_summary or "signature" in result.trend_summary

    def test_insufficient_data_summary(self):
        result = analyse_corpus([])
        assert "insufficient" in result.trend_summary.lower()


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_declining_corpus_pipeline(self):
        """Complete declining corpus should produce a full CorpusTemporalProfile."""
        profiles = make_corpus_declining(start_year=1970, n_periods=6)
        result = analyse_corpus(profiles, period_size=5)
        assert len(result.periods) == 6
        assert len(result.creative_freedom_index) == 6
        assert len(result.cultural_pressure_index) == 6
        assert result.trend_summary != ""
        assert result.prediction_reasoning != ""

    def test_corpus_with_nested_field_profiles(self):
        """Profiles with nested dot-path fields should work."""
        profiles = [
            {"year": 2000, "lsii_score": 0.3, "psychosomatic": {"valence": 0.5}},
            {"year": 2005, "lsii_score": 0.4, "psychosomatic": {"valence": 0.4}},
            {"year": 2010, "lsii_score": 0.5, "psychosomatic": {"valence": 0.3}},
        ]
        result = analyse_corpus(profiles, period_size=5)
        assert isinstance(result, CorpusTemporalProfile)
        assert len(result.lsii_trend) >= 2
