"""
Tests for core/psychosomatics.py

Uses synthetic TrajectoryProfile and FingerprintReport data to test
the psychosomatic profiler without requiring audio files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from unittest.mock import MagicMock

from core.divergence import (
    TrajectoryProfile, LatesonginversionResult, DivergenceVector
)
from core.fingerprints import FingerprintReport, FingerprintMatch
from core.psychosomatics import build_psychosomatic_profile, PsychosomaticProfile


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_divergence_vector(**kwargs):
    defaults = dict(
        spectral_centroid_delta=0.0,
        spectral_flux_delta=0.0,
        spectral_complexity_delta=0.0,
        dynamic_range_delta=0.0,
        dynamic_energy_delta=0.0,
        harmonic_tension_delta=0.0,
        harmonic_tonality_delta=0.0,
        harmonic_complexity_delta=0.0,
        temporal_groove_delta=0.0,
        temporal_syncopation_delta=0.0,
        temporal_onset_density_delta=0.0,
    )
    defaults.update(kwargs)
    return DivergenceVector(**defaults)


def make_lsii(lsii_score=0.1, direction="consistent", dominant_axis="spectral_brightness",
               flag_level="none"):
    return LatesonginversionResult(
        lsii=lsii_score,
        direction=direction,
        dominant_axis=dominant_axis,
        divergence=make_divergence_vector(),
        q1_q3_baseline={},
        q4_values={},
        trajectory_description="Consistent",
        inversion_description="",
        flag_level=flag_level,
        flag_notes="",
    )


def make_trajectory(
    valence_arc=None, energy_arc=None, complexity_arc=None,
    coherence_arc=None, tension_arc=None, lsii_score=0.1,
    direction="consistent", quarters=None
):
    valence_arc = valence_arc or [0.5, 0.5, 0.5, 0.5]
    energy_arc = energy_arc or [0.5, 0.5, 0.5, 0.5]
    complexity_arc = complexity_arc or [0.5, 0.5, 0.5, 0.5]
    coherence_arc = coherence_arc or [0.7, 0.7, 0.7, 0.7]
    tension_arc = tension_arc or [0.3, 0.3, 0.3, 0.3]
    quarters = quarters or [
        {'tempo_bpm': 100, 'onset_density': 2.0},
        {'tempo_bpm': 100, 'onset_density': 2.0},
        {'tempo_bpm': 100, 'onset_density': 2.0},
        {'tempo_bpm': 100, 'onset_density': 2.0},
    ]
    return TrajectoryProfile(
        quarters=quarters,
        valence_arc=valence_arc,
        energy_arc=energy_arc,
        complexity_arc=complexity_arc,
        coherence_arc=coherence_arc,
        tension_arc=tension_arc,
        lsii_result=make_lsii(lsii_score=lsii_score, direction=direction),
    )


def make_fingerprints(manufacturing_markers=None, authenticity_markers=None):
    return FingerprintReport(
        likely_instruments=[],
        likely_era=[],
        likely_techniques=[],
        production_context="",
        authenticity_markers=authenticity_markers or [],
        manufacturing_markers=manufacturing_markers or [],
    )


# ── Structure tests ─────────────────────────────────────────────────────────────

def test_profile_returns_expected_type():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert isinstance(result, PsychosomaticProfile)


def test_profile_all_fields_present():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)

    for attr in [
        'valence', 'arousal', 'coherence', 'authenticity_index',
        'complexity', 'tension_resolution_ratio', 'relational_density',
        'predicted_physical_responses', 'predicted_emotional_states',
        'priming_vectors', 'prestige_signals', 'identity_capture_risk',
        'stage1_priming_detected', 'stage1_evidence',
        'stage2_prestige_attached', 'stage2_evidence',
        'stage3_tag_risk', 'stage3_evidence',
        'lsii_psychosomatic_reading',
        'authentic_emission_score', 'manufacturing_score', 'creative_residue',
        'somatic_summary', 'mechanism_summary', 'elder_reading',
    ]:
        assert hasattr(result, attr), f"Missing field: {attr}"


def test_float_fields_in_expected_range():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)

    assert -1.0 <= result.valence <= 1.0
    assert 0.0 <= result.arousal <= 1.0
    assert 0.0 <= result.coherence <= 1.0
    assert 0.0 <= result.authenticity_index <= 1.0
    assert 0.0 <= result.complexity <= 1.0
    assert 0.0 <= result.identity_capture_risk <= 1.0
    assert 0.0 <= result.authentic_emission_score <= 1.0
    assert 0.0 <= result.manufacturing_score <= 1.0
    assert 0.0 <= result.creative_residue <= 1.0


def test_elder_reading_is_non_empty_string():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert isinstance(result.elder_reading, str)
    assert len(result.elder_reading) > 20


# ── LSII-driven tests ──────────────────────────────────────────────────────────

def test_high_lsii_mentioned_in_elder_reading():
    """High LSII should cause the elder_reading to mention the late shift."""
    traj = make_trajectory(
        lsii_score=0.7,
        direction="darker and quieter",
        tension_arc=[0.2, 0.25, 0.2, 0.6],  # Q4 spikes
    )
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    reading_lower = result.elder_reading.lower()
    # Should mention the shift or inversion
    assert any(word in reading_lower for word in ['shift', 'step', 'outside', 'frame', 'final']), \
        f"Elder reading should mention late shift, got: {result.elder_reading}"


def test_low_lsii_produces_consistent_reading():
    """Low LSII should produce a reading that notes consistency."""
    traj = make_trajectory(lsii_score=0.05, direction="consistent")
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    reading = result.lsii_psychosomatic_reading.lower()
    assert 'consistent' in reading or 'no' in reading or 'established' in reading


def test_extreme_lsii_reading_present():
    traj = make_trajectory(
        lsii_score=0.9,
        direction="drastically brighter and louder",
    )
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    # LSII of 0.9 should produce extreme language in the reading
    assert 'inversion' in result.lsii_psychosomatic_reading.lower() or \
           'extreme' in result.lsii_psychosomatic_reading.lower() or \
           'different' in result.lsii_psychosomatic_reading.lower()


# ── Manufacturing / authenticity tests ────────────────────────────────────────

def test_heavy_manufacturing_markers_raise_manufacturing_score():
    fp = make_fingerprints(manufacturing_markers=[
        "professional studio quality",
        "heavy limiting detected",
        "commercial mastering",
        "loudness maximised",
    ])
    traj = make_trajectory()
    result = build_psychosomatic_profile(traj, fp)
    assert result.manufacturing_score > 0.3, \
        f"Expected manufacturing_score > 0.3, got {result.manufacturing_score}"


def test_no_manufacturing_markers_low_score():
    fp = make_fingerprints(manufacturing_markers=[])
    traj = make_trajectory(coherence_arc=[0.8, 0.8, 0.8, 0.8])
    result = build_psychosomatic_profile(traj, fp)
    assert result.manufacturing_score == 0.0


def test_authentic_markers_raise_authentic_emission():
    fp = make_fingerprints(
        authenticity_markers=["preserved dynamics", "natural transients", "live performance noise"],
        manufacturing_markers=[],
    )
    # High coherence + moderate LSII = authentic
    traj = make_trajectory(coherence_arc=[0.8, 0.85, 0.9, 0.8], lsii_score=0.3)
    result = build_psychosomatic_profile(traj, fp)
    assert result.authentic_emission_score > 0.4


def test_elder_reading_efficient_for_heavy_manufacturing():
    fp = make_fingerprints(manufacturing_markers=[
        "professional studio quality",
        "heavy limiting",
        "loudness maximised",
        "commercial mastering",
    ])
    traj = make_trajectory(
        complexity_arc=[0.2, 0.2, 0.2, 0.2],
        coherence_arc=[0.6, 0.6, 0.6, 0.6],
        lsii_score=0.05,
    )
    result = build_psychosomatic_profile(traj, fp)
    reading_lower = result.elder_reading.lower()
    assert any(word in reading_lower for word in ['efficient', 'accomplished', 'job', 'commercial', 'manufactured']), \
        f"Elder reading should mention efficiency/manufacturing, got: {result.elder_reading}"


# ── Stage detection tests ──────────────────────────────────────────────────────

def test_stage1_detected_by_bpm_tempo():
    traj = make_trajectory(
        quarters=[
            {'tempo_bpm': 128, 'onset_density': 8.0},
            *[{'tempo_bpm': 128, 'onset_density': 3.0}] * 3,
        ]
    )
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.stage1_priming_detected is True
    assert len(result.stage1_evidence) > 0


def test_stage1_detected_by_high_q1_energy():
    traj = make_trajectory(energy_arc=[0.85, 0.6, 0.6, 0.6])
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.stage1_priming_detected is True


def test_stage2_detected_by_low_complexity_high_production():
    fp = make_fingerprints(
        manufacturing_markers=["professional studio quality", "loudness maximised"],
        authenticity_markers=[],
    )
    traj = make_trajectory(complexity_arc=[0.15, 0.15, 0.15, 0.15])
    result = build_psychosomatic_profile(traj, fp)
    assert result.stage2_prestige_attached is True


def test_no_stages_when_no_triggers():
    fp = make_fingerprints()
    traj = make_trajectory(
        quarters=[{'tempo_bpm': 90, 'onset_density': 2.0}] * 4,
        energy_arc=[0.4, 0.4, 0.4, 0.4],
        complexity_arc=[0.6, 0.6, 0.6, 0.6],
    )
    result = build_psychosomatic_profile(traj, fp)
    # With no triggers, no meaningful mechanism summary
    assert result.mechanism_summary == "No significant conditioning mechanisms detected." or \
           result.stage1_priming_detected is False


# ── Arousal / valence direction tests ──────────────────────────────────────────

def test_high_energy_arc_produces_high_arousal():
    traj = make_trajectory(energy_arc=[0.85, 0.9, 0.88, 0.87])
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.arousal > 0.7


def test_low_energy_arc_produces_low_arousal():
    traj = make_trajectory(energy_arc=[0.2, 0.15, 0.18, 0.2])
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.arousal < 0.4


def test_high_tension_produces_negative_valence():
    traj = make_trajectory(
        tension_arc=[0.85, 0.9, 0.88, 0.9],
        energy_arc=[0.4, 0.4, 0.4, 0.4],
    )
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.valence < 0.0


def test_low_tension_high_energy_positive_valence():
    traj = make_trajectory(
        tension_arc=[0.1, 0.1, 0.1, 0.1],
        energy_arc=[0.8, 0.8, 0.8, 0.8],
    )
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.valence > 0.0


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_handles_empty_arcs_gracefully():
    """Empty arcs should not raise, should return defaults."""
    lsii = make_lsii()
    traj = TrajectoryProfile(
        quarters=[],
        valence_arc=[],
        energy_arc=[],
        complexity_arc=[],
        coherence_arc=[],
        tension_arc=[],
        lsii_result=lsii,
    )
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)  # Should not raise
    assert isinstance(result, PsychosomaticProfile)


def test_handles_none_stem_features():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp, stem_features=None)
    assert isinstance(result, PsychosomaticProfile)


def test_physical_responses_are_non_empty():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert len(result.predicted_physical_responses) > 0


def test_emotional_states_are_non_empty():
    traj = make_trajectory()
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert len(result.predicted_emotional_states) > 0


def test_tension_resolution_ratio_unresolved():
    traj = make_trajectory(tension_arc=[0.2, 0.2, 0.2, 0.5])
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.tension_resolution_ratio > 1.0


def test_tension_resolution_ratio_resolving():
    traj = make_trajectory(tension_arc=[0.6, 0.6, 0.6, 0.2])
    fp = make_fingerprints()
    result = build_psychosomatic_profile(traj, fp)
    assert result.tension_resolution_ratio < 1.0
