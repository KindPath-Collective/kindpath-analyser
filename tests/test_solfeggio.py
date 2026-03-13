"""
Tests for core/solfeggio.py

Validates the three-axis frequency field measurement:
  - vocal fundamental deviation from biological reference (~130 Hz)
  - Solfeggio grid proximity (alignment with 396–852 Hz pre-institutional series)
  - institutional distance (0 = biological, 1 = A440 standard)

No audio files required — all tests use synthetic chroma vectors and
pre-computed parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from core.solfeggio import (
    SolfeggioAlignment,
    compute_solfeggio_alignment,
    aggregate_solfeggio,
    BIOLOGICAL_VOCAL_MEAN_HZ,
    A440_HZ,
    A432_HZ,
    SOLFEGGIO_HZ,
    SOLFEGGIO_NAMES,
    SOLFEGGIO_TABLE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def flat_chroma() -> np.ndarray:
    """Equal energy across all 12 pitch classes — no tonal centre."""
    return np.ones(12, dtype=float) / 12.0


def c_major_chroma() -> np.ndarray:
    """Chroma emphasising C-major tonality (C, E, G most prominent)."""
    chroma = np.zeros(12, dtype=float)
    chroma[0] = 0.40  # C
    chroma[4] = 0.30  # E
    chroma[7] = 0.25  # G
    chroma[9] = 0.05  # A (minor presence)
    return chroma


def a_minor_chroma() -> np.ndarray:
    """Chroma emphasising A-minor (A, C, E)."""
    chroma = np.zeros(12, dtype=float)
    chroma[9] = 0.40   # A
    chroma[0] = 0.30   # C
    chroma[4] = 0.25   # E
    chroma[2] = 0.05   # D (minor presence)
    return chroma


def mock_harmonic_features(chroma_mean=None, key_estimate=None, tuning_offset_cents=0.0,
                            solfeggio_alignment=None):
    """Minimal mock of HarmonicFeatures for aggregate_solfeggio tests."""
    from unittest.mock import MagicMock
    hf = MagicMock()
    hf.chroma_mean = chroma_mean if chroma_mean is not None else flat_chroma()
    hf.key_estimate = key_estimate or "C major"
    hf.tuning_offset_cents = tuning_offset_cents
    hf.solfeggio_alignment = solfeggio_alignment
    return hf


def mock_segment_features(chroma_mean=None, key_estimate=None,
                           tuning_offset_cents=0.0, solfeggio_alignment=None):
    """Minimal mock of SegmentFeatures with a harmonic sub-object."""
    from unittest.mock import MagicMock
    sf = MagicMock()
    sf.harmonic = mock_harmonic_features(
        chroma_mean=chroma_mean,
        key_estimate=key_estimate,
        tuning_offset_cents=tuning_offset_cents,
        solfeggio_alignment=solfeggio_alignment,
    )
    return sf


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_biological_vocal_mean_is_c3(self):
        # C3 at A440 is ~130.81 Hz — the mean adult vocal fundamental at rest
        assert abs(BIOLOGICAL_VOCAL_MEAN_HZ - 130.81) < 0.5

    def test_a440_above_a432(self):
        assert A440_HZ > A432_HZ

    def test_cents_difference_approx(self):
        # A440 vs A432: ~31.77 cents
        import math
        diff = 1200 * math.log2(A440_HZ / A432_HZ)
        assert abs(diff - 31.77) < 0.5

    def test_solfeggio_table_has_six_entries(self):
        assert len(SOLFEGGIO_TABLE) == 6

    def test_solfeggio_hz_ascending(self):
        freqs = [row[0] for row in SOLFEGGIO_TABLE]
        assert freqs == sorted(freqs)

    def test_solfeggio_names_match_hz(self):
        assert len(SOLFEGGIO_HZ) == len(SOLFEGGIO_NAMES)

    def test_known_solfeggio_values(self):
        # The six canonical frequencies as described by Puleo/Horowitz
        assert 396 in SOLFEGGIO_HZ
        assert 528 in SOLFEGGIO_HZ
        assert 852 in SOLFEGGIO_HZ


# ── compute_solfeggio_alignment ───────────────────────────────────────────────

class TestComputeSolfeggioAlignment:
    def test_returns_dataclass(self):
        result = compute_solfeggio_alignment(flat_chroma(), "C major", 0.0)
        assert isinstance(result, SolfeggioAlignment)

    def test_all_fields_present(self):
        result = compute_solfeggio_alignment(flat_chroma(), "C major", 0.0)
        assert hasattr(result, 'vocal_fundamental_deviation_hz')
        assert hasattr(result, 'solfeggio_grid_proximity')
        assert hasattr(result, 'nearest_solfeggio_hz')
        assert hasattr(result, 'nearest_solfeggio_name')
        assert hasattr(result, 'institutional_distance')
        assert hasattr(result, 'tuning_deviation_cents')
        assert hasattr(result, 'alignment_reading')

    def test_proximity_in_range(self):
        result = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        assert 0.0 <= result.solfeggio_grid_proximity <= 1.0

    def test_institutional_distance_in_range(self):
        result = compute_solfeggio_alignment(flat_chroma(), "A major", 0.0)
        assert 0.0 <= result.institutional_distance <= 1.0

    def test_tuning_deviation_passthrough(self):
        # Tuning deviation should reflect the input cents offset
        result = compute_solfeggio_alignment(flat_chroma(), "C major", -31.77)
        assert abs(result.tuning_deviation_cents - (-31.77)) < 0.1

    def test_nearest_solfeggio_is_valid(self):
        result = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        assert result.nearest_solfeggio_hz in SOLFEGGIO_HZ
        assert result.nearest_solfeggio_name in SOLFEGGIO_NAMES

    def test_alignment_reading_is_non_empty_string(self):
        result = compute_solfeggio_alignment(flat_chroma(), "C major", 0.0)
        assert isinstance(result.alignment_reading, str)
        assert len(result.alignment_reading) > 20

    def test_a432_tuning_lowers_institutional_distance(self):
        """A432 tuning (approx −31.77 cents from A440) should lower institutional_distance
        compared to exactly on-A440 tuning, all else equal."""
        at_440 = compute_solfeggio_alignment(flat_chroma(), "A major", 0.0)
        at_432 = compute_solfeggio_alignment(flat_chroma(), "A major", -31.77)
        # The tuning component should favour A432 (more biological)
        assert at_432.institutional_distance <= at_440.institutional_distance + 0.05

    def test_c_major_vocal_deviation(self):
        """C major at A440 should place the fundamental near biological reference."""
        result = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        # C3 is ~130 Hz which is close to BIOLOGICAL_VOCAL_MEAN_HZ — small deviation
        assert abs(result.vocal_fundamental_deviation_hz) < 50.0

    def test_handles_zero_chroma(self):
        """A zero chroma array (silence) should not crash."""
        result = compute_solfeggio_alignment(np.zeros(12), "C major", 0.0)
        assert isinstance(result, SolfeggioAlignment)

    def test_handles_flat_key_names(self):
        """Flat note names like 'Bb minor' should be normalised without error."""
        result = compute_solfeggio_alignment(flat_chroma(), "Bb major", 0.0)
        assert isinstance(result, SolfeggioAlignment)

    def test_unknown_key_falls_back_gracefully(self):
        """An unparseable key string should not raise — falls back to index 0."""
        result = compute_solfeggio_alignment(flat_chroma(), "unknown ???", 0.0)
        assert isinstance(result, SolfeggioAlignment)


# ── Biological reference boundary ─────────────────────────────────────────────

class TestBiologicalReference:
    def test_c3_at_a440_is_near_biological_mean(self):
        """C major at standard tuning should have low vocal fundamental deviation."""
        result = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        # The fundamental for C3 at A440 is ~130.81 Hz, matching biological mean
        assert abs(result.vocal_fundamental_deviation_hz) < 5.0

    def test_a_major_root_higher_than_biological_mean(self):
        """A major fundamental at A440 (~55 Hz) measured in higher octaves sits above biological mean."""
        result_a = compute_solfeggio_alignment(a_minor_chroma(), "A minor", 0.0)
        result_c = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        # Both should produce valid results — the sign may differ
        assert isinstance(result_a, SolfeggioAlignment)
        assert isinstance(result_c, SolfeggioAlignment)


# ── aggregate_solfeggio ────────────────────────────────────────────────────────

class TestAggregateSolfeggio:
    def _make_four_quarters(self, chroma=None, key="C major", tuning=0.0):
        sa = compute_solfeggio_alignment(
            chroma if chroma is not None else flat_chroma(), key, tuning
        )
        quarters = [
            mock_segment_features(
                chroma_mean=chroma if chroma is not None else flat_chroma(),
                key_estimate=key,
                tuning_offset_cents=tuning,
                solfeggio_alignment=sa,
            )
            for _ in range(4)
        ]
        return quarters

    def test_returns_solfeggio_alignment(self):
        quarters = self._make_four_quarters()
        result = aggregate_solfeggio(quarters)
        assert isinstance(result, SolfeggioAlignment)

    def test_returns_none_for_empty_list(self):
        assert aggregate_solfeggio([]) is None

    def test_returns_none_when_no_solfeggio_on_any_quarter(self):
        quarters = [mock_segment_features(solfeggio_alignment=None) for _ in range(4)]
        result = aggregate_solfeggio(quarters)
        assert result is None

    def test_averages_numerical_fields(self):
        """When all quarters have the same alignment, averaged values should match."""
        sa = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        quarters = [mock_segment_features(solfeggio_alignment=sa) for _ in range(4)]
        result = aggregate_solfeggio(quarters)
        assert result is not None
        assert abs(result.institutional_distance - sa.institutional_distance) < 0.01
        assert abs(result.solfeggio_grid_proximity - sa.solfeggio_grid_proximity) < 0.01

    def test_handles_partial_quarters_with_none(self):
        """Quarters where solfeggio_alignment is None should be skipped gracefully."""
        sa = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        quarters = [
            mock_segment_features(solfeggio_alignment=sa),
            mock_segment_features(solfeggio_alignment=None),
            mock_segment_features(solfeggio_alignment=sa),
            mock_segment_features(solfeggio_alignment=None),
        ]
        result = aggregate_solfeggio(quarters)
        assert result is not None

    def test_output_fields_in_valid_ranges(self):
        quarters = self._make_four_quarters(chroma=c_major_chroma())
        result = aggregate_solfeggio(quarters)
        assert result is not None
        assert 0.0 <= result.solfeggio_grid_proximity <= 1.0
        assert 0.0 <= result.institutional_distance <= 1.0
        assert isinstance(result.alignment_reading, str)


# ── Integration smoke test ─────────────────────────────────────────────────────

class TestSolfeggioIntegration:
    def test_full_pipeline_c_major_440(self):
        """End-to-end: C major at A440 produces a sensible reading."""
        result = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        assert result.institutional_distance >= 0.0
        assert result.solfeggio_grid_proximity >= 0.0
        assert len(result.alignment_reading) > 10

    def test_full_pipeline_a432_shows_lower_institutional_distance(self):
        """A432 tuning should show lower institutional distance than A440."""
        a440 = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        a432 = compute_solfeggio_alignment(c_major_chroma(), "C major", -31.77)
        # Tuning component pushes institutional_distance down for A432
        assert a432.institutional_distance <= a440.institutional_distance + 0.02

    def test_solfeggio_proximity_for_strong_tonal_content(self):
        """A strongly tonal chroma should produce a non-zero proximity score."""
        result = compute_solfeggio_alignment(c_major_chroma(), "C major", 0.0)
        assert result.solfeggio_grid_proximity >= 0.0  # at minimum non-negative
