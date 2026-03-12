"""
Tests for core/divergence.py — LSII computation and trajectory analysis.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {name}. Run tests/generate_fixtures.py first.")
    return path


from core.ingestion import load
from core.segmentation import segment
from core.feature_extractor import extract
from core.divergence import compute_trajectory, LatesonginversionResult, TrajectoryProfile


class TestDivergence:

    def _get_trajectory(self, fixture_name: str) -> TrajectoryProfile:
        path = _fixture(fixture_name)
        record = load(path)
        seg_result = segment(record)
        features = [extract(s, s.sample_rate) for s in seg_result.quarters]
        return compute_trajectory(features)

    def test_returns_trajectory_profile(self):
        traj = self._get_trajectory('sine_60s.wav')
        assert isinstance(traj, TrajectoryProfile)

    def test_lsii_result_in_trajectory(self):
        traj = self._get_trajectory('sine_60s.wav')
        assert hasattr(traj, 'lsii_result')
        assert isinstance(traj.lsii_result, LatesonginversionResult)

    def test_lsii_score_range(self):
        traj = self._get_trajectory('sine_60s.wav')
        assert 0.0 <= traj.lsii_result.lsii <= 1.0

    def test_consistent_track_has_low_lsii(self):
        """A pure sine wave with no Q4 variation should have low LSII."""
        traj = self._get_trajectory('sine_60s.wav')
        assert traj.lsii_result.lsii < 0.4, \
            f"Expected low LSII for consistent sine, got {traj.lsii_result.lsii:.3f}"

    def test_lsii_test_fixture_has_elevated_lsii(self):
        """The LSII test track has a different Q4 — should register divergence."""
        traj = self._get_trajectory('lsii_test.wav')
        # Q4 is at 880Hz, 0.2 amplitude vs 440Hz, 0.5 amplitude in Q1-Q3
        # This should produce a measurable LSII score
        assert traj.lsii_result.lsii > 0.1, \
            f"Expected elevated LSII for divergent Q4, got {traj.lsii_result.lsii:.3f}"

    def test_flag_level_is_valid_string(self):
        traj = self._get_trajectory('sine_60s.wav')
        valid_flags = {'none', 'low', 'moderate', 'high', 'extreme'}
        assert traj.lsii_result.flag_level.lower() in valid_flags, \
            f"Unexpected flag level: {traj.lsii_result.flag_level}"

    def test_direction_is_string(self):
        traj = self._get_trajectory('lsii_test.wav')
        assert isinstance(traj.lsii_result.direction, str)
        assert len(traj.lsii_result.direction) > 0

    def test_trajectory_has_four_quarters(self):
        traj = self._get_trajectory('sine_60s.wav')
        assert len(traj.quarters) == 4

    def test_dominant_axis_is_string(self):
        traj = self._get_trajectory('lsii_test.wav')
        assert isinstance(traj.lsii_result.dominant_axis, str)

    def test_divergence_vector_fields(self):
        traj = self._get_trajectory('lsii_test.wav')
        dv = traj.lsii_result.divergence
        assert hasattr(dv, 'spectral_centroid_delta')
        assert hasattr(dv, 'dynamic_energy_delta')
        assert hasattr(dv, 'harmonic_tension_delta')

    def test_compressed_track_lsii(self):
        """Compressed track should return valid LSII without error."""
        traj = self._get_trajectory('compressed.wav')
        assert 0.0 <= traj.lsii_result.lsii <= 1.0
